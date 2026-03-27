# risk_manager.py
"""
Pre-trade risk manager with dynamic position sizing.

Dynamic sizing tiers (score-based multiplier on base risk %):
    score < 0.85  → 0.5x  (half size — lower conviction)
    0.85–0.90     → 1.0x  (full size)
    0.90–0.95     → 1.25x (high conviction)
    0.95+         → 1.5x  (max conviction)

All limits configurable via .env.
"""

import os
from dataclasses import dataclass
from typing import Dict, Tuple

PAPER_MODE             = os.getenv("PAPER_MODE", "true").lower() == "true"
RISK_PCT_PER_TRADE     = float(os.getenv("RISK_PCT_PER_TRADE",   "1.0"))
MAX_OPEN_POSITIONS     = int(os.getenv("MAX_OPEN_POSITIONS",     "3"))
DAILY_LOSS_LIMIT_R     = float(os.getenv("DAILY_LOSS_LIMIT_R",   "3.0"))
MAX_DD_PCT             = float(os.getenv("MAX_DD_PCT",           "10.0"))
MIN_SIGNAL_CONFIDENCE  = float(os.getenv("MIN_SIGNAL_CONFIDENCE","0.80"))
MIN_SIGNAL_SCORE       = float(os.getenv("MIN_SIGNAL_SCORE",     "0.85"))
STARTING_BALANCE       = float(os.getenv("STARTING_BALANCE",     "1000.0"))

# Dynamic sizing multipliers per score bucket
SIZING_TIERS = [
    (0.95, 1.50),   # score >= 0.95 → 1.5x
    (0.90, 1.25),   # score >= 0.90 → 1.25x
    (0.85, 1.00),   # score >= 0.85 → 1.0x
    (0.00, 0.50),   # score >= 0.00 → 0.5x (fallback)
]


def score_to_multiplier(score: float) -> float:
    for threshold, mult in SIZING_TIERS:
        if score >= threshold:
            return mult
    return 0.5


@dataclass
class RiskDecision:
    approved:         bool
    reason:           str
    size_usd:         float = 0.0
    risk_usd:         float = 0.0
    size_multiplier:  float = 1.0   # actual multiplier applied


class RiskManager:
    """
    Stateful risk manager. Integrates with StrategyFilter for kill-switch checks.
    """

    def __init__(self, strategy_filter=None):
        self.balance:         float = STARTING_BALANCE
        self.high_water_mark: float = STARTING_BALANCE
        self.daily_r:         float = 0.0
        self.daily_date:      str   = ""
        self.open_positions:  Dict[str, object] = {}
        self.halted:          bool  = False
        self.halt_reason:     str   = ""
        self.strategy_filter  = strategy_filter   # optional StrategyFilter

    def _refresh_daily(self):
        from datetime import datetime, timezone
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self.daily_date:
            self.daily_r    = 0.0
            self.daily_date = today

    def _compute_size(self, entry: float, stop: float,
                      side: str, score: float) -> Tuple[float, float, float]:
        """
        Returns (size_usd, risk_usd, multiplier).
        Applies score-based dynamic sizing on top of base RISK_PCT_PER_TRADE.
        """
        if entry <= 0:
            return 0.0, 0.0, 1.0
        stop_pct = abs(entry - stop) / entry
        if stop_pct <= 0:
            return 0.0, 0.0, 1.0

        multiplier = score_to_multiplier(score)
        effective_risk_pct = RISK_PCT_PER_TRADE * multiplier
        risk_usd = self.balance * (effective_risk_pct / 100.0)
        size_usd = risk_usd / stop_pct

        return round(size_usd, 2), round(risk_usd, 2), multiplier

    def check_signal(self, signal) -> RiskDecision:
        self._refresh_daily()

        if self.halted:
            return RiskDecision(False, f"HALTED: {self.halt_reason}")

        confidence  = float(getattr(signal, "confidence", 0.0))
        meta        = signal.meta or {}
        total_score = float(meta.get("total_score", 0.0))
        setup_family= meta.get("setup_family", meta.get("regime_local", ""))
        htf_regime  = meta.get("regime_htf_1h", "")

        if confidence < MIN_SIGNAL_CONFIDENCE:
            return RiskDecision(False,
                f"confidence {confidence:.2f} < min {MIN_SIGNAL_CONFIDENCE}")

        if total_score < MIN_SIGNAL_SCORE:
            return RiskDecision(False,
                f"score {total_score:.2f} < min {MIN_SIGNAL_SCORE}")

        # Strategy kill-switch check
        if self.strategy_filter:
            allowed, reason = self.strategy_filter.is_allowed(
                signal.coin, setup_family, htf_regime
            )
            if not allowed:
                return RiskDecision(False, f"strategy_filter: {reason}")

        if len(self.open_positions) >= MAX_OPEN_POSITIONS:
            return RiskDecision(False,
                f"max positions ({MAX_OPEN_POSITIONS}) reached")

        if signal.coin in self.open_positions:
            return RiskDecision(False, f"already open on {signal.coin}")

        if self.daily_r <= -DAILY_LOSS_LIMIT_R:
            self.halted      = True
            self.halt_reason = f"daily loss limit -{DAILY_LOSS_LIMIT_R}R hit"
            return RiskDecision(False, self.halt_reason)

        dd_pct = (self.high_water_mark - self.balance) / self.high_water_mark * 100
        if dd_pct >= MAX_DD_PCT:
            self.halted      = True
            self.halt_reason = f"max drawdown {dd_pct:.1f}% >= {MAX_DD_PCT}%"
            return RiskDecision(False, self.halt_reason)

        size_usd, risk_usd, multiplier = self._compute_size(
            signal.entry_price, signal.stop_price, signal.side, total_score
        )
        if size_usd <= 0:
            return RiskDecision(False, "could not compute valid position size")

        mode = "PAPER" if PAPER_MODE else "LIVE"
        return RiskDecision(
            approved=True,
            reason=(f"approved [{mode}] "
                    f"conf={confidence:.2f} score={total_score:.2f} "
                    f"size={multiplier:.2f}x"),
            size_usd=size_usd,
            risk_usd=risk_usd,
            size_multiplier=multiplier,
        )

    def record_open(self, position):
        self.open_positions[position.coin] = position

    def record_partial(self, position, r_gained: float):
        """FIX: only account for the 50% partial slice."""
        pnl = r_gained * position.r_value * 0.5
        self.daily_r += r_gained * 0.5
        self.balance += pnl
        self._update_hwm()

    def record_close(self, position, runner_r: float):
        """
        FIX: runner_r is the R on the remaining 50% runner only.
        Partial was already accounted for in record_partial.
        """
        pnl = runner_r * position.r_value * 0.5
        self.daily_r += runner_r * 0.5
        self.balance += pnl
        self._update_hwm()
        self.open_positions.pop(position.coin, None)

    def record_full_loss(self, position):
        """
        Called when stopped out before any partial — full -1R loss.
        """
        pnl = -1.0 * position.r_value
        self.daily_r += -1.0
        self.balance += pnl
        self._update_hwm()
        self.open_positions.pop(position.coin, None)

    def _update_hwm(self):
        if self.balance > self.high_water_mark:
            self.high_water_mark = self.balance

    def reset_daily_halt(self):
        self._refresh_daily()
        if self.halted and "daily loss" in self.halt_reason:
            self.halted      = False
            self.halt_reason = ""

    def status_summary(self) -> str:
        self._refresh_daily()
        dd_pct = (self.high_water_mark - self.balance) / self.high_water_mark * 100
        return (
            f"Balance: ${self.balance:.2f}  HWM: ${self.high_water_mark:.2f}  "
            f"DD: {dd_pct:.1f}%  Daily R: {self.daily_r:+.2f}R  "
            f"Open: {len(self.open_positions)}  Halted: {self.halted}"
        )
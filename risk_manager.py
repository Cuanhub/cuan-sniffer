"""
Pre-trade risk manager — live-capital version.

Active-path accounting support:
- entry fee deduction
- funding debit/credit while open
- exit fee handling on partial / close / full loss
- all balance effects reflected in daily_r using fee/funding translated to R
"""

import os
from dataclasses import dataclass
from typing import Dict, Tuple

PAPER_MODE = os.getenv("PAPER_MODE", "true").lower() == "true"

RISK_PCT_PER_TRADE = float(os.getenv("RISK_PCT_PER_TRADE", "1.00"))

MAX_OPEN_POSITIONS = int(os.getenv("MAX_OPEN_POSITIONS", "4"))
DAILY_LOSS_LIMIT_R = float(os.getenv("DAILY_LOSS_LIMIT_R", "3.0"))
MAX_DD_PCT = float(os.getenv("MAX_DD_PCT", "15.0"))

MIN_SIGNAL_CONFIDENCE = float(os.getenv("MIN_SIGNAL_CONFIDENCE", "0.85"))
MIN_SIGNAL_SCORE = float(os.getenv("MIN_SIGNAL_SCORE", "0.85"))

STARTING_BALANCE = float(os.getenv("STARTING_BALANCE", "1000.0"))
MAX_FULL_LOSS_R = float(os.getenv("MAX_FULL_LOSS_R", "-1.5"))

CONFIDENCE_SIZING_TIERS = [
    (0.95, 1.25),
    (0.90, 1.10),
    (0.85, 1.00),
    (0.75, 0.85),
    (0.65, 0.70),
    (0.60, 0.50),
]


def confidence_to_multiplier(confidence: float) -> float:
    for threshold, mult in CONFIDENCE_SIZING_TIERS:
        if confidence >= threshold:
            return mult
    return 0.0


@dataclass
class RiskDecision:
    approved: bool
    reason: str
    size_usd: float = 0.0
    risk_usd: float = 0.0
    size_multiplier: float = 1.0


class RiskManager:
    def __init__(self, strategy_filter=None):
        self.balance: float = STARTING_BALANCE
        self.high_water_mark: float = STARTING_BALANCE
        self.daily_r: float = 0.0
        self.daily_date: str = ""
        self.open_positions: Dict[str, object] = {}
        self.halted: bool = False
        self.halt_reason: str = ""
        self.strategy_filter = strategy_filter

    def _refresh_daily(self):
        from datetime import datetime, timezone
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self.daily_date:
            self.daily_r = 0.0
            self.daily_date = today

    def _usd_to_r(self, position, usd_amount: float) -> float:
        r_value = float(getattr(position, "r_value", 0.0) or 0.0)
        if r_value <= 0:
            return 0.0
        return usd_amount / r_value

    def _compute_size(
        self,
        entry: float,
        stop: float,
        multiplier: float,
    ) -> Tuple[float, float]:
        if entry <= 0:
            return 0.0, 0.0

        stop_pct = abs(entry - stop) / entry
        if stop_pct <= 0:
            return 0.0, 0.0

        if multiplier <= 0:
            return 0.0, 0.0

        effective_risk_pct = RISK_PCT_PER_TRADE * multiplier
        risk_usd = self.balance * (effective_risk_pct / 100.0)
        size_usd = risk_usd / stop_pct

        return round(size_usd, 2), round(risk_usd, 2)

    def check_signal(self, signal) -> RiskDecision:
        self._refresh_daily()

        if self.halted:
            return RiskDecision(False, f"HALTED: {self.halt_reason}")

        confidence = float(getattr(signal, "confidence", 0.0))
        meta = signal.meta or {}
        total_score = float(meta.get("total_score", 0.0))
        setup_family = meta.get("setup_family", meta.get("regime_local", ""))

        if self.strategy_filter is None:
            print(
                "[RISK] WARNING: strategy_filter is None — session blocking inactive. "
                "Ensure Executor passes a StrategyFilter instance."
            )

        if confidence < MIN_SIGNAL_CONFIDENCE:
            return RiskDecision(
                False,
                f"confidence {confidence:.2f} < min {MIN_SIGNAL_CONFIDENCE:.2f}",
            )

        if total_score < MIN_SIGNAL_SCORE:
            return RiskDecision(
                False,
                f"score {total_score:.2f} < min {MIN_SIGNAL_SCORE:.2f}",
            )

        if self.strategy_filter:
            allowed, reason = self.strategy_filter.is_allowed(
                signal.coin,
                signal.side,
                setup_family,
            )
            if not allowed:
                return RiskDecision(False, f"strategy_filter: {reason}")

        if len(self.open_positions) >= MAX_OPEN_POSITIONS:
            return RiskDecision(False, f"max positions ({MAX_OPEN_POSITIONS}) reached")

        if signal.coin in self.open_positions:
            return RiskDecision(False, f"already open on {signal.coin}")

        if self.daily_r <= -DAILY_LOSS_LIMIT_R:
            self.halted = True
            self.halt_reason = f"daily loss limit -{DAILY_LOSS_LIMIT_R}R hit"
            return RiskDecision(False, self.halt_reason)

        dd_pct = (
            (self.high_water_mark - self.balance) / self.high_water_mark * 100
            if self.high_water_mark > 0 else 0.0
        )
        if dd_pct >= MAX_DD_PCT:
            self.halted = True
            self.halt_reason = f"max drawdown {dd_pct:.1f}% >= {MAX_DD_PCT:.1f}%"
            return RiskDecision(False, self.halt_reason)

        multiplier = confidence_to_multiplier(confidence)
        if multiplier <= 0:
            return RiskDecision(
                False,
                f"confidence {confidence:.2f} too low for any sizing tier",
            )

        size_usd, risk_usd = self._compute_size(
            signal.entry_price,
            signal.stop_price,
            multiplier,
        )
        if size_usd <= 0 or risk_usd <= 0:
            return RiskDecision(False, "could not compute valid position size")

        mode = "PAPER" if PAPER_MODE else "LIVE"
        return RiskDecision(
            approved=True,
            reason=(
                f"approved [{mode}] "
                f"conf={confidence:.2f} "
                f"score={total_score:.2f} "
                f"size={multiplier:.2f}x"
            ),
            size_usd=size_usd,
            risk_usd=risk_usd,
            size_multiplier=multiplier,
        )

    def record_open(self, position):
        self.open_positions[position.coin] = position

    def apply_entry_fee(self, position, fee_usd: float):
        if fee_usd == 0:
            return
        self.balance -= fee_usd
        self.daily_r -= self._usd_to_r(position, fee_usd)

    def apply_funding(self, position, funding_usd: float):
        """
        Positive funding_usd = cost. Negative = credit.
        """
        if funding_usd == 0:
            return
        self.balance -= funding_usd
        self.daily_r -= self._usd_to_r(position, funding_usd)

    def record_partial(self, position, gross_r: float, fee_usd: float = 0.0):
        gross_pnl = gross_r * position.r_value * 0.5
        net_pnl = gross_pnl - fee_usd

        self.daily_r += gross_r * 0.5 - self._usd_to_r(position, fee_usd)
        self.balance += net_pnl
        self._update_hwm()

    def record_close(self, position, runner_r: float, fee_usd: float = 0.0):
        gross_pnl = runner_r * position.r_value * 0.5
        net_pnl = gross_pnl - fee_usd

        self.daily_r += runner_r * 0.5 - self._usd_to_r(position, fee_usd)
        self.balance += net_pnl
        self._update_hwm()
        self.open_positions.pop(position.coin, None)

    def record_full_loss(self, position, realized_r: float = -1.0, fee_usd: float = 0.0):
        if realized_r < MAX_FULL_LOSS_R:
            print(
                f"[RISK] WARNING: record_full_loss received realized_r={realized_r:.3f} "
                f"for {getattr(position, 'coin', '?')} — clamped to {MAX_FULL_LOSS_R}. "
                f"Check paper_trader._close_full_loss for R calculation drift."
            )
            realized_r = MAX_FULL_LOSS_R

        gross_pnl = realized_r * position.r_value
        net_pnl = gross_pnl - fee_usd

        self.daily_r += realized_r - self._usd_to_r(position, fee_usd)
        self.balance += net_pnl
        self._update_hwm()
        self.open_positions.pop(position.coin, None)

    def record_cancelled(self, position):
        self.open_positions.pop(position.coin, None)

    def _update_hwm(self):
        if self.balance > self.high_water_mark:
            self.high_water_mark = self.balance

    def reset_daily_halt(self):
        self._refresh_daily()
        if self.halted and "daily loss" in self.halt_reason:
            self.halted = False
            self.halt_reason = ""

    def status_summary(self) -> str:
        self._refresh_daily()
        dd_pct = (
            (self.high_water_mark - self.balance) / self.high_water_mark * 100
            if self.high_water_mark > 0 else 0.0
        )
        return (
            f"Balance: ${self.balance:.2f}  "
            f"HWM: ${self.high_water_mark:.2f}  "
            f"DD: {dd_pct:.1f}%  "
            f"Daily R: {self.daily_r:+.2f}R  "
            f"Open: {len(self.open_positions)}  "
            f"Halted: {self.halted}"
        )
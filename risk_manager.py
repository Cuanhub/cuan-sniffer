"""
Pre-trade risk manager — live-capital version.

v2 — Audited.

BUG FIX: strategy_filter.is_allowed() was called with wrong argument
order: (coin, side, setup_family) but signature is (coin, setup_family,
htf_regime). This meant the filter was tracking side strings as setup
families and setup families as HTF regimes — all kill-switch logic was
operating on garbage keys.

Fixed call now passes: (coin, setup_family, htf_regime).

Active-path accounting (unchanged):
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
MAX_OPEN_POSITIONS_INTRADAY = int(
    os.getenv("MAX_OPEN_POSITIONS_INTRADAY", str(MAX_OPEN_POSITIONS))
)
MAX_OPEN_POSITIONS_SWING = int(
    os.getenv("MAX_OPEN_POSITIONS_SWING", str(MAX_OPEN_POSITIONS))
)

RISK_PCT_MULT_INTRADAY = float(os.getenv("RISK_PCT_MULT_INTRADAY", "1.0"))
RISK_PCT_MULT_SWING = float(os.getenv("RISK_PCT_MULT_SWING", "1.0"))

DAILY_LOSS_LIMIT_R = float(os.getenv("DAILY_LOSS_LIMIT_R", "3.0"))
MAX_DD_PCT = float(os.getenv("MAX_DD_PCT", "15.0"))

MIN_SIGNAL_CONFIDENCE = float(os.getenv("MIN_SIGNAL_CONFIDENCE", "0.64"))
MIN_SIGNAL_SCORE = float(os.getenv("MIN_SIGNAL_SCORE", "0.68"))

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
    track: str = "intraday"


class RiskManager:
    def __init__(self, strategy_filter=None):
        self.balance: float = STARTING_BALANCE
        self.ledger_balance: float = STARTING_BALANCE
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

    @staticmethod
    def _normalize_close_fraction(close_fraction: float, default: float) -> float:
        try:
            frac = float(close_fraction)
        except (TypeError, ValueError):
            frac = float(default)
        if frac <= 0:
            frac = float(default)
        return max(0.0, min(1.0, frac))

    def _compute_size(
        self,
        entry: float,
        stop: float,
        multiplier: float,
        track_mult: float,
    ) -> Tuple[float, float]:
        if entry <= 0:
            return 0.0, 0.0

        stop_pct = abs(entry - stop) / entry
        if stop_pct <= 0:
            return 0.0, 0.0

        if multiplier <= 0:
            return 0.0, 0.0

        effective_risk_pct = RISK_PCT_PER_TRADE * multiplier * track_mult
        risk_usd = self.balance * (effective_risk_pct / 100.0)
        size_usd = risk_usd / stop_pct

        return round(size_usd, 2), round(risk_usd, 2)

    @staticmethod
    def _signal_track(signal) -> str:
        meta = getattr(signal, "meta", None) or {}
        explicit = str(meta.get("execution_track", "")).strip().lower()
        if explicit in {"intraday", "swing"}:
            return explicit

        timeframe = str(meta.get("timeframe", "")).strip().lower()
        setup_family = str(
            meta.get("setup_family", meta.get("regime_local", ""))
        ).strip().lower()

        if timeframe in {"1h", "4h"} or setup_family == "swing":
            return "swing"
        return "intraday"

    @staticmethod
    def _position_track(position) -> str:
        explicit = str(getattr(position, "execution_track", "")).strip().lower()
        if explicit in {"intraday", "swing"}:
            return explicit

        timeframe = str(getattr(position, "timeframe", "")).strip().lower()
        setup_family = str(getattr(position, "setup_family", "")).strip().lower()

        if timeframe in {"1h", "4h"} or setup_family == "swing":
            return "swing"
        return "intraday"

    def check_signal(self, signal) -> RiskDecision:
        self._refresh_daily()
        track = self._signal_track(signal)
        track_mult = RISK_PCT_MULT_SWING if track == "swing" else RISK_PCT_MULT_INTRADAY

        if track_mult <= 0:
            return RiskDecision(False, f"{track} risk budget disabled (multiplier <= 0)", track=track)

        if self.halted:
            return RiskDecision(False, f"HALTED: {self.halt_reason}", track=track)

        confidence = float(getattr(signal, "confidence", 0.0))
        meta = signal.meta or {}
        total_score = float(meta.get("total_score", 0.0))
        setup_family = str(
            meta.get("setup_family", meta.get("regime_local", ""))
        ).strip()
        htf_regime = str(meta.get("regime_htf_1h", "")).strip()

        if self.strategy_filter is None:
            print(
                "[RISK] WARNING: strategy_filter is None — session blocking inactive. "
                "Ensure Executor passes a StrategyFilter instance."
            )

        if confidence < MIN_SIGNAL_CONFIDENCE:
            return RiskDecision(
                False,
                f"confidence {confidence:.2f} < min {MIN_SIGNAL_CONFIDENCE:.2f}",
                track=track,
            )

        # Use engine's regime-adaptive effective_threshold when present (already self-gated by engine).
        # Fall back to MIN_SIGNAL_SCORE for signals without metadata (backward compat / safety).
        _score_floor = float(meta.get("effective_threshold", MIN_SIGNAL_SCORE))
        if total_score < _score_floor:
            _regime = meta.get("market_regime", "unknown")
            return RiskDecision(
                False,
                f"score {total_score:.2f} < threshold {_score_floor:.2f} (regime={_regime})",
                track=track,
            )

        # ── Strategy filter — FIXED argument order ───────────────────
        # Old (BUGGY): is_allowed(coin, side, setup_family)
        # New (FIXED):  is_allowed(coin, setup_family, htf_regime)
        if self.strategy_filter:
            allowed, reason = self.strategy_filter.is_allowed(
                signal.coin,
                setup_family,
                htf_regime,
            )
            if not allowed:
                return RiskDecision(False, f"strategy_filter: {reason}", track=track)

        if len(self.open_positions) >= MAX_OPEN_POSITIONS:
            return RiskDecision(False, f"max positions ({MAX_OPEN_POSITIONS}) reached", track=track)

        open_intraday = sum(
            1 for p in self.open_positions.values()
            if self._position_track(p) == "intraday"
        )
        open_swing = sum(
            1 for p in self.open_positions.values()
            if self._position_track(p) == "swing"
        )

        if track == "intraday" and open_intraday >= MAX_OPEN_POSITIONS_INTRADAY:
            return RiskDecision(
                False,
                f"intraday max positions ({MAX_OPEN_POSITIONS_INTRADAY}) reached",
                track=track,
            )
        if track == "swing" and open_swing >= MAX_OPEN_POSITIONS_SWING:
            return RiskDecision(
                False,
                f"swing max positions ({MAX_OPEN_POSITIONS_SWING}) reached",
                track=track,
            )

        if signal.coin in self.open_positions:
            return RiskDecision(False, f"already open on {signal.coin}", track=track)

        if self.daily_r <= -DAILY_LOSS_LIMIT_R:
            self.halted = True
            self.halt_reason = f"daily loss limit -{DAILY_LOSS_LIMIT_R}R hit"
            return RiskDecision(False, self.halt_reason, track=track)

        dd_pct = (
            (self.high_water_mark - self.balance) / self.high_water_mark * 100
            if self.high_water_mark > 0 else 0.0
        )
        if dd_pct >= MAX_DD_PCT:
            self.halted = True
            self.halt_reason = f"max drawdown {dd_pct:.1f}% >= {MAX_DD_PCT:.1f}%"
            return RiskDecision(False, self.halt_reason, track=track)

        multiplier = confidence_to_multiplier(confidence)
        if multiplier <= 0:
            return RiskDecision(
                False,
                f"confidence {confidence:.2f} too low for any sizing tier",
                track=track,
            )

        size_usd, risk_usd = self._compute_size(
            signal.entry_price,
            signal.stop_price,
            multiplier,
            track_mult,
        )
        if size_usd <= 0 or risk_usd <= 0:
            return RiskDecision(False, "could not compute valid position size", track=track)

        mode = "PAPER" if PAPER_MODE else "LIVE"
        return RiskDecision(
            approved=True,
            reason=(
                f"approved [{mode}] "
                f"track={track} "
                f"conf={confidence:.2f} "
                f"score={total_score:.2f} "
                f"size={multiplier:.2f}x "
                f"budget={track_mult:.2f}x"
            ),
            size_usd=size_usd,
            risk_usd=risk_usd,
            size_multiplier=multiplier,
            track=track,
        )

    def record_open(self, position):
        self.open_positions[position.coin] = position

    def apply_entry_fee(self, position, fee_usd: float):
        if fee_usd == 0:
            return
        self.balance -= fee_usd
        self.ledger_balance -= fee_usd
        self.daily_r -= self._usd_to_r(position, fee_usd)

    def apply_funding(self, position, funding_usd: float):
        """
        Positive funding_usd = cost. Negative = credit.
        """
        if funding_usd == 0:
            return
        self.balance -= funding_usd
        self.ledger_balance -= funding_usd
        self.daily_r -= self._usd_to_r(position, funding_usd)

    def record_partial(
        self,
        position,
        gross_r: float,
        fee_usd: float = 0.0,
        close_fraction: float = 1.0,
    ):
        frac = self._normalize_close_fraction(close_fraction, default=1.0)
        gross_pnl = gross_r * position.r_value * frac
        net_pnl = gross_pnl - fee_usd

        self.daily_r += gross_r * frac - self._usd_to_r(position, fee_usd)
        self.balance += net_pnl
        self.ledger_balance += net_pnl
        self._update_hwm()

    def record_close(
        self,
        position,
        runner_r: float,
        fee_usd: float = 0.0,
        close_fraction: float = 1.0,
    ):
        frac = self._normalize_close_fraction(close_fraction, default=1.0)
        gross_pnl = runner_r * position.r_value * frac
        net_pnl = gross_pnl - fee_usd

        self.daily_r += runner_r * frac - self._usd_to_r(position, fee_usd)
        self.balance += net_pnl
        self.ledger_balance += net_pnl
        self._update_hwm()
        self.open_positions.pop(position.coin, None)

    def record_full_loss(self, position, realized_r: float = -1.0, fee_usd: float = 0.0):
        if realized_r < MAX_FULL_LOSS_R:
            print(
                f"[RISK] WARNING: record_full_loss received realized_r={realized_r:.3f} "
                f"for {getattr(position, 'coin', '?')} — clamped to {MAX_FULL_LOSS_R}. "
                f"Check paper_trader._close_full_loss for R calculation drift."
            )

        r_value = float(getattr(position, "r_value", 0.0) or 0.0)
        gross_pnl = realized_r * r_value
        min_gross_pnl = MAX_FULL_LOSS_R * r_value
        gross_pnl = max(gross_pnl, min_gross_pnl)
        realized_r_effective = (gross_pnl / r_value) if r_value > 0 else realized_r
        net_pnl = gross_pnl - fee_usd

        self.daily_r += realized_r_effective - self._usd_to_r(position, fee_usd)
        self.balance += net_pnl
        self.ledger_balance += net_pnl
        self._update_hwm()
        self.open_positions.pop(position.coin, None)

    def record_live_full_close(
        self,
        position,
        realized_r: float,
        fee_usd: float = 0.0,
        close_fraction: float = 1.0,
    ):
        """
        Accounting path for a full close in live mode.
        """
        frac = self._normalize_close_fraction(close_fraction, default=1.0)
        if realized_r < MAX_FULL_LOSS_R:
            print(
                f"[RISK] WARNING: record_live_full_close received realized_r={realized_r:.3f} "
                f"for {getattr(position, 'coin', '?')} — clamped to {MAX_FULL_LOSS_R}."
            )

        r_value = float(getattr(position, "r_value", 0.0) or 0.0)
        gross_pnl = realized_r * r_value * frac
        min_gross_pnl = MAX_FULL_LOSS_R * r_value * frac
        gross_pnl = max(gross_pnl, min_gross_pnl)
        gross_r_component = (gross_pnl / r_value) if r_value > 0 else realized_r * frac
        net_pnl = gross_pnl - fee_usd

        self.daily_r += gross_r_component - self._usd_to_r(position, fee_usd)
        self.balance += net_pnl
        self.ledger_balance += net_pnl
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
        open_intraday = sum(
            1 for p in self.open_positions.values()
            if self._position_track(p) == "intraday"
        )
        open_swing = sum(
            1 for p in self.open_positions.values()
            if self._position_track(p) == "swing"
        )
        return (
            f"Balance: ${self.balance:.2f}  "
            f"HWM: ${self.high_water_mark:.2f}  "
            f"DD: {dd_pct:.1f}%  "
            f"Daily R: {self.daily_r:+.2f}R  "
            f"Open: {len(self.open_positions)} "
            f"(intra={open_intraday}/{MAX_OPEN_POSITIONS_INTRADAY}, "
            f"swing={open_swing}/{MAX_OPEN_POSITIONS_SWING})  "
            f"Halted: {self.halted}"
        )

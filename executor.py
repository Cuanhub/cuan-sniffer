"""
Order executor — LIVE-ONLY, wallet-authoritative version.

Design:
- Executor owns normal runtime entry/exit decisions
- Wallet/venue state is the final authority for close confirmation
- LivePositionMonitor reconciles venue-flat positions that may have
  closed outside the normal executor path (restart, venue-side TP/SL, etc.)

Notes:
- No PaperTrader
- No PAPER_MODE branching
- Explicit per-cycle live exit loop
- All live closes require wallet confirmation
"""

import csv
import os
import threading
import time
from math import ceil
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from live_position_monitor import LivePositionMonitor
from order_tracker import OrderTracker, OrderKind, OrderLifecycleState  # TODO: OrderLifecycleState unused here — remove if no external callers depend on re-export
from protection_manager import ProtectionManager
from position import Position, PositionState, CloseReason
from risk_manager import (
    STARTING_BALANCE,
    MAX_OPEN_POSITIONS,
    MAX_OPEN_POSITIONS_INTRADAY,
    MAX_OPEN_POSITIONS_SWING,
    RiskManager,
)
from trade_log import init_trade_log, append_trade
from strategy_filter import StrategyFilter
from bootstrap import bootstrap_state, replay_strategy_filter
from execution_backend import ExecutionBackend
from execution_backend_factory import build_execution_backend

# ── Cooldowns ───────────────────────────────────────────────────────────
SIGNAL_COOLDOWN_SEC = int(os.getenv("SIGNAL_COOLDOWN_SEC", "300"))
COIN_REENTRY_COOLDOWN_SEC = int(os.getenv("COIN_REENTRY_COOLDOWN_SEC", "900"))
BUCKET_REENTRY_COOLDOWN_SEC = int(os.getenv("BUCKET_REENTRY_COOLDOWN_SEC", "300"))

# ── Entry buffer (session-scaled) ──────────────────────────────────────
ENTRY_BUFFER_ATR_MULT = float(os.getenv("ENTRY_BUFFER_ATR_MULT", "5.0"))

_DEFAULT_SESSION_ATR_SCALES = {
    "dead_zone": 0.60,
    "asia_open": 1.00,
    "asia_late": 1.00,
    "london_open": 1.00,
    "london_late": 1.00,
    "london_pm": 1.00,
    "ny_open": 1.20,
    "ny_pm": 1.00,
}


def _load_session_atr_scales() -> Dict[str, float]:
    scales = dict(_DEFAULT_SESSION_ATR_SCALES)
    for session_name in list(scales):
        env_key = f"SESSION_ATR_SCALE_{session_name.upper()}"
        env_val = os.getenv(env_key)
        if env_val is not None:
            try:
                scales[session_name] = float(env_val)
            except ValueError:
                pass
    return scales


SESSION_ATR_BUFFER_SCALE: Dict[str, float] = _load_session_atr_scales()

# ── Move / drift limits ───────────────────────────────────────────────
MAX_MOVE_STOP_FRAC = float(os.getenv("MAX_MOVE_STOP_FRAC", "2.5"))
ADVERSE_DRIFT_DEFAULT = float(os.getenv("ADVERSE_DRIFT_DEFAULT", "0.50"))
ADVERSE_DRIFT_ALIGNED = float(os.getenv("ADVERSE_DRIFT_ALIGNED", "0.75"))
ADVERSE_DRIFT_ALIGNED_STRONG = float(os.getenv("ADVERSE_DRIFT_ALIGNED_STRONG", "0.85"))

# ── Stale signal filter ───────────────────────────────────────────────
# SIGNAL_STALE_ATR_MULT: reject entry when price has moved more than this
# many ATR from the signal's entry_price since the signal was generated.
# Replaces the previously hardcoded MAX_SIGNAL_TO_ENTRY_ATR constant.
# Backward-compat: falls back to MAX_SIGNAL_TO_ENTRY_ATR if set, else 3.0.
SIGNAL_STALE_ATR_MULT = float(
    os.getenv("SIGNAL_STALE_ATR_MULT",
              os.getenv("MAX_SIGNAL_TO_ENTRY_ATR", "3.0"))
)
# SIGNAL_MOMENTUM_REENTRY_ATR: if > 0, allow entry even after a stale rejection
# when price is moving in the signal direction and the move is within this many ATR.
# Useful for strong-trend signals where price runs away then pulls back.
# Default 0.0 = disabled (fully backward compatible).
SIGNAL_MOMENTUM_REENTRY_ATR = float(os.getenv("SIGNAL_MOMENTUM_REENTRY_ATR", "0.0"))
MIN_EXECUTION_EFFECTIVE_RR = float(
    os.getenv("MIN_EXECUTION_EFFECTIVE_RR", os.getenv("MIN_EFFECTIVE_RR", "1.75"))
)

# ── Overextension / trailing / venue flatness ─────────────────────────
MAX_RECENT_MOVE_ATR = float(os.getenv("MAX_RECENT_MOVE_ATR", "1.50"))
TRAIL_ATR_MULT = float(os.getenv("TRAIL_ATR_MULT", "1.0"))
MAX_FULL_LOSS_R = float(os.getenv("MAX_FULL_LOSS_R", "-1.5"))
LIVE_FLAT_EPSILON_SZ = float(os.getenv("LIVE_FLAT_EPSILON_SZ", "1e-9"))
LIVE_EXIT_RETRY_COOLDOWN_SEC = float(os.getenv("LIVE_EXIT_RETRY_COOLDOWN_SEC", "10"))
LIVE_TINY_POSITION_USD = float(os.getenv("LIVE_TINY_POSITION_USD", "5.0"))
LIVE_EXIT_UNKNOWN_MAX_RETRIES = int(os.getenv("LIVE_EXIT_UNKNOWN_MAX_RETRIES", "4"))
LIVE_EXIT_UNKNOWN_BACKOFF_SEC = float(os.getenv("LIVE_EXIT_UNKNOWN_BACKOFF_SEC", "20"))
LIVE_EXIT_UNKNOWN_ESCALATE_EVERY = int(os.getenv("LIVE_EXIT_UNKNOWN_ESCALATE_EVERY", "3"))
AUTO_REPLACE_MISSING_PROTECTION_ON_BOOT = (
    os.getenv("AUTO_REPLACE_MISSING_PROTECTION_ON_BOOT", "true").lower() == "true"
)

# ── Stop construction (entry-time, structure + volatility) ─────────────
STOP_ATR_FLOOR_MULT_INTRADAY = float(os.getenv("STOP_ATR_FLOOR_MULT_INTRADAY", "1.10"))
STOP_ATR_FLOOR_MULT_SWING = float(os.getenv("STOP_ATR_FLOOR_MULT_SWING", "1.40"))
STOP_BUFFER_ATR_MULT = float(os.getenv("STOP_BUFFER_ATR_MULT", "0.10"))
MIN_STOP_ATR_REJECT = float(os.getenv("MIN_STOP_ATR_REJECT", "0.80"))
MIN_STOP_REDESIGN_RR = float(os.getenv("MIN_STOP_REDESIGN_RR", "1.8"))

# ── Continuation cap ──────────────────────────────────────────────────
CONTINUATION_MAX_SIZE_MULT = float(os.getenv("CONTINUATION_MAX_SIZE_MULT", "1.50"))

# ── Market regime gating/sizing ───────────────────────────────────────
BLOCK_CONTINUATION_IN_CHOP = (
    os.getenv("BLOCK_CONTINUATION_IN_CHOP", "true").lower() == "true"
)
# Block reversal entries when BOTH HTF and macro are opposed to the signal direction.
# Reversal longs in htf_down+macro_down and reversal shorts in htf_up+macro_up have
# negative expectancy in trending markets — structure is stacked against them.
BLOCK_REVERSAL_AGAINST_DUAL_TREND = (
    os.getenv("BLOCK_REVERSAL_AGAINST_DUAL_TREND", "true").lower() == "true"
)
WEAK_TREND_SIZE_MULT = float(os.getenv("WEAK_TREND_SIZE_MULT", "0.70"))
FULL_TP_MODE = os.getenv("FULL_TP_MODE", "true").lower() == "true"
# Partial-TP runner model: close a fraction at +PARTIAL_TP_R, move stop to BE, run to TP.
# Only active when FULL_TP_MODE=false. Setting ENABLE_PARTIAL_TP=true automatically
# implies FULL_TP_MODE=false for the partial trigger path.
ENABLE_PARTIAL_TP = os.getenv("ENABLE_PARTIAL_TP", "false").lower() == "true"
PARTIAL_TP_R = float(os.getenv("PARTIAL_TP_R", "1.0"))
PARTIAL_CLOSE_FRACTION = float(os.getenv("PARTIAL_CLOSE_FRACTION", "0.4"))
MOVE_STOP_TO_BREAKEVEN_AFTER_PARTIAL = (
    os.getenv("MOVE_STOP_TO_BREAKEVEN_AFTER_PARTIAL", "true").lower() == "true"
)
SCORE_SIZE_BASE_MULT = float(os.getenv("SCORE_SIZE_BASE_MULT", "1.00"))
SCORE_SIZE_MID_THRESHOLD = float(os.getenv("SCORE_SIZE_MID_THRESHOLD", "0.75"))
SCORE_SIZE_MID_MULT = float(os.getenv("SCORE_SIZE_MID_MULT", "1.25"))
SCORE_SIZE_HIGH_THRESHOLD = float(os.getenv("SCORE_SIZE_HIGH_THRESHOLD", "0.90"))
SCORE_SIZE_HIGH_MULT = float(os.getenv("SCORE_SIZE_HIGH_MULT", "1.50"))
CONT_STRONG_TREND_BONUS_MULT = float(os.getenv("CONT_STRONG_TREND_BONUS_MULT", "1.15"))
SCORE_SIZE_OVERLAY_MAX_MULT = float(os.getenv("SCORE_SIZE_OVERLAY_MAX_MULT", "1.50"))

# ── Portfolio replacement (conservative, full-book only) ────────────
ENABLE_POSITION_REPLACEMENT = (
    os.getenv("ENABLE_POSITION_REPLACEMENT", "true").lower() == "true"
)
POSITION_REPLACEMENT_MIN_SCORE_DELTA = float(
    os.getenv("POSITION_REPLACEMENT_MIN_SCORE_DELTA", "0.12")
)
POSITION_REPLACEMENT_PROTECT_PARTIALED = (
    os.getenv("POSITION_REPLACEMENT_PROTECT_PARTIALED", "true").lower() == "true"
)
POSITION_REPLACEMENT_PROTECT_NEAR_TP_R = float(
    os.getenv("POSITION_REPLACEMENT_PROTECT_NEAR_TP_R", "0.8")
)
POSITION_REPLACEMENT_PROTECT_IN_PROFIT_R = float(
    os.getenv("POSITION_REPLACEMENT_PROTECT_IN_PROFIT_R", "0.5")
)
POSITION_REPLACEMENT_PREFERRED_BONUS = float(
    os.getenv("POSITION_REPLACEMENT_PREFERRED_BONUS", "0.05")
)

# ── Session blocking ──────────────────────────────────────────────────
def _parse_session_set(env_name: str, default: str) -> set:
    raw = os.getenv(env_name, default)
    if raw is None:
        return set()
    raw = raw.strip()
    if not raw:
        return set()
    lowered = raw.lower()
    if lowered in {"none", "null", "false", "off", "no"}:
        return set()
    return {s.strip().lower() for s in raw.split(",") if s.strip()}


MAJORS_BLOCKED_SESSIONS = _parse_session_set("MAJORS_BLOCKED_SESSIONS", "dead_zone")
SOL_BETA_BLOCKED_SESSIONS = _parse_session_set("SOL_BETA_BLOCKED_SESSIONS", "dead_zone")
ALT_BETA_BLOCKED_SESSIONS = _parse_session_set("ALT_BETA_BLOCKED_SESSIONS", "dead_zone")
OTHER_BLOCKED_SESSIONS = _parse_session_set("OTHER_BLOCKED_SESSIONS", "dead_zone")
GLOBAL_BLOCKED_SESSIONS = _parse_session_set("BLOCKED_SESSIONS", "")

SOFT_BLOCKED_SESSIONS = _parse_session_set("SOFT_BLOCKED_SESSIONS", "ny_pm")
SESSION_OVERRIDE_MIN_SCORE = float(os.getenv("SESSION_OVERRIDE_MIN_SCORE", "0.80"))
SESSION_OVERRIDE_FAMILIES = {
    s.strip().lower()
    for s in os.getenv("SESSION_OVERRIDE_FAMILIES", "continuation").split(",")
    if s.strip()
}
# When true, swing-timeframe signals (timeframe in {1h, 4h} or setup_family==swing)
# bypass the session soft block entirely without needing to meet SESSION_OVERRIDE_MIN_SCORE.
# Hard blocks (BLOCKED_SESSIONS / per-bucket) are still enforced.
SWING_SESSION_OVERRIDE = os.getenv("SWING_SESSION_OVERRIDE", "false").lower() == "true"

DEAD_ZONE_SOFT_OVERRIDE_ENABLED = (
    os.getenv("DEAD_ZONE_SOFT_OVERRIDE_ENABLED", "true").lower() == "true"
)
DEAD_ZONE_OVERRIDE_MIN_SCORE = float(os.getenv("DEAD_ZONE_OVERRIDE_MIN_SCORE", "0.84"))
DEAD_ZONE_OVERRIDE_FAMILIES = {
    s.strip().lower()
    for s in os.getenv("DEAD_ZONE_OVERRIDE_FAMILIES", "continuation").split(",")
    if s.strip()
}
DEAD_ZONE_OVERRIDE_REQUIRE_TREND_ALIGN = (
    os.getenv("DEAD_ZONE_OVERRIDE_REQUIRE_TREND_ALIGN", "true").lower() == "true"
)

# ── Factor buckets ────────────────────────────────────────────────────
MAJORS_BUCKET_COINS = {
    c.strip().upper()
    for c in os.getenv("FACTOR_BUCKET_MAJORS_COINS", "BTC,ETH").split(",")
    if c.strip()
}
SOL_BETA_BUCKET_COINS = {
    c.strip().upper()
    for c in os.getenv("FACTOR_BUCKET_SOL_BETA_COINS", "SOL,JUP,JTO,WIF,PYTH").split(",")
    if c.strip()
}
ALT_BETA_BUCKET_COINS = {
    c.strip().upper()
    for c in os.getenv("FACTOR_BUCKET_ALT_BETA_COINS", "ARB").split(",")
    if c.strip()
}

_MAJORS_TOTAL = int(os.getenv("MAX_BUCKET_MAJORS", "2"))
_SOL_BETA_TOTAL = int(os.getenv("MAX_BUCKET_SOL_BETA", "2"))
_ALT_BETA_TOTAL = int(os.getenv("MAX_BUCKET_ALT_BETA", "1"))
_OTHER_TOTAL = int(os.getenv("MAX_BUCKET_OTHER", "1"))

MAX_BUCKET_MAJORS_LONG = int(os.getenv("MAX_BUCKET_MAJORS_LONG", str(_MAJORS_TOTAL)))
MAX_BUCKET_MAJORS_SHORT = int(os.getenv("MAX_BUCKET_MAJORS_SHORT", str(_MAJORS_TOTAL)))
MAX_BUCKET_SOL_BETA_LONG = int(os.getenv("MAX_BUCKET_SOL_BETA_LONG", str(_SOL_BETA_TOTAL)))
MAX_BUCKET_SOL_BETA_SHORT = int(os.getenv("MAX_BUCKET_SOL_BETA_SHORT", str(_SOL_BETA_TOTAL)))
MAX_BUCKET_ALT_BETA_LONG = int(os.getenv("MAX_BUCKET_ALT_BETA_LONG", str(_ALT_BETA_TOTAL)))
MAX_BUCKET_ALT_BETA_SHORT = int(os.getenv("MAX_BUCKET_ALT_BETA_SHORT", str(_ALT_BETA_TOTAL)))
MAX_BUCKET_OTHER_LONG = int(os.getenv("MAX_BUCKET_OTHER_LONG", str(_OTHER_TOTAL)))
MAX_BUCKET_OTHER_SHORT = int(os.getenv("MAX_BUCKET_OTHER_SHORT", str(_OTHER_TOTAL)))

INTRADAY_BUCKET_CAP_MULT = float(os.getenv("INTRADAY_BUCKET_CAP_MULT", "1.0"))
SWING_BUCKET_CAP_MULT = float(os.getenv("SWING_BUCKET_CAP_MULT", "1.0"))

# ── Missed-signal logging ─────────────────────────────────────────────
MISSED_LOG_FILE = os.getenv("MISSED_LOG_FILE", "missed_signals.csv")
LOG_MISSED = os.getenv("LOG_MISSED", "true").lower() == "true"
REJECT_MEMORY_TTL_SEC = int(os.getenv("REJECT_MEMORY_TTL_SEC", "300"))  # master enable/disable (0=off)
REJECT_MEMORY_MAX_ITEMS = int(os.getenv("REJECT_MEMORY_MAX_ITEMS", "5000"))
REJECT_MEMORY_FAMILIES = (
    "stale_or_drift",
    "insufficient_margin",
    "session_soft_blocked",
)

# Per-family TTLs — shorter for drift (new candle resets naturally), longer for session/margin.
REJECT_MEMORY_TTL_STALE_OR_DRIFT  = int(os.getenv("REJECT_MEMORY_TTL_STALE_OR_DRIFT",  "120"))
REJECT_MEMORY_TTL_SESSION_BLOCKED = int(os.getenv("REJECT_MEMORY_TTL_SESSION_BLOCKED", "300"))
REJECT_MEMORY_TTL_MARGIN          = int(os.getenv("REJECT_MEMORY_TTL_MARGIN",          "300"))

_REJECT_TTL_BY_FAMILY: Dict[str, int] = {
    "stale_or_drift":       REJECT_MEMORY_TTL_STALE_OR_DRIFT,
    "session_soft_blocked": REJECT_MEMORY_TTL_SESSION_BLOCKED,
    "insufficient_margin":  REJECT_MEMORY_TTL_MARGIN,
}
_REJECT_TTL_MAX = max(_REJECT_TTL_BY_FAMILY.values(), default=REJECT_MEMORY_TTL_SEC)

MISSED_FIELDS = [
    "timestamp",
    "signal_id",
    "coin",
    "side",
    "entry_price",
    "stop_price",
    "tp_price",
    "confidence",
    "total_score",
    "session",
    "regime",
    "reject_reason",
    "current_price",
    "price_move_r",
]


@dataclass
class ExecutorResult:
    traded: bool
    reason: str
    position_id: str = ""
    fill_price: float = 0.0
    fill_slippage_bps: float = 0.0
    fill_ratio: float = 0.0

    def __bool__(self) -> bool:
        return self.traded


class Executor:
    def __init__(
        self,
        notify_fn: Optional[Callable[[str], None]] = None,
        backend: Optional[ExecutionBackend] = None,
        signal_engine=None,
    ):
        self._live_mode = os.getenv("PAPER_MODE", "true").lower() != "true"
        self.venue_sync_unhealthy: bool = False
        self.notify = notify_fn or print
        self.backend = backend or build_execution_backend(debug=True)
        self.signal_engine = signal_engine
        self.order_tracker = OrderTracker()
        self.protection_manager = ProtectionManager(
            backend=self.backend,
            notify_fn=self.notify,
            persist_fn=lambda p: append_trade(p, paper_mode=False),
            order_tracker=self.order_tracker,
        )

        self.strategy_filter = StrategyFilter()
        self.risk = RiskManager(strategy_filter=self.strategy_filter)
        self.live_monitor = LivePositionMonitor(
            backend=self.backend,
            risk_manager=self.risk,
            notify_fn=self.notify,
            order_tracker=self.order_tracker,
            signal_engine=self.signal_engine,
        )

        init_trade_log()

        self._cooldowns: Dict[Tuple[str, str, str], float] = {}
        self._coin_last_fill_ts: Dict[Tuple[str, str, str], float] = {}
        self._bucket_last_fill_ts: Dict[Tuple[str, str, str], float] = {}
        self._session_entry_slippage_bps: List[float] = []
        self._session_exit_slippage_bps: List[float] = []
        self._recent_prices: Dict[str, Deque[Tuple[float, float]]] = {}
        self._pending_positions: Dict[str, bool] = {}
        self._pending_positions_lock = threading.Lock()
        self._reject_memory: Dict[Tuple[str, str, str, str, str, str, str, str, str, str], float] = {}
        self._venue_margin_cache: Optional[Tuple[float, float, float]] = None
        self._venue_margin_cache_ts: float = 0.0
        self._balance_ready: bool = False
        self._venue_equity_hwm: float = 0.0
        self._last_margin_query_meta: Dict[str, Any] = {}

        boot = bootstrap_state(
            STARTING_BALANCE,
            paper_mode=False,
            venue_checker=self._make_venue_checker(),
        )
        replay_strategy_filter(self.strategy_filter, STARTING_BALANCE)

        if hasattr(self.risk, "ledger_balance"):
            self.risk.ledger_balance = boot.realized_balance
        if self._live_mode:
            self.risk.balance = 0.0
            self.risk.high_water_mark = 0.0
            self._venue_equity_hwm = 0.0
        else:
            self.risk.balance = boot.realized_balance
            self.risk.high_water_mark = boot.high_water_mark
            self._venue_equity_hwm = self.risk.balance
        self.risk.daily_r = boot.daily_r

        restored_positions: List[Position] = []
        for pos in boot.open_positions:
            setattr(pos, "allow_reconcile_close", True)
            setattr(pos, "reconciled_from_venue", False)
            setattr(pos, "bootstrap_restored", True)
            self.risk.open_positions[pos.coin] = pos
            self.live_monitor.register(pos, allow_reconcile=True)
            restored_positions.append(pos)
            print(
                f"[BOOTSTRAP] Restored tracking {pos.coin} {pos.side} "
                f"entry={pos.entry_price:.4f} (LIVE — venue-verified, reconcilable)"
            )
            self._audit_bootstrap_protection(pos)

        for (coin, side), opened_ts in boot.open_timestamps.items():
            if opened_ts > 0:
                for track in ("intraday", "swing"):
                    key = (coin, side, track)
                    self._cooldowns[key] = opened_ts
                    self._coin_last_fill_ts[key] = opened_ts

                bucket = self._factor_bucket(coin)
                for track in ("intraday", "swing"):
                    bucket_key = (bucket, side, track)
                    existing = self._bucket_last_fill_ts.get(bucket_key, 0.0)
                    if opened_ts > existing:
                        self._bucket_last_fill_ts[bucket_key] = opened_ts

        _BOOT_BALANCE_MAX_ATTEMPTS = 3
        _BOOT_BALANCE_RETRY_SEC = 5
        for _attempt in range(_BOOT_BALANCE_MAX_ATTEMPTS):
            self._refresh_runtime_balance_from_venue(source="boot")
            if self.risk.balance > 0:
                self._balance_ready = True
                break
            if _attempt < _BOOT_BALANCE_MAX_ATTEMPTS - 1:
                print(
                    f"[EXECUTOR] Boot balance fetch returned 0 "
                    f"(attempt {_attempt + 1}/{_BOOT_BALANCE_MAX_ATTEMPTS}) "
                    f"— retrying in {_BOOT_BALANCE_RETRY_SEC}s"
                )
                time.sleep(_BOOT_BALANCE_RETRY_SEC)
        if not self._balance_ready:
            print(
                f"[EXECUTOR][WARNING] Boot balance could not be confirmed after "
                f"{_BOOT_BALANCE_MAX_ATTEMPTS} attempts "
                f"— signals will be GATED until venue sync recovers"
            )
        self._boot_balance = self.risk.balance
        self._boot_hwm = self.risk.high_water_mark
        self._boot_daily_r = boot.daily_r
        self._boot_daily_pnl = boot.daily_pnl
        self._boot_open = restored_positions
        self._boot_all_time_pnl = boot.all_time_pnl
        self._boot_closed_today = boot.closed_today
        self._boot_total_closed = boot.total_trades

        self._print_boot_banner()

    def _print_boot_banner(self):
        if GLOBAL_BLOCKED_SESSIONS:
            hard_display = f"global:{','.join(sorted(GLOBAL_BLOCKED_SESSIONS))}"
        else:
            hard_display = (
                f"majors={','.join(sorted(MAJORS_BLOCKED_SESSIONS)) or 'none'} "
                f"sol_beta={','.join(sorted(SOL_BETA_BLOCKED_SESSIONS)) or 'none'} "
                f"alt_beta={','.join(sorted(ALT_BETA_BLOCKED_SESSIONS)) or 'none'} "
                f"other={','.join(sorted(OTHER_BLOCKED_SESSIONS)) or 'none'}"
            )
        soft_display = ",".join(sorted(SOFT_BLOCKED_SESSIONS)) or "none"

        print(
            f"[EXECUTOR] LIVE | "
            f"Balance: ${self._boot_balance:.2f} | "
            f"HWM: ${self._boot_hwm:.2f} | "
            f"Open restored: {len(self._boot_open)} | "
            f"Max positions: {MAX_OPEN_POSITIONS} | "
            f"Track max open: intraday={MAX_OPEN_POSITIONS_INTRADAY} "
            f"swing={MAX_OPEN_POSITIONS_SWING} | "
            f"Bucket caps (L/S): "
            f"majors={MAX_BUCKET_MAJORS_LONG}/{MAX_BUCKET_MAJORS_SHORT}, "
            f"sol_beta={MAX_BUCKET_SOL_BETA_LONG}/{MAX_BUCKET_SOL_BETA_SHORT}, "
            f"alt_beta={MAX_BUCKET_ALT_BETA_LONG}/{MAX_BUCKET_ALT_BETA_SHORT}, "
            f"other={MAX_BUCKET_OTHER_LONG}/{MAX_BUCKET_OTHER_SHORT} | "
            f"Track cap mult: intraday={INTRADAY_BUCKET_CAP_MULT:.2f} "
            f"swing={SWING_BUCKET_CAP_MULT:.2f} | "
            f"Cooldown: {SIGNAL_COOLDOWN_SEC}s | "
            f"Coin reentry: {COIN_REENTRY_COOLDOWN_SEC}s | "
            f"Bucket reentry: {BUCKET_REENTRY_COOLDOWN_SEC}s | "
            f"Entry buffer: {ENTRY_BUFFER_ATR_MULT}xATR (session-scaled) | "
            f"Max move: {MAX_MOVE_STOP_FRAC}x stop_dist | "
            f"Drift: default={ADVERSE_DRIFT_DEFAULT} aligned={ADVERSE_DRIFT_ALIGNED} | "
            f"Stop model: floor(intra/swing)={STOP_ATR_FLOOR_MULT_INTRADAY:.2f}/"
            f"{STOP_ATR_FLOOR_MULT_SWING:.2f}xATR "
            f"buffer={STOP_BUFFER_ATR_MULT:.2f}xATR "
            f"min_stop={MIN_STOP_ATR_REJECT:.2f}xATR | "
            f"Mkt regime: block_cont_chop={BLOCK_CONTINUATION_IN_CHOP} "
            f"weak_mult={WEAK_TREND_SIZE_MULT:.2f}x | "
            f"Hard-blocked: {hard_display} | "
            f"Soft-blocked: {soft_display} (override>={SESSION_OVERRIDE_MIN_SCORE}) | "
            f"Dead-zone mode: {'SOFT' if DEAD_ZONE_SOFT_OVERRIDE_ENABLED else 'HARD'} "
            f"(override>={DEAD_ZONE_OVERRIDE_MIN_SCORE})"
        )
        if ENABLE_PARTIAL_TP:
            print(
                f"[EXIT_MODE] partial_tp enabled — "
                f"trigger={PARTIAL_TP_R}R close={PARTIAL_CLOSE_FRACTION*100:.0f}% "
                f"be_move={MOVE_STOP_TO_BREAKEVEN_AFTER_PARTIAL} "
                f"runner→TP (no ATR trail)"
            )
        elif FULL_TP_MODE:
            print("[EXIT_MODE] full_tp_mode enabled — hold until TP, no partials")
        else:
            print("[EXIT_MODE] legacy partial mode — 50% at 1R + ATR trail")

        # ── Risk stress snapshot ───────────────────────────────────────────────
        _risk_pct = float(os.getenv("RISK_PCT_PER_TRADE", "1.00"))
        _daily_halt_r = float(os.getenv("DAILY_LOSS_LIMIT_R", "3.0"))
        _dd_halt_pct = float(os.getenv("MAX_DD_PCT", "15.0"))
        _taker_bps = float(os.getenv("TAKER_FEE_BPS", "4.5"))
        _balance = self._boot_balance if self._boot_balance > 0 else float(
            os.getenv("STARTING_BALANCE", "1000")
        )
        _risk_usd = _balance * (_risk_pct / 100.0)
        _max_risk_usd = MAX_OPEN_POSITIONS * _risk_usd
        _worst_dd_usd = MAX_OPEN_POSITIONS * _risk_usd * abs(MAX_FULL_LOSS_R)
        _worst_dd_pct = (_worst_dd_usd / _balance * 100.0) if _balance > 0 else 0.0
        _daily_halt_usd = _daily_halt_r * _risk_usd
        print(
            f"[STRESS] Risk/trade: ${_risk_usd:.2f} ({_risk_pct:.2f}% of ${_balance:.2f}) | "
            f"Max simultaneous: ${_max_risk_usd:.2f} ({MAX_OPEN_POSITIONS} slots × {_risk_pct:.2f}%) | "
            f"Worst-case DD (all {MAX_FULL_LOSS_R:.1f}R): -${_worst_dd_usd:.2f} "
            f"({_worst_dd_pct:.1f}% of balance) | "
            f"Daily halt: -{_daily_halt_r:.1f}R (≈-${_daily_halt_usd:.2f}) | "
            f"DD halt: {_dd_halt_pct:.1f}% | "
            f"Fee/round-trip (2×taker): {2*_taker_bps:.1f}bps"
        )

    def _make_venue_checker(self):
        if not hasattr(self.backend, "_get_venue_position_size"):
            return None

        def checker(coin: str, side: str) -> bool:
            try:
                sz = self.backend._get_venue_position_size(coin, side)
                if sz is None:
                    return True
                return sz > 0
            except Exception:
                return True

        return checker

    def _get_hard_blocked_sessions(self, bucket: str) -> set:
        if GLOBAL_BLOCKED_SESSIONS:
            return GLOBAL_BLOCKED_SESSIONS
        if bucket == "majors":
            return MAJORS_BLOCKED_SESSIONS
        if bucket == "sol_beta":
            return SOL_BETA_BLOCKED_SESSIONS
        if bucket == "alt_beta":
            return ALT_BETA_BLOCKED_SESSIONS
        return OTHER_BLOCKED_SESSIONS

    def _can_override_soft_block(self, signal, session: str) -> bool:
        # Swing-timeframe signals bypass soft block unconditionally when enabled.
        # Hard session blocks (BLOCKED_SESSIONS / per-bucket) are NOT bypassed here.
        if SWING_SESSION_OVERRIDE and self._signal_timeframe_class(signal) == "swing":
            return True

        if session == "dead_zone" and DEAD_ZONE_SOFT_OVERRIDE_ENABLED:
            min_score = DEAD_ZONE_OVERRIDE_MIN_SCORE
            families = DEAD_ZONE_OVERRIDE_FAMILIES
            require_trend = DEAD_ZONE_OVERRIDE_REQUIRE_TREND_ALIGN
        elif session in SOFT_BLOCKED_SESSIONS:
            min_score = SESSION_OVERRIDE_MIN_SCORE
            families = SESSION_OVERRIDE_FAMILIES
            require_trend = True
        else:
            return False

        meta = signal.meta or {}
        score = float(meta.get("total_score", getattr(signal, "confidence", 0.0)) or 0.0)
        if score < min_score:
            return False

        setup_family = str(
            meta.get("setup_family", meta.get("regime_local", ""))
        ).strip().lower()
        if setup_family not in families:
            return False

        if require_trend and not self._is_trend_aligned(signal):
            return False

        return True

    @staticmethod
    def _is_trend_aligned(signal) -> bool:
        side = Executor._side_str(signal.side)
        regime = str(getattr(signal, "regime", "")).lower()

        if side == "LONG":
            return "htf_up" in regime and "macro_up" in regime
        return "htf_down" in regime and "macro_down" in regime

    def _score_size_overlay(
        self,
        signal,
        setup_family: str,
        market_regime: str,
    ) -> Tuple[float, float, float, float, bool]:
        meta = getattr(signal, "meta", None) or {}
        total_score = float(meta.get("total_score", getattr(signal, "confidence", 0.0)) or 0.0)

        score_mult = SCORE_SIZE_BASE_MULT
        if total_score >= SCORE_SIZE_HIGH_THRESHOLD:
            score_mult = SCORE_SIZE_HIGH_MULT
        elif total_score >= SCORE_SIZE_MID_THRESHOLD:
            score_mult = SCORE_SIZE_MID_MULT

        trend_bonus_applied = (
            str(setup_family or "").strip().lower() == "continuation"
            and str(market_regime or "").strip().lower() == "strong_trend"
            and self._is_trend_aligned(signal)
        )
        trend_bonus_mult = CONT_STRONG_TREND_BONUS_MULT if trend_bonus_applied else 1.0
        overlay_mult = min(SCORE_SIZE_OVERLAY_MAX_MULT, score_mult * trend_bonus_mult)

        return total_score, score_mult, trend_bonus_mult, overlay_mult, trend_bonus_applied

    def boot_status_message(self) -> str:
        pct = (
            self._boot_all_time_pnl / STARTING_BALANCE * 100
            if STARTING_BALANCE > 0 else 0.0
        )
        dd_pct = (
            (self._boot_hwm - self._boot_balance) / self._boot_hwm * 100
            if self._boot_hwm > 0 else 0.0
        )

        lines = [
            "🚀 *Cuan Sniffer — 🔴 LIVE*",
            "",
            "```",
            f"  Balance     : ${self._boot_balance:.2f}  ({pct:+.1f}%)",
            f"  All-time P&L: ${self._boot_all_time_pnl:+.2f}",
            f"  HWM         : ${self._boot_hwm:.2f}",
            f"  Drawdown    : {dd_pct:.1f}%",
            f"  Total closed: {self._boot_total_closed} trades",
            f"  Today       : {self._boot_closed_today} closed  "
            f"{self._boot_daily_r:+.2f}R  ${self._boot_daily_pnl:+.2f}",
            "```",
        ]

        if self._boot_open:
            lines.append("")
            lines.append(f"*📂 {len(self._boot_open)} open position(s) restored:*")
            lines.append("```")
            for pos in self._boot_open:
                state_tag = "⚡partial" if pos.partial_closed else "open"
                lines.append(
                    f"  {pos.coin:<6} {str(pos.side):<5} "
                    f"entry={pos.entry_price:.4f} stop={pos.stop_price:.4f} [{state_tag}]"
                )
            lines.append("```")
        else:
            lines.append("\n_No open positions_")

        sf = self.strategy_filter.summary()
        if sf.strip() not in ("No data yet.", ""):
            lines.append(f"\n*🔒 Filters:*\n```\n{sf}\n```")

        lines.append(f"\n🕒 `{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}`")
        return "\n".join(lines)

    def on_signal(self, signal, sig_id: int = 0) -> ExecutorResult:
        signal_side = self._side_str(signal.side)
        meta = signal.meta or {}
        session = str(meta.get("session", "")).strip().lower()
        coin = str(signal.coin).upper()
        bucket = self._factor_bucket(coin)
        track = self._signal_track(signal)
        now = time.time()

        if not self._try_mark_coin_pending_open(coin):
            reason = f"pending_open_race:{coin}"
            print(f"[EXECUTOR] {coin} {signal_side} REJECTED — {reason}")
            self._log_missed(signal, sig_id, reason)
            return ExecutorResult(traded=False, reason=reason)

        try:
            if self._has_open_position_for_coin(coin):
                reason = f"pending_or_open_position:{coin}"
                print(f"[EXECUTOR] {coin} {signal_side} REJECTED — {reason}")
                self._log_missed(signal, sig_id, reason)
                return ExecutorResult(traded=False, reason=reason)

            if self._live_mode and not self._balance_ready:
                reason = "startup_balance_not_ready"
                print(
                    f"[EXECUTOR] {coin} {signal_side} GATED — "
                    f"balance not initialized (waiting for venue sync)"
                )
                self._log_missed(signal, sig_id, reason)
                return ExecutorResult(traded=False, reason=reason)

            hard_blocked = self._get_hard_blocked_sessions(bucket)
            if session in hard_blocked:
                if not (session == "dead_zone" and DEAD_ZONE_SOFT_OVERRIDE_ENABLED):
                    reason = f"session_blocked:{session}"
                    print(f"[EXECUTOR] {coin} {signal_side} REJECTED — {reason}")
                    self._log_missed(signal, sig_id, reason)
                    return ExecutorResult(traded=False, reason=reason)

            session_soft_blocked = (
                (session in SOFT_BLOCKED_SESSIONS)
                or (session == "dead_zone" and DEAD_ZONE_SOFT_OVERRIDE_ENABLED)
            )
            if session_soft_blocked:
                required_score = (
                    DEAD_ZONE_OVERRIDE_MIN_SCORE if session == "dead_zone"
                    else SESSION_OVERRIDE_MIN_SCORE
                )
                if self._can_override_soft_block(signal, session):
                    print(
                        f"[EXECUTOR] {coin} {signal_side} SESSION OVERRIDE — "
                        f"{session} bypassed "
                        f"(conf={float(getattr(signal, 'confidence', 0)):.2f}, "
                        f"family={meta.get('setup_family', '?')}, "
                        f"regime={getattr(signal, 'regime', '?')})"
                    )
                else:
                    reason = f"session_soft_blocked:{session}"
                    reason = self._apply_reject_throttle(signal, signal_side, reason)
                    print(
                        f"[EXECUTOR] {coin} {signal_side} REJECTED — {reason} "
                        f"(override_min={required_score:.2f})"
                    )
                    self._log_missed(signal, sig_id, reason)
                    return ExecutorResult(traded=False, reason=reason)

            cooldown_key = (coin, signal_side, track)
            last_fired = self._cooldowns.get(cooldown_key, 0.0)
            if now - last_fired < SIGNAL_COOLDOWN_SEC:
                remaining = int(SIGNAL_COOLDOWN_SEC - (now - last_fired))
                reason = f"cooldown ({remaining}s remaining)"
                print(f"[EXECUTOR] {coin} {signal_side} COOLDOWN — {remaining}s")
                self._log_missed(signal, sig_id, reason)
                return ExecutorResult(traded=False, reason=reason)

            coin_fill_key = (coin, signal_side, track)
            recent_coin_fill = self._coin_last_fill_ts.get(coin_fill_key, 0.0)
            if now - recent_coin_fill < COIN_REENTRY_COOLDOWN_SEC:
                remaining = int(COIN_REENTRY_COOLDOWN_SEC - (now - recent_coin_fill))
                reason = f"coin_reentry_cooldown:{coin}:{signal_side} ({remaining}s remaining)"
                print(f"[EXECUTOR] {coin} {signal_side} REJECTED — {reason}")
                self._log_missed(signal, sig_id, reason)
                return ExecutorResult(traded=False, reason=reason)

            bucket_key = (bucket, signal_side, track)
            recent_bucket_fill = self._bucket_last_fill_ts.get(bucket_key, 0.0)
            if now - recent_bucket_fill < BUCKET_REENTRY_COOLDOWN_SEC:
                remaining = int(BUCKET_REENTRY_COOLDOWN_SEC - (now - recent_bucket_fill))
                reason = (
                    f"bucket_reentry_cooldown:{bucket}:{signal_side}:{track} "
                    f"({remaining}s remaining)"
                )
                print(f"[EXECUTOR] {coin} {signal_side} REJECTED — {reason}")
                self._log_missed(signal, sig_id, reason)
                return ExecutorResult(traded=False, reason=reason)

            dir_limit = self._track_bucket_dir_limit(bucket, signal_side, track)
            same_bucket_same_dir = sum(
                1
                for p in self.risk.open_positions.values()
                if self._factor_bucket(getattr(p, "coin", "")) == bucket
                and self._side_str(getattr(p, "side", "")) == signal_side
                and self._position_track(p) == track
            )
            if dir_limit <= 0:
                reason = f"bucket_dir_limit_disabled:{bucket}:{signal_side}:{track}"
                print(f"[EXECUTOR] {coin} {signal_side} REJECTED — {reason}")
                self._log_missed(signal, sig_id, reason)
                return ExecutorResult(traded=False, reason=reason)

            if same_bucket_same_dir >= dir_limit:
                reason = (
                    f"bucket_dir_limit:{bucket}:{signal_side}:{track} "
                    f"({same_bucket_same_dir}/{dir_limit})"
                )
                print(f"[EXECUTOR] {coin} {signal_side} REJECTED — {reason}")
                self._log_missed(signal, sig_id, reason)
                return ExecutorResult(traded=False, reason=reason)

            stop_reject = self._apply_entry_stop_redesign(signal=signal, track=track)
            if stop_reject:
                reason = f"stop_redesign_reject: {stop_reject}"
                print(f"[EXECUTOR] {coin} {signal_side} REJECTED — {reason}")
                self._log_missed(signal, sig_id, reason)
                return ExecutorResult(traded=False, reason=reason)

            setup_family = self._signal_setup_family(signal)
            market_regime = self._signal_market_regime(signal)
            market_meta = getattr(signal, "meta", None) or {}
            print(
                f"[REGIME_GATE] {coin} {signal_side} family={setup_family or '?'} "
                f"mkt={market_regime} "
                f"atr_ratio={float(market_meta.get('atr_ratio', 0.0)):.3f} "
                f"trend_proxy={int(float(market_meta.get('trend_strength_proxy', 0)))} "
                f"breakout_atr={float(market_meta.get('breakout_dist_atr', 0.0)):.3f}"
            )

            if (
                BLOCK_CONTINUATION_IN_CHOP
                and setup_family == "continuation"
                and market_regime == "chop"
            ):
                reason = "market_regime_block:continuation_in_chop"
                print(f"[EXECUTOR] {coin} {signal_side} REJECTED — {reason}")
                self._log_missed(signal, sig_id, reason)
                return ExecutorResult(traded=False, reason=reason)

            if BLOCK_REVERSAL_AGAINST_DUAL_TREND and setup_family == "reversal":
                htf = str(market_meta.get("regime_htf_1h", "")).strip().lower()
                macro = str(market_meta.get("regime_macro_4h", "")).strip().lower()
                if (
                    (signal_side == "LONG" and htf == "down" and macro == "down")
                    or (signal_side == "SHORT" and htf == "up" and macro == "up")
                ):
                    reason = f"market_regime_block:reversal_against_dual_trend(htf={htf},macro={macro})"
                    print(f"[EXECUTOR] {coin} {signal_side} REJECTED — {reason}")
                    self._log_missed(signal, sig_id, reason)
                    return ExecutorResult(traded=False, reason=reason)

            self._refresh_runtime_balance_from_venue(source=f"pre_entry:{coin}")
            if self._live_mode and self.venue_sync_unhealthy:
                reason = "venue_sync_unhealthy:missing_or_zero_equity"
                print(f"[EXECUTOR][WARNING] {coin} {signal_side} REJECTED — {reason}")
                self._log_missed(signal, sig_id, reason)
                return ExecutorResult(traded=False, reason=reason)
            decision = self.risk.check_signal(signal)
            if not decision.approved:
                if self._is_capacity_reject_reason(decision.reason):
                    replaced, replace_reason = self._try_replace_position_for_signal(
                        signal,
                        decision.reason,
                    )
                    if replaced:
                        decision = self.risk.check_signal(signal)
                        if not decision.approved:
                            reason = f"replacement_post_check_failed:{decision.reason}"
                            print(f"[EXECUTOR] {coin} {signal_side} REJECTED — {reason}")
                            self._log_missed(signal, sig_id, reason)
                            return ExecutorResult(traded=False, reason=reason)
                    else:
                        reason = f"{decision.reason} | replacement={replace_reason}"
                        print(f"[EXECUTOR] {coin} {signal_side} REJECTED — {reason}")
                        self._log_missed(signal, sig_id, reason)
                        return ExecutorResult(traded=False, reason=reason)
                else:
                    print(f"[EXECUTOR] {coin} {signal_side} REJECTED — {decision.reason}")
                    self._log_missed(signal, sig_id, decision.reason)
                    return ExecutorResult(traded=False, reason=decision.reason)

            (
                score_for_sizing,
                score_mult,
                trend_bonus_mult,
                overlay_mult,
                trend_bonus_applied,
            ) = self._score_size_overlay(
                signal=signal,
                setup_family=setup_family,
                market_regime=market_regime,
            )
            old_size_overlay = float(getattr(decision, "size_usd", 0.0) or 0.0)
            old_risk_overlay = float(getattr(decision, "risk_usd", 0.0) or 0.0)
            if old_size_overlay > 0 and old_risk_overlay > 0 and overlay_mult > 0:
                decision.size_usd = round(old_size_overlay * overlay_mult, 2)
                decision.risk_usd = round(old_risk_overlay * overlay_mult, 2)
                decision.size_multiplier = round(float(decision.size_multiplier) * overlay_mult, 4)
                decision.reason += (
                    f" | score_mult={score_mult:.2f}x"
                    f" trend_bonus={'on' if trend_bonus_applied else 'off'}"
                    f" final_mult={overlay_mult:.2f}x"
                )
            print(
                f"[EXECUTOR] {coin} sizing overlay: "
                f"score={score_for_sizing:.3f} "
                f"score_mult={score_mult:.2f}x "
                f"trend_bonus={'on' if trend_bonus_applied else 'off'}"
                f"({trend_bonus_mult:.2f}x) "
                f"final_mult={overlay_mult:.2f}x "
                f"size ${old_size_overlay:.2f}->{float(getattr(decision, 'size_usd', 0.0) or 0.0):.2f}"
            )

            margin_reject = self._apply_available_margin_sizing(decision, coin=coin)
            if margin_reject:
                reason = self._apply_reject_throttle(signal, signal_side, margin_reject)
                print(f"[EXECUTOR] {coin} {signal_side} REJECTED — {reason}")
                self._log_missed(signal, sig_id, reason)
                return ExecutorResult(traded=False, reason=reason)

            if (
                setup_family == "continuation"
                and decision.size_multiplier > CONTINUATION_MAX_SIZE_MULT
            ):
                original_mult = decision.size_multiplier
                scale = CONTINUATION_MAX_SIZE_MULT / original_mult
                decision.size_usd = round(decision.size_usd * scale, 2)
                decision.risk_usd = round(decision.risk_usd * scale, 2)
                decision.size_multiplier = CONTINUATION_MAX_SIZE_MULT
                decision.reason += f" | continuation_cap={CONTINUATION_MAX_SIZE_MULT:.2f}x"
                print(
                    f"[EXECUTOR] {coin} continuation cap: "
                    f"{original_mult:.2f}x → {CONTINUATION_MAX_SIZE_MULT:.2f}x"
                )

            if market_regime == "weak_trend" and WEAK_TREND_SIZE_MULT < 1.0:
                if WEAK_TREND_SIZE_MULT <= 0:
                    reason = "market_regime_invalid_weak_trend_mult"
                    print(f"[EXECUTOR] {coin} {signal_side} REJECTED — {reason}")
                    self._log_missed(signal, sig_id, reason)
                    return ExecutorResult(traded=False, reason=reason)

                decision.size_usd = round(decision.size_usd * WEAK_TREND_SIZE_MULT, 2)
                decision.risk_usd = round(decision.risk_usd * WEAK_TREND_SIZE_MULT, 2)
                decision.size_multiplier = round(decision.size_multiplier * WEAK_TREND_SIZE_MULT, 4)
                decision.reason += f" | weak_trend_mult={WEAK_TREND_SIZE_MULT:.2f}x"
                print(
                    f"[EXECUTOR] {coin} weak-trend size: "
                    f"multiplied by {WEAK_TREND_SIZE_MULT:.2f}x"
                )
                if decision.size_usd <= 0 or decision.risk_usd <= 0:
                    reason = "market_regime_weak_trend_size_too_small"
                    print(f"[EXECUTOR] {coin} {signal_side} REJECTED — {reason}")
                    self._log_missed(signal, sig_id, reason)
                    return ExecutorResult(traded=False, reason=reason)

            entry_reject = self._validate_entry(signal, session)
            if entry_reject:
                reason = self._apply_reject_throttle(signal, signal_side, entry_reject)
                print(f"[EXECUTOR] {coin} {signal_side} REJECTED — {reason}")
                self._log_missed(signal, sig_id, reason)
                return ExecutorResult(traded=False, reason=reason)

            print(f"[EXECUTOR] {coin} {signal_side} APPROVED — {decision.reason}")

            pending_position_id = f"pending_{coin}_{sig_id}_{int(now * 1000)}"
            entry_track_id = self.order_tracker.create_order(
                position_id=pending_position_id,
                coin=coin,
                side=signal_side,
                order_kind=OrderKind.ENTRY,
                requested_price=float(signal.entry_price),
                requested_size_usd=float(decision.size_usd),
                source="executor_entry",
            )
            self.order_tracker.mark_submitted(entry_track_id)

            fill = self.backend.execute_entry(
                coin=coin,
                side=signal_side,
                price=float(signal.entry_price),
                size_usd=decision.size_usd,
                confidence=float(signal.confidence),
            )

            if not fill.filled:
                raw_reason = str(fill.reject_reason or "")
                reason = f"fill_rejected: {fill.reject_reason}"
                self.order_tracker.mark_failed(
                    entry_track_id,
                    reason=str(fill.reject_reason or "entry_rejected"),
                )
                reason = self._apply_reject_throttle(signal, signal_side, raw_reason or reason)
                print(f"[EXECUTOR] {coin} FILL REJECTED — {reason}")
                self._log_missed(signal, sig_id, reason)
                return ExecutorResult(traded=False, reason=reason)

            self._session_entry_slippage_bps.append(fill.slippage_bps)

            entry_order_ids = self._extract_order_ids(fill)
            position = self._build_position(
                signal=signal,
                sig_id=sig_id,
                decision=decision,
                fill_price=fill.fill_price,
                fill_size_usd=fill.fill_size_usd,
                entry_order_ids=entry_order_ids,
            )

            setattr(position, "allow_reconcile_close", False)
            setattr(position, "reconciled_from_venue", False)
            setattr(position, "bootstrap_restored", False)

            self.order_tracker.bind_position(entry_track_id, position.position_id)
            self._apply_fill_to_tracked_order(
                entry_track_id,
                fill,
                fallback_price=float(signal.entry_price),
            )

            entry_fee_usd = 0.0
            if hasattr(fill, "meta") and isinstance(fill.meta, dict):
                entry_fee_usd = float(fill.meta.get("estimated_fee_usd", 0.0))
            elif hasattr(self.backend, "estimate_entry_fee"):
                entry_fee_usd = float(self.backend.estimate_entry_fee(fill.fill_size_usd))

            position.entry_fee_usd = round(entry_fee_usd, 8)
            position.exit_fees_usd = 0.0
            position.funding_usd = 0.0
            position.total_fees_usd = round(entry_fee_usd, 8)
            position.last_funding_ts = datetime.now(timezone.utc)

            slip_tag = f" (slip={fill.slippage_bps:.1f}bps)" if fill.slippage_bps > 0 else ""
            partial_tag = ""
            if fill.fill_ratio is not None and fill.fill_ratio < 0.99:
                partial_tag = f" [partial fill: {fill.fill_ratio * 100:.0f}%]"

            print(
                f"[EXECUTOR] FILLED {coin} {signal_side} "
                f"@ {fill.fill_price:.4f}{slip_tag} fee=${entry_fee_usd:.2f}{partial_tag}"
            )

            self.risk.record_open(position)
            self.live_monitor.register(position, allow_reconcile=False)
            self.risk.apply_entry_fee(position, entry_fee_usd)
            self._ensure_native_protection(position, entry_fill=fill, source="entry_fill")
            append_trade(position, paper_mode=False)
            if self.signal_engine is not None and hasattr(self.signal_engine, "record_emitted_signal"):
                try:
                    self.signal_engine.record_emitted_signal(signal, entry_price=float(fill.fill_price))
                except Exception as e:
                    print(f"[DEDUP_RECORD] executor-side record skipped: {e}")

            print(
                f"[LIVE] OPEN {coin} {signal_side} "
                f"entry={fill.fill_price:.4f} size=${fill.fill_size_usd:.0f} "
                f"fee=${entry_fee_usd:.2f}"
            )

            self._cooldowns[cooldown_key] = now
            self._coin_last_fill_ts[coin_fill_key] = now
            self._bucket_last_fill_ts[bucket_key] = now

            return ExecutorResult(
                traded=True,
                reason=decision.reason,
                position_id=position.position_id,
                fill_price=fill.fill_price,
                fill_slippage_bps=fill.slippage_bps,
                fill_ratio=fill.fill_ratio if fill.fill_ratio is not None else 1.0,
            )
        finally:
            self._clear_coin_pending_open(coin)

    def _apply_entry_stop_redesign(self, signal, track: str) -> Optional[str]:
        meta = signal.meta if isinstance(signal.meta, dict) else {}
        coin = str(signal.coin).upper()
        side = self._side_str(signal.side)

        entry = float(getattr(signal, "entry_price", 0.0) or 0.0)
        structural_stop = float(getattr(signal, "stop_price", 0.0) or 0.0)
        tp = float(getattr(signal, "tp_price", 0.0) or 0.0)
        atr = float(meta.get("atr", 0.0) or 0.0)

        if entry <= 0 or structural_stop <= 0 or tp <= 0:
            return "invalid_signal_levels"
        if atr <= 0:
            return "invalid_atr_for_stop_redesign"

        if side == "LONG":
            if structural_stop >= entry:
                return (
                    f"invalid_structural_stop_long "
                    f"(stop={structural_stop:.6f} >= entry={entry:.6f})"
                )
            if tp <= entry:
                return (
                    f"invalid_tp_long "
                    f"(tp={tp:.6f} <= entry={entry:.6f})"
                )
        elif side == "SHORT":
            if structural_stop <= entry:
                return (
                    f"invalid_structural_stop_short "
                    f"(stop={structural_stop:.6f} <= entry={entry:.6f})"
                )
            if tp >= entry:
                return (
                    f"invalid_tp_short "
                    f"(tp={tp:.6f} >= entry={entry:.6f})"
                )
        else:
            return f"invalid_side:{side}"

        floor_mult = (
            STOP_ATR_FLOOR_MULT_SWING if track == "swing"
            else STOP_ATR_FLOOR_MULT_INTRADAY
        )
        floor_mult = max(0.0, float(floor_mult))
        buffer_mult = max(0.0, float(STOP_BUFFER_ATR_MULT))
        min_stop_atr = max(0.0, float(MIN_STOP_ATR_REJECT))

        atr_floor_dist = floor_mult * atr
        min_stop_dist = min_stop_atr * atr
        buffer_dist = buffer_mult * atr

        if side == "LONG":
            atr_floor_stop = entry - atr_floor_dist
            floor_applied_stop = min(structural_stop, atr_floor_stop)
            buffered_stop = floor_applied_stop - buffer_dist
            final_stop = buffered_stop
            if (entry - final_stop) < min_stop_dist:
                final_stop = entry - min_stop_dist
        else:
            atr_floor_stop = entry + atr_floor_dist
            floor_applied_stop = max(structural_stop, atr_floor_stop)
            buffered_stop = floor_applied_stop + buffer_dist
            final_stop = buffered_stop
            if (final_stop - entry) < min_stop_dist:
                final_stop = entry + min_stop_dist

        final_stop_dist = abs(entry - final_stop)
        if final_stop_dist <= 0:
            return "invalid_final_stop_distance"

        tp_dist = abs(tp - entry)
        final_rr = tp_dist / final_stop_dist if final_stop_dist > 0 else 0.0
        min_rr = float(MIN_STOP_REDESIGN_RR)
        min_rr_tolerance = 0.05
        min_rr_effective = max(0.0, min_rr - min_rr_tolerance)
        if final_rr < min_rr_effective:
            return (
                f"final_rr_below_min "
                f"(final_rr={final_rr:.2f} < min_rr={min_rr_effective:.2f})"
            )

        signal.stop_price = float(final_stop)

        meta["stop_structural"] = round(structural_stop, 8)
        meta["stop_atr_floor"] = round(atr_floor_stop, 8)
        meta["stop_buffered"] = round(buffered_stop, 8)
        meta["stop_final"] = round(final_stop, 8)
        meta["rr_final"] = round(final_rr, 4)
        meta["stop_floor_mult"] = round(floor_mult, 4)
        meta["stop_buffer_mult"] = round(buffer_mult, 4)
        meta["stop_min_atr_reject"] = round(min_stop_atr, 4)
        meta["stop_track"] = track
        signal.meta = meta

        print(
            f"[STOP_REDESIGN] {coin} {side} track={track} "
            f"structural_stop={structural_stop:.6f} "
            f"atr_floor_stop={atr_floor_stop:.6f} "
            f"buffered_stop={buffered_stop:.6f} "
            f"final_stop={final_stop:.6f} "
            f"final_rr={final_rr:.2f}"
        )

        return None

    def _validate_entry(self, signal, session: str = "") -> Optional[str]:
        meta = signal.meta or {}
        coin = str(signal.coin).upper()
        atr = float(meta.get("atr", 0.0))
        signal_price = float(signal.entry_price)
        stop_price = float(signal.stop_price)
        tp_price = float(signal.tp_price)
        side = self._side_str(signal.side)

        if signal_price <= 0 or stop_price <= 0 or tp_price <= 0:
            return "invalid_signal_levels_for_rr_guard"

        current_price = self.backend.get_mid_price(coin)
        if current_price is None or float(current_price) <= 0:
            print(
                f"[RR_GUARD] {coin} {side} reject=missing_market_price_for_rr_guard "
                f"score={float(meta.get('total_score', getattr(signal, 'confidence', 0.0)) or 0.0):.3f} "
                f"entry={signal_price:.6f} tp={tp_price:.6f} sl={stop_price:.6f}"
            )
            return "missing_market_price_for_rr_guard"
        current_price = float(current_price)
        score = float(meta.get("total_score", getattr(signal, "confidence", 0.0)) or 0.0)

        if (side == "LONG" and current_price >= tp_price) or (
            side == "SHORT" and current_price <= tp_price
        ):
            print(
                f"[RR_GUARD] {coin} {side} reject=tp_already_consumed "
                f"score={score:.3f} current={current_price:.6f} "
                f"entry={signal_price:.6f} tp={tp_price:.6f} sl={stop_price:.6f}"
            )
            return "tp_already_consumed"

        self._record_recent_price(coin, current_price)

        price_move = abs(current_price - signal_price)
        stop_dist = abs(signal_price - stop_price)

        if atr > 0:
            move_atr = price_move / atr
            if move_atr > SIGNAL_STALE_ATR_MULT:
                # Momentum re-entry override: price ran away (stale) but has since pulled
                # back within SIGNAL_MOMENTUM_REENTRY_ATR of the original entry price.
                # Only applies when price is on the correct side (not through the entry).
                if SIGNAL_MOMENTUM_REENTRY_ATR > 0:
                    price_ran_then_pulled_back = (
                        (side == "LONG" and current_price >= signal_price)
                        or (side == "SHORT" and current_price <= signal_price)
                    )
                    if price_ran_then_pulled_back and move_atr <= SIGNAL_MOMENTUM_REENTRY_ATR:
                        pass  # allow — price pulled back into acceptable re-entry window
                    else:
                        return (
                            f"signal_stale_move ({move_atr:.2f} ATR > {SIGNAL_STALE_ATR_MULT:.2f})"
                        )
                else:
                    return (
                        f"signal_stale_move ({move_atr:.2f} ATR > {SIGNAL_STALE_ATR_MULT:.2f})"
                    )

        if atr > 0:
            session_scale = SESSION_ATR_BUFFER_SCALE.get(session, 1.0)
            effective_mult = ENTRY_BUFFER_ATR_MULT * session_scale
            atr_buffer = effective_mult * atr

            if price_move > atr_buffer:
                return (
                    f"entry_outside_atr_buffer "
                    f"(move={price_move:.4f} > {atr_buffer:.4f} = "
                    f"{effective_mult:.1f}xATR [{session}])"
                )

        if stop_dist > 0:
            move_frac = price_move / stop_dist

            if move_frac > MAX_MOVE_STOP_FRAC:
                if side == "LONG" and current_price > signal_price:
                    return f"too_late_long (moved {move_frac:.2f}x stop_dist toward TP)"
                if side == "SHORT" and current_price < signal_price:
                    return f"too_late_short (moved {move_frac:.2f}x stop_dist toward TP)"

            trend_aligned = self._is_trend_aligned(signal)
            drift_limit = ADVERSE_DRIFT_ALIGNED if trend_aligned else ADVERSE_DRIFT_DEFAULT
            market_regime = self._signal_market_regime(signal)
            if trend_aligned and market_regime == "strong_trend":
                drift_limit = max(drift_limit, ADVERSE_DRIFT_ALIGNED_STRONG)

            if side == "LONG" and current_price < signal_price and move_frac > drift_limit:
                return (
                    f"adverse_drift_long "
                    f"(price moved {move_frac:.0%} of stop_dist toward stop) "
                    f"[aligned={trend_aligned}, limit={drift_limit:.0%}]"
                )
            if side == "SHORT" and current_price > signal_price and move_frac > drift_limit:
                return (
                    f"adverse_drift_short "
                    f"(price moved {move_frac:.0%} of stop_dist toward stop) "
                    f"[aligned={trend_aligned}, limit={drift_limit:.0%}]"
                )

        if side == "LONG":
            rr_num = tp_price - current_price
            rr_den = current_price - stop_price
        else:
            rr_num = current_price - tp_price
            rr_den = stop_price - current_price

        if rr_den <= 0:
            print(
                f"[RR_GUARD] {coin} {side} reject=fill_rr_invalid_geometry "
                f"score={score:.3f} current={current_price:.6f} entry={signal_price:.6f} "
                f"tp={tp_price:.6f} sl={stop_price:.6f} den={rr_den:.8f}"
            )
            return "fill_rr_invalid_geometry"

        effective_rr = rr_num / rr_den
        if effective_rr < MIN_EXECUTION_EFFECTIVE_RR:
            print(
                f"[RR_GUARD] {coin} {side} reject=fill_rr_below_threshold "
                f"score={score:.3f} rr={effective_rr:.3f} min_rr={MIN_EXECUTION_EFFECTIVE_RR:.3f} "
                f"current={current_price:.6f} entry={signal_price:.6f} "
                f"tp={tp_price:.6f} sl={stop_price:.6f}"
            )
            return (
                f"fill_rr_below_threshold "
                f"(rr={effective_rr:.2f} < {MIN_EXECUTION_EFFECTIVE_RR:.2f})"
            )

        if atr <= 0:
            print(
                f"[RR_GUARD] {coin} {side} atr_unavailable_for_stale_checks "
                f"score={score:.3f} rr={effective_rr:.3f} "
                f"current={current_price:.6f} entry={signal_price:.6f}"
            )
            return None

        recent_move_atr = self._recent_move_in_atr(coin, current_price, atr)
        if recent_move_atr >= MAX_RECENT_MOVE_ATR:
            return (
                f"overextended_move "
                f"({recent_move_atr:.2f} ATR >= {MAX_RECENT_MOVE_ATR:.2f} ATR)"
            )

        return None

    def _record_recent_price(self, coin: str, price: float):
        now = time.time()
        dq = self._recent_prices.setdefault(coin, deque())
        dq.append((now, float(price)))
        cutoff = now - 900
        while dq and dq[0][0] < cutoff:
            dq.popleft()

    def _recent_move_in_atr(self, coin: str, current_price: float, atr: float) -> float:
        if atr <= 0:
            return 0.0
        dq = self._recent_prices.get(coin)
        if not dq:
            return 0.0
        lookback_price = dq[0][1]
        return abs(current_price - lookback_price) / atr

    def _factor_bucket(self, coin: str) -> str:
        c = str(coin).upper()
        if c in MAJORS_BUCKET_COINS:
            return "majors"
        if c in SOL_BETA_BUCKET_COINS:
            return "sol_beta"
        if c in ALT_BETA_BUCKET_COINS:
            return "alt_beta"
        return "other"

    @staticmethod
    def _coin_key(coin: str) -> str:
        return str(coin or "").upper().strip()

    @staticmethod
    def _position_terminal_lock(pos: Position):
        lock = getattr(pos, "_terminal_lock", None)
        if lock is None:
            lock = threading.Lock()
            setattr(pos, "_terminal_lock", lock)
        return lock

    def _is_coin_pending_open(self, coin: str) -> bool:
        key = self._coin_key(coin)
        with self._pending_positions_lock:
            return bool(self._pending_positions.get(key, False))

    def _try_mark_coin_pending_open(self, coin: str) -> bool:
        key = self._coin_key(coin)
        with self._pending_positions_lock:
            if self._pending_positions.get(key, False):
                return False
            self._pending_positions[key] = True
            return True

    def _clear_coin_pending_open(self, coin: str):
        key = self._coin_key(coin)
        with self._pending_positions_lock:
            self._pending_positions.pop(key, None)

    def _has_open_position_for_coin(self, coin: str) -> bool:
        key = self._coin_key(coin)
        risk_pos = self.risk.open_positions.get(key)
        if risk_pos is not None and risk_pos.state != PositionState.CLOSED:
            return True
        for pos in self.live_monitor.positions.values():
            if self._coin_key(getattr(pos, "coin", "")) != key:
                continue
            if pos.state != PositionState.CLOSED:
                return True
        return False

    @staticmethod
    def _reject_reason_family(reason: str) -> str:
        text = str(reason or "").strip().lower()
        if not text:
            return ""
        if text.startswith("session_soft_blocked:"):
            return "session_soft_blocked"
        if "insufficient_available_margin" in text or "insufficient margin" in text:
            return "insufficient_margin"
        if (
            "signal_stale_move" in text
            or "entry_outside_atr_buffer" in text
            or "too_late_" in text
            or "adverse_drift_" in text
            or "overextended_move" in text
        ):
            return "stale_or_drift"
        return ""

    def _reject_signature(self, signal, signal_side: str) -> Tuple[str, str, str, str, str, str, str, str, str]:
        meta = getattr(signal, "meta", None) or {}
        coin = self._coin_key(getattr(signal, "coin", ""))
        side = self._side_str(signal_side)
        track = self._signal_track(signal)
        setup_family = str(meta.get("setup_family", meta.get("regime_local", ""))).strip().lower()
        session = str(meta.get("session", "")).strip().lower()
        market_regime = str(meta.get("market_regime", "unknown")).strip().lower()
        total_score = float(meta.get("total_score", getattr(signal, "confidence", 0.0)) or 0.0)
        score_bucket = str(int(round(total_score * 20.0)))
        entry_price = float(getattr(signal, "entry_price", 0.0) or 0.0)
        atr = float(meta.get("atr", 0.0) or 0.0)
        if atr > 0 and entry_price > 0:
            price_bucket = str(int(round(entry_price / atr)))
        else:
            price_bucket = str(int(round(entry_price * 1000.0)))
        live_mid = self.backend.get_mid_price(coin)
        if live_mid is not None and float(live_mid) > 0:
            if atr > 0 and entry_price > 0:
                live_gap_bucket = str(int(round((float(live_mid) - entry_price) / atr)))
            else:
                live_gap_bucket = str(int(round(float(live_mid) * 1000.0)))
        else:
            live_gap_bucket = "na"
        margin_bucket = "na"
        snapshot_fn = getattr(self, "_venue_margin_snapshot", None)
        venue_margin = snapshot_fn(coin) if callable(snapshot_fn) else None
        if venue_margin is not None:
            try:
                margin_bucket = str(int(round(float(venue_margin[2]) / 25.0)))
            except Exception:
                margin_bucket = "na"
        return (
            coin,
            side,
            track,
            setup_family,
            session,
            market_regime,
            score_bucket,
            live_gap_bucket,
            margin_bucket,
        )

    def _active_reject_cooldown(
        self,
        signal,
        signal_side: str,
        family: str = "",
        signature: Optional[Tuple[str, ...]] = None,
    ) -> Tuple[str, int]:
        if REJECT_MEMORY_TTL_SEC <= 0:
            return "", 0
        requested_family = str(family or "").strip().lower()
        if signature is None:
            signature = self._reject_signature(signal, signal_side)
        now = time.time()
        families = [requested_family] if requested_family else list(REJECT_MEMORY_FAMILIES)
        for fam in families:
            if fam not in REJECT_MEMORY_FAMILIES:
                continue
            key = signature + (fam,)
            ts = float(self._reject_memory.get(key, 0.0) or 0.0)
            if ts <= 0:
                continue
            elapsed = now - ts
            ttl = _REJECT_TTL_BY_FAMILY.get(fam, REJECT_MEMORY_TTL_SEC)
            if elapsed < ttl:
                remaining = int(max(1.0, ttl - elapsed))
                return fam, remaining
        return "", 0

    def _remember_reject(
        self,
        signal,
        signal_side: str,
        reason: str,
        signature: Optional[Tuple[str, ...]] = None,
    ):
        family = self._reject_reason_family(reason)
        if not family or REJECT_MEMORY_TTL_SEC <= 0:
            return

        now = time.time()
        if signature is None:
            signature = self._reject_signature(signal, signal_side)
        key = signature + (family,)
        # Only record the first rejection time — do NOT overwrite on repeat polls.
        # Overwriting resets the TTL window on every 30s poll cycle, creating
        # perpetual lockout as long as the same stale candle is in the feature frame.
        if key not in self._reject_memory:
            self._reject_memory[key] = now

        cutoff = now - max(1, _REJECT_TTL_MAX * 2)
        if len(self._reject_memory) > REJECT_MEMORY_MAX_ITEMS:
            stale_keys = [k for k, ts in self._reject_memory.items() if ts < cutoff]
            for stale_key in stale_keys:
                self._reject_memory.pop(stale_key, None)
        if len(self._reject_memory) > REJECT_MEMORY_MAX_ITEMS:
            oldest_key = min(self._reject_memory.items(), key=lambda item: item[1])[0]
            self._reject_memory.pop(oldest_key, None)

    def _apply_reject_throttle(self, signal, signal_side: str, reason: str) -> str:
        family = self._reject_reason_family(reason)
        if not family:
            return reason
        signature = self._reject_signature(signal, signal_side)
        cool_family, cool_remaining = self._active_reject_cooldown(
            signal,
            signal_side,
            family=family,
            signature=signature,
        )
        if cool_family:
            return f"reject_memory_cooldown:{cool_family} ({cool_remaining}s remaining)"
        self._remember_reject(signal, signal_side, reason, signature=signature)
        return reason

    def _bucket_dir_limit(self, bucket: str, side: str) -> int:
        s = side.upper()
        if bucket == "majors":
            return MAX_BUCKET_MAJORS_LONG if s == "LONG" else MAX_BUCKET_MAJORS_SHORT
        if bucket == "sol_beta":
            return MAX_BUCKET_SOL_BETA_LONG if s == "LONG" else MAX_BUCKET_SOL_BETA_SHORT
        if bucket == "alt_beta":
            return MAX_BUCKET_ALT_BETA_LONG if s == "LONG" else MAX_BUCKET_ALT_BETA_SHORT
        return MAX_BUCKET_OTHER_LONG if s == "LONG" else MAX_BUCKET_OTHER_SHORT

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

    @staticmethod
    def _signal_setup_family(signal) -> str:
        meta = getattr(signal, "meta", None) or {}
        return str(meta.get("setup_family", meta.get("regime_local", ""))).strip().lower()

    @staticmethod
    def _signal_market_regime(signal) -> str:
        meta = getattr(signal, "meta", None) or {}
        tag = str(meta.get("market_regime", "unknown")).strip().lower()
        if tag in {"chop", "weak_trend", "strong_trend"}:
            return tag
        return "unknown"

    @staticmethod
    def _signal_total_score(signal) -> float:
        meta = getattr(signal, "meta", None) or {}
        return float(meta.get("total_score", getattr(signal, "confidence", 0.0)) or 0.0)

    @staticmethod
    def _regime_component(regime_text: str, prefix: str, default: str = "") -> str:
        for token in str(regime_text or "").strip().lower().split("|"):
            if token.startswith(prefix):
                return token[len(prefix):].strip()
        return default

    def _position_market_regime(self, pos: Position) -> str:
        regime = str(getattr(pos, "regime", "")).strip().lower()
        tag = self._regime_component(regime, "mkt_", default="unknown")
        if tag in {"chop", "weak_trend", "strong_trend"}:
            return tag
        return "unknown"

    def _is_position_trend_aligned(self, pos: Position) -> bool:
        regime = str(getattr(pos, "regime", "")).strip().lower()
        side = self._side_str(getattr(pos, "side", ""))
        if side == "LONG":
            return "htf_up" in regime and "macro_up" in regime
        return "htf_down" in regime and "macro_down" in regime

    def _position_live_r(self, pos: Position) -> Optional[float]:
        px = float(getattr(pos, "current_price", 0.0) or 0.0)
        if px <= 0:
            mid = self.backend.get_mid_price(pos.coin)
            px = float(mid or 0.0)
        if px <= 0:
            return None
        try:
            return float(pos.current_r(px))
        except Exception:
            return None

    @staticmethod
    def _position_remaining_to_tp_r(pos: Position, current_r: Optional[float]) -> Optional[float]:
        if current_r is None:
            return None
        target_r = float(getattr(pos, "rr_planned", 0.0) or 0.0)
        if target_r <= 0:
            return None
        return max(0.0, target_r - current_r)

    def _signal_replacement_quality(self, signal) -> float:
        score = self._signal_total_score(signal)
        setup_family = self._signal_setup_family(signal)
        market_regime = self._signal_market_regime(signal)
        if (
            setup_family == "continuation"
            and market_regime == "strong_trend"
            and self._is_trend_aligned(signal)
        ):
            score += POSITION_REPLACEMENT_PREFERRED_BONUS
        elif setup_family == "continuation" and market_regime == "weak_trend":
            score -= 0.03
        elif setup_family == "reversal":
            score -= 0.03

        if market_regime == "chop":
            score -= 0.10
        return score

    def _position_replacement_quality(self, pos: Position) -> float:
        score = float(getattr(pos, "total_score", 0.0) or getattr(pos, "confidence", 0.0) or 0.0)
        setup_family = str(getattr(pos, "setup_family", "")).strip().lower()
        market_regime = self._position_market_regime(pos)

        if (
            setup_family == "continuation"
            and market_regime == "strong_trend"
            and self._is_position_trend_aligned(pos)
        ):
            score += POSITION_REPLACEMENT_PREFERRED_BONUS
        elif setup_family == "continuation" and market_regime == "weak_trend":
            score -= 0.05
        elif setup_family == "reversal":
            score -= 0.03
        elif setup_family not in {"continuation", "reversal", "swing"}:
            score -= 0.04

        if market_regime == "chop":
            score -= 0.12
        return score

    @staticmethod
    def _is_capacity_reject_reason(reason: str) -> bool:
        text = str(reason or "").strip().lower()
        return (
            text.startswith("max positions (")
            or text.startswith("intraday max positions (")
            or text.startswith("swing max positions (")
        )

    def _find_replaceable_position_for_signal(
        self,
        signal,
        reject_reason: str,
    ) -> Tuple[Optional[Position], str]:
        if not ENABLE_POSITION_REPLACEMENT:
            return None, "replacement_disabled"
        if not self._is_capacity_reject_reason(reject_reason):
            return None, "reject_not_capacity"

        incoming_coin = self._coin_key(getattr(signal, "coin", ""))
        incoming_track = self._signal_track(signal)
        incoming_quality = self._signal_replacement_quality(signal)
        reason_text = str(reject_reason or "").strip().lower()
        restrict_track = "intraday" if reason_text.startswith("intraday max positions") else (
            "swing" if reason_text.startswith("swing max positions") else ""
        )

        weakest: Optional[Position] = None
        weakest_quality = 10_000.0
        weakest_diag = ""

        for pos in self.risk.open_positions.values():
            if pos is None:
                continue
            if pos.state == PositionState.CLOSED:
                continue
            if self._coin_key(getattr(pos, "coin", "")) == incoming_coin:
                continue

            pos_track = self._position_track(pos)
            if restrict_track and pos_track != restrict_track:
                print(
                    f"[PORTFOLIO_REPLACE] protect {pos.coin} {pos.side} "
                    f"reason=track_mismatch incoming={incoming_track} open={pos_track}"
                )
                continue

            if str(getattr(pos, "exit_requested_reason", "")).strip():
                print(
                    f"[PORTFOLIO_REPLACE] protect {pos.coin} {pos.side} "
                    "reason=exit_already_pending"
                )
                continue

            if POSITION_REPLACEMENT_PROTECT_PARTIALED and bool(getattr(pos, "partial_closed", False)):
                print(
                    f"[PORTFOLIO_REPLACE] protect {pos.coin} {pos.side} "
                    "reason=partial_closed"
                )
                continue

            current_r = self._position_live_r(pos)
            if (
                current_r is not None
                and POSITION_REPLACEMENT_PROTECT_IN_PROFIT_R > 0
                and current_r >= POSITION_REPLACEMENT_PROTECT_IN_PROFIT_R
            ):
                print(
                    f"[PORTFOLIO_REPLACE] protect {pos.coin} {pos.side} "
                    f"reason=in_profit current_r={current_r:.2f} "
                    f"threshold={POSITION_REPLACEMENT_PROTECT_IN_PROFIT_R:.2f}"
                )
                continue

            tp_remaining_r = self._position_remaining_to_tp_r(pos, current_r)
            if (
                tp_remaining_r is not None
                and POSITION_REPLACEMENT_PROTECT_NEAR_TP_R > 0
                and tp_remaining_r <= POSITION_REPLACEMENT_PROTECT_NEAR_TP_R
            ):
                print(
                    f"[PORTFOLIO_REPLACE] protect {pos.coin} {pos.side} "
                    f"reason=near_tp remaining_r={tp_remaining_r:.2f} "
                    f"threshold={POSITION_REPLACEMENT_PROTECT_NEAR_TP_R:.2f}"
                )
                continue

            pos_quality = self._position_replacement_quality(pos)
            if pos_quality < weakest_quality:
                weakest = pos
                weakest_quality = pos_quality
                weakest_diag = (
                    f"coin={pos.coin} side={pos.side} "
                    f"setup={getattr(pos, 'setup_family', '')} "
                    f"mkt={self._position_market_regime(pos)} "
                    f"score={float(getattr(pos, 'total_score', 0.0) or 0.0):.3f} "
                    f"quality={pos_quality:.3f}"
                )

        if weakest is None:
            return None, "no_replaceable_open_position"

        required_quality = weakest_quality + POSITION_REPLACEMENT_MIN_SCORE_DELTA
        if incoming_quality < required_quality:
            return (
                None,
                (
                    f"incoming_not_stronger incoming_quality={incoming_quality:.3f} "
                    f"required={required_quality:.3f} weakest=({weakest_diag})"
                ),
            )

        return (
            weakest,
            (
                f"incoming_quality={incoming_quality:.3f} "
                f"weakest_quality={weakest_quality:.3f} "
                f"delta={incoming_quality - weakest_quality:.3f}"
            ),
        )

    def _try_replace_position_for_signal(
        self,
        signal,
        reject_reason: str,
    ) -> Tuple[bool, str]:
        candidate, detail = self._find_replaceable_position_for_signal(signal, reject_reason)
        if candidate is None:
            print(
                f"[PORTFOLIO_REPLACE] rejected incoming {signal.coin} {self._side_str(signal.side)} "
                f"reason={detail}"
            )
            return False, detail

        exit_px = float(self.backend.get_mid_price(candidate.coin) or 0.0)
        if exit_px <= 0:
            exit_px = float(getattr(candidate, "current_price", 0.0) or 0.0)
        if exit_px <= 0:
            exit_px = float(getattr(candidate, "entry_price", 0.0) or 0.0)
        if exit_px <= 0:
            msg = f"replace_failed_no_price:{candidate.coin}"
            print(f"[PORTFOLIO_REPLACE] {msg}")
            return False, msg

        print(
            f"[PORTFOLIO_REPLACE] replacing {candidate.coin} {candidate.side} "
            f"for incoming {signal.coin} {self._side_str(signal.side)} "
            f"detail={detail}"
        )
        self._live_close_runner(candidate, exit_px, CloseReason.MANUAL)

        still_open = bool(
            candidate.coin in self.risk.open_positions
            and self.risk.open_positions[candidate.coin].state != PositionState.CLOSED
        )
        if still_open:
            msg = f"replace_close_not_confirmed:{candidate.coin}"
            print(f"[PORTFOLIO_REPLACE] {msg}")
            return False, msg

        print(
            f"[PORTFOLIO_REPLACE] replaced {candidate.coin} {candidate.side} "
            f"with incoming {signal.coin} {self._side_str(signal.side)}"
        )
        return True, f"replaced:{candidate.coin}"

    def _track_bucket_dir_limit(self, bucket: str, side: str, track: str) -> int:
        base_limit = self._bucket_dir_limit(bucket, side)
        mult = SWING_BUCKET_CAP_MULT if track == "swing" else INTRADAY_BUCKET_CAP_MULT
        if mult <= 0:
            return 0
        return max(1, int(ceil(base_limit * mult)))

    def record_exit_slippage(self, slippage_bps: float):
        self._session_exit_slippage_bps.append(slippage_bps)

    def slippage_summary(self) -> str:
        entry_n = len(self._session_entry_slippage_bps)
        exit_n = len(self._session_exit_slippage_bps)

        if entry_n == 0 and exit_n == 0:
            return "No fills this session."

        entry_avg = sum(self._session_entry_slippage_bps) / entry_n if entry_n else 0.0
        exit_avg = sum(self._session_exit_slippage_bps) / exit_n if exit_n else 0.0
        total_entry_cost = sum(self._session_entry_slippage_bps)
        total_exit_cost = sum(self._session_exit_slippage_bps)

        return (
            f"Entry: {entry_n} fills, avg {entry_avg:.1f}bps "
            f"(total {total_entry_cost:.0f}bps) | "
            f"Exit: {exit_n} fills, avg {exit_avg:.1f}bps "
            f"(total {total_exit_cost:.0f}bps)"
        )

    def update(self):
        self._sync_tracked_positions()
        self._update_live_positions()
        self.live_monitor.update()
        self._refresh_runtime_balance_from_venue(source="update")
        self._audit_open_protection()
        self.risk.reset_daily_halt()

    def status_message(self) -> str:
        lines = ["🔴 *Live Trader Status*\n", f"💰 `{self.risk.status_summary()}`\n"]

        if self.risk.open_positions:
            lines.append("*Tracked live positions:*")
            for pos in self.risk.open_positions.values():
                lines.append(
                    f"`{pos.coin} {self._side_str(pos.side)} entry={pos.entry_price:.4f} "
                    f"size=${pos.size_usd:.2f}`"
                )
        else:
            lines.append("_No tracked live positions_")

        sf = self.strategy_filter.summary()
        if sf.strip() not in ("No data yet.", ""):
            lines.append(f"\n*Filters:*\n```\n{sf}\n```")

        lines.append(f"📊 _{self.slippage_summary()}_")
        return "\n".join(lines)

    def shutdown(self):
        self.backend.shutdown()
        print(
            f"[EXECUTOR] Shutdown — cooldowns: {len(self._cooldowns)} | "
            f"Slippage: {self.slippage_summary()}"
        )

    def _log_missed(self, signal, sig_id: int, reason: str):
        if not LOG_MISSED:
            return

        meta = signal.meta or {}
        signal_price = float(signal.entry_price)
        stop_dist = abs(signal_price - float(signal.stop_price))

        current_price = self.backend.get_mid_price(signal.coin)
        price_move_r = 0.0
        if current_price is not None and stop_dist > 0:
            if self._side_str(signal.side) == "LONG":
                price_move_r = (current_price - signal_price) / stop_dist
            else:
                price_move_r = (signal_price - current_price) / stop_dist

        file_exists = os.path.exists(MISSED_LOG_FILE)
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "signal_id": sig_id,
            "coin": signal.coin,
            "side": self._side_str(signal.side),
            "entry_price": round(signal_price, 8),
            "stop_price": round(float(signal.stop_price), 8),
            "tp_price": round(float(signal.tp_price), 8),
            "confidence": round(float(signal.confidence), 3),
            "total_score": round(float(meta.get("total_score", 0.0)), 3),
            "session": meta.get("session", ""),
            "regime": signal.regime,
            "reject_reason": reason[:200],
            "current_price": round(current_price, 8) if current_price else "",
            "price_move_r": round(price_move_r, 3),
        }

        try:
            with open(MISSED_LOG_FILE, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=MISSED_FIELDS)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)
        except Exception as e:
            print(f"[EXECUTOR] Missed log write failed: {e}")

    def _build_position(
        self,
        signal,
        sig_id: int,
        decision,
        fill_price: Optional[float] = None,
        fill_size_usd: Optional[float] = None,
        entry_order_ids: Optional[List[str]] = None,
    ) -> Position:
        meta = signal.meta or {}
        pos_id = f"{signal.coin}_{int(datetime.now(timezone.utc).timestamp() * 1000)}"
        setup_family = meta.get("setup_family", meta.get("regime_local", ""))

        actual_entry = float(fill_price if fill_price is not None else signal.entry_price)
        actual_size = float(fill_size_usd if fill_size_usd is not None else decision.size_usd)

        stop_dist = abs(actual_entry - float(signal.stop_price))
        if actual_entry > 0 and stop_dist > 0:
            stop_pct = stop_dist / actual_entry
            actual_risk = actual_size * stop_pct
        else:
            actual_risk = decision.risk_usd

        pos = Position(
            position_id=pos_id,
            coin=signal.coin,
            side=self._side_str(signal.side),
            signal_id=sig_id,
            entry_price=actual_entry,
            stop_price=float(signal.stop_price),
            tp_price=float(signal.tp_price),
            original_stop_price=float(signal.stop_price),
            original_tp_price=float(signal.tp_price),
            atr=float(meta.get("atr", 0.0)),
            size_usd=actual_size,
            risk_usd=round(actual_risk, 2),
            r_value=round(actual_risk, 2),
            size_multiplier=decision.size_multiplier,
            peak_price=actual_entry,
            regime=signal.regime,
            setup_family=setup_family,
            htf_regime=meta.get("regime_htf_1h", ""),
            confidence=float(signal.confidence),
            total_score=float(meta.get("total_score", 0.0)),
            session=meta.get("session", ""),
            timeframe=str(meta.get("timeframe", "15m") or "15m"),
            execution_track=decision.track,
            entry_order_ids=self._encode_order_ids(entry_order_ids),
        )
        setattr(pos, "allow_reconcile_close", False)
        setattr(pos, "reconciled_from_venue", False)
        setattr(pos, "bootstrap_restored", False)
        return pos

    @staticmethod
    def _side_str(side) -> str:
        return str(side).upper().replace("ORDERSIDE.", "").replace("POSITIONSIDE.", "")

    @staticmethod
    def _r_vs_original(pos: Position, price: float) -> float:
        original_stop = getattr(pos, "original_stop_price", None) or pos.stop_price
        stop_dist = abs(pos.entry_price - original_stop)
        if stop_dist <= 0:
            return 0.0
        if pos.side == "LONG":
            return (price - pos.entry_price) / stop_dist
        return (pos.entry_price - price) / stop_dist

    @staticmethod
    def _extract_order_ids(fill) -> List[str]:
        if fill is None or not hasattr(fill, "meta") or not isinstance(fill.meta, dict):
            return []
        raw = fill.meta.get("oids", [])
        if not isinstance(raw, list):
            return []
        return [str(oid) for oid in raw if oid is not None]

    @staticmethod
    def _encode_order_ids(order_ids: Optional[List[str]]) -> str:
        if not order_ids:
            return ""
        cleaned = [str(oid).strip() for oid in order_ids if str(oid).strip()]
        return "|".join(cleaned)

    def _append_exit_order_ids(self, pos: Position, order_ids: List[str]):
        if not order_ids:
            return
        existing = [
            chunk.strip()
            for chunk in str(getattr(pos, "exit_order_ids", "")).split("|")
            if chunk.strip()
        ]
        merged = existing + [oid for oid in order_ids if oid]
        pos.exit_order_ids = self._encode_order_ids(merged)

    @staticmethod
    def _has_native_protection_metadata(pos: Position) -> bool:
        return bool(
            str(getattr(pos, "stop_order_id", "")).strip()
            or str(getattr(pos, "tp_order_id", "")).strip()
        )

    @staticmethod
    def _coin_size_from_fill(fill) -> float:
        if fill is None:
            return 0.0
        meta = getattr(fill, "meta", {})
        if isinstance(meta, dict):
            total_sz = float(meta.get("total_sz", 0.0) or 0.0)
            if total_sz > 0:
                return total_sz

        fill_notional = float(getattr(fill, "fill_size_usd", 0.0) or 0.0)
        fill_px = float(getattr(fill, "fill_price", 0.0) or 0.0)
        if fill_notional > 0 and fill_px > 0:
            return fill_notional / fill_px
        return 0.0

    @staticmethod
    def _first_order_id(order_ids: List[str]) -> str:
        for oid in order_ids:
            cleaned = str(oid).strip()
            if cleaned:
                return cleaned
        return ""

    @staticmethod
    def _order_kind_for_exit_reason(reason: str) -> OrderKind:
        if str(reason).strip().lower() == "partial":
            return OrderKind.PARTIAL_EXIT
        return OrderKind.FULL_EXIT

    def _apply_fill_to_tracked_order(self, tracked_order_id: str, fill, fallback_price: float = 0.0):
        if not tracked_order_id or fill is None:
            return

        order_ids = self._extract_order_ids(fill)
        venue_order_id = self._first_order_id(order_ids)
        if venue_order_id:
            self.order_tracker.mark_acknowledged(
                tracked_order_id,
                venue_order_id=venue_order_id,
            )

        fill_ratio = getattr(fill, "fill_ratio", None)
        if fill_ratio is None or fill_ratio <= 0:
            fill_ratio = 1.0

        fill_price = float(getattr(fill, "fill_price", 0.0) or fallback_price or 0.0)
        fill_size_usd = float(getattr(fill, "fill_size_usd", 0.0) or 0.0)
        fill_size_coin = self._coin_size_from_fill(fill)

        if fill_ratio < 0.999:
            self.order_tracker.mark_partially_filled(
                tracked_order_id,
                fill_price=fill_price,
                fill_size_usd=fill_size_usd,
                fill_size_coin=fill_size_coin,
                fill_ratio=fill_ratio,
                venue_order_id=venue_order_id,
            )
            return

        self.order_tracker.mark_filled(
            tracked_order_id,
            fill_price=fill_price,
            fill_size_usd=fill_size_usd,
            fill_size_coin=fill_size_coin,
            fill_ratio=fill_ratio,
            venue_order_id=venue_order_id,
        )

    def _resolve_position_size_coin(self, pos: Position, entry_fill=None) -> float:
        signed = self._get_venue_signed_size(pos.coin)
        side = str(pos.side).upper()

        if signed is not None:
            if side == "LONG":
                if signed > LIVE_FLAT_EPSILON_SZ:
                    return abs(signed)
                if signed < -LIVE_FLAT_EPSILON_SZ:
                    return 0.0
            else:
                if signed < -LIVE_FLAT_EPSILON_SZ:
                    return abs(signed)
                if signed > LIVE_FLAT_EPSILON_SZ:
                    return 0.0

        from_fill = self._coin_size_from_fill(entry_fill)
        if from_fill > 0:
            return from_fill

        px = self.backend.get_mid_price(pos.coin)
        if px is None or px <= 0:
            px = float(getattr(pos, "entry_price", 0.0) or 0.0)
        if px <= 0:
            return 0.0
        notional = float(getattr(pos, "size_usd", 0.0) or 0.0)
        return max(0.0, notional / px)

    @staticmethod
    def _split_pipe_ids(raw: str) -> List[str]:
        return [chunk.strip() for chunk in str(raw or "").split("|") if chunk.strip()]

    @staticmethod
    def _extract_prefixed_error(error_blob: str, prefix: str) -> str:
        prefix = f"{prefix.lower()}:"
        for chunk in str(error_blob or "").split("|"):
            text = chunk.strip()
            if text.lower().startswith(prefix):
                return text[len(prefix):].strip()
        return ""

    def _notify_protection_issue(self, pos: Position, issue: str):
        self.protection_manager.notify_issue(pos, issue=issue, source="executor")

    def _ensure_native_protection(self, pos: Position, entry_fill=None, source: str = "") -> bool:
        return self.protection_manager.place_after_entry(
            pos=pos,
            entry_fill=entry_fill,
            source=source or "runtime",
        )

    def _hydrate_order_tracker_from_position(self, pos: Position):
        side = self._side_str(pos.side)
        for oid in self._split_pipe_ids(getattr(pos, "entry_order_ids", "")):
            self.order_tracker.ensure_venue_order(
                position_id=pos.position_id,
                coin=pos.coin,
                side=side,
                order_kind=OrderKind.ENTRY,
                venue_order_id=oid,
                source="bootstrap_entry",
                filled=True,
            )

        stop_oid = str(getattr(pos, "stop_order_id", "") or "").strip()
        if stop_oid:
            self.order_tracker.ensure_venue_order(
                position_id=pos.position_id,
                coin=pos.coin,
                side=side,
                order_kind=OrderKind.STOP_LOSS,
                venue_order_id=stop_oid,
                source="bootstrap_protection",
                filled=False,
            )

        tp_oid = str(getattr(pos, "tp_order_id", "") or "").strip()
        if tp_oid:
            self.order_tracker.ensure_venue_order(
                position_id=pos.position_id,
                coin=pos.coin,
                side=side,
                order_kind=OrderKind.TAKE_PROFIT,
                venue_order_id=tp_oid,
                source="bootstrap_protection",
                filled=False,
            )

    def _audit_bootstrap_protection(self, pos: Position):
        self._hydrate_order_tracker_from_position(pos)

        has_meta = self._has_native_protection_metadata(pos)
        if has_meta:
            if not str(getattr(pos, "protection_status", "")).strip():
                pos.protection_status = (
                    "protected"
                    if str(getattr(pos, "stop_order_id", "")).strip()
                    and str(getattr(pos, "tp_order_id", "")).strip()
                    else "partial"
                )
            pos.venue_protection_mode = True
            pos.allow_reconcile_close = True
            self.protection_manager.audit_position(pos, source="bootstrap_audit")
            return

        pos.venue_protection_mode = True
        pos.protection_status = "missing"
        if not str(getattr(pos, "protection_error", "")).strip():
            pos.protection_error = "missing_protection_metadata_on_boot"

        self.protection_manager.notify_issue(
            pos,
            issue=pos.protection_error,
            source="bootstrap_audit",
        )

        if AUTO_REPLACE_MISSING_PROTECTION_ON_BOOT:
            self._ensure_native_protection(
                pos,
                entry_fill=None,
                source="bootstrap_repair",
            )

        self.protection_manager.audit_position(pos, source="bootstrap_audit")
        append_trade(pos, paper_mode=False)

    def _audit_open_protection(self):
        self.protection_manager.audit_open_positions(
            self._iter_active_positions(),
            source="runtime_audit",
        )

    def _sync_tracked_positions(self):
        for pos in list(self.live_monitor.positions.values()):
            if pos.state == PositionState.CLOSED:
                continue
            if pos.coin not in self.risk.open_positions:
                self.risk.open_positions[pos.coin] = pos
                print(
                    f"[EXECUTOR] Synced monitor->risk for {pos.coin} "
                    f"{pos.side} ({pos.position_id})"
                )

        for pos in list(self.risk.open_positions.values()):
            if pos.state == PositionState.CLOSED:
                continue
            if pos.position_id not in self.live_monitor.positions:
                self.live_monitor.register(
                    pos,
                    allow_reconcile=bool(getattr(pos, "allow_reconcile_close", False)),
                )
                print(
                    f"[EXECUTOR] Synced risk->monitor for {pos.coin} "
                    f"{pos.side} ({pos.position_id})"
                )

    def _iter_active_positions(self) -> List[Position]:
        merged: Dict[str, Position] = {}

        for pos in self.risk.open_positions.values():
            if pos.state != PositionState.CLOSED:
                merged[pos.position_id] = pos

        for pos in self.live_monitor.positions.values():
            if pos.state != PositionState.CLOSED:
                merged[pos.position_id] = pos

        return list(merged.values())

    def _update_live_positions(self):
        active_positions = self._iter_active_positions()

        if not active_positions:
            return

        for pos in active_positions:
            price = self.backend.get_mid_price(pos.coin)
            if price is None or price <= 0:
                print(f"[EXIT_CHECK] {pos.coin} skipped — missing price")
                continue

            pos.update_price(price)
            pos.first_update_pending = False

            print(
                f"[EXIT_CHECK] {pos.coin} {pos.side} "
                f"px={price:.6f} entry={pos.entry_price:.6f} "
                f"stop={pos.stop_price:.6f} tp={pos.tp_price:.6f} "
                f"partial={pos.partial_closed}"
            )

            self._evaluate_live_position(pos, price)

    def _evaluate_live_position(self, pos: Position, price: float):
        if (
            pos.state == PositionState.CLOSED
            or bool(getattr(pos, "_terminalized", False))
            or bool(getattr(pos, "_terminalizing", False))
        ):
            return

        side = pos.side
        stop_hit = (
            (side == "LONG" and price <= pos.stop_price)
            or (side == "SHORT" and price >= pos.stop_price)
        )

        if stop_hit:
            # Deterministic stop behavior: never gate stop execution.
            if pos.partial_closed is True:
                print(f"[EXIT_TRIGGER] {pos.coin} STOP_HIT_EXECUTED mode=STOP_RUNNER no_gating")
                self._live_close_runner(pos, price, CloseReason.STOP_RUNNER)
            elif pos.partial_closed is False:
                print(f"[EXIT_TRIGGER] {pos.coin} STOP_HIT_EXECUTED mode=STOP_FULL no_gating")
                self._live_close_full_stop(pos, price)
            else:
                # Defensive only; should never occur.
                print(
                    f"[EXIT_TRIGGER] {pos.coin} STOP_BLOCKED reason=invalid_partial_state "
                    f"value={pos.partial_closed}"
                )
            return

        tp_hit = (
            (side == "LONG" and price >= pos.tp_price)
            or (side == "SHORT" and price <= pos.tp_price)
        )

        # FULL_TP_MODE: hold entire position until TP fires, no partials.
        if FULL_TP_MODE and not ENABLE_PARTIAL_TP and not bool(pos.partial_closed):
            if tp_hit:
                print(f"[EXIT_TRIGGER] {pos.coin} TP_FULL mode=full_tp")
                self._live_close_runner(pos, price, CloseReason.TP_FULL)
            return

        # Partial-TP runner model: trigger partial at PARTIAL_TP_R, runner to TP with BE stop.
        if ENABLE_PARTIAL_TP and not bool(pos.partial_closed):
            if pos.current_r(price) >= PARTIAL_TP_R:
                print(
                    f"[EXIT_TRIGGER] {pos.coin} PARTIAL_TP "
                    f"r={pos.current_r(price):.2f} threshold={PARTIAL_TP_R}"
                )
                self._live_take_partial(pos, price)
            elif tp_hit:
                # TP hit before partial threshold was reached (price gapped through 1R to TP).
                print(f"[EXIT_TRIGGER] {pos.coin} TP_FULL mode=partial_tp_gapped")
                self._live_close_runner(pos, price, CloseReason.TP_FULL)
            return

        # Legacy partial model (FULL_TP_MODE=false, ENABLE_PARTIAL_TP=false): 50% at 1R + ATR trail.
        if not ENABLE_PARTIAL_TP and not bool(pos.partial_closed) and pos.current_r(price) >= 1.0:
            print(f"[EXIT_TRIGGER] {pos.coin} PARTIAL mode=legacy")
            self._live_take_partial(pos, price)

        if pos.partial_closed:
            if tp_hit:
                print(f"[EXIT_TRIGGER] {pos.coin} TP_FULL")
                self._live_close_runner(pos, price, CloseReason.TP_FULL)
                return

            # ENABLE_PARTIAL_TP: runner holds static BE stop — no trailing.
            # Legacy mode: ATR-trail the stop.
            if not ENABLE_PARTIAL_TP:
                self._live_update_trailing_stop(pos)

    def _live_execute_exit(self, pos: Position, price: float, size_usd: float, reason: str):
        if (
            pos.state == PositionState.CLOSED
            or bool(getattr(pos, "_terminalized", False))
            or bool(getattr(pos, "_terminalizing", False))
        ):
            print(
                f"[LIVE] EXIT SKIP {pos.coin} {pos.side} reason={reason} "
                "already_closed"
            )
            return None

        order_kind = self._order_kind_for_exit_reason(reason)
        requested_size_coin = 0.0
        if price is not None and float(price) > 0:
            requested_size_coin = float(size_usd) / float(price)

        exit_track_id = self.order_tracker.create_order(
            position_id=pos.position_id,
            coin=pos.coin,
            side=pos.side,
            order_kind=order_kind,
            requested_price=float(price or 0.0),
            requested_size_usd=float(size_usd or 0.0),
            requested_size_coin=float(requested_size_coin),
            source=f"executor_exit:{reason}",
        )
        self.order_tracker.mark_submitted(exit_track_id)

        fill = self.backend.execute_exit(
            coin=pos.coin,
            side=pos.side,
            price=price,
            size_usd=size_usd,
            reason=reason,
        )
        if not fill.filled:
            reject_reason = str(getattr(fill, "reject_reason", "") or "")
            reject_meta = getattr(fill, "meta", {})
            if not isinstance(reject_meta, dict):
                reject_meta = {}
            print(
                f"[LIVE] EXIT REJECTED {pos.coin} {pos.side} reason={reason} "
                f"reject={reject_reason}"
            )
            self.order_tracker.mark_failed(
                exit_track_id,
                reason=reject_reason or "exit_rejected",
            )
            self._handle_exit_reject(
                pos=pos,
                reason_value=reason,
                requested_size_usd=size_usd,
                reject_reason=reject_reason,
                tracked_order_id=exit_track_id,
                reject_meta=reject_meta,
            )
            return None

        self._apply_fill_to_tracked_order(
            exit_track_id,
            fill,
            fallback_price=float(price or 0.0),
        )
        self.record_exit_slippage(fill.slippage_bps)
        self._append_exit_order_ids(pos, self._extract_order_ids(fill))
        return fill

    def _mark_exit_requested(self, pos: Position, reason_value: str):
        now = datetime.now(timezone.utc)
        if getattr(pos, "exit_requested_at", None) is None:
            pos.exit_requested_at = now
        if str(getattr(pos, "exit_requested_reason", "")).strip().lower() != reason_value.lower():
            pos.exit_requested_reason = reason_value
        pos.exit_trigger_source = "executor_triggered"
        pos.wallet_flat_confirmed = False
        pos.wallet_flat_confirmed_at = None
        pos.reconciled_from_venue = False
        # Keep reconciler enabled until wallet state confirms flat.
        pos.allow_reconcile_close = True

    def _persist_exit_intent(self, pos: Position, reason_value: str):
        self._mark_exit_requested(pos, reason_value)
        append_trade(pos, paper_mode=False)

    def _capture_exit_attempt(self, pos: Position, fill, reason_value: str):
        now = datetime.now(timezone.utc)
        meta = getattr(fill, "meta", {})
        if not isinstance(meta, dict):
            meta = {}
        fee_usd = float(meta.get("estimated_fee_usd", 0.0))
        pos.pending_exit_reason = reason_value
        pos.pending_exit_fill_price = float(getattr(fill, "fill_price", 0.0) or 0.0)
        pos.pending_exit_fill_size_usd = float(getattr(fill, "fill_size_usd", 0.0) or 0.0)
        pos.pending_exit_fee_usd = fee_usd
        pos.pending_exit_slippage_bps = float(getattr(fill, "slippage_bps", 0.0) or 0.0)
        pos.pending_exit_recorded_at = now

    def _exit_retry_cooling(self, pos: Position, reason_value: str) -> bool:
        if (
            pos.state == PositionState.CLOSED
            or bool(getattr(pos, "_terminalized", False))
            or bool(getattr(pos, "_terminalizing", False))
        ):
            return True

        next_retry_ts = float(getattr(pos, "next_exit_retry_not_before", 0.0) or 0.0)
        if next_retry_ts > 0 and time.time() < next_retry_ts:
            return True

        signed = self._get_venue_signed_size(pos.coin)
        if signed is not None and abs(float(signed)) <= LIVE_FLAT_EPSILON_SZ:
            reconcile_reason = (
                str(getattr(pos, "exit_requested_reason", "")).strip().lower()
                or reason_value.lower()
                or CloseReason.STOP_FULL.value
            )
            self.live_monitor._finalize_closed_position(
                pos,
                source="wallet_flat_reconcile",
                reason=reconcile_reason,
            )
            return True

        if str(getattr(pos, "exit_requested_reason", "")).strip().lower() != reason_value.lower():
            return False
        ts = getattr(pos, "exit_requested_at", None)
        if ts is None:
            return False
        elapsed = (datetime.now(timezone.utc) - ts).total_seconds()
        return elapsed < LIVE_EXIT_RETRY_COOLDOWN_SEC

    def _get_venue_signed_size(self, coin: str) -> Optional[float]:
        fn = getattr(self.backend, "_get_venue_signed_size", None)
        if fn is None:
            return None
        try:
            szi = fn(coin)
        except Exception as e:
            print(f"[LIVE] venue signed size check failed for {coin}: {e}")
            return None
        if szi is None:
            return None
        return float(szi)

    @staticmethod
    def _extract_float_from_paths(blob: Dict[str, Any], paths: List[Tuple[str, ...]]) -> Optional[float]:
        for path in paths:
            node: Any = blob
            valid = True
            for key in path:
                if not isinstance(node, dict) or key not in node:
                    valid = False
                    break
                node = node.get(key)
            if not valid or node in (None, ""):
                continue
            try:
                return float(node)
            except (TypeError, ValueError):
                continue
        return None

    def _apply_runtime_balance_from_venue(
        self,
        venue_margin: Tuple[float, float, float],
        source: str = "",
    ) -> float:
        venue_equity, used_margin_val, available_margin = venue_margin
        local_balance = float(getattr(self.risk, "balance", 0.0) or 0.0)

        if float(venue_equity or 0.0) <= 0:
            if self._live_mode:
                self.venue_sync_unhealthy = True
                self.risk.balance = 0.0
                print(
                    f"[EXECUTOR][WARNING] LIVE venue sync unhealthy: "
                    f"non-positive venue equity ({float(venue_equity or 0.0):.4f}) "
                    f"source={source or 'runtime'} -> runtime_balance forced to 0.0 "
                    f"(used_margin=${float(used_margin_val):.4f}, available_margin=${float(available_margin):.4f})"
                )
                return 0.0

            print(
                f"[EXECUTOR] runtime balance fallback -> local balance "
                f"(source={source or 'runtime'}, reason=non_positive_venue_equity, "
                f"local=${local_balance:.2f})"
            )
            return local_balance

        self.venue_sync_unhealthy = False
        self._balance_ready = True
        local_ledger_balance = float(
            getattr(self.risk, "ledger_balance", local_balance) or 0.0
        )
        runtime_balance = float(venue_equity)
        self.risk.balance = runtime_balance

        venue_hwm = float(getattr(self, "_venue_equity_hwm", 0.0) or 0.0)
        if source == "boot" or venue_hwm <= 0:
            venue_hwm = runtime_balance
        elif runtime_balance > venue_hwm:
            venue_hwm = runtime_balance
        self._venue_equity_hwm = venue_hwm
        self.risk.high_water_mark = venue_hwm

        print(
            f"[EXECUTOR] balance_sync source={source or 'runtime'} "
            f"venue_equity=${runtime_balance:.2f} "
            f"used_margin=${float(used_margin_val):.2f} "
            f"available_margin=${float(available_margin):.2f} "
            f"local_ledger=${local_ledger_balance:.2f} "
            f"risk.balance set to ${runtime_balance:.2f}"
        )
        return runtime_balance

    def _refresh_runtime_balance_from_venue(self, source: str = "") -> Optional[Tuple[float, float, float]]:
        venue_margin = self._get_venue_available_margin()
        if venue_margin is None:
            if self._live_mode:
                self.venue_sync_unhealthy = True
                self.risk.balance = 0.0
                print(
                    f"[EXECUTOR][WARNING] LIVE venue sync unhealthy: "
                    f"venue margin unavailable source={source or 'runtime'} "
                    "-> runtime_balance forced to 0.0"
                )
                return None
            print(
                f"[EXECUTOR] runtime balance fallback -> local balance "
                f"(source={source or 'runtime'}, reason=missing_venue_margin_snapshot, "
                f"local=${float(getattr(self.risk, 'balance', 0.0) or 0.0):.2f})"
            )
            return None
        self._log_margin_account_debug(source=source, venue_margin=venue_margin)
        self._apply_runtime_balance_from_venue(venue_margin, source=source)
        return venue_margin

    def _log_margin_account_debug(self, source: str, venue_margin: Tuple[float, float, float]):
        if not (source == "boot" or str(source).startswith("pre_entry:")):
            return
        meta = getattr(self, "_last_margin_query_meta", {}) or {}
        venue_label = str(meta.get("venue", "")).strip() or (
            "TESTNET" if os.getenv("HL_TESTNET", "true").lower() == "true" else "MAINNET"
        )
        query_address = str(meta.get("query_address", "")).strip() or "unknown"
        account_address = str(meta.get("account_address", "")).strip() or os.getenv("HL_ACCOUNT_ADDRESS", "").strip()
        vault_or_subaccount = str(meta.get("vault_or_subaccount", "")).strip() or (
            os.getenv("HL_VAULT_ADDRESS", "").strip() or "none"
        )
        raw_margin_summary = meta.get("raw_margin_summary", {})
        venue_equity, used_margin, available_margin = venue_margin
        print(
            f"[MARGIN_ACCOUNT_DEBUG] source={source} "
            f"venue={venue_label} queried_account={query_address} "
            f"account_address={account_address} vault_or_subaccount={vault_or_subaccount} "
            f"equity=${float(venue_equity):.2f} available_margin=${float(available_margin):.2f} "
            f"used_margin=${float(used_margin):.2f} raw_margin_summary={raw_margin_summary}"
        )

    def _venue_margin_snapshot_with_warning(self, reason: str = "") -> Optional[Tuple[float, float, float]]:
        snapshot = getattr(self, "_venue_margin_cache", None)
        if snapshot is None:
            return None
        age_sec = max(0.0, time.time() - float(getattr(self, "_venue_margin_cache_ts", 0.0) or 0.0))
        print(
            f"[EXECUTOR] venue margin fallback (cached) reason={reason or 'unknown'} "
            f"age={age_sec:.1f}s"
        )
        return snapshot

    def _venue_margin_snapshot(self, coin: str) -> Optional[Tuple[float, float, float]]:
        # Margin is account-level (not coin-specific); keep coin for API compatibility.
        _ = coin
        return getattr(self, "_venue_margin_cache", None)

    def _get_venue_available_margin(self) -> Optional[Tuple[float, float, float]]:
        """
        Delegates to backend.get_margin_summary() which uses the proven
        self.info.user_state() path (same as _get_venue_signed_size).
        Returns None on any failure — callers skip the margin constraint.
        """
        fn = getattr(self.backend, "get_margin_summary", None)
        if not callable(fn):
            print("[EXECUTOR] backend has no get_margin_summary — skipping margin constraint")
            return self._venue_margin_snapshot_with_warning("backend_no_get_margin_summary")

        try:
            result = fn()
        except Exception as e:
            print(f"[EXECUTOR] get_margin_summary raised: {e}")
            return self._venue_margin_snapshot_with_warning("get_margin_summary_exception")

        if result is None:
            return self._venue_margin_snapshot_with_warning("get_margin_summary_returned_none")

        self._venue_margin_cache = result
        self._venue_margin_cache_ts = time.time()
        return result

    def _apply_available_margin_sizing(self, decision, coin: str) -> Optional[str]:
        venue_margin = self._get_venue_available_margin()
        if venue_margin is None:
            if self._live_mode:
                return "insufficient_available_margin"
            return None

        venue_equity, used_margin, available_balance = venue_margin
        self._apply_runtime_balance_from_venue(venue_margin, source=f"margin:{coin}")
        if available_balance <= 0:
            return "insufficient_available_margin"

        old_size = float(getattr(decision, "size_usd", 0.0) or 0.0)
        old_risk = float(getattr(decision, "risk_usd", 0.0) or 0.0)
        if old_size <= 0 or old_risk <= 0:
            return None

        # Available margin is the primary constraint. Keep a small safety buffer to avoid
        # false rejects from rounding/fees at the venue.
        margin_buffer_mult = 0.95
        max_by_available = max(0.0, float(available_balance) * margin_buffer_mult)

        # Venue equity is a secondary constraint to avoid outsized notional from stale local state.
        max_by_equity = max_by_available
        if float(venue_equity) > 0:
            max_by_equity = min(max_by_available, float(venue_equity) * margin_buffer_mult)

        requested_notional = old_size
        required_margin = requested_notional
        if max_by_equity <= 0:
            return "insufficient_available_margin"

        # Minimum viable trade check: near-threshold requests are clamped, not rejected.
        target_notional = min(requested_notional, max_by_equity)
        if target_notional <= 0:
            return "insufficient_available_margin"

        scale = target_notional / requested_notional if requested_notional > 0 else 0.0
        decision.size_usd = round(target_notional, 2)
        decision.risk_usd = round(old_risk * scale, 2)
        if decision.risk_usd <= 0 and target_notional > 0:
            decision.risk_usd = 0.01

        print(
            f"[ MARGIN ] equity={float(venue_equity):.2f} "
            f"available={float(available_balance):.2f} "
            f"requested_notional={requested_notional:.2f} "
            f"required_margin={required_margin:.2f} "
            f"queried_account={str((getattr(self, '_last_margin_query_meta', {}) or {}).get('query_address', 'unknown'))}"
        )
        print(
            f"[EXECUTOR] {coin} margin sizing cap: "
            f"total=${venue_equity:.2f} used=${used_margin:.2f} "
            f"avail=${available_balance:.2f} scale={scale:.3f}x "
            f"size ${old_size:.2f}->{decision.size_usd:.2f}"
        )
        if scale < 1.0:
            decision.reason += (
                f" | avail_margin_cap={scale:.2f}x"
                f" avail=${available_balance:.2f}"
            )

        if decision.size_usd <= 0 or decision.risk_usd <= 0:
            return "insufficient_available_margin"
        return None

    def _get_venue_size(self, pos: Position) -> Optional[float]:
        signed = self._get_venue_signed_size(pos.coin)
        if signed is not None:
            if str(pos.side).upper() == "LONG":
                return abs(signed) if signed > 0 else 0.0
            return abs(signed) if signed < 0 else 0.0

        fn = getattr(self.backend, "_get_venue_position_size", None)
        if fn is None:
            return None
        try:
            sz = fn(pos.coin, pos.side)
        except Exception as e:
            print(f"[LIVE] venue size check failed for {pos.coin} {pos.side}: {e}")
            return None
        if sz is None:
            return None
        return float(sz)

    def _is_venue_flat(self, pos: Position) -> Optional[bool]:
        signed = self._get_venue_signed_size(pos.coin)
        if signed is not None:
            return abs(signed) <= LIVE_FLAT_EPSILON_SZ

        venue_sz = self._get_venue_size(pos)
        if venue_sz is None:
            return None
        if venue_sz > LIVE_FLAT_EPSILON_SZ:
            return False

        fn = getattr(self.backend, "_get_venue_position_size", None)
        if fn is None:
            return True

        opposite = "SHORT" if str(pos.side).upper() == "LONG" else "LONG"
        try:
            opp_sz = fn(pos.coin, opposite)
        except Exception as e:
            print(f"[LIVE] venue opposite-side check failed for {pos.coin} {opposite}: {e}")
            return None
        if opp_sz is None:
            return None
        return abs(float(opp_sz)) <= LIVE_FLAT_EPSILON_SZ

    def _handle_exit_reject(
        self,
        pos: Position,
        reason_value: str,
        requested_size_usd: float,
        reject_reason: str,
        tracked_order_id: str = "",
        reject_meta: Optional[Dict[str, float]] = None,
    ):
        if (
            pos.state == PositionState.CLOSED
            or bool(getattr(pos, "_terminalized", False))
            or bool(getattr(pos, "_terminalizing", False))
        ):
            return

        normalized = str(reject_reason).strip().lower()
        if normalized != "size_rounds_to_zero_or_no_position":
            return

        reject_meta = reject_meta if isinstance(reject_meta, dict) else {}
        signed = self._get_venue_signed_size(pos.coin)
        signed_hint = reject_meta.get("signed_venue_size")
        if signed is None and signed_hint is not None:
            try:
                signed = float(signed_hint)
            except (TypeError, ValueError):
                signed = None
        if signed is None and bool(reject_meta.get("wallet_flat_hint", False)):
            signed = 0.0

        side = str(pos.side).upper()
        if signed is None:
            pos.exit_trigger_source = "wallet_desync_detected"
            unknown_count = int(getattr(pos, "exit_unknown_reject_count", 0) or 0) + 1
            pos.exit_unknown_reject_count = unknown_count
            capped_count = min(max(1, unknown_count), max(1, LIVE_EXIT_UNKNOWN_MAX_RETRIES))
            backoff_sec = max(
                LIVE_EXIT_RETRY_COOLDOWN_SEC,
                LIVE_EXIT_UNKNOWN_BACKOFF_SEC * capped_count,
            )
            pos.next_exit_retry_not_before = time.time() + backoff_sec
            print(
                f"[LIVE] EXIT_DIAG {pos.coin} {side} reason={reason_value} "
                f"branch=state_unknown action=backoff retry_in={backoff_sec:.0f}s "
                f"unknown_count={unknown_count}"
            )
            if unknown_count == 1 or unknown_count % max(1, LIVE_EXIT_UNKNOWN_ESCALATE_EVERY) == 0:
                append_trade(pos, paper_mode=False)
            return

        if abs(signed) <= LIVE_FLAT_EPSILON_SZ:
            requested_reason = str(getattr(pos, "exit_requested_reason", "") or "").strip().lower()
            reconcile_reason = requested_reason or CloseReason.STOP_FULL.value
            pos.allow_reconcile_close = True
            pos.reconciled_from_venue = False
            pos.exit_unknown_reject_count = 0
            pos.next_exit_retry_not_before = 0.0

            self.live_monitor._finalize_closed_position(
                pos,
                source="wallet_flat_reconcile",
                reason=reconcile_reason,
            )
            if pos.state == PositionState.CLOSED:
                print(
                    f"[EXIT_RECONCILED_IMMEDIATE] coin={pos.coin} reason=wallet_flat"
                )
                return

            pos.exit_trigger_source = "executor_reconcile_pending"
            pos.next_exit_retry_not_before = time.time() + LIVE_EXIT_RETRY_COOLDOWN_SEC
            print(
                f"[LIVE] EXIT_DIAG {pos.coin} {side} reason={reason_value} "
                f"branch=wallet_flat signed_szi={signed:.8f} action=mark_reconcile_pending"
            )
            append_trade(pos, paper_mode=False)
            return

        pos.exit_unknown_reject_count = 0
        pos.next_exit_retry_not_before = time.time() + LIVE_EXIT_RETRY_COOLDOWN_SEC
        same_side_open = signed > LIVE_FLAT_EPSILON_SZ if side == "LONG" else signed < -LIVE_FLAT_EPSILON_SZ
        opposite_side_open = signed < -LIVE_FLAT_EPSILON_SZ if side == "LONG" else signed > LIVE_FLAT_EPSILON_SZ

        mid = self.backend.get_mid_price(pos.coin)
        est_notional = abs(signed) * mid if mid is not None and mid > 0 else 0.0

        if opposite_side_open:
            pos.exit_trigger_source = "wallet_desync_detected"
            append_trade(pos, paper_mode=False)
            print(
                f"[LIVE] EXIT_DIAG {pos.coin} {side} reason={reason_value} "
                f"branch=side_mismatch signed_szi={signed:.8f} action=keep_tracked"
            )
            return

        if same_side_open and mid is not None and mid > 0 and est_notional <= LIVE_TINY_POSITION_USD:
            pos.exit_trigger_source = "wallet_desync_detected"
            append_trade(pos, paper_mode=False)
            print(
                f"[LIVE] EXIT_DIAG {pos.coin} {side} reason={reason_value} "
                f"branch=tiny_same_side signed_szi={signed:.8f} est_notional=${est_notional:.2f} "
                f"req_notional=${requested_size_usd:.2f} action=keep_tracked"
            )
            return

        if same_side_open:
            pos.exit_trigger_source = "wallet_desync_detected"
            append_trade(pos, paper_mode=False)
            print(
                f"[LIVE] EXIT_DIAG {pos.coin} {side} reason={reason_value} "
                f"branch=same_side_still_open signed_szi={signed:.8f} est_notional=${est_notional:.2f} "
                "action=keep_tracked"
            )
            return

        pos.exit_trigger_source = "wallet_desync_detected"
        append_trade(pos, paper_mode=False)
        print(
            f"[LIVE] EXIT_DIAG {pos.coin} {side} reason={reason_value} "
            f"branch=state_unknown signed_szi={signed:.8f} action=keep_tracked"
        )

    def _expected_remaining_size_after_partial(self, pos: Position) -> float:
        total_coin = self._position_entry_size_coin(pos)
        partial_fraction = float(getattr(pos, "partial_close_fraction", 0.0) or 0.0)
        partial_fraction = max(0.0, min(1.0, partial_fraction))
        remaining_coin = 0.0
        if total_coin > 0 and partial_fraction > 0:
            remaining_coin = total_coin * max(0.0, 1.0 - partial_fraction)
        if remaining_coin <= 0:
            partial_closed_usd = float(getattr(pos, "partial_close_size_usd", 0.0) or 0.0)
            if partial_closed_usd > 0 and float(getattr(pos, "entry_price", 0.0) or 0.0) > 0:
                remaining_coin = max(
                    0.0,
                    total_coin - (partial_closed_usd / float(getattr(pos, "entry_price", 0.0))),
                )
        if remaining_coin <= 0 and total_coin > 0:
            # Legacy fallback for rows created before fractional partial accounting.
            _frac = PARTIAL_CLOSE_FRACTION if ENABLE_PARTIAL_TP else 0.5
            remaining_coin = total_coin * max(0.0, 1.0 - _frac)
        return remaining_coin

    def _resolve_remaining_runner_size_usd(self, pos: Position, fallback_price: float) -> float:
        legacy_remaining_coin = self._expected_remaining_size_after_partial(pos)

        remaining_coin = self._get_venue_size(pos)
        if remaining_coin is None or remaining_coin <= LIVE_FLAT_EPSILON_SZ:
            remaining_coin = legacy_remaining_coin

        px = float(fallback_price or 0.0)
        if px <= 0:
            px = float(self.backend.get_mid_price(pos.coin) or 0.0)
        if px <= 0:
            px = float(getattr(pos, "current_price", 0.0) or 0.0)
        if px <= 0:
            px = float(getattr(pos, "entry_price", 0.0) or 0.0)
        if px <= 0:
            fallback_usd = float(getattr(pos, "size_usd", 0.0) or 0.0) * 0.5
            return max(0.0, fallback_usd)

        resolved_size_usd = float(remaining_coin) * px
        if resolved_size_usd <= 0:
            fallback_usd = float(getattr(pos, "size_usd", 0.0) or 0.0) * 0.5
            return max(0.0, fallback_usd)

        return resolved_size_usd

    @staticmethod
    def _resolve_exit_notional_usd(fill, fallback_size_usd: float) -> float:
        fallback = max(0.0, float(fallback_size_usd or 0.0))
        if fill is None:
            return fallback
        notional = float(getattr(fill, "fill_size_usd", 0.0) or 0.0)
        if notional > 0:
            return notional
        return fallback

    @staticmethod
    def _position_entry_size_coin(pos: Position) -> float:
        entry_px = float(getattr(pos, "entry_price", 0.0) or 0.0)
        total_usd = float(getattr(pos, "size_usd", 0.0) or 0.0)
        if entry_px <= 0 or total_usd <= 0:
            return 0.0
        return max(0.0, total_usd / entry_px)

    def _resolve_exit_size_coin(self, fill, fallback_size_coin: float = 0.0) -> float:
        size_coin = self._coin_size_from_fill(fill)
        if size_coin > 0:
            return size_coin
        return max(0.0, float(fallback_size_coin or 0.0))

    def _close_fraction_from_coin(self, pos: Position, close_size_coin: float, default: float = 0.5) -> float:
        total = self._position_entry_size_coin(pos)
        if total <= 0:
            return max(0.0, min(1.0, float(default)))
        frac = float(close_size_coin or 0.0) / total
        if frac <= 0:
            frac = float(default)
        return max(0.0, min(1.0, frac))

    def _confirm_partial_remaining(self, pos: Position) -> Optional[bool]:
        venue_sz = self._get_venue_size(pos)
        if venue_sz is None:
            return None

        if venue_sz <= LIVE_FLAT_EPSILON_SZ:
            return False

        expected_remaining = self._expected_remaining_size_after_partial(pos)
        if expected_remaining <= 0:
            return True

        lower_bound = expected_remaining * 0.25
        upper_bound = expected_remaining * 1.75

        return lower_bound <= venue_sz <= upper_bound

    def _duration_minutes(self, pos: Position) -> float:
        end = pos.closed_at or datetime.now(timezone.utc)
        delta = end - pos.opened_at
        return max(0.0, delta.total_seconds() / 60.0)

    def _notify_live_partial(
        self,
        pos: Position,
        fill_price: float,
        net_partial_r: float,
        net_partial_usd: float,
        closed_notional_usd: float,
        remaining_notional_usd: float,
    ):
        try:
            msg = (
                f"✂️ *{pos.coin} {pos.side} PARTIAL TP* 🔴 LIVE\n\n"
                f"💰 Exit price: `{fill_price:.5g}`\n"
                f"📥 Entry: `{pos.entry_price:.5g}`\n\n"
                f"```\n"
                f"  R captured  : {net_partial_r:+.2f}R\n"
                f"  P&L         : ${net_partial_usd:+.2f}\n"
                f"  Closed      : ${closed_notional_usd:.0f}\n"
                f"  Remaining   : ${remaining_notional_usd:.0f}\n"
                f"  New stop    : {pos.stop_price:.5g}\n"
                f"```\n"
                f"🆔 `{pos.position_id}`\n"
                f"🕒 `{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}`"
            )
            self.notify(msg)
        except Exception as e:
            print(f"[EXECUTOR] live partial notify failed: {e}")

    def _notify_live_close(
        self,
        pos: Position,
        reason: CloseReason,
        fill_price: float,
        realized_r: float,
        pnl_usd: float,
        slip_bps: float,
    ):
        try:
            duration_min = self._duration_minutes(pos)
            hrs = int(duration_min // 60)
            mins = int(duration_min % 60)
            duration_str = f"{hrs}h {mins}m" if hrs > 0 else f"{mins}m"

            if reason == CloseReason.TP_FULL:
                title = "🎯"
                label = "TARGET HIT"
            elif reason in {CloseReason.STOP_FULL, CloseReason.STOP_RUNNER}:
                title = "🛑"
                label = "STOPPED"
            else:
                title = "⏹"
                label = reason.value.upper()

            msg = (
                f"{title} *{pos.coin} {pos.side} {label}* 🔴 LIVE\n\n"
                f"💰 Exit: `{fill_price:.5g}`\n"
                f"📥 Entry: `{pos.entry_price:.5g}`\n\n"
                f"```\n"
                f"  Reason  : {reason.value}\n"
                f"  R       : {realized_r:+.2f}R\n"
                f"  P&L     : ${pnl_usd:+.2f}\n"
                f"  Size    : ${pos.size_usd:.0f}\n"
                f"  Slip    : {slip_bps:.1f}bps\n"
                f"  Hold    : {duration_str}\n"
                f"```\n"
                f"🆔 `{pos.position_id}`\n"
                f"🕒 `{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}`"
            )
            self.notify(msg)
        except Exception as e:
            print(f"[EXECUTOR] live close notify failed: {e}")

    def _live_take_partial(self, pos: Position, price: float):
        if (
            pos.state == PositionState.CLOSED
            or bool(getattr(pos, "_terminalized", False))
            or bool(getattr(pos, "_terminalizing", False))
        ):
            return

        if self._exit_retry_cooling(pos, "partial"):
            return

        request_px = float(price or 0.0)
        if request_px <= 0:
            request_px = float(self.backend.get_mid_price(pos.coin) or 0.0)
        if request_px <= 0:
            request_px = float(getattr(pos, "current_price", 0.0) or 0.0)
        if request_px <= 0:
            request_px = float(getattr(pos, "entry_price", 0.0) or 0.0)

        _partial_frac = PARTIAL_CLOSE_FRACTION if ENABLE_PARTIAL_TP else 0.5
        live_coin = self._get_venue_size(pos)
        if live_coin is not None and live_coin > LIVE_FLAT_EPSILON_SZ:
            requested_partial_coin = float(live_coin) * _partial_frac
        else:
            requested_partial_coin = self._position_entry_size_coin(pos) * _partial_frac

        if requested_partial_coin > LIVE_FLAT_EPSILON_SZ and request_px > 0:
            requested_partial_usd = requested_partial_coin * request_px
        else:
            requested_partial_usd = max(0.0, float(getattr(pos, "size_usd", 0.0) or 0.0) * _partial_frac)
            requested_partial_coin = (
                (requested_partial_usd / request_px)
                if request_px > 0 and requested_partial_usd > 0
                else 0.0
            )
        self._persist_exit_intent(pos, "partial")
        fill = self._live_execute_exit(
            pos=pos,
            price=price,
            size_usd=requested_partial_usd,
            reason="partial",
        )
        if fill is None:
            return

        partial_confirmed = self._confirm_partial_remaining(pos)
        if partial_confirmed is False:
            print(
                f"[LIVE] PARTIAL UNCONFIRMED {pos.coin} {pos.side} — "
                "venue state inconsistent; keeping pre-partial state for retry/reconcile"
            )
            return
        if partial_confirmed is None:
            print(
                f"[LIVE] PARTIAL UNCERTAIN {pos.coin} {pos.side} — "
                "venue state unknown; skipping local partial state mutation this cycle"
            )
            return

        fill_price = float(fill.fill_price)
        exit_fee_usd = float((fill.meta or {}).get("estimated_fee_usd", 0.0))
        filled_partial_coin = self._resolve_exit_size_coin(fill, requested_partial_coin)
        close_fraction = self._close_fraction_from_coin(
            pos,
            filled_partial_coin,
            default=_partial_frac,
        )
        filled_partial_usd = self._resolve_exit_notional_usd(
            fill,
            filled_partial_coin * fill_price if fill_price > 0 else requested_partial_usd,
        )

        gross_r = pos.current_r(fill_price)
        gross_partial_usd = gross_r * close_fraction * pos.r_value
        net_partial_usd = gross_partial_usd - exit_fee_usd
        net_partial_r = net_partial_usd / pos.r_value if pos.r_value > 0 else 0.0

        pos.partial_r += net_partial_r
        pos.pnl_usd += net_partial_usd
        pos.partial_close_size_usd = round(filled_partial_usd, 8)
        pos.partial_close_fraction = round(close_fraction, 8)
        pos.partial_closed = True
        pos.state = PositionState.PARTIAL
        pos.current_price = fill_price
        if MOVE_STOP_TO_BREAKEVEN_AFTER_PARTIAL:
            pos.move_stop_to_breakeven()
            expected_remaining_coin = self._expected_remaining_size_after_partial(pos)
            be_replace_ok = self.protection_manager.replace_stop_after_partial(
                pos=pos,
                fallback_size_coin=expected_remaining_coin,
                source="partial_be",
            )
            if not be_replace_ok:
                print(
                    f"[LIVE] PARTIAL {pos.coin} {pos.side} WARNING — "
                    "breakeven stop replacement failed; local stop moved only"
                )
        pos.exit_fees_usd = float(getattr(pos, "exit_fees_usd", 0.0)) + exit_fee_usd
        pos.total_fees_usd = round(pos.entry_fee_usd + pos.exit_fees_usd + pos.funding_usd, 8)
        pos.reconciled_from_venue = False

        self.risk.record_partial(
            pos,
            gross_r=gross_r,
            fee_usd=exit_fee_usd,
            close_fraction=close_fraction,
        )
        append_trade(pos, paper_mode=False)

        print(
            f"[LIVE] PARTIAL {pos.coin} {pos.side} fill={fill_price:.4f} "
            f"gross={gross_r:+.2f}R net={net_partial_r:+.2f}R fee=${exit_fee_usd:.2f} "
            f"close_frac={close_fraction:.4f}"
        )

        remaining_notional_usd = max(
            0.0,
            float(getattr(pos, "size_usd", 0.0) or 0.0) - filled_partial_usd,
        )
        self._notify_live_partial(
            pos=pos,
            fill_price=fill_price,
            net_partial_r=net_partial_r,
            net_partial_usd=net_partial_usd,
            closed_notional_usd=filled_partial_usd,
            remaining_notional_usd=remaining_notional_usd,
        )

    def _live_update_trailing_stop(self, pos: Position):
        trail_dist = TRAIL_ATR_MULT * pos.atr
        if trail_dist <= 0:
            return
        if pos.side == "LONG":
            new_stop = pos.peak_price - trail_dist
            if new_stop > pos.stop_price:
                pos.stop_price = new_stop
        else:
            new_stop = pos.peak_price + trail_dist
            if new_stop < pos.stop_price:
                pos.stop_price = new_stop

    def _live_close_full_stop(self, pos: Position, price: float):
        if (
            pos.state == PositionState.CLOSED
            or bool(getattr(pos, "_terminalized", False))
            or bool(getattr(pos, "_terminalizing", False))
        ):
            return

        if self._exit_retry_cooling(pos, CloseReason.STOP_FULL.value):
            return

        self._persist_exit_intent(pos, CloseReason.STOP_FULL.value)
        fill = self._live_execute_exit(
            pos=pos,
            price=price,
            size_usd=pos.size_usd,
            reason=CloseReason.STOP_FULL.value,
        )
        if fill is None:
            return

        self._capture_exit_attempt(pos, fill, CloseReason.STOP_FULL.value)
        append_trade(pos, paper_mode=False)

        flat = self._is_venue_flat(pos)
        if flat is False:
            print(
                f"[LIVE] EXIT PARTIAL {pos.coin} STOP_FULL — still open on venue; "
                "will retry next cycle"
            )
            return
        if flat is None:
            print(
                f"[LIVE] EXIT UNCERTAIN {pos.coin} STOP_FULL — venue state unknown; "
                "keeping position tracked"
            )
            return

        terminal_lock = self._position_terminal_lock(pos)
        with terminal_lock:
            if (
                pos.state == PositionState.CLOSED
                or bool(getattr(pos, "_terminalized", False))
                or bool(getattr(pos, "_terminalizing", False))
            ):
                return

            setattr(pos, "_terminalizing", True)
            try:
                fill_price = float(fill.fill_price)
                exit_fee_usd = float((fill.meta or {}).get("estimated_fee_usd", 0.0))
                slip_bps = float(getattr(fill, "slippage_bps", 0.0) or 0.0)

                actual_r_gross = max(self._r_vs_original(pos, fill_price), MAX_FULL_LOSS_R)
                fee_r = exit_fee_usd / pos.r_value if pos.r_value > 0 else 0.0
                actual_r_net = actual_r_gross - fee_r
            except Exception:
                setattr(pos, "_terminalizing", False)
                raise

            pos.runner_r += actual_r_net
            pos.pnl_usd += actual_r_gross * pos.r_value - exit_fee_usd
            pos.state = PositionState.CLOSED
            pos.close_reason = CloseReason.STOP_FULL
            pos.closed_at = datetime.now(timezone.utc)
            pos.updated_at = pos.closed_at
            pos.current_price = fill_price
            pos.exit_fees_usd = float(getattr(pos, "exit_fees_usd", 0.0)) + exit_fee_usd
            pos.total_fees_usd = round(pos.entry_fee_usd + pos.exit_fees_usd + pos.funding_usd, 8)
            pos.reconciled_from_venue = False
            pos.allow_reconcile_close = False
            pos.wallet_flat_confirmed = True
            pos.wallet_flat_confirmed_at = pos.closed_at
            pos.exit_trigger_source = "executor_triggered"
            setattr(pos, "_terminalized", True)
            setattr(pos, "_terminalizing", False)
            self.order_tracker.reconcile_position_close(
                pos,
                close_reason=CloseReason.STOP_FULL,
                source=pos.exit_trigger_source,
            )

            self.risk.record_live_full_close(pos, realized_r=actual_r_gross, fee_usd=exit_fee_usd)
            self._record_strategy_outcome(pos, won=False)
            append_trade(pos, paper_mode=False)
            self.live_monitor.unregister(pos.position_id)
            self.risk.open_positions.pop(pos.coin, None)

            print(
                f"[LIVE] CLOSE {pos.coin} STOP_FULL fill={fill_price:.4f} "
                f"net_r={actual_r_net:+.2f} fee=${exit_fee_usd:.2f} pnl=${pos.pnl_usd:+.2f}"
            )

            self._notify_live_close(
                pos=pos,
                reason=CloseReason.STOP_FULL,
                fill_price=fill_price,
                realized_r=actual_r_net,
                pnl_usd=pos.pnl_usd,
                slip_bps=slip_bps,
            )

    def _live_close_runner(self, pos: Position, price: float, reason: CloseReason):
        if (
            pos.state == PositionState.CLOSED
            or bool(getattr(pos, "_terminalized", False))
            or bool(getattr(pos, "_terminalizing", False))
        ):
            return

        if self._exit_retry_cooling(pos, reason.value):
            return

        self._persist_exit_intent(pos, reason.value)
        runner_size_usd = self._resolve_remaining_runner_size_usd(
            pos,
            fallback_price=price,
        )
        fill = self._live_execute_exit(
            pos=pos,
            price=price,
            size_usd=runner_size_usd,
            reason=reason.value,
        )
        if fill is None:
            return

        self._capture_exit_attempt(pos, fill, reason.value)
        append_trade(pos, paper_mode=False)

        flat = self._is_venue_flat(pos)
        if flat is False:
            print(
                f"[LIVE] EXIT PARTIAL {pos.coin} {reason.value.upper()} — still open on venue; "
                "will retry next cycle"
            )
            return
        if flat is None:
            print(
                f"[LIVE] EXIT UNCERTAIN {pos.coin} {reason.value.upper()} — venue state unknown; "
                "keeping position tracked"
            )
            return

        terminal_lock = self._position_terminal_lock(pos)
        with terminal_lock:
            if (
                pos.state == PositionState.CLOSED
                or bool(getattr(pos, "_terminalized", False))
                or bool(getattr(pos, "_terminalizing", False))
            ):
                return

            setattr(pos, "_terminalizing", True)
            try:
                fill_price = float(fill.fill_price)
                exit_fee_usd = float((fill.meta or {}).get("estimated_fee_usd", 0.0))
                slip_bps = float(getattr(fill, "slippage_bps", 0.0) or 0.0)
                request_px = float(price or 0.0)
                if request_px <= 0:
                    request_px = fill_price
                if request_px <= 0:
                    request_px = float(self.backend.get_mid_price(pos.coin) or 0.0)
                requested_runner_coin = (
                    (runner_size_usd / request_px)
                    if request_px > 0 and runner_size_usd > 0
                    else 0.0
                )
                filled_runner_coin = self._resolve_exit_size_coin(fill, requested_runner_coin)
                if not bool(getattr(pos, "partial_closed", False)):
                    default_runner_frac = 1.0
                else:
                    partial_frac = float(getattr(pos, "partial_close_fraction", 0.0) or 0.0)
                    if partial_frac <= 0:
                        partial_size_usd = float(getattr(pos, "partial_close_size_usd", 0.0) or 0.0)
                        entry_px = float(getattr(pos, "entry_price", 0.0) or 0.0)
                        total_coin = self._position_entry_size_coin(pos)
                        if partial_size_usd > 0 and entry_px > 0 and total_coin > 0:
                            partial_frac = (partial_size_usd / entry_px) / total_coin
                    if partial_frac <= 0:
                        partial_frac = 0.5
                    default_runner_frac = max(0.0, 1.0 - max(0.0, min(1.0, partial_frac)))
                close_fraction = self._close_fraction_from_coin(
                    pos,
                    filled_runner_coin,
                    default=default_runner_frac,
                )

                actual_runner_r_gross = self._r_vs_original(pos, fill_price)
                gross_runner_usd = actual_runner_r_gross * close_fraction * pos.r_value
                net_runner_usd = gross_runner_usd - exit_fee_usd
                net_runner_r = net_runner_usd / pos.r_value if pos.r_value > 0 else 0.0
            except Exception:
                setattr(pos, "_terminalizing", False)
                raise

            pos.runner_r += net_runner_r
            pos.pnl_usd += net_runner_usd
            pos.state = PositionState.CLOSED
            pos.close_reason = reason
            pos.closed_at = datetime.now(timezone.utc)
            pos.updated_at = pos.closed_at
            pos.current_price = fill_price
            pos.exit_fees_usd = float(getattr(pos, "exit_fees_usd", 0.0)) + exit_fee_usd
            pos.total_fees_usd = round(pos.entry_fee_usd + pos.exit_fees_usd + pos.funding_usd, 8)
            pos.reconciled_from_venue = False
            pos.allow_reconcile_close = False
            pos.wallet_flat_confirmed = True
            pos.wallet_flat_confirmed_at = pos.closed_at
            pos.exit_trigger_source = "executor_triggered"
            setattr(pos, "_terminalized", True)
            setattr(pos, "_terminalizing", False)
            self.order_tracker.reconcile_position_close(
                pos,
                close_reason=reason,
                source=pos.exit_trigger_source,
            )

            if reason == CloseReason.TP_FULL and not bool(getattr(pos, "partial_closed", False)):
                self.risk.record_live_full_close(
                    pos,
                    realized_r=actual_runner_r_gross,
                    fee_usd=exit_fee_usd,
                    close_fraction=close_fraction,
                )
            else:
                self.risk.record_close(
                    pos,
                    runner_r=actual_runner_r_gross,
                    fee_usd=exit_fee_usd,
                    close_fraction=close_fraction,
                )
            if reason not in {CloseReason.MANUAL}:
                self._record_strategy_outcome(pos, won=pos.realized_r > 0)
            append_trade(pos, paper_mode=False)
            self.live_monitor.unregister(pos.position_id)
            self.risk.open_positions.pop(pos.coin, None)

            print(
                f"[LIVE] CLOSE {pos.coin} {reason.value.upper()} fill={fill_price:.4f} "
                f"total_r={pos.realized_r:+.2f} fee=${exit_fee_usd:.2f} pnl=${pos.pnl_usd:+.2f}"
            )

            self._notify_live_close(
                pos=pos,
                reason=reason,
                fill_price=fill_price,
                realized_r=pos.realized_r,
                pnl_usd=pos.pnl_usd,
                slip_bps=slip_bps,
            )

    def _record_strategy_outcome(self, pos: Position, won: bool):
        strategy_filter = getattr(self.risk, "strategy_filter", None)
        if strategy_filter is None:
            pass
        else:
            try:
                strategy_filter.record_outcome(
                    pos.coin,
                    pos.setup_family,
                    pos.htf_regime,
                    won=won,
                )
            except Exception as e:
                print(f"[EXECUTOR] strategy_filter outcome update failed: {e}")

        try:
            signal_engine = getattr(self, "signal_engine", None)
            recorder = getattr(signal_engine, "record_directional_outcome", None)
            if signal_engine is not None and callable(recorder):
                recorder(
                    coin=str(getattr(pos, "coin", "")).upper(),
                    side=str(getattr(pos, "side", "")).upper(),
                    won=bool(won),
                    setup_family=str(getattr(pos, "setup_family", "")).strip().lower(),
                )
        except Exception as e:
            print(f"[EXECUTOR] directional outcome update failed: {e}")

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from enum import Enum


class PositionState(str, Enum):
    OPEN = "open"
    PARTIAL = "partial"
    CLOSED = "closed"
    CANCELLED = "cancelled"


class CloseReason(str, Enum):
    TP_FULL = "tp_full"
    STOP_FULL = "stop_full"
    STOP_RUNNER = "stop_runner"
    MANUAL = "manual"
    DAILY_HALT = "daily_halt"
    DD_HALT = "dd_halt"
    STALE_SIGNAL = "stale_signal"


def _utc() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Position:
    # Identity
    position_id: str
    coin: str
    side: str
    signal_id: int

    # Levels
    entry_price: float
    stop_price: float
    tp_price: float
    atr: float

    # ORIGINAL levels must never mutate
    original_stop_price: float = 0.0
    original_tp_price: float = 0.0

    # Sizing
    size_usd: float = 0.0
    risk_usd: float = 0.0
    r_value: float = 0.0

    # State
    state: PositionState = PositionState.OPEN
    close_reason: Optional[CloseReason] = None

    # Tracking
    partial_closed: bool = False
    breakeven_moved: bool = False
    current_price: float = 0.0
    peak_price: float = 0.0

    # First-cycle stale detection
    first_update_pending: bool = True
    stale_invalidated: bool = False

    # P&L
    partial_r: float = 0.0
    runner_r: float = 0.0
    pnl_usd: float = 0.0
    partial_close_size_usd: float = 0.0
    partial_close_fraction: float = 0.0

    # ── Fee / funding accounting (v2) ────────────────────────────────
    # Previously set dynamically by executor — now declared so they
    # appear in to_dict() and survive serialisation round-trips.
    entry_fee_usd: float = 0.0
    exit_fees_usd: float = 0.0
    funding_usd: float = 0.0
    total_fees_usd: float = 0.0
    last_funding_ts: Optional[datetime] = None

    # Timestamps
    opened_at: datetime = field(default_factory=_utc)
    closed_at: Optional[datetime] = None
    updated_at: datetime = field(default_factory=_utc)

    # Signal metadata
    regime: str = ""
    setup_family: str = ""
    htf_regime: str = ""
    confidence: float = 0.0
    total_score: float = 0.0
    session: str = ""
    timeframe: str = "15m"
    execution_track: str = "intraday"
    size_multiplier: float = 1.0
    entry_order_ids: str = ""
    exit_order_ids: str = ""
    stop_order_id: str = ""
    tp_order_id: str = ""
    protection_status: str = ""
    protection_placed_at: Optional[datetime] = None
    protection_error: str = ""
    venue_protection_mode: bool = False
    allow_reconcile_close: bool = False
    reconciled_from_venue: bool = False
    exit_requested_reason: str = ""
    exit_requested_at: Optional[datetime] = None
    exit_trigger_source: str = ""
    wallet_flat_confirmed: bool = False
    wallet_flat_confirmed_at: Optional[datetime] = None
    pending_exit_reason: str = ""
    pending_exit_fill_price: float = 0.0
    pending_exit_fill_size_usd: float = 0.0
    pending_exit_fee_usd: float = 0.0
    pending_exit_slippage_bps: float = 0.0
    pending_exit_recorded_at: Optional[datetime] = None

    def __post_init__(self):
        if self.original_stop_price == 0.0:
            self.original_stop_price = self.stop_price
        if self.original_tp_price == 0.0:
            self.original_tp_price = self.tp_price
        if self.current_price == 0.0:
            self.current_price = self.entry_price
        if self.peak_price == 0.0:
            self.peak_price = self.entry_price

    @property
    def stop_dist(self) -> float:
        """
        Always use ORIGINAL stop distance for R math.
        Current stop_price may move to breakeven or trail.
        """
        return abs(self.entry_price - self.original_stop_price)

    @property
    def current_stop_dist(self) -> float:
        return abs(self.entry_price - self.stop_price)

    @property
    def tp_dist(self) -> float:
        return abs(self.original_tp_price - self.entry_price)

    @property
    def rr_planned(self) -> float:
        return self.tp_dist / self.stop_dist if self.stop_dist > 0 else 0.0

    @property
    def realized_r(self) -> float:
        return self.partial_r + self.runner_r

    @property
    def is_open(self) -> bool:
        return self.state in (PositionState.OPEN, PositionState.PARTIAL)

    @property
    def is_closed(self) -> bool:
        return self.state == PositionState.CLOSED

    def current_r(self, price: float) -> float:
        """
        R multiple relative to ORIGINAL stop distance.
        This stays stable even after stop moves to breakeven.
        """
        if self.stop_dist <= 0:
            return 0.0
        if self.side == "LONG":
            return (price - self.entry_price) / self.stop_dist
        return (self.entry_price - price) / self.stop_dist

    def update_price(self, price: float):
        self.current_price = price
        self.updated_at = _utc()

        if self.side == "LONG":
            self.peak_price = max(self.peak_price or price, price)
        else:
            self.peak_price = min(self.peak_price or price, price)

    def move_stop_to_breakeven(self):
        self.stop_price = self.entry_price
        self.breakeven_moved = True

    def invalidate_as_stale(self):
        self.state = PositionState.CANCELLED
        self.close_reason = CloseReason.STALE_SIGNAL
        self.stale_invalidated = True
        self.closed_at = _utc()
        self.pnl_usd = 0.0
        self.partial_r = 0.0
        self.runner_r = 0.0

    def to_dict(self) -> dict:
        return {
            "position_id": self.position_id,
            "coin": self.coin,
            "side": self.side,
            "signal_id": self.signal_id,
            "entry_price": self.entry_price,
            "stop_price": self.stop_price,
            "tp_price": self.tp_price,
            "original_stop_price": self.original_stop_price,
            "original_tp_price": self.original_tp_price,
            "atr": self.atr,
            "size_usd": self.size_usd,
            "risk_usd": self.risk_usd,
            "r_value": self.r_value,
            "state": self.state.value,
            "close_reason": self.close_reason.value if self.close_reason else "",
            "partial_r": round(self.partial_r, 4),
            "runner_r": round(self.runner_r, 4),
            "realized_r": round(self.realized_r, 4),
            "pnl_usd": round(self.pnl_usd, 2),
            "partial_close_size_usd": round(self.partial_close_size_usd, 8),
            "partial_close_fraction": round(self.partial_close_fraction, 8),
            "size_multiplier": self.size_multiplier,
            "partial_closed": self.partial_closed,
            "breakeven_moved": self.breakeven_moved,
            "current_price": self.current_price,
            "first_update_pending": self.first_update_pending,
            "stale_invalidated": self.stale_invalidated,
            # ── Fee fields (v2) ──────────────────────────────────────
            "entry_fee_usd": round(self.entry_fee_usd, 8),
            "exit_fees_usd": round(self.exit_fees_usd, 8),
            "funding_usd": round(self.funding_usd, 8),
            "total_fees_usd": round(self.total_fees_usd, 8),
            # ─────────────────────────────────────────────────────────
            "regime": self.regime,
            "setup_family": self.setup_family,
            "htf_regime": self.htf_regime,
            "confidence": self.confidence,
            "total_score": self.total_score,
            "session": self.session,
            "timeframe": self.timeframe,
            "execution_track": self.execution_track,
            "rr_planned": round(self.rr_planned, 3),
            "entry_order_ids": self.entry_order_ids,
            "exit_order_ids": self.exit_order_ids,
            "stop_order_id": self.stop_order_id,
            "tp_order_id": self.tp_order_id,
            "protection_status": self.protection_status,
            "protection_placed_at": (
                self.protection_placed_at.isoformat()
                if self.protection_placed_at else ""
            ),
            "protection_error": self.protection_error,
            "venue_protection_mode": self.venue_protection_mode,
            "allow_reconcile_close": self.allow_reconcile_close,
            "reconciled_from_venue": self.reconciled_from_venue,
            "exit_requested_reason": self.exit_requested_reason,
            "exit_requested_at": (
                self.exit_requested_at.isoformat() if self.exit_requested_at else ""
            ),
            "exit_trigger_source": self.exit_trigger_source,
            "wallet_flat_confirmed": self.wallet_flat_confirmed,
            "wallet_flat_confirmed_at": (
                self.wallet_flat_confirmed_at.isoformat()
                if self.wallet_flat_confirmed_at else ""
            ),
            "pending_exit_reason": self.pending_exit_reason,
            "pending_exit_fill_price": round(self.pending_exit_fill_price, 8),
            "pending_exit_fill_size_usd": round(self.pending_exit_fill_size_usd, 8),
            "pending_exit_fee_usd": round(self.pending_exit_fee_usd, 8),
            "pending_exit_slippage_bps": round(self.pending_exit_slippage_bps, 4),
            "pending_exit_recorded_at": (
                self.pending_exit_recorded_at.isoformat()
                if self.pending_exit_recorded_at else ""
            ),
            "opened_at": self.opened_at.isoformat(),
            "closed_at": self.closed_at.isoformat() if self.closed_at else "",
        }

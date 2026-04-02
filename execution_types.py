# execution_types.py
"""
Shared data types for the execution layer.

These are the contracts between the signal engine, the executor,
and any execution backend (paper or live). Every backend must
speak in these types.

IMPORTANT — size_usd semantics:
    size_usd = NOTIONAL EXPOSURE, not margin posted.

    Example: SOL at $140, 10x leverage, risking $20 (2% of $1000 balance)
      stop_distance_pct = 1.4%
      size_usd = risk_usd / stop_distance_pct = $20 / 0.014 = $1,428 notional
      margin_required = $1,428 / 10 = $142.80

    This means:
      - pnl_usd = percent_move * size_usd  (correct, uses notional)
      - pnl_r = price_move / stop_distance  (correct, pure R-multiple)
      - risk_usd = size_usd * stop_distance_pct (what you actually lose at -1R)

    Both pnl_usd and pnl_r are consistent because they're both derived
    from the same entry/exit/stop triangle — just expressed differently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any
from datetime import datetime, timezone


class OrderSide(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class FillStatus(str, Enum):
    FILLED = "FILLED"
    PARTIAL = "PARTIAL"
    REJECTED = "REJECTED"
    PENDING = "PENDING"


class PositionStatus(str, Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    STOPPED = "STOPPED"
    TP_HIT = "TP_HIT"
    EXPIRED = "EXPIRED"


@dataclass
class OrderRequest:
    """What the executor sends to a backend."""
    coin: str
    side: OrderSide
    order_type: OrderType
    price: float                    # entry price (limit) or reference (market)
    size_usd: float                 # NOTIONAL position size in USD
    stop_price: float
    tp_price: float
    risk_usd: float                 # actual dollar risk at -1R (size * stop_dist_pct)
    confidence: float
    signal_id: str
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def stop_distance_pct(self) -> float:
        """Stop distance as fraction of entry price."""
        if self.price <= 0:
            return 0.0
        return abs(self.price - self.stop_price) / self.price


@dataclass
class FillResult:
    """What the backend returns after attempting to fill."""
    signal_id: str
    status: FillStatus
    fill_price: float               # actual fill price (includes slippage)
    fill_size_usd: float            # actual filled notional size
    requested_size_usd: float       # what was requested
    slippage_bps: float             # realized slippage in basis points
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    reject_reason: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def fill_ratio(self) -> float:
        if self.requested_size_usd <= 0:
            return 0.0
        return self.fill_size_usd / self.requested_size_usd


@dataclass
class Position:
    """A live or paper position tracked by the backend."""
    position_id: str
    coin: str
    side: OrderSide
    entry_price: float
    size_usd: float                 # NOTIONAL exposure
    stop_price: float
    tp_price: float
    risk_usd: float = 0.0          # actual dollar risk at -1R
    status: PositionStatus = PositionStatus.OPEN
    entry_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    pnl_usd: float = 0.0
    pnl_r: float = 0.0
    signal_id: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def stop_dist(self) -> float:
        if self.side == OrderSide.LONG:
            return self.entry_price - self.stop_price
        return self.stop_price - self.entry_price

    @property
    def is_open(self) -> bool:
        return self.status == PositionStatus.OPEN

    def mark_to_market(self, current_price: float) -> float:
        """
        Unrealized PnL in USD.
        Uses notional exposure: pnl = percent_move * size_usd
        """
        if self.entry_price <= 0:
            return 0.0
        if self.side == OrderSide.LONG:
            return (current_price - self.entry_price) / self.entry_price * self.size_usd
        return (self.entry_price - current_price) / self.entry_price * self.size_usd

    def close(self, exit_price: float, status: PositionStatus = PositionStatus.CLOSED):
        """
        Close the position and compute realized PnL.

        pnl_usd: dollar P&L based on notional exposure
            = percent_move * size_usd

        pnl_r: R-multiple based on price move vs stop distance
            = price_move / stop_distance

        Both are consistent — pnl_usd / risk_usd ≈ pnl_r
        (exact when no slippage on exit).
        """
        self.exit_price = exit_price
        self.exit_time = datetime.now(timezone.utc)
        self.status = status

        # Dollar PnL from notional
        self.pnl_usd = self.mark_to_market(exit_price)

        # R-multiple from price geometry
        sd = self.stop_dist
        if sd > 0:
            if self.side == OrderSide.LONG:
                self.pnl_r = (exit_price - self.entry_price) / sd
            else:
                self.pnl_r = (self.entry_price - exit_price) / sd
        else:
            self.pnl_r = 0.0


@dataclass
class AccountState:
    """Snapshot of account / portfolio state."""
    balance_usd: float              # realized balance (cash)
    equity_usd: float               # balance + unrealized PnL
    open_positions: int
    total_exposure_usd: float       # sum of open position notional sizes
    max_positions: int
    daily_pnl_usd: float = 0.0
    total_pnl_usd: float = 0.0
    unrealized_pnl_usd: float = 0.0  # sum of open position MTM
# paper_backend.py
"""
Paper trading execution backend.

FIXES from audit:
  1. Risk check uses risk_usd from OrderRequest, not notional vs balance
  2. AccountState.equity_usd includes unrealized MTM from open positions
  3. Trade log now persists fill details (slippage, fill_ratio, requested vs filled)
  4. Exit simulation includes gap-through behavior for volatile markets
  5. Explicit documentation that size_usd = notional exposure
"""

from __future__ import annotations

import csv
import os
import random
import uuid
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict

from execution_backend import ExecutionBackend
from execution_types import (
    OrderRequest,
    OrderSide,
    OrderType,
    FillResult,
    FillStatus,
    Position,
    PositionStatus,
    AccountState,
)


# ── Defaults ───────────────────────────────────────────────────────────────────

DEFAULT_BALANCE = 10_000.0
DEFAULT_MAX_POSITIONS = 5
DEFAULT_MAX_RISK_PCT = 2.0             # max risk_usd as % of balance per trade

# Slippage simulation
DEFAULT_SLIPPAGE_MIN_BPS = 5           # best case: 0.05%
DEFAULT_SLIPPAGE_MAX_BPS = 25          # worst case: 0.25%
DEFAULT_SLIPPAGE_MEAN_BPS = 12         # average: ~0.12%

# Partial fill simulation
DEFAULT_MIN_FILL_RATIO = 0.60
DEFAULT_FILL_BASE = 0.80
DEFAULT_FILL_CONFIDENCE_BONUS = 0.15

# Gap-through simulation
DEFAULT_GAP_THROUGH_PROBABILITY = 0.08  # 8% chance of gap-through on stop/TP
DEFAULT_GAP_THROUGH_MAX_MULT = 0.3     # gap can extend up to 30% beyond stop distance

# Persistence
TRADES_FILE = "trades.csv"
BALANCE_FILE = "paper_balance.txt"

# Trade log fields — richer than before for audit trail
TRADE_LOG_FIELDS = [
    "timestamp", "action", "position_id", "signal_id",
    "coin", "side", "entry_price", "exit_price",
    "size_usd", "risk_usd", "stop_price", "tp_price",
    "status", "pnl_usd", "pnl_r", "balance",
    # NEW: fill detail fields for paper/live comparison
    "requested_price", "requested_size_usd",
    "fill_price", "fill_size_usd", "fill_ratio", "slippage_bps",
    "gap_through", "gap_amount_bps",
]


class PaperBackend(ExecutionBackend):
    """
    Paper execution backend with realistic fill simulation.

    All size_usd values are NOTIONAL EXPOSURE (not margin).
    Risk checks use risk_usd (actual dollars at risk at -1R).
    """

    def __init__(
        self,
        balance: Optional[float] = None,
        max_positions: int = DEFAULT_MAX_POSITIONS,
        max_risk_pct: float = DEFAULT_MAX_RISK_PCT,
        slippage_min_bps: float = DEFAULT_SLIPPAGE_MIN_BPS,
        slippage_max_bps: float = DEFAULT_SLIPPAGE_MAX_BPS,
        slippage_mean_bps: float = DEFAULT_SLIPPAGE_MEAN_BPS,
        min_fill_ratio: float = DEFAULT_MIN_FILL_RATIO,
        fill_base: float = DEFAULT_FILL_BASE,
        fill_confidence_bonus: float = DEFAULT_FILL_CONFIDENCE_BONUS,
        gap_through_prob: float = DEFAULT_GAP_THROUGH_PROBABILITY,
        gap_through_max_mult: float = DEFAULT_GAP_THROUGH_MAX_MULT,
        persist: bool = True,
    ):
        self.max_positions = max_positions
        self.max_risk_pct = max_risk_pct

        # Slippage config
        self.slippage_min_bps = slippage_min_bps
        self.slippage_max_bps = slippage_max_bps
        self.slippage_mean_bps = slippage_mean_bps

        # Fill simulation config
        self.min_fill_ratio = min_fill_ratio
        self.fill_base = fill_base
        self.fill_confidence_bonus = fill_confidence_bonus

        # Gap-through config
        self.gap_through_prob = gap_through_prob
        self.gap_through_max_mult = gap_through_max_mult

        # State
        self._positions: Dict[str, Position] = {}
        self._closed: List[Position] = []
        self._persist = persist
        self._daily_pnl: float = 0.0
        self._total_pnl: float = 0.0

        # Load or initialize balance
        if balance is not None:
            self._balance = balance
        else:
            self._balance = self._load_balance()

        # Restore open positions from trade log
        self._restore_positions()

    # ── Order submission ───────────────────────────────────────────────────

    def submit_order(self, order: OrderRequest) -> FillResult:
        # Pre-flight checks
        reject = self._pre_flight(order)
        if reject:
            return reject

        # Simulate slippage
        slippage_bps = self._simulate_slippage()
        fill_price = self._apply_slippage(order.price, order.side, slippage_bps)

        # Simulate partial fill
        fill_ratio = self._simulate_fill_ratio(order.confidence)
        fill_size = order.size_usd * fill_ratio

        # Minimum viable position check
        if fill_size < 10.0:
            return FillResult(
                signal_id=order.signal_id,
                status=FillStatus.REJECTED,
                fill_price=0.0,
                fill_size_usd=0.0,
                requested_size_usd=order.size_usd,
                slippage_bps=0.0,
                reject_reason="fill_size_below_minimum",
            )

        # Scale risk_usd proportionally to actual fill
        fill_risk_usd = order.risk_usd * fill_ratio

        # Create position
        pos_id = f"paper_{uuid.uuid4().hex[:12]}"
        position = Position(
            position_id=pos_id,
            coin=order.coin,
            side=OrderSide(order.side),
            entry_price=fill_price,
            size_usd=fill_size,
            stop_price=order.stop_price,
            tp_price=order.tp_price,
            risk_usd=fill_risk_usd,
            signal_id=order.signal_id,
            meta=order.meta,
        )

        self._positions[pos_id] = position

        status = FillStatus.FILLED if fill_ratio >= 0.99 else FillStatus.PARTIAL

        fill_result = FillResult(
            signal_id=order.signal_id,
            status=status,
            fill_price=fill_price,
            fill_size_usd=fill_size,
            requested_size_usd=order.size_usd,
            slippage_bps=slippage_bps,
            meta={
                "position_id": pos_id,
                "fill_ratio": round(fill_ratio, 3),
                "risk_usd": round(fill_risk_usd, 2),
            },
        )

        # Log with full fill details
        self._log_trade(position, "OPEN", fill_detail={
            "requested_price": round(order.price, 8),
            "requested_size_usd": round(order.size_usd, 2),
            "fill_price": round(fill_price, 8),
            "fill_size_usd": round(fill_size, 2),
            "fill_ratio": round(fill_ratio, 3),
            "slippage_bps": round(slippage_bps, 2),
            "gap_through": False,
            "gap_amount_bps": 0.0,
        })

        return fill_result

    # ── Exit checking ──────────────────────────────────────────────────────

    def check_exits(self, price_feeds: dict[str, float]) -> List[Position]:
        closed_this_cycle: List[Position] = []

        for pos_id, pos in list(self._positions.items()):
            if not pos.is_open:
                continue

            current_price = price_feeds.get(pos.coin)
            if current_price is None:
                continue

            exit_price: Optional[float] = None
            exit_status: Optional[PositionStatus] = None
            gap_through = False
            gap_amount_bps = 0.0

            if pos.side == OrderSide.LONG:
                if current_price <= pos.stop_price:
                    exit_price = pos.stop_price
                    exit_status = PositionStatus.STOPPED

                    # Gap-through simulation: price may have gapped below stop
                    gap_through, gap_price = self._simulate_gap_through(
                        pos.stop_price, pos.entry_price, "stop", OrderSide.LONG
                    )
                    if gap_through:
                        exit_price = gap_price
                        gap_amount_bps = abs(gap_price - pos.stop_price) / pos.stop_price * 10_000

                elif current_price >= pos.tp_price:
                    exit_price = pos.tp_price
                    exit_status = PositionStatus.TP_HIT

                    # TP gap-through works in your favor
                    gap_through, gap_price = self._simulate_gap_through(
                        pos.tp_price, pos.entry_price, "tp", OrderSide.LONG
                    )
                    if gap_through:
                        exit_price = gap_price
                        gap_amount_bps = abs(gap_price - pos.tp_price) / pos.tp_price * 10_000

            else:  # SHORT
                if current_price >= pos.stop_price:
                    exit_price = pos.stop_price
                    exit_status = PositionStatus.STOPPED

                    gap_through, gap_price = self._simulate_gap_through(
                        pos.stop_price, pos.entry_price, "stop", OrderSide.SHORT
                    )
                    if gap_through:
                        exit_price = gap_price
                        gap_amount_bps = abs(gap_price - pos.stop_price) / pos.stop_price * 10_000

                elif current_price <= pos.tp_price:
                    exit_price = pos.tp_price
                    exit_status = PositionStatus.TP_HIT

                    gap_through, gap_price = self._simulate_gap_through(
                        pos.tp_price, pos.entry_price, "tp", OrderSide.SHORT
                    )
                    if gap_through:
                        exit_price = gap_price
                        gap_amount_bps = abs(gap_price - pos.tp_price) / pos.tp_price * 10_000

            if exit_price is not None and exit_status is not None:
                # Apply exit slippage (reduced vs entry)
                if not gap_through:
                    exit_slip = self._simulate_slippage() * 0.5
                    exit_side = OrderSide.SHORT if pos.side == OrderSide.LONG else OrderSide.LONG
                    exit_price = self._apply_slippage(exit_price, exit_side, exit_slip)

                pos.close(exit_price, exit_status)
                self._balance += pos.pnl_usd
                self._daily_pnl += pos.pnl_usd
                self._total_pnl += pos.pnl_usd

                del self._positions[pos_id]
                self._closed.append(pos)
                closed_this_cycle.append(pos)

                self._log_trade(pos, "CLOSE", fill_detail={
                    "requested_price": round(pos.stop_price if exit_status == PositionStatus.STOPPED else pos.tp_price, 8),
                    "requested_size_usd": round(pos.size_usd, 2),
                    "fill_price": round(exit_price, 8),
                    "fill_size_usd": round(pos.size_usd, 2),
                    "fill_ratio": 1.0,
                    "slippage_bps": round(exit_slip if not gap_through else 0.0, 2),
                    "gap_through": gap_through,
                    "gap_amount_bps": round(gap_amount_bps, 2),
                })
                self._save_balance()

        return closed_this_cycle

    # ── Position queries ───────────────────────────────────────────────────

    def get_positions(self, coin: Optional[str] = None) -> List[Position]:
        positions = list(self._positions.values())
        if coin:
            positions = [p for p in positions if p.coin == coin]
        return positions

    def get_account(self, price_feeds: Optional[dict[str, float]] = None) -> AccountState:
        """
        FIX: equity now includes unrealized PnL from open positions.

        If price_feeds is provided, marks open positions to market.
        Otherwise equity = balance (no mark).
        """
        open_positions = [p for p in self._positions.values() if p.is_open]
        total_exposure = sum(p.size_usd for p in open_positions)

        # Compute unrealized PnL if we have current prices
        unrealized = 0.0
        if price_feeds:
            for p in open_positions:
                current = price_feeds.get(p.coin)
                if current is not None:
                    unrealized += p.mark_to_market(current)

        return AccountState(
            balance_usd=self._balance,
            equity_usd=self._balance + unrealized,
            open_positions=len(open_positions),
            total_exposure_usd=total_exposure,
            max_positions=self.max_positions,
            daily_pnl_usd=self._daily_pnl,
            total_pnl_usd=self._total_pnl,
            unrealized_pnl_usd=unrealized,
        )

    def close_position(self, position_id: str, price: float) -> Optional[Position]:
        pos = self._positions.get(position_id)
        if pos is None or not pos.is_open:
            return None

        pos.close(price, PositionStatus.CLOSED)
        self._balance += pos.pnl_usd
        self._daily_pnl += pos.pnl_usd
        self._total_pnl += pos.pnl_usd

        del self._positions[position_id]
        self._closed.append(pos)
        self._log_trade(pos, "CLOSE")
        self._save_balance()
        return pos

    def shutdown(self):
        self._save_balance()
        print(f"[PAPER] Shutdown — balance=${self._balance:.2f}, "
              f"open={len(self._positions)}, total_pnl=${self._total_pnl:.2f}")

    # ── Slippage simulation ────────────────────────────────────────────────

    def _simulate_slippage(self) -> float:
        return random.triangular(
            self.slippage_min_bps,
            self.slippage_max_bps,
            self.slippage_mean_bps,
        )

    def _apply_slippage(self, price: float, side: OrderSide, bps: float) -> float:
        slip_pct = bps / 10_000
        if side == OrderSide.LONG:
            return price * (1 + slip_pct)
        return price * (1 - slip_pct)

    # ── Fill simulation ────────────────────────────────────────────────────

    def _simulate_fill_ratio(self, confidence: float) -> float:
        conf = max(0.50, min(0.95, confidence))
        conf_bonus = self.fill_confidence_bonus * (conf - 0.50) / 0.45
        jitter = random.uniform(-0.10, 0.10)
        ratio = self.fill_base + conf_bonus + jitter
        return max(self.min_fill_ratio, min(1.0, ratio))

    # ── Gap-through simulation ─────────────────────────────────────────────

    def _simulate_gap_through(
        self,
        level: float,
        entry: float,
        level_type: str,  # "stop" or "tp"
        side: OrderSide,
    ) -> tuple[bool, float]:
        """
        Simulate gap-through: price jumps past stop/TP level.

        8% probability per exit event.
        Gap extends 0-30% of the stop distance beyond the level.

        Stop gap-through always hurts you.
        TP gap-through always helps you.
        """
        if random.random() > self.gap_through_prob:
            return False, level

        stop_dist = abs(entry - level)
        gap_size = random.uniform(0.01, self.gap_through_max_mult) * stop_dist

        if level_type == "stop":
            # Stop gap: price goes further against you
            if side == OrderSide.LONG:
                return True, level - gap_size   # worse fill for long stop
            else:
                return True, level + gap_size   # worse fill for short stop
        else:
            # TP gap: price goes further in your favor
            if side == OrderSide.LONG:
                return True, level + gap_size   # better fill for long TP
            else:
                return True, level - gap_size   # better fill for short TP

    # ── Pre-flight validation ──────────────────────────────────────────────

    def _pre_flight(self, order: OrderRequest) -> Optional[FillResult]:
        """
        FIX: Risk check now uses risk_usd (actual dollar risk at -1R),
        NOT notional size vs balance. A leveraged position can have
        notional much larger than balance while only risking 1-2%.
        """

        # Max positions
        open_count = sum(1 for p in self._positions.values() if p.is_open)
        if open_count >= self.max_positions:
            return FillResult(
                signal_id=order.signal_id,
                status=FillStatus.REJECTED,
                fill_price=0.0,
                fill_size_usd=0.0,
                requested_size_usd=order.size_usd,
                slippage_bps=0.0,
                reject_reason=f"max_positions_reached ({self.max_positions})",
            )

        # FIX: Compare RISK (not notional) against risk budget
        max_risk_usd = self._balance * (self.max_risk_pct / 100)
        if order.risk_usd > max_risk_usd * 1.5:
            return FillResult(
                signal_id=order.signal_id,
                status=FillStatus.REJECTED,
                fill_price=0.0,
                fill_size_usd=0.0,
                requested_size_usd=order.size_usd,
                slippage_bps=0.0,
                reject_reason=(
                    f"risk_exceeds_limit "
                    f"(risk=${order.risk_usd:.2f} > "
                    f"max=${max_risk_usd:.2f} * 1.5)"
                ),
            )

        # Duplicate coin check
        for pos in self._positions.values():
            if pos.is_open and pos.coin == order.coin and pos.side.value == order.side.value:
                return FillResult(
                    signal_id=order.signal_id,
                    status=FillStatus.REJECTED,
                    fill_price=0.0,
                    fill_size_usd=0.0,
                    requested_size_usd=order.size_usd,
                    slippage_bps=0.0,
                    reject_reason=f"duplicate_position_{order.coin}_{order.side.value}",
                )

        # Insufficient balance — check against risk, not notional
        if self._balance < order.risk_usd:
            return FillResult(
                signal_id=order.signal_id,
                status=FillStatus.REJECTED,
                fill_price=0.0,
                fill_size_usd=0.0,
                requested_size_usd=order.size_usd,
                slippage_bps=0.0,
                reject_reason=(
                    f"insufficient_balance "
                    f"(balance=${self._balance:.2f} < "
                    f"risk=${order.risk_usd:.2f})"
                ),
            )

        return None

    # ── Persistence ────────────────────────────────────────────────────────

    def _save_balance(self):
        if not self._persist:
            return
        try:
            with open(BALANCE_FILE, "w") as f:
                f.write(str(round(self._balance, 2)))
        except Exception as e:
            print(f"[PAPER] Balance save failed: {e}")

    def _load_balance(self) -> float:
        if not self._persist or not os.path.exists(BALANCE_FILE):
            return DEFAULT_BALANCE
        try:
            with open(BALANCE_FILE) as f:
                return float(f.read().strip())
        except Exception:
            return DEFAULT_BALANCE

    def _log_trade(
        self,
        pos: Position,
        action: str,
        fill_detail: Optional[Dict] = None,
    ):
        """
        FIX: Now logs full fill details for later audit trail.
        New fields: requested_price, requested_size_usd, fill_price,
        fill_size_usd, fill_ratio, slippage_bps, gap_through, gap_amount_bps
        """
        if not self._persist:
            return

        file_exists = os.path.exists(TRADES_FILE)
        fd = fill_detail or {}

        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "position_id": pos.position_id,
            "signal_id": pos.signal_id,
            "coin": pos.coin,
            "side": pos.side.value,
            "entry_price": round(pos.entry_price, 8),
            "exit_price": round(pos.exit_price, 8) if pos.exit_price else "",
            "size_usd": round(pos.size_usd, 2),
            "risk_usd": round(pos.risk_usd, 2),
            "stop_price": round(pos.stop_price, 8),
            "tp_price": round(pos.tp_price, 8),
            "status": pos.status.value,
            "pnl_usd": round(pos.pnl_usd, 2),
            "pnl_r": round(pos.pnl_r, 3),
            "balance": round(self._balance, 2),
            # Fill detail fields
            "requested_price": fd.get("requested_price", ""),
            "requested_size_usd": fd.get("requested_size_usd", ""),
            "fill_price": fd.get("fill_price", ""),
            "fill_size_usd": fd.get("fill_size_usd", ""),
            "fill_ratio": fd.get("fill_ratio", ""),
            "slippage_bps": fd.get("slippage_bps", ""),
            "gap_through": fd.get("gap_through", ""),
            "gap_amount_bps": fd.get("gap_amount_bps", ""),
        }

        try:
            with open(TRADES_FILE, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=TRADE_LOG_FIELDS)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)
        except Exception as e:
            print(f"[PAPER] Trade log failed: {e}")

    def _restore_positions(self):
        if not self._persist or not os.path.exists(TRADES_FILE):
            return

        opens: Dict[str, dict] = {}
        closed_ids: set = set()

        try:
            with open(TRADES_FILE, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    pid = row.get("position_id", "")
                    action = row.get("action", "")
                    if action == "OPEN":
                        opens[pid] = row
                    elif action == "CLOSE":
                        closed_ids.add(pid)
        except Exception as e:
            print(f"[PAPER] Position restore failed: {e}")
            return

        restored = 0
        for pid, row in opens.items():
            if pid in closed_ids:
                continue
            try:
                pos = Position(
                    position_id=pid,
                    coin=row["coin"],
                    side=OrderSide(row["side"]),
                    entry_price=float(row["entry_price"]),
                    size_usd=float(row["size_usd"]),
                    stop_price=float(row["stop_price"]),
                    tp_price=float(row["tp_price"]),
                    risk_usd=float(row.get("risk_usd", 0)),
                    signal_id=row.get("signal_id", ""),
                    entry_time=datetime.fromisoformat(row["timestamp"]),
                )
                self._positions[pid] = pos
                restored += 1
            except Exception as e:
                print(f"[PAPER] Skip restoring {pid}: {e}")

        if restored > 0:
            print(f"[PAPER] Restored {restored} open position(s)")

    def reset_daily_pnl(self):
        self._daily_pnl = 0.0
from __future__ import annotations

import csv
import fcntl
import os
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, List, Optional


ORDERS_FILE = os.getenv("ORDERS_FILE", "orders.csv")
LOCK_FILE = f"{ORDERS_FILE}.lock"


class OrderKind(str, Enum):
    ENTRY = "entry"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    PARTIAL_EXIT = "partial_exit"
    FULL_EXIT = "full_exit"


class OrderLifecycleState(str, Enum):
    CREATED = "created"
    SUBMITTED = "submitted"
    ACKNOWLEDGED = "acknowledged"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    FAILED = "failed"
    RECONCILED_FROM_VENUE = "reconciled_from_venue"


TERMINAL_ORDER_STATES = {
    OrderLifecycleState.FILLED,
    OrderLifecycleState.CANCELLED,
    OrderLifecycleState.FAILED,
    OrderLifecycleState.RECONCILED_FROM_VENUE,
}


ORDER_HEADER = [
    "order_id",
    "position_id",
    "coin",
    "side",
    "order_kind",
    "state",
    "venue_order_id",
    "requested_price",
    "requested_size_usd",
    "requested_size_coin",
    "fill_price",
    "fill_size_usd",
    "fill_size_coin",
    "fill_ratio",
    "reject_reason",
    "source",
    "note",
    "created_at",
    "submitted_at",
    "acknowledged_at",
    "filled_at",
    "cancelled_at",
    "failed_at",
    "reconciled_at",
    "updated_at",
]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value) if value not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _parse_dt(raw: str) -> Optional[datetime]:
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


@dataclass
class TrackedOrder:
    order_id: str
    position_id: str
    coin: str
    side: str
    order_kind: OrderKind
    state: OrderLifecycleState = OrderLifecycleState.CREATED
    venue_order_id: str = ""

    requested_price: float = 0.0
    requested_size_usd: float = 0.0
    requested_size_coin: float = 0.0

    fill_price: float = 0.0
    fill_size_usd: float = 0.0
    fill_size_coin: float = 0.0
    fill_ratio: float = 0.0

    reject_reason: str = ""
    source: str = ""
    note: str = ""

    created_at: datetime = field(default_factory=_utc_now)
    submitted_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None
    failed_at: Optional[datetime] = None
    reconciled_at: Optional[datetime] = None
    updated_at: datetime = field(default_factory=_utc_now)

    def to_row(self) -> Dict[str, str]:
        return {
            "order_id": self.order_id,
            "position_id": self.position_id,
            "coin": self.coin,
            "side": self.side,
            "order_kind": self.order_kind.value,
            "state": self.state.value,
            "venue_order_id": self.venue_order_id,
            "requested_price": str(self.requested_price),
            "requested_size_usd": str(self.requested_size_usd),
            "requested_size_coin": str(self.requested_size_coin),
            "fill_price": str(self.fill_price),
            "fill_size_usd": str(self.fill_size_usd),
            "fill_size_coin": str(self.fill_size_coin),
            "fill_ratio": str(self.fill_ratio),
            "reject_reason": self.reject_reason,
            "source": self.source,
            "note": self.note,
            "created_at": self.created_at.isoformat() if self.created_at else "",
            "submitted_at": self.submitted_at.isoformat() if self.submitted_at else "",
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else "",
            "filled_at": self.filled_at.isoformat() if self.filled_at else "",
            "cancelled_at": self.cancelled_at.isoformat() if self.cancelled_at else "",
            "failed_at": self.failed_at.isoformat() if self.failed_at else "",
            "reconciled_at": self.reconciled_at.isoformat() if self.reconciled_at else "",
            "updated_at": self.updated_at.isoformat() if self.updated_at else "",
        }

    @staticmethod
    def from_row(row: Dict[str, str]) -> "TrackedOrder":
        return TrackedOrder(
            order_id=row.get("order_id", ""),
            position_id=row.get("position_id", ""),
            coin=row.get("coin", ""),
            side=row.get("side", ""),
            order_kind=OrderKind(row.get("order_kind", OrderKind.ENTRY.value)),
            state=OrderLifecycleState(row.get("state", OrderLifecycleState.CREATED.value)),
            venue_order_id=row.get("venue_order_id", ""),
            requested_price=_safe_float(row.get("requested_price")),
            requested_size_usd=_safe_float(row.get("requested_size_usd")),
            requested_size_coin=_safe_float(row.get("requested_size_coin")),
            fill_price=_safe_float(row.get("fill_price")),
            fill_size_usd=_safe_float(row.get("fill_size_usd")),
            fill_size_coin=_safe_float(row.get("fill_size_coin")),
            fill_ratio=_safe_float(row.get("fill_ratio")),
            reject_reason=row.get("reject_reason", ""),
            source=row.get("source", ""),
            note=row.get("note", ""),
            created_at=_parse_dt(row.get("created_at", "")) or _utc_now(),
            submitted_at=_parse_dt(row.get("submitted_at", "")),
            acknowledged_at=_parse_dt(row.get("acknowledged_at", "")),
            filled_at=_parse_dt(row.get("filled_at", "")),
            cancelled_at=_parse_dt(row.get("cancelled_at", "")),
            failed_at=_parse_dt(row.get("failed_at", "")),
            reconciled_at=_parse_dt(row.get("reconciled_at", "")),
            updated_at=_parse_dt(row.get("updated_at", "")) or _utc_now(),
        )


class OrderTracker:
    def __init__(self, orders_file: str = ORDERS_FILE):
        self.orders_file = orders_file
        self.lock_file = f"{orders_file}.lock"
        self.orders: Dict[str, TrackedOrder] = {}
        self._seq = 0
        self._load()

    @contextmanager
    def _file_lock(self):
        Path(self.lock_file).touch(exist_ok=True)
        with open(self.lock_file, "r+") as lock_f:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)

    def _load(self):
        if not os.path.exists(self.orders_file):
            return

        latest_by_id: Dict[str, TrackedOrder] = {}
        with self._file_lock():
            if not os.path.exists(self.orders_file):
                return
            with open(self.orders_file, newline="") as f:
                for row in csv.DictReader(f):
                    order_id = row.get("order_id", "")
                    if not order_id:
                        continue
                    try:
                        latest_by_id[order_id] = TrackedOrder.from_row(row)
                    except Exception:
                        continue

        self.orders = latest_by_id

    def _next_order_id(self) -> str:
        self._seq += 1
        return f"ord_{int(time.time() * 1000)}_{self._seq}"

    def _save(self):
        rows = [o.to_row() for o in self.orders.values()]

        with self._file_lock():
            target = Path(self.orders_file)
            fd, tmp_path = tempfile.mkstemp(
                prefix=f"{target.name}.",
                suffix=".tmp",
                dir=str(target.parent or Path(".")),
                text=True,
            )
            try:
                with os.fdopen(fd, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=ORDER_HEADER)
                    writer.writeheader()
                    writer.writerows(rows)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, self.orders_file)
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

    def _persist(self, order: TrackedOrder):
        order.updated_at = _utc_now()
        self.orders[order.order_id] = order
        self._save()

    def create_order(
        self,
        position_id: str,
        coin: str,
        side: str,
        order_kind: OrderKind,
        requested_price: float = 0.0,
        requested_size_usd: float = 0.0,
        requested_size_coin: float = 0.0,
        source: str = "",
        note: str = "",
    ) -> str:
        order_id = self._next_order_id()
        order = TrackedOrder(
            order_id=order_id,
            position_id=position_id,
            coin=str(coin).upper(),
            side=str(side).upper(),
            order_kind=order_kind,
            state=OrderLifecycleState.CREATED,
            requested_price=float(requested_price or 0.0),
            requested_size_usd=float(requested_size_usd or 0.0),
            requested_size_coin=float(requested_size_coin or 0.0),
            source=source,
            note=note,
        )
        self._persist(order)
        return order_id

    def bind_position(self, order_id: str, position_id: str):
        order = self.orders.get(order_id)
        if order is None:
            return
        order.position_id = position_id
        self._persist(order)

    def mark_submitted(self, order_id: str, venue_order_id: str = ""):
        order = self.orders.get(order_id)
        if order is None:
            return
        now = _utc_now()
        order.state = OrderLifecycleState.SUBMITTED
        order.submitted_at = order.submitted_at or now
        if venue_order_id:
            order.venue_order_id = str(venue_order_id)
        self._persist(order)

    def mark_acknowledged(self, order_id: str, venue_order_id: str = "", note: str = ""):
        order = self.orders.get(order_id)
        if order is None:
            return
        now = _utc_now()
        order.state = OrderLifecycleState.ACKNOWLEDGED
        order.acknowledged_at = order.acknowledged_at or now
        if venue_order_id:
            order.venue_order_id = str(venue_order_id)
        if note:
            order.note = note
        self._persist(order)

    def mark_partially_filled(
        self,
        order_id: str,
        fill_price: float,
        fill_size_usd: float,
        fill_size_coin: float,
        fill_ratio: float,
        venue_order_id: str = "",
    ):
        order = self.orders.get(order_id)
        if order is None:
            return
        now = _utc_now()
        order.state = OrderLifecycleState.PARTIALLY_FILLED
        order.acknowledged_at = order.acknowledged_at or now
        order.filled_at = now
        if venue_order_id:
            order.venue_order_id = str(venue_order_id)
        order.fill_price = float(fill_price or 0.0)
        order.fill_size_usd = float(fill_size_usd or 0.0)
        order.fill_size_coin = float(fill_size_coin or 0.0)
        order.fill_ratio = float(fill_ratio or 0.0)
        self._persist(order)

    def mark_filled(
        self,
        order_id: str,
        fill_price: float,
        fill_size_usd: float,
        fill_size_coin: float,
        fill_ratio: float = 1.0,
        venue_order_id: str = "",
    ):
        order = self.orders.get(order_id)
        if order is None:
            return
        now = _utc_now()
        order.state = OrderLifecycleState.FILLED
        order.acknowledged_at = order.acknowledged_at or now
        order.filled_at = now
        if venue_order_id:
            order.venue_order_id = str(venue_order_id)
        order.fill_price = float(fill_price or 0.0)
        order.fill_size_usd = float(fill_size_usd or 0.0)
        order.fill_size_coin = float(fill_size_coin or 0.0)
        order.fill_ratio = float(fill_ratio or 1.0)
        self._persist(order)

    def mark_cancelled(self, order_id: str, reason: str = "", note: str = ""):
        order = self.orders.get(order_id)
        if order is None:
            return
        order.state = OrderLifecycleState.CANCELLED
        order.cancelled_at = _utc_now()
        if reason:
            order.reject_reason = reason
        if note:
            order.note = note
        self._persist(order)

    def mark_failed(self, order_id: str, reason: str = "", note: str = ""):
        order = self.orders.get(order_id)
        if order is None:
            return
        order.state = OrderLifecycleState.FAILED
        order.failed_at = _utc_now()
        if reason:
            order.reject_reason = reason
        if note:
            order.note = note
        self._persist(order)

    def mark_reconciled_from_venue(self, order_id: str, note: str = ""):
        order = self.orders.get(order_id)
        if order is None:
            return
        order.state = OrderLifecycleState.RECONCILED_FROM_VENUE
        order.reconciled_at = _utc_now()
        if note:
            order.note = note
        self._persist(order)

    def get_order(self, order_id: str) -> Optional[TrackedOrder]:
        return self.orders.get(order_id)

    def orders_for_position(
        self,
        position_id: str,
        kinds: Optional[Iterable[OrderKind]] = None,
    ) -> List[TrackedOrder]:
        kind_set = set(kinds) if kinds else None
        out: List[TrackedOrder] = []
        for order in self.orders.values():
            if order.position_id != position_id:
                continue
            if kind_set is not None and order.order_kind not in kind_set:
                continue
            out.append(order)
        return out

    def ensure_venue_order(
        self,
        position_id: str,
        coin: str,
        side: str,
        order_kind: OrderKind,
        venue_order_id: str,
        source: str = "",
        filled: bool = False,
    ) -> str:
        venue_order_id = str(venue_order_id or "").strip()
        if not venue_order_id:
            return ""

        for order in self.orders.values():
            if (
                order.position_id == position_id
                and order.order_kind == order_kind
                and str(order.venue_order_id).strip() == venue_order_id
            ):
                return order.order_id

        order_id = self.create_order(
            position_id=position_id,
            coin=coin,
            side=side,
            order_kind=order_kind,
            source=source,
            note="hydrated_from_position",
        )
        self.mark_submitted(order_id, venue_order_id=venue_order_id)
        self.mark_acknowledged(order_id, venue_order_id=venue_order_id, note="hydrated_from_position")
        if filled:
            self.mark_filled(
                order_id,
                fill_price=0.0,
                fill_size_usd=0.0,
                fill_size_coin=0.0,
                fill_ratio=1.0,
                venue_order_id=venue_order_id,
            )
        return order_id

    def reconcile_position_close(self, pos, close_reason, source: str = ""):
        position_orders = self.orders_for_position(getattr(pos, "position_id", ""))
        if not position_orders:
            return

        reason_str = str(getattr(close_reason, "value", close_reason) or "").strip().lower()
        triggered_kind: Optional[OrderKind] = None
        if reason_str in {"stop_full", "stop_runner"}:
            triggered_kind = OrderKind.STOP_LOSS
        elif reason_str == "tp_full":
            triggered_kind = OrderKind.TAKE_PROFIT

        for order in position_orders:
            if order.state in TERMINAL_ORDER_STATES:
                continue

            if order.order_kind in {OrderKind.STOP_LOSS, OrderKind.TAKE_PROFIT}:
                if triggered_kind and order.order_kind == triggered_kind:
                    self.mark_reconciled_from_venue(
                        order.order_id,
                        note=f"{source}:{reason_str or 'closed'}",
                    )
                else:
                    self.mark_cancelled(
                        order.order_id,
                        reason=f"position_closed:{reason_str or 'closed'}",
                        note=source,
                    )
                continue

            if order.order_kind in {OrderKind.PARTIAL_EXIT, OrderKind.FULL_EXIT}:
                self.mark_reconciled_from_venue(
                    order.order_id,
                    note=f"{source}:{reason_str or 'closed'}",
                )

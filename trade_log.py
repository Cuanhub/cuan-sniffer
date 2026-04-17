"""
Persists every trade state change to trades.csv.
"""

import csv
import fcntl
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List

from position import Position

TRADES_FILE = os.getenv("TRADES_FILE", "trades.csv")
LOCK_FILE = f"{TRADES_FILE}.lock"

HEADER = [
    "position_id", "coin", "side", "signal_id",
    "entry_price", "stop_price", "tp_price",
    "original_stop_price", "original_tp_price",
    "atr",
    "size_usd", "risk_usd", "r_value",
    "size_multiplier",
    "partial_r", "runner_r", "realized_r", "pnl_usd", "rr_planned",
    "partial_close_size_usd", "partial_close_fraction",
    "state", "close_reason",
    "partial_closed", "breakeven_moved",
    "first_update_pending", "stale_invalidated",
    "regime", "setup_family", "htf_regime",
    "confidence", "total_score", "session",
    "timeframe", "execution_track",
    "entry_order_ids", "exit_order_ids",
    "stop_order_id", "tp_order_id",
    "protection_status", "protection_placed_at", "protection_error",
    "venue_protection_mode",
    "allow_reconcile_close", "reconciled_from_venue",
    "exit_requested_reason", "exit_requested_at", "exit_trigger_source",
    "wallet_flat_confirmed", "wallet_flat_confirmed_at",
    "pending_exit_reason", "pending_exit_fill_price", "pending_exit_fill_size_usd",
    "pending_exit_fee_usd", "pending_exit_slippage_bps", "pending_exit_recorded_at",
    "entry_fee_usd", "exit_fees_usd", "funding_usd", "total_fees_usd",
    "last_funding_ts",
    "opened_at", "closed_at",
    "paper_mode",
]


@contextmanager
def _trades_lock():
    """
    Serialize all readers/writers touching trades.csv.
    """
    Path(LOCK_FILE).touch(exist_ok=True)
    with open(LOCK_FILE, "r+") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)


def _normalize_row(row: Dict) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in row.items():
        if isinstance(v, bool):
            out[k] = "true" if v else "false"
        elif v is None:
            out[k] = ""
        else:
            out[k] = str(v)
    return out


def _merge_fieldnames(rows: List[Dict], preferred: List[str]) -> List[str]:
    fieldnames: List[str] = []
    for name in preferred:
        if name not in fieldnames:
            fieldnames.append(name)
    for row in rows:
        for name in row.keys():
            if name not in fieldnames:
                fieldnames.append(name)
    return fieldnames


def _atomic_write_rows(rows: List[Dict], fieldnames: List[str]):
    target = Path(TRADES_FILE)
    fd, tmp_path = tempfile.mkstemp(
        prefix=f"{target.name}.",
        suffix=".tmp",
        dir=str(target.parent or Path(".")),
        text=True,
    )
    try:
        with os.fdopen(fd, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, TRADES_FILE)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def load_trade_rows() -> List[Dict[str, str]]:
    if not os.path.exists(TRADES_FILE):
        return []
    with _trades_lock():
        if not os.path.exists(TRADES_FILE):
            return []
        with open(TRADES_FILE, newline="") as f:
            return list(csv.DictReader(f))


def upsert_trade_row(row: Dict):
    normalized = _normalize_row(row)
    with _trades_lock():
        existing: List[Dict[str, str]] = []
        updated = False

        if os.path.exists(TRADES_FILE):
            with open(TRADES_FILE, newline="") as f:
                for current in csv.DictReader(f):
                    if current.get("position_id") == normalized.get("position_id"):
                        existing.append(normalized)
                        updated = True
                    else:
                        existing.append(current)

        if not updated:
            existing.append(normalized)

        fieldnames = _merge_fieldnames(existing, HEADER)
        _atomic_write_rows(existing, fieldnames)


def init_trade_log():
    with _trades_lock():
        if os.path.exists(TRADES_FILE):
            return
        _atomic_write_rows([], HEADER)
    print(f"[TRADE_LOG] Initialized {TRADES_FILE}")


def append_trade(position: Position, paper_mode: bool = True):
    paper_mode_str = str(paper_mode).lower()

    row = {
        "position_id": position.position_id,
        "coin": position.coin,
        "side": position.side,
        "signal_id": position.signal_id,
        "entry_price": position.entry_price,
        "stop_price": position.stop_price,
        "tp_price": position.tp_price,
        "original_stop_price": position.original_stop_price,
        "original_tp_price": position.original_tp_price,
        "atr": position.atr,
        "size_usd": position.size_usd,
        "risk_usd": position.risk_usd,
        "r_value": position.r_value,
        "size_multiplier": position.size_multiplier,
        "partial_r": round(position.partial_r, 4),
        "runner_r": round(position.runner_r, 4),
        "realized_r": round(position.realized_r, 4),
        "pnl_usd": round(position.pnl_usd, 2),
        "partial_close_size_usd": round(position.partial_close_size_usd, 8),
        "partial_close_fraction": round(position.partial_close_fraction, 8),
        "rr_planned": round(position.rr_planned, 3),
        "state": position.state.value,
        "close_reason": position.close_reason.value if position.close_reason else "",
        "partial_closed": position.partial_closed,
        "breakeven_moved": position.breakeven_moved,
        "first_update_pending": position.first_update_pending,
        "stale_invalidated": position.stale_invalidated,
        "regime": position.regime,
        "setup_family": position.setup_family,
        "htf_regime": position.htf_regime,
        "confidence": position.confidence,
        "total_score": position.total_score,
        "session": position.session,
        "timeframe": position.timeframe,
        "execution_track": position.execution_track,
        "entry_order_ids": position.entry_order_ids,
        "exit_order_ids": position.exit_order_ids,
        "stop_order_id": position.stop_order_id,
        "tp_order_id": position.tp_order_id,
        "protection_status": position.protection_status,
        "protection_placed_at": (
            position.protection_placed_at.isoformat()
            if position.protection_placed_at else ""
        ),
        "protection_error": position.protection_error,
        "venue_protection_mode": position.venue_protection_mode,
        "allow_reconcile_close": position.allow_reconcile_close,
        "reconciled_from_venue": position.reconciled_from_venue,
        "exit_requested_reason": position.exit_requested_reason,
        "exit_requested_at": (
            position.exit_requested_at.isoformat() if position.exit_requested_at else ""
        ),
        "exit_trigger_source": position.exit_trigger_source,
        "wallet_flat_confirmed": position.wallet_flat_confirmed,
        "wallet_flat_confirmed_at": (
            position.wallet_flat_confirmed_at.isoformat()
            if position.wallet_flat_confirmed_at else ""
        ),
        "pending_exit_reason": position.pending_exit_reason,
        "pending_exit_fill_price": round(position.pending_exit_fill_price, 8),
        "pending_exit_fill_size_usd": round(position.pending_exit_fill_size_usd, 8),
        "pending_exit_fee_usd": round(position.pending_exit_fee_usd, 8),
        "pending_exit_slippage_bps": round(position.pending_exit_slippage_bps, 4),
        "pending_exit_recorded_at": (
            position.pending_exit_recorded_at.isoformat()
            if position.pending_exit_recorded_at else ""
        ),
        "entry_fee_usd": round(position.entry_fee_usd, 8),
        "exit_fees_usd": round(position.exit_fees_usd, 8),
        "funding_usd": round(position.funding_usd, 8),
        "total_fees_usd": round(position.total_fees_usd, 8),
        "last_funding_ts": (
            position.last_funding_ts.isoformat() if position.last_funding_ts else ""
        ),
        "opened_at": position.opened_at.isoformat(),
        "closed_at": position.closed_at.isoformat() if position.closed_at else "",
        "paper_mode": paper_mode_str,
    }
    upsert_trade_row(row)

# trade_log.py
"""
Persists every position state change to trades.csv.
FIX: uses the passed paper_mode argument (not os.getenv).
FIX: upsert on position_id so partial + final close update the same row.
"""

import csv
import os
from typing import List

from position import Position

TRADES_FILE = os.getenv("TRADES_FILE", "trades.csv")

HEADER = [
    "position_id", "coin", "side", "signal_id",
    "entry_price", "stop_price", "tp_price", "atr",
    "size_usd", "risk_usd", "r_value", "size_multiplier",
    "partial_r", "runner_r", "realized_r", "pnl_usd", "rr_planned",
    "state", "close_reason",
    "partial_closed", "breakeven_moved",
    "setup_family", "htf_regime", "regime", "confidence", "total_score", "session",
    "opened_at", "closed_at",
    "paper_mode",
]


def init_trade_log():
    if os.path.exists(TRADES_FILE):
        return
    with open(TRADES_FILE, "w", newline="") as f:
        csv.writer(f).writerow(HEADER)
    print(f"[TRADE_LOG] Initialized {TRADES_FILE}")


def append_trade(position: Position, paper_mode: bool = True):
    """
    Upsert by position_id.
    FIX: uses the passed paper_mode arg directly.
    Called on open, partial close, and final close.
    """
    row = {
        "position_id":    position.position_id,
        "coin":           position.coin,
        "side":           position.side,
        "signal_id":      position.signal_id,
        "entry_price":    position.entry_price,
        "stop_price":     position.stop_price,
        "tp_price":       position.tp_price,
        "atr":            position.atr,
        "size_usd":       position.size_usd,
        "risk_usd":       position.risk_usd,
        "r_value":        position.r_value,
        "size_multiplier":position.size_multiplier,
        "partial_r":      round(position.partial_r,    4),
        "runner_r":       round(position.runner_r,     4),
        "realized_r":     round(position.realized_r,   4),
        "pnl_usd":        round(position.pnl_usd,      2),
        "rr_planned":     round(position.rr_planned,   3),
        "state":          position.state.value,
        "close_reason":   position.close_reason.value if position.close_reason else "",
        "partial_closed": position.partial_closed,
        "breakeven_moved":position.breakeven_moved,
        "setup_family":   position.setup_family,
        "htf_regime":     position.htf_regime,
        "regime":         position.regime,
        "confidence":     position.confidence,
        "total_score":    position.total_score,
        "session":        position.session,
        "opened_at":      position.opened_at.isoformat(),
        "closed_at":      position.closed_at.isoformat() if position.closed_at else "",
        "paper_mode":     str(paper_mode).lower(),   # FIX: use passed arg
    }

    existing: List[dict] = []
    updated = False

    if os.path.exists(TRADES_FILE):
        with open(TRADES_FILE, newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r.get("position_id") == position.position_id:
                    existing.append(row)
                    updated = True
                else:
                    existing.append(r)

    if not updated:
        existing.append(row)

    with open(TRADES_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(existing)
"""
Append-only, idempotent CSV storage for the cascade-fade forward
collector. Two files per symbol per UTC day, mirroring this codebase's
existing shadow_research_candidates/outcomes split: events.csv logs each
detected cascade immediately (unique event_id); resolutions.csv logs the
outcome once known, matched back to event_id. Never overwrites; a
restart must not duplicate a row for an event_id already on disk.
"""

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Set

DATA_DIR = Path(__file__).resolve().parent / "data"

EVENT_FIELDNAMES = [
    "event_id", "symbol", "detected_at_utc", "cascade_move_pct", "cascade_vol_ratio",
    "fade_side", "entry_price", "stop_price", "tp1_price", "tp2_price",
]
RESOLUTION_FIELDNAMES = [
    "event_id", "symbol", "resolved_at_utc", "filled", "exit_reason",
    "avg_exit_price", "gross_pnl_pct", "fees_pct", "net_pnl_pct", "pnl_r", "bars_held",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _today_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _event_path(symbol: str, date_str: str) -> Path:
    return DATA_DIR / f"{symbol.upper()}_{date_str}_events.csv"


def _resolution_path(symbol: str, date_str: str) -> Path:
    return DATA_DIR / f"{symbol.upper()}_{date_str}_resolutions.csv"


def _existing_ids(path: Path, id_col: str = "event_id") -> Set[str]:
    if not path.exists():
        return set()
    seen = set()
    with open(path, "r", newline="") as f:
        for row in csv.DictReader(f):
            v = row.get(id_col, "")
            if v:
                seen.add(v)
    return seen


def make_event_id(symbol: str, detected_ts_ms: int) -> str:
    return f"{symbol.upper()}_{detected_ts_ms}"


def append_event(row: Dict[str, Any], date_str: str = None) -> bool:
    date_str = date_str or _today_str()
    path = _event_path(row["symbol"], date_str)
    if row["event_id"] in _existing_ids(path):
        return False
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=EVENT_FIELDNAMES)
        if is_new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in EVENT_FIELDNAMES})
    return True


def append_resolution(row: Dict[str, Any], date_str: str = None) -> bool:
    date_str = date_str or _today_str()
    path = _resolution_path(row["symbol"], date_str)
    if row["event_id"] in _existing_ids(path):
        return False
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESOLUTION_FIELDNAMES)
        if is_new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in RESOLUTION_FIELDNAMES})
    return True


def find_unresolved_event_ids(days_back: int = 2) -> Set[str]:
    """Scan recent event files for event_ids that don't yet have a
    matching row in the corresponding resolution file."""
    from datetime import timedelta
    unresolved = set()
    for d in range(days_back):
        date_str = (datetime.now(timezone.utc) - timedelta(days=d)).strftime("%Y-%m-%d")
        for event_path in DATA_DIR.glob(f"*_{date_str}_events.csv"):
            symbol = event_path.name.split("_")[0]
            resolved_ids = _existing_ids(_resolution_path(symbol, date_str))
            event_ids = _existing_ids(event_path)
            unresolved |= (event_ids - resolved_ids)
    return unresolved


def load_event(event_id: str, days_back: int = 2) -> Dict[str, Any]:
    """Find and return the full event row for event_id by scanning recent
    event files (event_id embeds the symbol so this is a small scan)."""
    from datetime import timedelta
    symbol = event_id.split("_")[0]
    for d in range(days_back):
        date_str = (datetime.now(timezone.utc) - timedelta(days=d)).strftime("%Y-%m-%d")
        path = _event_path(symbol, date_str)
        if not path.exists():
            continue
        with open(path, "r", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("event_id") == event_id:
                    return row
    return {}

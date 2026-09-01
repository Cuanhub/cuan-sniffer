"""
Append-only, idempotent CSV storage for the funding-data collector.

One file per symbol per UTC day under research/funding/data/. Never
overwrites; re-running the collector after a crash/restart must not
duplicate rows for a timestamp already on disk -- resume is safe by
construction, not by assumption.
"""

import csv
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Set

DATA_DIR = Path(__file__).resolve().parent / "data"

FIELDNAMES = [
    "timestamp_utc",
    "symbol",
    "funding_rate",
    "realized_funding_rate",
    "funding_interval_hours",
    "mark_price",
    "oracle_price",
    "open_interest",
    "open_interest_usd",
    "best_bid",
    "best_ask",
    "mid",
    "spread_bps",
    "volume_24h_usd",
    "raw_response_json",
]


def _file_path(symbol: str, when: datetime) -> Path:
    date_str = when.strftime("%Y-%m-%d")
    return DATA_DIR / f"{symbol.upper()}_{date_str}.csv"


def _existing_timestamps(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    seen = set()
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = row.get("timestamp_utc", "")
            if ts:
                seen.add(ts)
    return seen


def append_row(row: Dict[str, Any]) -> bool:
    """
    Append one row for row["symbol"] / row["timestamp_utc"]. Returns False
    (no-op) if a row for that exact timestamp already exists in the target
    file -- this is what makes a restart safe to just re-run.
    """
    symbol = str(row["symbol"]).upper()
    ts_str = row["timestamp_utc"]
    when = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = _file_path(symbol, when)

    if ts_str in _existing_timestamps(path):
        return False

    is_new_file = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if is_new_file:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in FIELDNAMES})
    return True


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

"""The sealed holdout: the most recent HOLDOUT_DAYS that development never sees.

A lab that tests many ideas on the same data will find something by chance,
so nothing here is believed until it also works on bars it was not developed
on. Development runs get the archive cut at the seal; only `lab holdout` sees
past it, at most once per experiment, and every opening is counted against
the seal (the holdout threshold is 0.05 / openings so far).

When HOLDOUT_DAYS of new data has piled up beyond the holdout window, the
seal rolls forward: the old holdout joins development, a new unseen window
begins, and its opening count starts at zero because nobody has seen it.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from labcore import paths

HOLDOUT_DAYS = 45
DAY_MS = 86_400_000


def _iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, timezone.utc).isoformat(timespec="seconds")


def load() -> Optional[Dict[str, Any]]:
    return json.loads(paths.SEAL.read_text()) if paths.SEAL.exists() else None


def current(newest_ms: int) -> Dict[str, Any]:
    """The seal in force, creating the first one if none exists."""
    s = load()
    if s is None:
        s = {"history": []}
    if not s.get("seal_id"):
        _new(s, newest_ms, 1)
    return s


def roll_if_due(newest_ms: int) -> Optional[str]:
    s = current(newest_ms)
    if newest_ms - s["seal_ms"] >= 2 * HOLDOUT_DAYS * DAY_MS:
        n = int(s["seal_id"][1:]) + 1
        s["history"].append({k: s[k] for k in ("seal_id", "seal_ms", "seal_utc", "created_utc")})
        _new(s, newest_ms, n)
        return s["seal_id"]
    _save(s)
    return None


def _new(s: Dict[str, Any], newest_ms: int, n: int) -> None:
    seal_ms = newest_ms - HOLDOUT_DAYS * DAY_MS
    seal_ms -= seal_ms % 3_600_000
    s.update({"seal_id": f"S{n}", "seal_ms": seal_ms, "seal_utc": _iso(seal_ms),
              "created_utc": _iso(int(datetime.now(timezone.utc).timestamp() * 1000))})
    _save(s)


def _save(s: Dict[str, Any]) -> None:
    paths.SEAL.parent.mkdir(parents=True, exist_ok=True)
    paths.SEAL.write_text(json.dumps(s, indent=1))

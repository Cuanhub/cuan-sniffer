"""The ledger: every experiment, including every failure, append-only.

The denominator is the whole point. Twenty experiments at p < 0.05 produce
one "discovery" by chance, so a lab that forgets its failures overstates
every success. Each line carries the SHA-256 of the line before it, so a
deleted or edited entry breaks the chain and `verify()` says where.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from labcore import paths

GENESIS = "0" * 64
EVENTS = ("historical", "registered", "null_calibrated", "dev_result",
          "holdout_result", "proposed", "operator_decision", "retired", "note")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _digest(line: str) -> str:
    return hashlib.sha256(line.encode()).hexdigest()


def entries(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    p = path or paths.LEDGER
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def append(event: str, exp_id: str, payload: Dict[str, Any],
           path: Optional[Path] = None) -> Dict[str, Any]:
    if event not in EVENTS:
        raise ValueError(f"unknown ledger event {event!r}")
    p = path or paths.LEDGER
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = p.read_text().splitlines() if p.exists() else []
    prev = _digest(lines[-1]) if lines else GENESIS
    row = {"ts": _now(), "event": event, "id": exp_id, "prev": prev,
           "payload": payload}
    line = json.dumps(row, sort_keys=True, default=str)
    with p.open("a") as f:
        f.write(line + "\n")
    return row


def verify(path: Optional[Path] = None) -> Optional[str]:
    """None if the chain is intact, else where it breaks."""
    p = path or paths.LEDGER
    if not p.exists():
        return None
    prev = GENESIS
    for n, line in enumerate(p.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("prev") != prev:
            return f"line {n}: chain broken -- an entry before it was edited or removed"
        prev = _digest(line)
    return None


def experiments(path: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    """Fold the event stream into one current record per experiment."""
    out: Dict[str, Dict[str, Any]] = {}
    for row in entries(path):
        e = out.setdefault(row["id"], {"id": row["id"], "events": []})
        e["events"].append(row["event"])
        e[row["event"]] = row["payload"]
        e["last_ts"] = row["ts"]
    return out


def next_id(path: Optional[Path] = None) -> str:
    nums = [int(k[1:4]) for k in experiments(path)
            if len(k) >= 4 and k[0] == "E" and k[1:4].isdigit()]
    return f"E{(max(nums) + 1) if nums else 1:03d}"


def holdout_spends(seal_id: str, path: Optional[Path] = None) -> int:
    return sum(1 for r in entries(path) if r["event"] == "holdout_result"
               and r["payload"].get("seal_id") == seal_id)


def tested_count(path: Optional[Path] = None) -> int:
    """Hypotheses that have touched real data, historical ones included --
    the honest denominator for anything reported as a discovery."""
    n = 0
    for e in experiments(path).values():
        if "historical" in e:
            n += int(e["historical"].get("hypotheses", 1))
        elif "dev_result" in e:
            n += len(e["dev_result"].get("framings", {})) or 1
    return n

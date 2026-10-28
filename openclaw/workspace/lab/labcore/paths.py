"""Where everything lives. One place, so nothing writes outside the lab."""

from __future__ import annotations

import os
from pathlib import Path

LAB = Path(os.getenv("CUAN_LAB_ROOT", Path(__file__).resolve().parents[1]))
DATA = LAB / "data"
ARCHIVE = DATA / "archive" / "1h"
ARCHIVE_15M = DATA / "archive" / "15m"
ARCHIVE_MANIFEST = DATA / "archive" / "manifest.json"
REVISIONS = DATA / "archive" / "revisions.jsonl"
LIVE = DATA / "live"                     # read-only replica of the live CSVs
EXPERIMENTS = LAB / "experiments"
RESULTS = LAB / "results"
LEDGER = LAB / "ledger" / "ledger.jsonl"
PROPOSALS = LAB / "proposals"
SEAL = LAB / "ledger" / "seal.json"

# The live repo's data directory, if this lab runs on the same machine. It is
# only ever READ (sync.py copies from it); nothing in the lab writes there.
LIVE_SOURCE = os.getenv("CUAN_LIVE_DATA", "").strip()

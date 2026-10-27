"""A read-only replica of the live system's research records.

Copies FROM the live repo's research/levels/data INTO this lab's copy of
research/levels/data, which is where tools.why and tools.boundary read. The
direction never reverses: nothing in the lab opens a file under
CUAN_LIVE_DATA for writing, and test_lab checks that statically.

The control files (_control_queue.json, _halt.json, ...) are deliberately
NOT in the list. They are instructions to the live trader, not records of
it; a lab that can read them is one edit away from writing them.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from labcore import paths

RECORDS = ("episodes.csv", "outcomes.csv", "labels.csv",
           "zone_episodes.csv", "zone_outcomes.csv")


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sync_live(source: str = "", dest: Optional[Path] = None) -> Dict[str, Any]:
    src = Path(source or paths.LIVE_SOURCE).expanduser()
    dst = dest or (paths.LAB / "research" / "levels" / "data")
    if not str(source or paths.LIVE_SOURCE) or not src.is_dir():
        return {"synced": False, "reason": "CUAN_LIVE_DATA not set or not a directory"}
    if src.resolve() == dst.resolve():
        return {"synced": False, "reason": "source and destination are the same directory"}
    dst.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in RECORDS:
        s = src / name
        if s.is_file():
            shutil.copyfile(s, dst / name)          # read source, write lab copy
            files[name] = {"bytes": s.stat().st_size, "sha256": _sha(dst / name)}
    manifest = {"synced": True, "source": str(src), "files": files,
                "at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (dst / "_sync_manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest

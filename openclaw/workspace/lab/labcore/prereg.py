"""Pre-registration: the design is fixed, and provably so, before the data.

`register` validates PREREG.md, runs the experiment's own tests and the
causality check, then git-commits the design and records its hashes in the
ledger. `run` refuses unless the PREREG and the code still hash to what was
registered -- an edited design is a new experiment, and its results may not
inherit the old registration's timestamp.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from labcore import paths

SECTIONS = ("Question", "Grounding", "Data", "Feature definition", "Framings",
            "Controls", "Decision rule", "What would make this wrong",
            "Graveyard check")
PLACEHOLDERS = re.compile(r"\b(TODO|TBD|XXX|FIXME)\b|<fill|\.\.\.\s*$", re.I | re.M)


def sections(text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    cur: Optional[str] = None
    for line in text.splitlines():
        m = re.match(r"^##\s+(.+?)\s*$", line)
        if m:
            cur = m.group(1)
            out[cur] = ""
        elif cur is not None:
            out[cur] += line + "\n"
    return out


def validate(text: str, framings: Dict[str, dict]) -> List[str]:
    problems = []
    found = sections(text)
    for s in SECTIONS:
        body = found.get(s)
        if body is None:
            problems.append(f"missing section '## {s}'")
        elif len(body.strip()) < 20:
            problems.append(f"section '## {s}' is empty or too thin")
    if PLACEHOLDERS.search(text):
        problems.append("contains a placeholder (TODO/TBD/XXX/<fill...>)")
    fr = found.get("Framings", "")
    for name in framings:
        if name not in fr:
            problems.append(f"framing '{name}' is in FRAMINGS but not described under '## Framings'")
    return problems


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def harness_digest() -> str:
    """One hash over every file that computes a number or a verdict."""
    h = hashlib.sha256()
    roots = [paths.LAB / "research" / "backtest", paths.LAB / "research" / "levels",
             paths.LAB / "labcore"]
    for root in roots:
        for p in sorted(root.rglob("*.py")):
            if "experiments" in p.parts and root.name == "backtest":
                continue
            h.update(str(p.relative_to(paths.LAB)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def git_commit(files: List[Path], message: str) -> Optional[str]:
    if not (paths.LAB / ".git").exists():
        return None
    rel = [str(Path(f).resolve().relative_to(paths.LAB.resolve())) for f in files]
    subprocess.run(["git", "add", *rel], cwd=paths.LAB, check=True, capture_output=True)
    r = subprocess.run(["git", "commit", "-m", message], cwd=paths.LAB,
                       capture_output=True, text=True)
    if r.returncode != 0 and "nothing to commit" not in (r.stdout + r.stderr):
        raise RuntimeError(f"git commit failed: {r.stderr.strip()[:200]}")
    out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=paths.LAB,
                         capture_output=True, text=True, check=True)
    return out.stdout.strip()

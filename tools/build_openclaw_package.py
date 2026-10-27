#!/usr/bin/env python3
"""Build the OpenClaw lab package from this repo's current source.

    python -m tools.build_openclaw_package            # -> dist/cuan-lab/ and dist/cuan-lab.tar.gz

The package is an OpenClaw workspace (openclaw/workspace/) with the lab
inside it, plus COPIES of the research code it runs -- research/levels (the
strategy's band logic), research/backtest (the harness), tools/why.py and
tools/boundary.py -- taken from this repo at build time, so the lab always
backtests the strategy that is actually live.

What it must never contain, each checked before anything is written out:
  - anything that can place an order: core/, the live trader, the Telegram
    control, the protection loop, the Hyperliquid exchange client
  - a .env, a private key, or this account's address
  - the live data directory (positions, sizes, control files)
A failed check aborts the build; there is no flag to skip one.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "openclaw" / "workspace"
DIST = REPO / "dist"

RESEARCH_FILES = [
    "research/__init__.py",
    "research/levels/*.py",
    "research/levels/SCHEMA.md",
    "research/backtest/*.py",
    "research/backtest/PREREG_*.md",
    "research/backtest/experiments/*.py",
    "tools/__init__.py",
    "tools/why.py",
    "tools/boundary.py",
    "tests/fixtures/zones/*.json",
    "tests/fixtures/hyperliquid/candle_snapshot_1h_eth.json",
    "tests/fixtures/hyperliquid/PROVENANCE.md",
    "test_backtest_engine.py",
    "test_backtest_trendline.py",
    "test_backtest_stats_and_data.py",
    "test_backtest_isolation.py",
    "test_backtest_experiment.py",
    "test_zones.py",
    "test_zone_pass.py",
    "test_zone_log.py",
    "test_research_levels_isolation.py",
]
# Evidence that may not exist yet (an experiment not yet run). Copied when
# present; never required.
OPTIONAL_FILES = [
    "research/backtest/results/*/*.json",
    "research/backtest/results/*/*.md",
]

FORBIDDEN_PATHS = ("core/", "tools/live_levels_trader.py", "tools/telegram_control.py",
                   "tools/protect_loop.py", "tools/retention.py", "notify.py",
                   "research/levels/data/", "deploy/", "archive/")
FORBIDDEN_IMPORTS = ("core", "notify", "tools.live_levels_trader", "tools.telegram_control",
                     "tools.protect_loop", "hyperliquid.exchange", "eth_account")
PRIVATE_KEY = re.compile(r"0x[0-9a-fA-F]{64}\b")
# Caches and environments a development run leaves in the source tree. Not
# source, not reviewable, and a cache can hold strings from whatever it
# analysed -- never shipped.
BUILD_ARTIFACTS = ("__pycache__", ".pytest_cache", ".mypy_cache", ".venv", ".git",
                   ".ruff_cache")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _account_address() -> str:
    addr = os.getenv("HL_ACCOUNT_ADDRESS", "").strip().lower()
    env = REPO / ".env"
    if not addr and env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("HL_ACCOUNT_ADDRESS="):
                addr = line.split("=", 1)[1].strip().strip("'\"").lower()
    return addr


def check(out: Path) -> list:
    """Every reason this package must not ship. Empty means clean."""
    problems = []
    addr = _account_address()
    for p in sorted(out.rglob("*")):
        if not p.is_file():
            continue
        rel = str(p.relative_to(out))
        lab_rel = rel[4:] if rel.startswith("lab/") else rel
        if any(part in BUILD_ARTIFACTS for part in p.relative_to(out).parts):
            problems.append(f"build artifact shipped: {rel}")
            continue
        if p.name == ".env" or p.name.startswith(".env."):
            problems.append(f"secrets file shipped: {rel}")
        placeholder = p.name == ".gitkeep" and p.stat().st_size == 0
        if any(lab_rel.startswith(f) for f in FORBIDDEN_PATHS) and not placeholder:
            problems.append(f"order-capable or live path shipped: {rel}")
        if p.suffix in (".py", ".md", ".json", ".txt", ".sh", ".ini", ".jsonl", ".csv"):
            text = p.read_text(errors="ignore")
            if addr and addr in text.lower():
                problems.append(f"account address appears in {rel}")
            if PRIVATE_KEY.search(text):
                problems.append(f"something shaped like a private key in {rel}")
        if p.suffix == ".py":
            import ast
            for node in ast.walk(ast.parse(p.read_text(), filename=rel)):
                names = ([a.name for a in node.names] if isinstance(node, ast.Import) else
                         [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
                for n in names:
                    if any(n == b or n.startswith(b + ".") for b in FORBIDDEN_IMPORTS):
                        problems.append(f"{rel} imports {n}")
    return problems


def build(out: Path = DIST / "cuan-lab", tarball: bool = True, log=print) -> Path:
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(SRC, out, ignore=shutil.ignore_patterns(*BUILD_ARTIFACTS, "*.pyc",
                                                            "data", "results"))
    shutil.copy2(REPO / "openclaw" / "README.md", out / "README.md")
    lab = out / "lab"
    n = 0
    for pattern in RESEARCH_FILES + OPTIONAL_FILES:
        hits = sorted(REPO.glob(pattern))
        if not hits and pattern in RESEARCH_FILES:
            raise SystemExit(f"build: nothing matched {pattern}")
        for src in hits:
            if "__pycache__" in src.parts:
                continue
            dst = lab / src.relative_to(REPO)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            n += 1
    (lab / "research" / "levels" / "data").mkdir(parents=True, exist_ok=True)
    (lab / "research" / "levels" / "data" / ".gitkeep").write_text("")
    for d in ("data", "results", "proposals"):
        (lab / d).mkdir(exist_ok=True)
        (lab / d / ".gitkeep").write_text("")

    problems = check(out)
    if problems:
        shutil.rmtree(out)
        raise SystemExit("BUILD REFUSED:\n  - " + "\n  - ".join(problems))

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                            text=True).stdout.strip()
    # research/levels/data is the live bot's runtime state, rewritten every
    # pass; it is not source and is never shipped, so it cannot make the
    # build "dirty".
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "research", "tools",
                                 "openclaw", ":!research/levels/data"], cwd=REPO,
                                capture_output=True, text=True).stdout.strip())
    files = {str(p.relative_to(out)): _sha(p) for p in sorted(out.rglob("*")) if p.is_file()}
    manifest = {"built_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "source_commit": commit, "source_dirty": dirty, "files": files}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    log(f"[BUILD] {out}: {len(files)} files ({n} copied from the repo), "
        f"source {commit[:10]}{' + uncommitted changes' if dirty else ''}")
    if tarball:
        tgz = out.parent / f"{out.name}.tar.gz"
        with tarfile.open(tgz, "w:gz") as tf:
            tf.add(out, arcname=out.name)
        log(f"[BUILD] {tgz} ({tgz.stat().st_size / 1e6:.1f} MB)")
    return out


if __name__ == "__main__":
    build()
    sys.exit(0)

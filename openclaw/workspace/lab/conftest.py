"""No test may touch the lab's real ledger, archive, results or the replica.

Autouse, so a new test cannot forget. Every path in labcore.paths and the
research data directory point into a temporary directory for each test.
"""

import shutil
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_lab(tmp_path, monkeypatch):
    from labcore import paths
    from research.levels import storage

    root = tmp_path / "lab"
    monkeypatch.setattr(paths, "DATA", root / "data")
    monkeypatch.setattr(paths, "ARCHIVE", root / "data" / "archive" / "1h")
    monkeypatch.setattr(paths, "ARCHIVE_MANIFEST", root / "data" / "archive" / "manifest.json")
    monkeypatch.setattr(paths, "REVISIONS", root / "data" / "archive" / "revisions.jsonl")
    monkeypatch.setattr(paths, "LIVE", root / "data" / "live")
    monkeypatch.setattr(paths, "RESULTS", root / "results")
    monkeypatch.setattr(paths, "LEDGER", root / "ledger" / "ledger.jsonl")
    monkeypatch.setattr(paths, "PROPOSALS", root / "proposals")
    monkeypatch.setattr(paths, "SEAL", root / "ledger" / "seal.json")
    monkeypatch.setattr(paths, "LIVE_SOURCE", "")
    monkeypatch.setattr(storage, "DATA_DIR", root / "research_data")
    yield root

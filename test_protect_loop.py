"""tools/protect_loop.py: the protection daemon's startup.

During the 2026-10-22 outage this logged "no venue ... cannot run" 117 times
-- a crash loop at the 30s ThrottleInterval, because it gave the process up on
the first construction failure.
"""
from __future__ import annotations

import pytest

import tools.protect_loop as PL


def test_a_transient_venue_failure_is_retried(monkeypatch):
    calls = {"n": 0}
    waits = []

    class Boom(Exception):
        pass

    def flaky(debug=False):
        calls["n"] += 1
        if calls["n"] < 3:
            raise Boom("429 Too Many Requests")
        return "VENUE"

    monkeypatch.setattr(PL, "Venue", flaky)
    got = PL.start_venue(sleep=waits.append)
    assert got == "VENUE"
    assert calls["n"] == 3
    assert waits == [15, 30], f"backoff was {waits}"


def test_it_gives_up_after_the_last_attempt(monkeypatch):
    waits = []

    def always(debug=False):
        raise RuntimeError("NameResolutionError")

    monkeypatch.setattr(PL, "Venue", always)
    with pytest.raises(RuntimeError):
        PL.start_venue(sleep=waits.append)
    assert len(waits) == PL.START_ATTEMPTS - 1, waits


def test_it_waits_long_enough_to_matter(monkeypatch):
    """The point is to stop cycling at the ThrottleInterval. Five attempts at
    15/30/45/60 is 150s in process, so a persistent outage restarts every ~3
    minutes instead of every 30 seconds."""
    total = sum(PL.START_BACKOFF_SEC * i for i in range(1, PL.START_ATTEMPTS))
    assert total >= 120, total


def test_a_first_attempt_success_does_not_sleep(monkeypatch):
    waits = []
    monkeypatch.setattr(PL, "Venue", lambda debug=False: "VENUE")
    assert PL.start_venue(sleep=waits.append) == "VENUE"
    assert waits == []

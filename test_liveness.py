"""tools/liveness.py: the system says so when it goes quiet."""

from __future__ import annotations

import os
from unittest import mock

import pytest

from tools import liveness


class Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


@pytest.fixture
def rig(tmp_path):
    clock = Clock()
    sent = []
    for name in liveness.STALE_SEC:
        p = tmp_path / f"{name}.log"
        p.write_text("x")
        os.utime(p, (clock.t, clock.t))
    with mock.patch.object(liveness.notify, "send",
                           lambda text, **k: sent.append(text) or True):
        yield clock, sent, tmp_path, liveness.Watch(
            now=clock, logs=tmp_path, state=tmp_path / "state.json")


def touch(tmp_path, clock, *names):
    for n in names or liveness.STALE_SEC:
        os.utime(tmp_path / f"{n}.log", (clock.t, clock.t))


def test_a_quiet_healthy_system_says_nothing(rig):
    clock, sent, logs, w = rig
    for _ in range(50):
        clock.t += 60
        touch(logs, clock)
        w.tick()
    assert sent == []


def test_a_service_that_stops_logging_is_reported_then_its_recovery(rig):
    clock, sent, logs, w = rig
    for _ in range(6):
        clock.t += 60
        touch(logs, clock, "trader", "scanner")
        w.tick()
    assert len(sent) == 1 and "protect has gone quiet" in sent[0]
    clock.t += 60
    touch(logs, clock)
    w.tick()
    assert "protect is logging again" in sent[-1]


def test_a_sleep_is_reported_on_wake_without_calling_everything_dead(rig):
    clock, sent, logs, w = rig
    w.tick()
    clock.t += 37 * 60                 # asleep: no tick, no log writes
    w.tick()
    assert len(sent) == 1 and "down for 37 min" in sent[0]


def test_a_gap_alert_is_retried_until_it_lands(rig):
    clock, sent, logs, w = rig
    w.tick()
    clock.t += 3600
    with mock.patch.object(liveness.notify, "send", lambda text, **k: False):
        w.tick()
    assert sent == []
    clock.t += 30
    touch(logs, clock)
    w.tick()
    assert len(sent) == 1 and "down for" in sent[0]


def test_downtime_across_a_restart_is_reported(rig):
    clock, sent, logs, w = rig
    w.tick()
    clock.t += 7200
    again = liveness.Watch(now=clock, logs=logs, state=logs / "state.json")
    again.tick()
    assert len(sent) == 1 and "2.0h" in sent[0]


def test_a_first_ever_run_reports_no_gap(rig):
    _, sent, _, w = rig
    w.tick()
    assert sent == []


def test_no_url_means_no_ping(rig, monkeypatch):
    clock, _, logs, w = rig
    monkeypatch.delenv("HEARTBEAT_PING_URL", raising=False)
    with mock.patch.object(liveness.requests, "get") as get:
        w.tick()
    get.assert_not_called()


def test_pings_are_paced_and_stop_while_anything_is_stale(rig, monkeypatch):
    clock, _, logs, w = rig
    monkeypatch.setenv("HEARTBEAT_PING_URL", "https://example.test/abc")
    with mock.patch.object(liveness.requests, "get") as get:
        w.tick()
        clock.t += 30
        touch(logs, clock)
        w.tick()
        assert get.call_count == 1, "pinged faster than PING_EVERY_SEC"
        for _ in range(5):
            clock.t += 60
            touch(logs, clock)
            w.tick()
        assert get.call_count == 2
        for _ in range(6):
            clock.t += 60
            touch(logs, clock, "trader", "scanner")
            w.tick()
        get.reset_mock()
        for _ in range(6):
            clock.t += 60
            touch(logs, clock, "trader", "scanner")
            w.tick()
        get.assert_not_called()


def test_a_failed_ping_never_prints_the_url_or_raises(rig, monkeypatch, capsys):
    _, _, _, w = rig
    monkeypatch.setenv("HEARTBEAT_PING_URL", "https://example.test/secret")
    with mock.patch.object(liveness.requests, "get",
                           side_effect=OSError("https://example.test/secret")):
        w.tick()
    assert "secret" not in capsys.readouterr().out

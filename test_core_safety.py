"""core/safety.py: the kill switch, the two halt states, and the limits.

Scenarios from the build brief covered here: flatten under API errors and
backoff, daily loss breach, drawdown breach, unknown position at startup.
"""

from __future__ import annotations

import pytest

from core import execute, safety
from core.safety import AUTO, MANUAL, EquityLog, HaltState
from tests.fake_venue import FakeHL, build_venue


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    monkeypatch.setattr(execute, "SETTLE_SEC", 0.0)
    monkeypatch.setattr(safety, "FLATTEN_PAUSE_SEC", 0.0)
    safety.reset_errors()
    yield
    safety.reset_errors()


# ── flatten_all ────────────────────────────────────────────────────────────

def test_flatten_all_closes_every_venue_position():
    fake = FakeHL(positions={"HYPE": 2.0, "ZRO": 50.0}, mid=10.0)
    res = safety.flatten_all(build_venue(fake))
    assert res.flat and fake.pos == {}
    assert set(res.closed) == {"HYPE", "ZRO"}


def test_flatten_all_closes_a_position_no_state_file_knows_about():
    """REVIEW.md finding 5. /closeall iterated state["owned"], so an orphan --
    the exact thing C-1 produces -- could not be closed from Telegram at all.
    """
    fake = FakeHL(positions={"ZRO": 50.0}, mid=2.0)
    res = safety.flatten_all(build_venue(fake), reason="closeall")
    assert res.flat and "ZRO" in res.closed


def test_flatten_all_closes_a_short():
    fake = FakeHL(positions={"HYPE": -3.0}, mid=10.0)
    assert safety.flatten_all(build_venue(fake)).flat
    assert fake.pos == {}


def test_flatten_all_cancels_resting_orders_first():
    fake = FakeHL(positions={"HYPE": 2.0}, mid=100.0)
    fake.rest_protection("HYPE", 2.0, stop_px=90.0, tp_px=110.0)
    fake.rest_plain_order("HYPE", 1.0, px=80.0)
    res = safety.flatten_all(build_venue(fake))
    assert res.flat
    assert fake.resting == [], "a resting bid could re-open what we just closed"


def test_flatten_all_retries_until_the_venue_confirms_flat():
    """One IOC is not a close: a partial fill leaves size behind."""
    fake = FakeHL(positions={"HYPE": 8.0}, mid=100.0)
    fake.fill_fraction = 0.5                 # halves the position each round
    res = safety.flatten_all(build_venue(fake))
    assert res.flat and fake.pos == {}
    assert res.closed["HYPE"] == pytest.approx(8.0, rel=0.01)
    assert len(fake.sent) > 1, "a single IOC would have left size behind"


def test_flatten_all_gives_up_early_on_provable_deadlock(monkeypatch):
    """A position that will not move must not burn the whole 120s window.

    Caught by this suite hanging for 122s: repeated partial closes converge on
    a residual below the asset's size precision, which no order can express,
    and the loop retried it until the deadline.
    """
    monkeypatch.setattr(safety, "MAX_FLATTEN_SEC", 60.0)
    fake = FakeHL(positions={"HYPE": 2.0}, mid=100.0)
    fake.order_fills = False
    started = safety.time.time()
    res = safety.flatten_all(build_venue(fake), reason="deadlock")
    assert not res.flat
    assert safety.time.time() - started < 5.0, "it waited out the deadline"
    assert "no progress" in res.detail


def test_flatten_all_survives_a_burst_of_api_errors():
    fake = FakeHL(positions={"HYPE": 2.0}, mid=100.0)
    fake.fail_info = 3                       # first reads 429
    res = safety.flatten_all(build_venue(fake))
    assert res.flat and fake.pos == {}


def test_flatten_all_is_not_throttled_by_the_rate_limit_brake(monkeypatch):
    """REVIEW.md finding 10. Every read the kill switch makes is `urgent`, so
    the 429 brake never sleeps on its behalf.
    """
    slept = []
    monkeypatch.setattr(safety.time, "sleep", lambda s: slept.append(s))
    monkeypatch.setattr("core.venue.time.sleep", lambda s: slept.append(s))
    monkeypatch.setattr(safety, "RATE_LIMIT_SLEEP_SEC", 99.0, raising=False)
    fake = FakeHL(positions={"HYPE": 2.0}, mid=100.0)
    fake.fail_info = 2
    safety.flatten_all(build_venue(fake))
    assert all(s == 0.0 for s in slept), f"the brake throttled a flatten: {slept}"


def test_flatten_all_reports_failure_rather_than_claiming_flat(monkeypatch):
    """A position that will not close must be reported as still open, loudly."""
    monkeypatch.setattr(safety, "MAX_FLATTEN_SEC", 0.05)
    fake = FakeHL(positions={"HYPE": 2.0}, mid=100.0)
    fake.order_fills = False                 # the venue accepts and does nothing
    res = safety.flatten_all(build_venue(fake), reason="test")
    assert not res.flat
    assert res.still_open.get("HYPE") == pytest.approx(2.0)
    assert "STILL OPEN" in res.detail


def test_flatten_all_alerts_when_it_fails(monkeypatch, captured_alerts):
    monkeypatch.setattr(safety, "MAX_FLATTEN_SEC", 0.05)
    fake = FakeHL(positions={"HYPE": 2.0}, mid=100.0)
    fake.order_fills = False
    safety.flatten_all(build_venue(fake), reason="test")
    blob = " ".join(str(p) for p in captured_alerts)
    assert "Flatten did not finish" in blob and "HYPE" in blob


def test_an_unreadable_venue_at_the_deadline_is_never_called_flat(monkeypatch):
    monkeypatch.setattr(safety, "MAX_FLATTEN_SEC", 0.05)
    fake = FakeHL(positions={"HYPE": 2.0}, mid=100.0)
    v = build_venue(fake)
    fake.fail_info_always = True             # permanently dark, not a burst
    res = safety.flatten_all(v)
    assert not res.flat, "claiming flat on an unreadable venue orphans a position"
    assert "cannot claim flat" in res.detail


def test_flatten_all_on_an_already_flat_account_sends_nothing():
    fake = FakeHL(positions={}, mid=100.0)
    res = safety.flatten_all(build_venue(fake))
    assert res.flat and fake.sent == []


# ── halt: exactly two states ───────────────────────────────────────────────

def test_an_auto_halt_clears_on_the_next_utc_day(tmp_path, monkeypatch):
    """The 2026-10-13 daily-loss halt was still blocking trading on 10-15."""
    st = HaltState(tmp_path / "halt.json")
    st.set(AUTO, "daily loss")
    assert st.active() is not None
    monkeypatch.setattr(HaltState, "_today", staticmethod(lambda: "2099-01-01"))
    assert st.active() is None, "an AUTO halt must clear itself"
    assert st.active() is None                # and stay cleared


def test_a_manual_halt_does_not_clear_itself(tmp_path, monkeypatch):
    st = HaltState(tmp_path / "halt.json")
    st.set(MANUAL, "drawdown")
    monkeypatch.setattr(HaltState, "_today", staticmethod(lambda: "2099-01-01"))
    assert st.active() is not None
    st.clear()
    assert st.active() is None


def test_there_is_no_third_halt_kind(tmp_path):
    st = HaltState(tmp_path / "halt.json")
    with pytest.raises(ValueError):
        st.set("verify_stops", "something")


def test_halting_flattens_and_requires_a_venue(tmp_path):
    """REVIEW.md finding 4: no backend=None path exists to pass."""
    fake = FakeHL(positions={"HYPE": 2.0}, mid=100.0)
    st = HaltState(tmp_path / "halt.json")
    res = safety.halt(build_venue(fake), st, "test halt", kind=AUTO)
    assert res.flat and fake.pos == {}
    assert st.active()["kind"] == AUTO
    with pytest.raises(TypeError):
        safety.halt(st, "no venue")          # type: ignore[call-arg,arg-type]


def test_a_halt_alerts_once_not_every_pass(tmp_path, captured_alerts):
    fake = FakeHL(positions={}, mid=100.0)
    v, st = build_venue(fake), HaltState(tmp_path / "halt.json")
    safety.halt(v, st, "first", kind=AUTO)
    before = len([p for p in captured_alerts if "Halted" in str(p)])
    safety.halt(v, st, "first", kind=AUTO)
    after = len([p for p in captured_alerts if "Halted" in str(p)])
    assert before == 1 and after == 1


def test_a_cleared_auto_halt_announces_the_resume(tmp_path, monkeypatch,
                                                  captured_alerts):
    st = HaltState(tmp_path / "halt.json")
    st.set(AUTO, "daily loss")
    monkeypatch.setattr(HaltState, "_today", staticmethod(lambda: "2099-01-01"))
    st.active()
    assert any("Resumed" in str(p) for p in captured_alerts)


# ── limits ─────────────────────────────────────────────────────────────────

def test_daily_loss_is_measured_against_the_day_start(tmp_path):
    log = EquityLog(tmp_path / "eq.json")
    fake = FakeHL(positions={}, usdc=1000.0)
    v = build_venue(fake)
    safety.check_limits(v, log)              # records day_start = 1000
    fake.usdc = 940.0                        # -6%
    daily = next(x for x in safety.check_limits(v, log) if x.name == "daily_loss")
    assert daily.breached and daily.halt_kind == AUTO
    assert "6.00%" in daily.detail


def test_daily_loss_inside_the_limit_is_not_a_breach(tmp_path):
    log = EquityLog(tmp_path / "eq.json")
    fake = FakeHL(positions={}, usdc=1000.0)
    v = build_venue(fake)
    safety.check_limits(v, log)
    fake.usdc = 960.0                        # -4%, limit is 5%
    daily = next(x for x in safety.check_limits(v, log) if x.name == "daily_loss")
    assert not daily.breached


def test_drawdown_is_measured_against_the_peak_and_is_manual(tmp_path):
    log = EquityLog(tmp_path / "eq.json")
    fake = FakeHL(positions={}, usdc=1000.0)
    v = build_venue(fake)
    safety.check_limits(v, log)              # peak = 1000
    fake.usdc = 880.0                        # -12%
    dd = next(x for x in safety.check_limits(v, log) if x.name == "drawdown")
    assert dd.breached and dd.halt_kind == MANUAL


def test_the_peak_survives_a_new_utc_day(tmp_path, monkeypatch):
    log = EquityLog(tmp_path / "eq.json")
    assert log.observe(1000.0) == (1000.0, 1000.0)
    monkeypatch.setattr(safety, "datetime", safety.datetime)
    # a new day resets day_start but must not reset the peak
    import datetime as _dt

    class _Tomorrow(_dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return _dt.datetime(2099, 1, 1, tzinfo=tz)

    monkeypatch.setattr(safety, "datetime", _Tomorrow)
    day_start, peak = log.observe(900.0)
    assert day_start == 900.0 and peak == 1000.0


def test_limits_are_log_only_until_the_equity_question_is_settled(tmp_path):
    """SIMPLIFY_PLAN 3C: a limit built on an unverified equity number can
    flatten a healthy account, so it reports and does not act yet."""
    assert safety.LIMITS_ENFORCE is False
    log, st = EquityLog(tmp_path / "eq.json"), HaltState(tmp_path / "halt.json")
    fake = FakeHL(positions={"HYPE": 2.0}, usdc=1000.0, mid=100.0)
    v = build_venue(fake)
    safety.check_limits(v, log)
    fake.usdc = 500.0                        # a 50% loss
    verdicts = safety.enforce_limits(v, st, log)
    assert any(x.breached for x in verdicts), "the breach must still be reported"
    assert st.active() is None, "but it must not halt while log-only"
    assert fake.pos.get("HYPE") == pytest.approx(2.0)


def test_enforcement_halts_and_flattens_when_switched_on(tmp_path, monkeypatch):
    monkeypatch.setattr(safety, "LIMITS_ENFORCE", True)
    log, st = EquityLog(tmp_path / "eq.json"), HaltState(tmp_path / "halt.json")
    fake = FakeHL(positions={"HYPE": 2.0}, usdc=1000.0, mid=100.0)
    v = build_venue(fake)
    safety.check_limits(v, log)
    fake.usdc = 500.0
    safety.enforce_limits(v, st, log)
    assert st.active() is not None and fake.pos == {}


def test_both_equity_readings_are_logged_while_a_position_is_open(tmp_path, capsys):
    log = EquityLog(tmp_path / "eq.json")
    fake = FakeHL(positions={"HYPE": 2.0}, usdc=1000.0, mid=100.0,
                  unrealized={"HYPE": -37.5})
    safety.check_limits(build_venue(fake), log)
    out = capsys.readouterr().out
    assert "usdc_total=1000.0000" in out and "with_unrealized=962.5000" in out


def test_an_unreadable_equity_is_not_a_breach(tmp_path):
    log = EquityLog(tmp_path / "eq.json")
    fake = FakeHL(positions={}, usdc=1000.0)
    v = build_venue(fake)
    fake.fail_info = 99
    assert safety.check_limits(v, log) == []


# ── error rate ─────────────────────────────────────────────────────────────

def test_the_error_rail_alerts_over_the_limit_once(captured_alerts):
    for i in range(safety.ERROR_LIMIT):
        safety.record_error("venue", i)
    assert not any("Error rate" in str(p) for p in captured_alerts)
    safety.record_error("venue", "one too many")
    assert sum("Error rate" in str(p) for p in captured_alerts) == 1
    for i in range(5):
        safety.record_error("venue", i)
    assert sum("Error rate" in str(p) for p in captured_alerts) == 1


def test_the_error_rail_never_raises_into_its_caller(monkeypatch):
    """An error reporter that becomes the error is worse than none."""
    def boom():
        raise RuntimeError("clock unavailable")

    monkeypatch.setattr(safety.time, "time", boom)
    assert safety.record_error("x", "y") == 0


def test_errors_age_out_of_the_window(monkeypatch):
    base = 1_000_000.0
    monkeypatch.setattr(safety.time, "time", lambda: base)
    for i in range(safety.ERROR_LIMIT):
        safety.record_error("venue", i)
    monkeypatch.setattr(safety.time, "time",
                        lambda: base + safety.ERROR_WINDOW_SEC + 1)
    assert safety.record_error("venue", "much later") == 1

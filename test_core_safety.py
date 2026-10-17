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


def test_the_drawdown_reference_is_a_rolling_window_not_an_all_time_peak(tmp_path):
    """The reference must follow the account down within the window.

    This replaces a test that asserted the opposite -- that the peak survives
    a new UTC day. That encoded an ALL-TIME high, which never decays: an
    account that falls 10% and stabilises then reads as breached for ever.
    The reference was deliberately moved off an all-time high (`7764dee`), and
    on 2026-10-13 a stale one kept the rail firing for two days against a high
    the account could no longer reach.
    """
    import datetime as _dt
    log = EquityLog(tmp_path / "eq.json")
    t0 = _dt.datetime(2026, 10, 13, 12, 0, tzinfo=_dt.timezone.utc)
    assert log.observe(1000.0, now=t0) == (1000.0, 1000.0)

    # Still inside the window: the old high is the reference.
    _, high = log.observe(900.0, now=t0 + _dt.timedelta(hours=1))
    assert high == 1000.0

    # Past the window: the 1000 reading is gone and 900 is the new high, so
    # the drawdown reads 0% rather than 10% for ever.
    day_start, high = log.observe(900.0,
                                  now=t0 + _dt.timedelta(
                                      hours=safety.DRAWDOWN_WINDOW_HOURS + 1))
    assert high == 900.0, "an all-time peak would strand the account here"
    assert day_start == 900.0, "a new UTC day resets the day-start"


def test_a_sample_is_taken_even_on_a_pass_that_refuses_to_trade(tmp_path):
    """From 2026-10-13 to 10-15 the halted trader took no equity readings at
    all, so it spent two days measured against a reference it could not
    reach. check_limits is what samples, and preflight calls it before every
    early return."""
    import inspect
    src = inspect.getsource(safety.check_limits)
    assert "log.observe" in src
    from core import safety as S
    pre = inspect.getsource(__import__("tools.live_levels_trader",
                                       fromlist=["x"]).preflight)
    assert "enforce_limits" in pre
    assert pre.index("enforce_limits") < pre.index("MAX_CONCURRENT"), (
        "the equity sample must be taken before the caps can return early")


def test_the_limits_act(tmp_path):
    """A breach halts and flattens. Enforcement is on by default."""
    assert safety.LIMITS_ENFORCE is True
    log, st = EquityLog(tmp_path / "eq.json"), HaltState(tmp_path / "halt.json")
    fake = FakeHL(positions={"HYPE": 2.0}, usdc=1000.0, mid=100.0)
    v = build_venue(fake)
    safety.check_limits(v, log)
    fake.usdc = 500.0                        # a 50% loss
    verdicts = safety.enforce_limits(v, st, log)
    assert any(x.breached for x in verdicts)
    assert st.active() is not None, "a breach must halt"
    assert fake.pos == {}, "a breach must flatten"


def test_a_breach_halts_with_the_kind_its_limit_carries(tmp_path):
    """Daily loss clears on the next UTC day; drawdown needs a human."""
    log, st = EquityLog(tmp_path / "eq.json"), HaltState(tmp_path / "halt.json")
    fake = FakeHL(positions={}, usdc=1000.0)
    v = build_venue(fake)
    safety.check_limits(v, log)
    fake.usdc = 940.0                        # -6%: daily loss only
    safety.enforce_limits(v, st, log)
    assert st.active()["kind"] == AUTO
    st.clear()
    fake.usdc = 880.0                        # -12%: drawdown too
    safety.enforce_limits(v, st, log)
    assert st.active()["kind"] == AUTO, "daily loss is checked first"


def test_equity_already_includes_unrealised_pnl():
    """The reading that lets the limits act, asserted against the recording.

    265 equity samples from the live trader's log. Where a position was open
    and nothing filled between two samples, total moved by exactly the flat
    leverage times the margin change -- i.e. by the whole change in position
    notional. So `total` marks to market and must NOT have unrealizedPnl added
    to it; doing so double-counts the open position.
    """
    import pathlib
    import statistics
    rows = [tuple(map(float, line.split())) for line in
            (pathlib.Path("tests/fixtures/hyperliquid/equity_hold_series.txt")
             .read_text().strip().splitlines())]
    assert len(rows) > 200, "the recorded series went missing"
    ratios = []
    prev = None
    for total, hold in rows:
        if prev and prev != (total, hold) and prev[1] > 0 and hold > 0:
            d_hold = hold - prev[1]
            if abs(d_hold) > 1e-9:
                ratios.append((total - prev[0]) / d_hold)
        prev = (total, hold)
    assert len(ratios) > 20
    assert statistics.median(ratios) == pytest.approx(5.0, abs=0.01), (
        "total no longer tracks position notional; re-derive before trusting "
        "the loss limits")


def test_equity_is_total_alone_and_never_total_plus_unrealised(tmp_path):
    """The double-count guard. There is deliberately no `with_unrealized`."""
    from core.venue import Equity
    assert not hasattr(Equity, "with_unrealized")
    log = EquityLog(tmp_path / "eq.json")
    fake = FakeHL(positions={"HYPE": 2.0}, usdc=1000.0, mid=100.0,
                  unrealized={"HYPE": -37.5})
    v = build_venue(fake)
    safety.check_limits(v, log)
    day_start, _ = log.observe(v.equity().usdc_total)
    assert day_start == pytest.approx(1000.0), (
        "the limit must read total alone; 962.5 would double-count")


def test_the_unrealised_figure_is_still_logged(tmp_path, capsys):
    log = EquityLog(tmp_path / "eq.json")
    fake = FakeHL(positions={"HYPE": 2.0}, usdc=1000.0, mid=100.0,
                  unrealized={"HYPE": -37.5})
    safety.check_limits(build_venue(fake), log)
    out = capsys.readouterr().out
    assert "equity 1000.0000" in out and "incl. unrealised -37.5000" in out


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


def test_the_error_alert_names_the_commonest_source(captured_alerts):
    """Ported from test_error_rate.py before it was retired: an operator
    needs to know WHICH thing is failing, not just that something is."""
    for _ in range(4):
        safety.record_error("open_orders", "429")
    for _ in range(2):
        safety.record_error("positions", "429")
    blob = " ".join(str(p) for p in captured_alerts)
    assert "Error rate" in blob
    assert "open_orders` x4" in blob and "positions` x2" in blob
    assert blob.index("open_orders") < blob.index("positions"), \
        "sources must be listed worst first"


def test_the_venue_read_paths_record_their_failures():
    """Also ported: the rail is worthless if the paths that fail never call
    it. These are the reads whose failure the operator must hear about."""
    import inspect
    from core import protect
    assert "record_error" in inspect.getsource(protect.ensure_stop)
    assert "record_error" in inspect.getsource(protect.run_once)
    assert "record_error" in inspect.getsource(safety.check_limits)

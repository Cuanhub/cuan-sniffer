"""core/streak.py: consecutive losing trades, grouped from real fills.

The grouping is the part that has to be right. `closedPnl` is per fill, so a
position closed in three fills is three rows, and counting rows instead of
trades is how the old stand-down miscounted. These tests run the grouper over
the 385 real recorded fills as well as over hand-built cases.
"""

from __future__ import annotations

import time

import pytest

from core import safety, streak
from core.streak import Trade, losing_run, trades_from_fills
from tests.fake_venue import FakeHL, build_venue, load


# ── grouping, against the real recording ──────────────────────────────────

def test_the_real_fills_group_into_whole_round_trips():
    fills = load("user_fills_30d.json")
    trades = trades_from_fills(fills)
    assert trades, "385 real fills produced no trades"
    # Every trade must contain at least one opening and one closing fill.
    assert all(t.fills >= 2 for t in trades), \
        "a trade with one fill means an open was paired with nothing"
    assert all(t.closed_ms >= t.opened_ms for t in trades)
    assert len(trades) < len(fills), "grouping did not collapse anything"


def test_grouping_collapses_multi_fill_entries():
    """The recorded data has an entry that filled in three parts and closed in
    one. That is one trade, not four."""
    fills = load("user_fills_30d.json")
    multi = [t for t in trades_from_fills(fills) if t.fills > 2]
    assert multi, "expected at least one multi-fill trade in the recording"


def test_a_trade_opened_before_the_window_is_excluded():
    """Its entry fees are not in the data, so its result would be flattering
    by an unknown amount -- and a loss limit must never be flattered."""
    fills = load("user_fills_30d.json")
    oldest = min(fills, key=lambda f: f["time"])
    assert float(oldest["startPosition"]) != 0.0, (
        "precondition: the recording starts mid-trade")
    trades = trades_from_fills(fills)
    assert all(t.opened_ms > oldest["time"] for t in trades)


def test_pnl_is_net_of_fees():
    """Compared against the same grouping with fees zeroed, so the excluded
    fills (opened before the window, or still open) fall out of both sides."""
    fills = load("user_fills_30d.json")
    free = [dict(f, fee="0.0") for f in fills]
    net = sum(t.pnl for t in trades_from_fills(fills))
    gross = sum(t.pnl for t in trades_from_fills(free))
    paid = gross - net
    assert paid > 0, "fees were not subtracted"
    # Every fee in the recording is positive, so what the trades paid must sit
    # between zero and the total fees on the file.
    assert paid <= sum(float(f["fee"]) for f in fills) + 1e-9


# ── grouping, on built cases ──────────────────────────────────────────────

def fill(coin, t, side, sz, start, pnl=0.0, fee=0.0, tid=0):
    return {"coin": coin, "time": t, "side": side, "sz": str(sz),
            "startPosition": str(start), "closedPnl": str(pnl),
            "fee": str(fee), "tid": tid}


def test_one_open_and_one_close_is_one_trade():
    trades = trades_from_fills([
        fill("HYPE", 100, "B", 2.0, 0.0, fee=0.1),
        fill("HYPE", 200, "A", 2.0, 2.0, pnl=-5.0, fee=0.1)])
    assert len(trades) == 1
    assert trades[0].pnl == pytest.approx(-5.2)
    assert trades[0].lost


def test_a_position_closed_in_three_fills_is_still_one_trade():
    trades = trades_from_fills([
        fill("HYPE", 100, "B", 3.0, 0.0, fee=0.1),
        fill("HYPE", 200, "A", 1.0, 3.0, pnl=-1.0, fee=0.1),
        fill("HYPE", 300, "A", 1.0, 2.0, pnl=-1.0, fee=0.1),
        fill("HYPE", 400, "A", 1.0, 1.0, pnl=-1.0, fee=0.1)])
    assert len(trades) == 1, "three closing fills became three trades"
    assert trades[0].pnl == pytest.approx(-3.4)
    assert trades[0].fills == 4


def test_two_round_trips_on_one_coin_are_two_trades():
    trades = trades_from_fills([
        fill("HYPE", 100, "B", 2.0, 0.0),
        fill("HYPE", 200, "A", 2.0, 2.0, pnl=-5.0),
        fill("HYPE", 300, "B", 2.0, 0.0),
        fill("HYPE", 400, "A", 2.0, 2.0, pnl=7.0)])
    assert [t.pnl for t in trades] == [pytest.approx(-5.0), pytest.approx(7.0)]


def test_trades_on_different_coins_do_not_merge():
    trades = trades_from_fills([
        fill("HYPE", 100, "B", 2.0, 0.0),
        fill("ZRO", 150, "B", 9.0, 0.0),
        fill("HYPE", 200, "A", 2.0, 2.0, pnl=-5.0),
        fill("ZRO", 250, "A", 9.0, 9.0, pnl=-3.0)])
    assert {t.coin for t in trades} == {"HYPE", "ZRO"}
    assert len(trades) == 2


def test_a_still_open_position_is_not_a_trade():
    trades = trades_from_fills([fill("HYPE", 100, "B", 2.0, 0.0)])
    assert trades == [], "an open position has no result yet"


def test_a_short_round_trip_is_grouped_too():
    trades = trades_from_fills([
        fill("HYPE", 100, "A", 2.0, 0.0),
        fill("HYPE", 200, "B", 2.0, -2.0, pnl=4.0)])
    assert len(trades) == 1 and not trades[0].lost


def test_trades_come_back_oldest_close_first():
    trades = trades_from_fills([
        fill("ZRO", 300, "B", 1.0, 0.0),
        fill("ZRO", 400, "A", 1.0, 1.0, pnl=-1.0),
        fill("HYPE", 100, "B", 1.0, 0.0),
        fill("HYPE", 200, "A", 1.0, 1.0, pnl=-1.0)])
    assert [t.closed_ms for t in trades] == [200, 400]


# ── the run ────────────────────────────────────────────────────────────────

def t(pnl, closed_ms=0, coin="HYPE"):
    return Trade(coin, closed_ms - 10, closed_ms, pnl, 2)


def test_the_run_is_the_losses_at_the_end():
    assert len(losing_run([t(5), t(-1), t(-2), t(-3)])) == 3


def test_a_win_ends_the_run():
    assert losing_run([t(-1), t(-2), t(5)]) == []


def test_a_scratch_trade_ends_the_run():
    """The old counter treated anything not strictly positive as a loss, so a
    break-even trade extended a run it should have ended."""
    assert losing_run([t(-1), t(-2), t(0.0)]) == []


def test_no_trades_is_no_run():
    assert losing_run([]) == []


# ── the stand-down ─────────────────────────────────────────────────────────

def losses(n, last_closed_ms, coin="HYPE"):
    """n losing round trips, the last closing at last_closed_ms."""
    out = []
    for i in range(n):
        closed = last_closed_ms - (n - 1 - i) * 3_600_000
        out.append(fill(coin, closed - 1000, "B", 1.0, 0.0))
        out.append(fill(coin, closed, "A", 1.0, 1.0, pnl=-1.0))
    return out


@pytest.fixture
def now_ms():
    return int(time.time() * 1000)


def test_three_losses_in_a_row_stand_the_bot_down(now_ms):
    fake = FakeHL()
    fake.fills_override = losses(3, now_ms - 60_000)
    sd = streak.current(build_venue(fake), now_ms=now_ms)
    assert sd is not None
    assert len(sd.run) == 3 and sd.hours_left == pytest.approx(24.0, abs=0.1)
    assert "3 losing trades in a row" in sd.reason


def test_two_losses_do_not(now_ms):
    fake = FakeHL()
    fake.fills_override = losses(2, now_ms - 60_000)
    assert streak.current(build_venue(fake), now_ms=now_ms) is None


def test_the_stand_down_expires_on_its_own(now_ms):
    """No flag to clear. Once the cooldown has passed since the last close it
    simply stops being returned -- the old one had to be cleared by hand."""
    fake = FakeHL()
    old = now_ms - int((streak.COOLDOWN_HOURS + 1) * 3_600_000)
    fake.fills_override = losses(3, old)
    assert streak.current(build_venue(fake), now_ms=now_ms) is None


def test_a_win_after_the_run_ends_it_immediately(now_ms):
    fake = FakeHL()
    rows = losses(3, now_ms - 3_600_000)
    rows += [fill("HYPE", now_ms - 2000, "B", 1.0, 0.0),
             fill("HYPE", now_ms - 1000, "A", 1.0, 1.0, pnl=9.0)]
    fake.fills_override = rows
    assert streak.current(build_venue(fake), now_ms=now_ms) is None


def test_a_losing_run_spread_across_coins_still_counts(now_ms):
    fake = FakeHL()
    rows = (losses(1, now_ms - 3 * 3_600_000, "HYPE")
            + losses(1, now_ms - 2 * 3_600_000, "ZRO")
            + losses(1, now_ms - 1 * 3_600_000, "NIL"))
    fake.fills_override = rows
    sd = streak.current(build_venue(fake), now_ms=now_ms)
    assert sd is not None and {x.coin for x in sd.run} == {"HYPE", "ZRO", "NIL"}


def test_an_unreadable_venue_does_not_stand_the_bot_down(now_ms):
    """Unlike a loss limit, the safe default is to allow trading: refusing on
    an unreadable venue would stop the bot on every API hiccup."""
    fake = FakeHL()
    fake.fills_override = losses(3, now_ms - 60_000)
    v = build_venue(fake)
    fake.fail_info_always = True
    assert streak.current(v, now_ms=now_ms) is None


def test_nothing_is_stored_so_a_wrong_answer_cannot_persist(now_ms, tmp_path):
    """The old counter lived in the state file and was fed by assumed_loss
    when the PnL query failed, so a rate limit manufactured losing trades and
    the count then drifted with no way to re-derive it."""
    fake = FakeHL()
    fake.fills_override = losses(3, now_ms - 60_000)
    v = build_venue(fake)
    assert streak.current(v, now_ms=now_ms) is not None
    fake.fills_override = []                 # the venue now shows no losses
    assert streak.current(v, now_ms=now_ms) is None, (
        "the run survived the fills that justified it")


# ── the operator override ──────────────────────────────────────────────────

def test_clearing_a_stand_down_is_not_undone_by_the_same_trades(now_ms,
                                                                tmp_path):
    fake = FakeHL()
    fake.fills_override = losses(3, now_ms - 60_000)
    v = build_venue(fake)
    ov = safety.StandDownOverride(tmp_path / "sd.json")
    assert safety.standing_down(v, ov) is not None
    ov.clear_now()
    assert safety.standing_down(v, ov) is None


def test_a_fresh_loss_after_a_clear_starts_a_new_run(now_ms, tmp_path):
    """Clearing silences the run that caused it, not the rail."""
    from core.meta import write_json
    path = tmp_path / "sd.json"
    cleared_at = now_ms - 10 * 3_600_000             # cleared ten hours ago
    write_json(path, {"ignore_before_ms": cleared_at})
    ov = safety.StandDownOverride(path)

    fake = FakeHL()
    v = build_venue(fake)
    fake.fills_override = losses(3, cleared_at - 60_000)   # all before it
    assert safety.standing_down(v, ov) is None

    fake.fills_override = losses(3, now_ms - 60_000)       # all after it
    assert safety.standing_down(v, ov) is not None, (
        "clearing must not disable the rail for good")


def test_no_override_file_means_no_trades_are_ignored(tmp_path):
    assert safety.StandDownOverride(tmp_path / "missing.json"
                                    ).ignore_before_ms() == 0

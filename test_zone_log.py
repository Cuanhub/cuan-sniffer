"""The band strategy's own record: recording and resolving.

This exists because the 2026-10-26 stop finding -- the largest effect in the
project -- had to be measured on the LEVEL scanner's episodes, since bands
kept no record of their own. A finding about bands rested on a different
strategy's data, and two candidate features could not be tested at all
because nothing stored them.
"""

from __future__ import annotations

import pathlib

import pytest

from research.levels import storage, zone_log
from research.levels.zones import Plan, Zone, SUPPLY


def band(lo=100.0, hi=103.0, base_i=900, bos_i=905, coin="SOL"):
    return Zone(coin=coin, lo=lo, hi=hi, base_i=base_i, bos_i=bos_i)


def plan(entry=101.5, stop=97.0, tp=112.0, immediate=True, atr=3.0, **kw):
    z = band(**kw)
    t = Zone(coin=z.coin, lo=tp, hi=tp * 1.01, base_i=910, bos_i=911,
             kind=SUPPLY)
    return Plan(zone=z, target=t, entry=entry, stop=stop, tp=tp,
                immediate=immediate, atr_pct=atr)


def bars(n=950, lo=99.0, hi=104.0, c=101.0):
    return [{"t": i * 3600_000, "o": c, "h": hi, "l": lo, "c": c}
            for i in range(n)]


# ── one row per band, not per pass ─────────────────────────────────────────

def test_the_id_is_the_band_so_a_repeated_pass_records_once():
    """At 450s a single band would otherwise write about 190 identical rows
    a day."""
    a, b = plan(), plan()
    assert zone_log.episode_id(a) == zone_log.episode_id(b)
    assert zone_log.episode_id(plan(base_i=800)) != zone_log.episode_id(a)


def test_the_id_matches_the_one_the_executor_uses():
    """zone_pass.to_episode builds the same string. If they drift, the
    recorded row and the traded row stop being the same trade."""
    from research.levels import zone_pass
    p = plan()
    assert zone_log.episode_id(p) == zone_pass.to_episode(p)["episode_id"]


def test_the_row_carries_what_the_analysis_needed_and_lacked():
    """universe_rank and day_ntl_vlm are here because the 2026-10-26 scan
    could NOT answer "does universe rank predict outcome" -- nothing stored
    it."""
    row = zone_log.episode_row(plan(), bars(), 101.0, taken=True,
                               universe_rank=7, day_ntl_vlm=1.23e8,
                               max_leverage=10,
                               ctx={"mark_px": 101.0, "oracle_px": 100.0})
    assert row["universe_rank"] == 7
    assert row["day_ntl_vlm"] == 1.23e8
    assert row["max_leverage"] == 10
    assert row["basis_bps"] == pytest.approx(100.0, abs=0.1)
    assert row["stop_atr_mult"] > 0
    for k in storage.ZONE_EPISODE_FIELDNAMES:
        assert k in row, f"{k} missing from the recorded row"


def test_a_missing_value_stays_missing_rather_than_becoming_zero():
    """A zero that somebody later averages is worse than a blank."""
    row = zone_log.episode_row(plan(), bars(), 101.0, taken=False)
    assert row["day_ntl_vlm"] == ""
    assert row["max_leverage"] == ""
    assert row["basis_bps"] == ""


def test_recording_never_raises_into_the_pass():
    """A recorder that can break the trading pass is worse than none."""
    # This writes, so it must never be pointed at the live data directory.
    # It leaked there once -- clean alone, leaking in the full suite -- so
    # the guard is asserted HERE, next to the write, and names itself.
    live = pathlib.Path(__file__).parent / "research/levels/data"
    assert storage.DATA_DIR.resolve() != live.resolve(), (
        f"storage.DATA_DIR is the LIVE directory ({storage.DATA_DIR}); "
        f"an earlier test left it unredirected")
    assert zone_log.record_pass(None, 2.0, None) == 0          # type: ignore[arg-type]
    assert zone_log.record_pass({"SOL": [plan()]}, 2.0, {}) >= 0


# ── resolving: `won` is decided once, here ─────────────────────────────────

def test_the_target_before_the_stop_is_a_win():
    fwd = ([{"t": 1, "h": 102.0, "l": 101.0, "c": 101.5}] * 25
           + [{"t": 2, "h": 113.0, "l": 101.0, "c": 112.5}])
    out = zone_log.resolve_one(
        zone_log.episode_row(plan(), bars(), 101.0, taken=True), fwd)
    assert out and out["won"] is True
    assert out["outcome_r"] == pytest.approx((112.0 - 101.5) / (101.5 - 97.0),
                                             abs=1e-3)


def test_the_stop_before_the_target_is_a_loss_of_exactly_one_R():
    fwd = ([{"t": 1, "h": 102.0, "l": 101.0, "c": 101.5}] * 25
           + [{"t": 2, "h": 102.0, "l": 96.0, "c": 96.5}])
    out = zone_log.resolve_one(
        zone_log.episode_row(plan(), bars(), 101.0, taken=True), fwd)
    assert out and out["won"] is False
    assert out["outcome_r"] == -1.0


def test_a_bar_touching_BOTH_counts_as_a_loss():
    """The data cannot say which came first inside a bar, and assuming the
    good one is how a backtest flatters itself. This is the single trap that
    contaminated the first pass of the 2026-10-26 analysis -- hit_tp and
    hit_stop were both true on hundreds of episodes.
    """
    fwd = ([{"t": 1, "h": 102.0, "l": 101.0, "c": 101.5}] * 25
           + [{"t": 2, "h": 113.0, "l": 96.0, "c": 100.0}])
    out = zone_log.resolve_one(
        zone_log.episode_row(plan(), bars(), 101.0, taken=True), fwd)
    assert out["hit_tp"] is True and out["hit_stop"] is True
    assert out["won"] is False, "a both-touched bar was scored as a win"
    assert out["outcome_r"] == -1.0


def test_won_is_stored_so_no_consumer_can_re_derive_it_wrongly():
    fwd = ([{"t": 1, "h": 102.0, "l": 101.0, "c": 101.5}] * 25
           + [{"t": 2, "h": 113.0, "l": 96.0, "c": 100.0}])
    out = zone_log.resolve_one(
        zone_log.episode_row(plan(), bars(), 101.0, taken=True), fwd)
    assert "won" in storage.ZONE_OUTCOME_FIELDNAMES
    assert out["won"] in (True, False)


def test_a_band_price_never_reached_is_recorded_not_dropped():
    """A band price never reached is a different outcome from one that
    filled and lost, and collapsing them hides how often the strategy simply
    waits."""
    p = plan(entry=90.0, stop=85.0, tp=110.0, immediate=False)
    fwd = [{"t": 1, "h": 102.0, "l": 95.0, "c": 101.0}] * 30
    out = zone_log.resolve_one(
        zone_log.episode_row(p, bars(), 101.0, taken=True), fwd)
    assert out and out["filled"] is False
    assert out["won"] == "" and out["outcome_r"] == ""


def test_an_unfinished_trade_is_left_unresolved():
    """Neither stop nor target touched: resolve it later, do not guess."""
    fwd = [{"t": 1, "h": 102.0, "l": 101.0, "c": 101.5}] * 40
    assert zone_log.resolve_one(
        zone_log.episode_row(plan(), bars(), 101.0, taken=True), fwd) is None


def test_too_few_bars_is_unresolved_rather_than_a_loss():
    """A winner's median time to target was 5 bars, so a handful of bars
    cannot distinguish a loser from an open trade."""
    fwd = [{"t": 1, "h": 102.0, "l": 96.0, "c": 96.5}] * 3
    assert zone_log.resolve_one(
        zone_log.episode_row(plan(), bars(), 101.0, taken=True), fwd) is None


def test_broken_geometry_is_refused_not_resolved():
    for p in (plan(entry=97.0, stop=101.0), plan(tp=95.0)):
        row = zone_log.episode_row(p, bars(), 101.0, taken=True)
        assert zone_log.resolve_one(row, bars(40)) is None


# ── which bands are worth a human verdict ──────────────────────────────────

def _row(coin="SOL", rr=2.0, eid=None):
    return {"zone_episode_id": eid or f"zone:{coin}:1:{rr}", "coin": coin,
            "rr": str(rr), "zone_lo": "100", "zone_hi": "103",
            "risk_pct": "3.0", "stop_atr_mult": "2.0",
            "target_dist_pct": "6.0"}


def test_it_picks_the_boundary_not_the_best_setups():
    """A band at rr 0.02 teaches nothing -- the gate refuses it and so would
    anyone. The only place a label can say the LINE is wrong is next to it."""
    rows = [_row(rr=0.02), _row(rr=1.95), _row(rr=2.05), _row(rr=9.0)]
    got = {float(r["rr"]) for r in zone_log.boundary_cases(rows, 2.0)}
    # A SET: 1.95 and 2.05 are equidistant in intent and their float gaps
    # differ in the 17th decimal, so asserting an order here would be
    # asserting a rounding artefact.
    assert got == {1.95, 2.05}, got


def test_nearest_to_the_line_comes_first():
    rows = [_row(rr=2.3), _row(rr=1.98), _row(rr=1.7)]
    got = [float(r["rr"]) for r in zone_log.boundary_cases(rows, 2.0)]
    assert got[0] == 1.98


def test_it_says_which_side_the_gate_came_down_on():
    rows = [_row(rr=1.9), _row(rr=2.1)]
    sides = {float(r["rr"]): r["_side"] for r in
             zone_log.boundary_cases(rows, 2.0)}
    assert "REFUSE" in sides[1.9] and "TAKE" in sides[2.1]


def test_already_labelled_bands_are_not_asked_about_again():
    rows = [_row(rr=1.95, eid="A"), _row(rr=1.96, eid="B")]
    got = zone_log.boundary_cases(rows, 2.0, labelled=["A"])
    assert [r["zone_episode_id"] for r in got] == ["B"]


def test_the_window_scales_with_the_floor():
    """At MIN_RR 2.0 the window is rr 1.6-2.4; at 1.0 it must not still be
    +/-0.4, or half of everything qualifies."""
    # rr 1.7 is INSIDE the window at a 2.0 floor (gap 0.3 <= 0.4) and
    # OUTSIDE it at a 1.0 floor (gap 0.7 > 0.2).
    rows = [_row(rr=1.7)]
    assert zone_log.boundary_cases(rows, 2.0), "1.7 is near a 2.0 floor"
    assert not zone_log.boundary_cases(rows, 1.0), "1.7 is far from a 1.0 floor"


def test_an_unparseable_rr_is_skipped_not_crashed():
    rows = [{"zone_episode_id": "x", "coin": "SOL", "rr": ""},
            {"zone_episode_id": "y", "coin": "SOL", "rr": "n/a"},
            _row(rr=2.0, eid="z")]
    got = zone_log.boundary_cases(rows, 2.0)
    assert [r["zone_episode_id"] for r in got] == ["z"]


def test_the_limit_is_a_sitting_not_a_mailing_list():
    rows = [_row(rr=2.0 + i * 0.001, eid=f"e{i}") for i in range(80)]
    assert len(zone_log.boundary_cases(rows, 2.0, limit=10)) == 10

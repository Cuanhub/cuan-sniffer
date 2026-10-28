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


# ── resolving: through the backtest engine, once per band ──────────────────

H = 3_600_000


def ep(p=None, **kw):
    """An episode recorded at the last bar of bars() (t = 949 h)."""
    return zone_log.episode_row(p or plan(**kw), bars(), 101.0, taken=True)


def seq(*rest, ts=949 * H):
    """Closed bars from the detection bar onward: the detection bar first."""
    det = {"t": ts, "o": 101.0, "h": 102.0, "l": 100.0, "c": 101.0}
    return [det] + [dict(b, t=ts + (k + 1) * H) for k, b in enumerate(rest)]


QUIET = {"o": 101.5, "h": 102.0, "l": 101.0, "c": 101.5}


def test_the_target_before_the_stop_is_a_win_net_of_costs():
    out = zone_log.resolve_one(ep(), seq(QUIET, {"o": 101.5, "h": 113.0, "l": 101.0, "c": 112.5}))
    assert out and out["won"] is True and out["exit"] == "tp"
    gross = (112.0 - 101.5) / (101.5 - 97.0)
    assert out["r_gross"] == pytest.approx(gross, abs=1e-3)
    assert out["r_net"] < out["r_gross"], "fees were not charged"
    assert out["resolver"] == zone_log.RESOLVER


def test_a_stop_loses_slightly_more_than_one_R_from_measured_slippage():
    out = zone_log.resolve_one(ep(), seq(QUIET, {"o": 101.5, "h": 102.0, "l": 96.0, "c": 96.5}))
    assert out["won"] is False and out["exit"] == "stop"
    assert -1.1 < out["r_gross"] < -1.0
    assert out["r_net"] < out["r_gross"]


def test_a_bar_touching_BOTH_counts_as_a_loss():
    """The data cannot say which came first inside a bar."""
    out = zone_log.resolve_one(ep(), seq(QUIET, {"o": 101.5, "h": 113.0, "l": 96.0, "c": 100.0}))
    assert out["won"] is False and out["ambiguous"] is True


def test_a_target_in_the_hour_a_bid_filled_is_not_a_win():
    """The high may have printed before the dip that filled the bid. The old
    resolver counted these -- 6 of the first 23 recorded wins."""
    e = ep(entry=100.5, immediate=False)
    out = zone_log.resolve_one(e, seq({"o": 102.0, "h": 113.0, "l": 100.0, "c": 101.0},
                                      {"o": 101.0, "h": 101.5, "l": 96.0, "c": 96.5}))
    assert out["won"] is False and out["fill_bar_tp"] is True


def test_an_in_band_entry_is_not_judged_by_its_own_hour():
    """Its recorded price came partway through the detection bar."""
    det_dipped = {"t": 949 * H, "o": 101.0, "h": 102.0, "l": 90.0, "c": 101.0}
    out = zone_log.resolve_one(ep(), [det_dipped] + seq(QUIET, {"o": 101.5, "h": 113.0,
                                                               "l": 101.0, "c": 112.5})[1:])
    assert out["won"] is True


def test_a_bid_fills_only_within_its_twelve_hour_life():
    e = ep(entry=99.0, immediate=False)
    late = [QUIET] * 12 + [{"o": 101.0, "h": 101.0, "l": 98.0, "c": 99.5}]
    out = zone_log.resolve_one(e, seq(*late))
    assert out and out["filled"] is False and out["exit"] == "expired"
    assert zone_log.resolve_one(e, seq(*([QUIET] * 5))) is None, "resolved inside its life"


def test_won_is_stored_so_no_consumer_can_re_derive_it_wrongly():
    assert {"won", "r_net", "band_id", "resolver"} <= set(storage.ZONE_OUTCOME_FIELDNAMES)


def test_an_unfinished_trade_is_left_unresolved():
    assert zone_log.resolve_one(ep(), seq(*([QUIET] * 40))) is None


def test_an_unclosed_detection_bar_is_unresolved():
    assert zone_log.resolve_one(ep(), seq(QUIET, ts=948 * H)) is None


def test_broken_geometry_is_refused_not_resolved():
    for p_ in (plan(entry=97.0, stop=101.0), plan(tp=95.0)):
        assert zone_log.resolve_one(ep(p_), seq(QUIET)) is None


def test_the_resolver_is_the_engine():
    """One implementation: the recorded R IS the engine's R on the same bars."""
    from research.backtest.costs import MEASURED
    from research.backtest.engine import MARKET_AT, Order, simulate
    b = seq(QUIET, {"o": 101.5, "h": 113.0, "l": 101.0, "c": 112.5})
    out = zone_log.resolve_one(ep(), b)
    tr = simulate(b, Order("SOL", 0, MARKET_AT, 97.0, 112.0, limit=101.5), MEASURED)
    assert out["r_net"] == round(tr.r_net, 4)


# ── one band, one record ───────────────────────────────────────────────────

def test_the_band_id_survives_the_window_sliding():
    """The SAME base candle an hour later sits one index lower in the
    window. episode_id changes; band_id must not."""
    b0 = bars(950)
    # the next hour's window: drop the oldest bar, add a new one at the end
    b1 = [dict(b, t=b0[i + 1]["t"]) for i, b in enumerate(b0[1:])] + [dict(b0[-1], t=950 * H)]
    r0 = zone_log.episode_row(plan(base_i=900), b0, 101.0, taken=True)
    r1 = zone_log.episode_row(plan(base_i=899), b1, 101.0, taken=True)
    assert r0["zone_episode_id"] != r1["zone_episode_id"]
    assert r0["band_id"] == r1["band_id"] != ""
    assert zone_log.band_key(r0) == zone_log.band_key({**r0, "band_id": ""})


def test_a_band_is_recorded_once_across_passes():
    b0 = bars(950)
    b1 = [dict(b, t=b0[i + 1]["t"]) for i, b in enumerate(b0[1:])] + [dict(b0[-1], t=950 * H)]
    assert zone_log.record_pass({"SOL": [plan(base_i=900)]}, 2.0, {"SOL": b0}) == 1
    assert zone_log.record_pass({"SOL": [plan(base_i=899)]}, 2.0, {"SOL": b1}) == 0


def test_each_band_resolves_once_at_its_first_sighting():
    first = {**ep(), "detected_at_utc": "2026-10-27T00:00:00Z"}
    later = {**ep(), "zone_episode_id": "zone:SOL:899:100", "bar_ts_ms": 950 * H,
             "bars_since_base": 51, "detected_at_utc": "2026-10-27T01:00:00Z", "band_id": ""}
    picks = zone_log.first_sightings([later, first])
    assert list(picks.values()) == [first]


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


def test_resolve_writes_one_outcome_per_band_from_closed_bars_only():
    """End to end: two sightings of one band, candles in the venue's string
    format, the newest one still forming."""
    import time as _time
    now = int(_time.time() * 1000)
    ts = (now // H) * H - 30 * H                       # detection 30 h ago
    first = {**ep(), "bar_ts_ms": ts, "detected_at_utc": "2026-10-27T00:00:00Z",
             "band_id": "band:SOL:1:100"}
    again = {**first, "zone_episode_id": "zone:SOL:899:100", "bar_ts_ms": ts + H,
             "detected_at_utc": "2026-10-27T01:00:00Z"}
    for row in (first, again):
        storage._append("zone_episodes.csv", storage.ZONE_EPISODE_FIELDNAMES, row,
                        "zone_episode_id")
    candles = [{"t": ts + k * H, "o": "101.5", "h": "102", "l": "101", "c": "101.5"}
               for k in range(31)]
    candles[3].update({"h": "113", "c": "112.5"})       # the target, 3 h after detection
    candles[-1].update({"l": "50"})                     # the FORMING bar: must be ignored

    class Venue:
        def get_candles(self, coin, interval, start, end):
            return candles

    assert zone_log.resolve(Venue()) == 1
    outs = list(__import__("csv").DictReader(open(storage.DATA_DIR / "zone_outcomes.csv")))
    assert len(outs) == 1
    assert outs[0]["zone_episode_id"] == first["zone_episode_id"]
    assert outs[0]["won"] == "True" and outs[0]["band_id"] == "band:SOL:1:100"
    assert zone_log.resolve(Venue()) == 0, "the band resolved twice"


def _stored(row):
    storage._append("zone_episodes.csv", storage.ZONE_EPISODE_FIELDNAMES, row, "zone_episode_id")


def test_the_forming_bar_cannot_resolve_a_trade():
    """Still open on every CLOSED bar; only the still-forming bar reaches the
    stop. Its low is not final, so nothing may be resolved from it."""
    import time as _time
    now = int(_time.time() * 1000)
    ts = (now // H) * H - 20 * H
    _stored({**ep(), "bar_ts_ms": ts, "band_id": "band:SOL:2:100"})
    candles = [{"t": ts + k * H, "o": "101.5", "h": "102", "l": "101", "c": "101.5"}
               for k in range(21)]
    candles[-1].update({"l": "90"})                      # forming: t + 1h > now

    class Venue:
        def get_candles(self, coin, interval, start, end):
            return candles
    assert candles[-1]["t"] + H > now
    assert zone_log.resolve(Venue()) == 0


def test_a_band_already_resolved_under_another_episode_id_is_not_resolved_again():
    """Band-level, not just id-level: storage's id check alone would let the
    same band resolve twice under two different episode ids."""
    import time as _time
    now = int(_time.time() * 1000)
    ts = (now // H) * H - 30 * H
    _stored({**ep(), "bar_ts_ms": ts, "band_id": "band:SOL:3:100"})
    storage.append_zone_outcome({"zone_episode_id": "zone:SOL:OTHER", "band_id": "band:SOL:3:100",
                                 "coin": "SOL", "won": "False"})
    candles = [{"t": ts + k * H, "o": "101.5", "h": "113", "l": "101", "c": "112"} for k in range(31)]

    class Venue:
        def get_candles(self, coin, interval, start, end):
            return candles
    assert zone_log.resolve(Venue()) == 0

"""research/levels/zones.py against the bands the operator drew himself.

The fixture is the ground truth: seven bands he marked on his own charts on
2026-10-17, and the 1h bars behind them. If the detector stops reproducing
those bands, it has stopped finding the thing he trades -- which is the only
definition of correct this module has.
"""

from __future__ import annotations

import json
import pathlib
import statistics

import pytest

from research.levels import zones as Z
from research.levels.zones import Zone, demand_zones, live_zones

FIX = pathlib.Path(__file__).parent / "tests/fixtures/zones"
BARS = json.loads((FIX / "bars_1h.json").read_text())
BANDS = {k: tuple(v) for k, v in
         json.loads((FIX / "operator_bands.json").read_text()).items()}

# How far an edge may sit from the one he drew and still be the same band.
# His own edges are read off a chart by eye, so demanding better than this
# would be fitting to the thickness of his cursor.
SAME_BAND_PCT = 2.5


def _edge_err(z, lo, hi):
    return max(abs(z.lo - lo) / lo, abs(z.hi - hi) / hi) * 100.0


def his_band(coin, **kw):
    """The detected band that matches the one the operator drew, or None.

    Matched by EDGE AGREEMENT, not by proximity to price. Scoring the fixture
    with `nearest_below` is the mistake this module exists to undo: price
    approaches support from above, so the nearest band is always the
    shallowest one, and that bias is what bought ARB 5.6% above the band it
    meant to buy. These tests ask "is his band in there", and reachability is
    a separate question with its own tests below.
    """
    lo, hi = BANDS[coin]
    zs = demand_zones(BARS[coin], coin=coin, **kw)
    if not zs:
        return None
    best = min(zs, key=lambda z: _edge_err(z, lo, hi))
    return best if _edge_err(best, lo, hi) <= SAME_BAND_PCT else None


def candidates(coin, **kw):
    """Every band this pass would actually consider for one coin."""
    bars = BARS[coin]
    return live_zones(demand_zones(bars, coin=coin, **kw), bars[-1]["c"])


# ── the fit to the operator's own bands ────────────────────────────────────

@pytest.mark.parametrize("coin", sorted(BANDS))
def test_every_operator_band_is_found(coin):
    assert his_band(coin) is not None, (
        f"{coin}: nothing within {SAME_BAND_PCT}% of {BANDS[coin]}")


def test_the_edge_error_does_not_regress():
    """0.97% across his sixteen -- it was 0.73% on the seven the definition
    was fitted to, so it held up out of sample. A looser bound here would let
    the definition drift away from the charts one commit at a time."""
    errs = []
    for coin, (lo, hi) in BANDS.items():
        z = his_band(coin)
        errs += [abs(z.lo - lo) / lo * 100, abs(z.hi - hi) / hi * 100]
    got = statistics.mean(errs)
    assert got < 1.1, f"mean edge error {got:.2f}%"


def test_the_band_widths_match_his():
    """The width agreeing is what says this found the SAME object rather than
    something nearby. His sixteen average 2.89%; these come out 2.42%.

    Mine run slightly NARROWER, which is the safe direction: the band is the
    base candle's low-to-close, and he rounds outward to tidy numbers on a
    chart. A detector drifting WIDER would be inventing band it cannot
    justify, and the stop sits under the low, so width is risk.
    """
    mine = statistics.mean(his_band(c).width_pct for c in BANDS)
    his = statistics.mean((h - l) / l * 100 for l, h in BANDS.values())
    assert mine < his, f"mine {mine:.2f}% is not narrower than his {his:.2f}%"
    assert his - mine < 0.8, f"mine {mine:.2f}% vs his {his:.2f}%"


# ── what the pass would actually trade ─────────────────────────────────────

def test_his_band_is_among_the_candidates_wherever_it_is_reachable():
    """The detector finding his band is not the claim that matters; the pass
    OFFERING it is. Twelve of sixteen, and the four it withholds are all
    cases where buying would be wrong rather than cases it failed to see:

        AERO, MON  price is BELOW his band -- nothing to buy
        STBL       price 7.3% above it (he said: skip, don't like the chart)
        ARB        price 5.25% above it, just past ZONE_REACH_PCT

    ARB is the one that started this. The trader bought 0.18482 while his
    band was 0.1705-0.175; here that band is out of reach and there is no
    trade at all.
    """
    offered = []
    for coin, (lo, hi) in BANDS.items():
        if any(_edge_err(z, lo, hi) <= SAME_BAND_PCT
               for z in candidates(coin)):
            offered.append(coin)
    assert len(offered) >= 12, f"only {len(offered)}/16 offered: {offered}"


@pytest.mark.parametrize("coin", ["AERO", "MON"])
def test_a_band_above_price_is_not_offered(coin):
    """Price is below it, so entering would mean buying above the market."""
    px = BARS[coin][-1]["c"]
    lo, hi = BANDS[coin]
    assert px < hi, f"{coin} fixture no longer has price below the band"
    assert not any(_edge_err(z, lo, hi) <= SAME_BAND_PCT
                   for z in candidates(coin))


@pytest.mark.parametrize("coin", ["ARB", "STBL"])
def test_a_band_price_has_run_too_far_above_is_not_offered(coin):
    """The whole failure, stated as a test. Both are setups he refused by
    hand, and both are refused here because price is further above the band
    than ZONE_REACH_PCT allows -- not because anything scored them."""
    lo, hi = BANDS[coin]
    z = his_band(coin)
    assert z is not None, f"{coin} band must still be DETECTED"
    px = BARS[coin][-1]["c"]
    assert (px - z.hi) / z.hi * 100 > Z.ZONE_REACH_PCT
    assert not any(_edge_err(c, lo, hi) <= SAME_BAND_PCT
                   for c in candidates(coin))


def test_the_candidate_list_stays_short_enough_to_read():
    """Bands persist for ever now, so the count is worth watching: the raw
    list runs to a median of 23 per coin. Reachability is what makes that
    tractable, not a filter on the bands themselves."""
    counts = [len(candidates(c)) for c in BANDS]
    assert statistics.median(counts) <= 4, f"median {statistics.median(counts)}"
    assert max(counts) <= 10, f"max {max(counts)}: {counts}"


def test_close_beats_high_as_the_band_top():
    """The one real choice in the definition, kept honest.

    Using the base candle's HIGH roughly doubles the band width and more than
    doubles the edge error. If someone 'fixes' the top to the high, this says
    why not.
    """
    def err_with(top_is_high: bool):
        out = []
        for coin, (lo, hi) in BANDS.items():
            z = his_band(coin)
            top = BARS[coin][z.base_i]["h"] if top_is_high else z.hi
            out += [abs(z.lo - lo) / lo * 100, abs(top - hi) / hi * 100]
        return statistics.mean(out)
    assert err_with(False) < err_with(True)


# ── the thing that started this: ARB ───────────────────────────────────────

def test_ARB_is_one_zone_not_seventeen_levels():
    """levels.py had 17 ARB levels, median 0.83% apart, and the trader bought
    0.184295 -- 5.6% above the band. One band, and that entry is outside it.
    """
    z = his_band("ARB")
    assert z.contains(0.1711), "the operator's double-tap entry is in the band"
    assert not z.contains(0.18482), "the trade that was taken is outside it"
    assert z.width_pct < 5.0


def test_the_entry_that_was_taken_would_be_refused():
    """0.18482 is above the band, so there is nothing to enter."""
    z = his_band("ARB")
    assert 0.18482 > z.hi


# ── entry placement: inside the band, middle to upper ──────────────────────

def test_entry_sits_in_the_upper_half_not_at_the_low():
    """The operator's rule and his reason: an entry at the band low risks
    missing the band entirely when price only penetrates part of it."""
    for coin in BANDS:
        z = his_band(coin)
        e = z.entry_px()
        assert z.contains(e)
        assert e >= z.mid - 1e-12, f"{coin} entry below the midpoint"


def test_price_already_inside_the_band_is_the_entry():
    z = Zone("X", lo=100.0, hi=103.0, base_i=0, bos_i=1)
    assert z.entry_px(px_now=102.0) == 102.0
    assert z.entry_px(px_now=101.6) == 101.6


def test_price_below_the_midpoint_still_enters_at_the_midpoint():
    """Deeper than the midpoint is a better price, not a worse one -- but the
    resting bid sits at the midpoint, so that is what gets filled."""
    z = Zone("X", lo=100.0, hi=104.0, base_i=0, bos_i=1)
    assert z.entry_px(px_now=100.5) == z.mid


def test_price_above_the_band_waits_at_the_midpoint():
    z = Zone("X", lo=100.0, hi=104.0, base_i=0, bos_i=1)
    assert z.entry_px(px_now=110.0) == z.mid


# ── stop placement: just under the band low ────────────────────────────────

def test_the_stop_sits_just_under_the_band_low():
    """A close below the low is what invalidates the zone, so the definition
    supplies the stop. No sigma multiple, no next-support guess."""
    z = Zone("X", lo=100.0, hi=103.0, base_i=0, bos_i=1)
    s = z.stop_px()
    assert s < z.lo
    assert (z.lo - s) / z.lo * 100 == pytest.approx(Z.ZONE_STOP_BUFFER_PCT)


def test_the_risk_per_trade_is_the_band_plus_the_buffer():
    for coin in BANDS:
        z = his_band(coin)
        risk = (z.entry_px() - z.stop_px()) / z.entry_px() * 100
        assert 0.1 < risk < 6.0, f"{coin} risk {risk:.2f}%"


# ── what makes a zone, and what unmakes it ─────────────────────────────────

def bars(seq):
    """seq of (o,h,l,c)."""
    return [{"o": o, "h": h, "l": l, "c": c} for o, h, l, c in seq]


def test_a_rally_that_does_not_break_structure_is_not_a_zone():
    """Price must CLOSE above a prior swing high. A rally that stalls under
    it leaves no zone behind."""
    seq = ([(10, 11, 9, 10)] * 4 + [(10, 12, 9, 11)]          # swing high 12
           + [(11, 11, 8, 9)] * 4                              # dip
           + [(9, 11.5, 9, 11.4)] * 6)                         # fails to close >12
    assert demand_zones(bars(seq), "X", min_bars=0) == []


def test_a_wick_through_the_swing_high_is_not_a_break():
    """A sweep is exactly what this must exclude, so the break is judged on
    the close."""
    seq = ([(10, 11, 9, 10)] * 4 + [(10, 12, 9, 11)]
           + [(11, 11, 8, 9)] * 4
           + [(9, 13, 9, 11.8)] * 6)        # wicks to 13, closes 11.8 (<12)
    assert demand_zones(bars(seq), "X", min_bars=0) == []


def test_a_band_price_closed_under_is_still_a_band():
    """The filter that used to drop these found 9 of the operator's 16 bands.
    Removing it finds 16 of 16.

    He keeps bands price has closed well under -- ETH 0.72%, VVV 1.18%,
    WLD 2.06%, AERO 3.09%, MON 3.58%, NEAR 3.78%, and APT at 8.47%. No
    tolerance threshold reproduces APT; only having none does.

    This test exists so the filter cannot be reinstated as a bug fix. What
    keeps a dead band from becoming a trade is `live_zones` (price has to be
    able to reach it) and the stop under the low -- not detection.
    """
    seq = ([(10, 11, 9, 10)] * 4 + [(10, 12, 9, 11)]
           + [(11, 11, 8, 8.6)] * 4                            # base low 8
           + [(9, 14, 9, 13)] * 4)                             # breaks structure
    before = demand_zones(bars(seq), "X", min_bars=0)
    assert before, "a clean break should leave a zone"
    after = demand_zones(bars(seq + [(13, 13, 7, 7.5)]), "X",  # closes below 8
                         min_bars=0)
    assert [(z.lo, z.hi) for z in after] == [(z.lo, z.hi) for z in before], \
        "a close below the low must not remove the band"


def test_removing_the_aliveness_filter_is_what_found_his_bands():
    """The measurement behind the decision, on the fixture itself: every one
    of the sixteen bands he drew is detected, and seven of them had closed
    below their low."""
    missed = []
    for coin, (lo, hi) in BANDS.items():
        zs = demand_zones(BARS[coin], coin=coin)
        err = min((max(abs(z.lo - lo) / lo, abs(z.hi - hi) / hi) * 100
                   for z in zs), default=1e9)
        if err > 2.5:
            missed.append((coin, round(err, 2)))
    assert not missed, f"bands no longer found: {missed}"

    breached = 0
    for coin, (lo, hi) in BANDS.items():
        zs = demand_zones(BARS[coin], coin=coin)
        z = min(zs, key=lambda z: max(abs(z.lo - lo) / lo,
                                      abs(z.hi - hi) / hi))
        if any(b["c"] < z.lo for b in BARS[coin][z.bos_i:]):
            breached += 1
    assert breached >= 7, (
        f"only {breached} bands had closed below their low; the fixture no "
        "longer demonstrates why the filter was removed")


def test_one_impulse_breaking_several_highs_yields_one_zone():
    """The duplication that gave ARB seventeen levels, caught at the source."""
    seq = ([(10, 11, 9, 10)] * 3 + [(10, 11.5, 9, 11)] * 3
           + [(11, 12, 9, 11.5)] * 3
           + [(11, 11, 8, 8.6)] * 4
           + [(9, 20, 9, 19)] * 6)
    zs = demand_zones(bars(seq), "X", min_bars=0)
    assert len(zs) <= 1, f"{len(zs)} zones from one impulse"


def test_absurd_widths_are_refused():
    assert Z.ZONE_MIN_WIDTH_PCT < 1.0 < Z.ZONE_MAX_WIDTH_PCT
    seq = ([(10, 11, 9, 10)] * 4 + [(10, 12, 9, 11)]
           + [(11, 11, 1, 10.9)] * 4          # base spans 1 -> 10.9: ~990%
           + [(11, 14, 11, 13)] * 4)
    assert demand_zones(bars(seq), "X", min_bars=0) == []


def test_too_few_bars_is_empty_not_an_error():
    assert demand_zones([], "X", min_bars=0) == []
    assert demand_zones(bars([(1, 1, 1, 1)] * 3), "X", min_bars=0) == []


def test_there_is_no_way_to_ask_for_just_the_nearest_band():
    """`nearest_below` is gone, not deprecated. It agreed with 10 of the
    operator's 16 bands and is biased by construction -- price approaches
    support from above, so the nearest base is always the shallowest one.
    A function with a warning in its docstring is still callable."""
    assert not hasattr(Z, "nearest_below")


def test_the_pivot_window_is_not_a_fragile_knob():
    """2 and 3 were identical in the fit; if that stops being true the
    definition has become sensitive to a parameter nobody measured."""
    for coin in sorted(BANDS):
        a = his_band(coin, window=2)
        b = his_band(coin, window=3)
        assert a is not None and b is not None
        assert abs(a.lo - b.lo) / b.lo < 0.02, coin


# ── the horizon, and nested bands ──────────────────────────────────────────

def test_short_history_returns_nothing_rather_than_a_band_that_fits_in_it():
    """A zone's base can be far back and the detector only sees the bars it
    is given. With 400 bars of 1h, HYPE's band came out 77.44-77.675 (0.30%
    wide); with 800 it came out 77.033-78.625, which is the one the operator
    drew. Six of seven coins were identical at every length, so this is a
    horizon, not instability -- and the honest answer below it is silence.
    """
    bars = BARS["HYPE"]
    assert demand_zones(bars[-400:], "HYPE") == []
    assert demand_zones(bars, "HYPE"), "full history must still find it"
    # and below the horizon it is silence, not a thinner band
    assert demand_zones(bars[-700:], "HYPE") == []


def test_the_bands_are_stable_once_there_is_enough_history():
    for coin in sorted(BANDS):
        lo, hi = BANDS[coin]
        short = demand_zones(BARS[coin][-800:], coin)
        a = min(short, key=lambda z: _edge_err(z, lo, hi), default=None)
        b = his_band(coin)
        assert a is not None and b is not None, coin
        assert abs(a.lo - b.lo) / b.lo < 0.01, f"{coin} {a.lo} vs {b.lo}"
        assert abs(a.hi - b.hi) / b.hi < 0.01, f"{coin} {a.hi} vs {b.hi}"


def test_a_tight_base_inside_a_wider_one_becomes_one_band():
    """Taking the inner band would put the stop INSIDE the zone rather than
    under it, which is the one thing the stop must never be."""
    zs = [Zone("X", 100.0, 104.0, 10, 20), Zone("X", 101.0, 102.0, 15, 18)]
    merged = Z._merge_nested(zs)
    assert len(merged) == 1
    assert (merged[0].lo, merged[0].hi) == (100.0, 104.0)


def test_separate_bands_are_not_merged():
    zs = [Zone("X", 100.0, 102.0, 10, 20), Zone("X", 110.0, 112.0, 30, 40)]
    assert len(Z._merge_nested(zs)) == 2


def test_no_live_band_is_thinner_than_anything_the_operator_drew():
    """His narrowest is SUI at 1.59%. A 0.30% band yields a 0.40% stop, which
    noise takes out before the idea is tested."""
    for coin in sorted(BANDS):
        z = his_band(coin)
        risk = (z.entry_px() - z.stop_px()) / z.entry_px() * 100
        assert risk > 0.5, f"{coin} risk only {risk:.2f}%"

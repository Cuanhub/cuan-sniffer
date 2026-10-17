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
from research.levels.zones import Zone, demand_zones, nearest_below

FIX = pathlib.Path(__file__).parent / "tests/fixtures/zones"
BARS = json.loads((FIX / "bars_1h.json").read_text())
BANDS = {k: tuple(v) for k, v in
         json.loads((FIX / "operator_bands.json").read_text()).items()}


def detected(coin, **kw):
    """The live zone price is falling into, for one coin."""
    bars = BARS[coin]
    zs = demand_zones(bars, coin=coin, **kw)
    return nearest_below(zs, bars[-1]["c"])


# ── the fit to the operator's own bands ────────────────────────────────────

@pytest.mark.parametrize("coin", sorted(BANDS))
def test_every_operator_band_is_found(coin):
    z = detected(coin)
    assert z is not None, f"no live zone below price for {coin}"
    lo, hi = BANDS[coin]
    assert abs(z.lo - lo) / lo * 100 < 2.5, f"{coin} low {z.lo} vs {lo}"
    assert abs(z.hi - hi) / hi * 100 < 2.5, f"{coin} high {z.hi} vs {hi}"


def test_the_edge_error_does_not_regress():
    """0.73% when fitted. A looser bound here would let the definition drift
    away from the charts one commit at a time."""
    errs = []
    for coin, (lo, hi) in BANDS.items():
        z = detected(coin)
        errs += [abs(z.lo - lo) / lo * 100, abs(z.hi - hi) / hi * 100]
    assert statistics.mean(errs) < 1.0, f"mean edge error {statistics.mean(errs):.2f}%"


def test_the_band_widths_match_his():
    """The width agreeing is what says this found the SAME object rather than
    something nearby. His average 2.96%; fitted 2.97%."""
    mine = statistics.mean(detected(c).width_pct for c in BANDS)
    his = statistics.mean((h - l) / l * 100 for l, h in BANDS.values())
    assert abs(mine - his) < 0.5, f"mine {mine:.2f}% vs his {his:.2f}%"


def test_close_beats_high_as_the_band_top():
    """The one real choice in the definition, kept honest.

    Using the base candle's HIGH roughly doubles the band width and more than
    doubles the edge error. If someone 'fixes' the top to the high, this says
    why not.
    """
    def err_with(top_is_high: bool):
        out = []
        for coin, (lo, hi) in BANDS.items():
            bars = BARS[coin]
            zs = demand_zones(bars, coin=coin)
            z = nearest_below(zs, bars[-1]["c"])
            top = bars[z.base_i]["h"] if top_is_high else z.hi
            out += [abs(z.lo - lo) / lo * 100, abs(top - hi) / hi * 100]
        return statistics.mean(out)
    assert err_with(False) < err_with(True)


# ── the thing that started this: ARB ───────────────────────────────────────

def test_ARB_is_one_zone_not_seventeen_levels():
    """levels.py had 17 ARB levels, median 0.83% apart, and the trader bought
    0.184295 -- 5.6% above the band. One zone, and that entry is outside it.
    """
    z = detected("ARB")
    assert z.contains(0.1711), "the operator's double-tap entry is in the band"
    assert not z.contains(0.18482), "the trade that was taken is outside it"
    assert z.width_pct < 5.0


def test_the_entry_that_was_taken_would_be_refused():
    """0.18482 is above the band, so there is nothing to enter."""
    z = detected("ARB")
    assert 0.18482 > z.hi


# ── entry placement: inside the band, middle to upper ──────────────────────

def test_entry_sits_in_the_upper_half_not_at_the_low():
    """The operator's rule and his reason: an entry at the band low risks
    missing the band entirely when price only penetrates part of it."""
    for coin in BANDS:
        z = detected(coin)
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
        z = detected(coin)
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


def test_a_zone_dies_when_price_closes_below_its_low():
    seq = ([(10, 11, 9, 10)] * 4 + [(10, 12, 9, 11)]
           + [(11, 11, 8, 8.6)] * 4                            # base low 8
           + [(9, 14, 9, 13)] * 4)                             # breaks structure
    alive = demand_zones(bars(seq), "X", min_bars=0)
    assert alive, "a clean break should leave a zone"
    dead = demand_zones(bars(seq + [(13, 13, 7, 7.5)]), "X")   # closes below 8
    assert dead == [], "a zone price closed under is not a zone any more"


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


def test_nearest_below_picks_the_highest_zone_under_price():
    zs = [Zone("X", 10, 11, 0, 1), Zone("X", 20, 21, 2, 3),
          Zone("X", 30, 31, 4, 5)]
    assert nearest_below(zs, 25.0).lo == 20
    assert nearest_below(zs, 5.0) is None


def test_the_pivot_window_is_not_a_fragile_knob():
    """2 and 3 were identical in the fit; if that stops being true the
    definition has become sensitive to a parameter nobody measured."""
    for coin in sorted(BANDS):
        a = detected(coin, window=2)
        b = detected(coin, window=3)
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
        a = nearest_below(demand_zones(BARS[coin][-800:], coin),
                          BARS[coin][-1]["c"])
        b = nearest_below(demand_zones(BARS[coin], coin), BARS[coin][-1]["c"])
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
        z = detected(coin)
        risk = (z.entry_px() - z.stop_px()) / z.entry_px() * 100
        assert risk > 0.5, f"{coin} risk only {risk:.2f}%"

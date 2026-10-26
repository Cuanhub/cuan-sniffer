"""The band pass: the gate, the episode contract, and the table.

The fixture is the operator's own sixteen bands and the 1h bars behind them,
the same ground truth test_zones.py uses.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from research.levels import zone_pass as ZP
from research.levels import zones as Z

FIX = pathlib.Path(__file__).parent / "tests/fixtures/zones"
BARS = json.loads((FIX / "bars_1h.json").read_text())
BANDS = {k: tuple(v) for k, v in
         json.loads((FIX / "operator_bands.json").read_text()).items()}

# What he said about each of these, by hand, on 2026-10-17.
VERDICT = {"STBL": "skip", "FARTCOIN": "take", "WLD": "take", "SUI": "take",
           "HYPE": "take", "BCH": "skip", "ENA": "skip", "ARB": "skip"}
FLOOR = 1.5


class FakeClient:
    """Serves the fixture bars, and counts calls."""

    def __init__(self, fail: set = frozenset()):
        self.calls = 0
        self.fail = set(fail)

    def get_candles(self, coin, tf, start, end):
        self.calls += 1
        if coin in self.fail:
            raise RuntimeError("venue said no")
        return [{"t": i * 3600_000, "o": b["o"], "h": b["h"], "l": b["l"],
                 "c": b["c"]} for i, b in enumerate(BARS.get(coin, []))]


def run(min_rr=FLOOR, coins=None, **kw):
    uni = [{"symbol": c} for c in (coins or sorted(BANDS))]
    return ZP.plans_by_coin(FakeClient(**kw), uni, min_rr=min_rr)


# ── the failure that caused the rebuild ────────────────────────────────────

def test_no_intent_enters_above_a_band_the_operator_drew():
    """The whole point. ARB was bought at 0.18482 at market while his band
    was 0.1705-0.175, 5.6% below it."""
    intents, _, _ = run()
    above = []
    for i in intents:
        hi = BANDS[i["coin"]][1]
        if i["suggested_entry"] > hi:
            above.append((i["coin"], i["suggested_entry"], hi))
    assert not above, f"entries above his band: {above}"


def test_ARB_produces_no_plan_at_all():
    """His call was skip, and here there is nothing to skip.

    Not a score: ARB's only two bands are 0.7% and 0.6% wide, under
    ZONE_MIN_WIDTH_PCT, so neither is a band this trades.
    """
    _, _, found = run(coins=["ARB"])
    assert "ARB" not in found


@pytest.mark.parametrize("coin", ["ARB", "STBL"])
def test_the_setups_he_refused_are_not_offered(coin):
    intents, _, _ = run(coins=[coin])
    assert intents == []


@pytest.mark.parametrize("coin", ["SUI"])
def test_the_setups_he_took_are_offered_inside_his_band(coin):
    intents, _, _ = run(coins=[coin])
    assert len(intents) == 1, intents
    lo, hi = BANDS[coin]
    e = intents[0]["suggested_entry"]
    assert e <= hi, f"{coin} entry {e} above his band top {hi}"


def test_FARTCOIN_is_now_refused_on_reward_risk_not_on_the_band():
    """FARTCOIN was in this list until the stop became volatility-scaled.

    Its band is still found and still correct -- what changed is that the
    stop now clears the coin's own noise, which widened the risk and took
    the ratio to about 0.98. At MIN_RR 2.0 that no longer pays for itself.

    This is the trade-off being made deliberately, not a regression. On a
    208-day walk-forward backtest the volatility-scaled stop took the
    strategy from -43.7R over 786 trades to +25.2R over 173, and the price
    of that is exactly this: fewer setups clear the bar. The band is
    offered on the candidate table either way, with the reason.
    """
    intents, rejected, found = run(coins=["FARTCOIN"])
    assert intents == [], "FARTCOIN cleared MIN_RR unexpectedly"
    assert found.get("FARTCOIN"), "the band itself must still be detected"
    lo, hi = BANDS["FARTCOIN"]
    p = found["FARTCOIN"][0]
    err = max(abs(p.zone.lo - lo) / lo, abs(p.zone.hi - hi) / hi) * 100
    assert err <= 2.5, f"it is no longer his band: {err:.2f}% off"
    assert p.rr < 2.0
    assert any("rr" in r[2] for r in rejected), rejected


# ── one coin, one intent ───────────────────────────────────────────────────

def test_a_coin_yields_at_most_one_intent():
    """BCH has five reachable bands. Admitting all of them would open five
    BCH positions for one idea."""
    intents, _, found = run()
    seen = [i["coin"] for i in intents]
    assert len(seen) == len(set(seen)), f"duplicate coins: {seen}"
    assert len(found.get("BCH", [])) > 1, "fixture no longer has the BCH case"


def test_the_intent_is_the_most_recent_qualifying_band():
    intents, _, found = run()
    for i in intents:
        plans = found[i["coin"]]
        ok = [p for p in plans if ZP.rejection(p, FLOOR) is None]
        assert i["zone_base_i"] == ok[0].zone.base_i, i["coin"]


def test_the_table_still_shows_the_bands_that_were_passed_over():
    """The operator asked to see the candidates. No selection rule reproduced
    his own choice better than 9 of 12, so showing only the winner would hide
    exactly the thing he needs to judge."""
    _, _, found = run()
    table = ZP.render_candidates(found, FLOOR)
    assert table.count("BCH") > 1
    assert "refused: rr" in table
    assert "NOW (in band)" in table or "bid, waiting" in table


# ── the gate ───────────────────────────────────────────────────────────────

def test_the_only_gate_is_reward_risk():
    """Recency, reach, side and the existence of a target are structural and
    settled in plans_for. A plan reaching the gate is a real band with a real
    target, so the one open question is whether it pays for its risk."""
    for coin in sorted(BANDS):
        _, _, found = run(coins=[coin])
        for p in found.get(coin, []):
            why = ZP.rejection(p, FLOOR)
            assert why is None or why.startswith("rr "), why


def test_raising_the_floor_only_removes_intents():
    a, _, _ = run(min_rr=1.5)
    b, _, _ = run(min_rr=3.0)
    assert {i["coin"] for i in b} <= {i["coin"] for i in a}


def test_a_refused_plan_is_reported_with_its_reason():
    _, rejected, _ = run()
    assert any("rr " in r[2] for r in rejected), rejected


# ── the episode contract the executor already speaks ───────────────────────

REQUIRED = ("coin", "episode_id", "level_price", "suggested_entry",
            "suggested_stop", "suggested_tp", "suggested_rr")


def test_every_intent_carries_what_the_executor_reads():
    intents, _, _ = run()
    assert intents
    for i in intents:
        for k in REQUIRED:
            assert k in i and i[k] not in (None, ""), (k, i)


def test_the_episode_id_is_stable_for_the_same_band():
    a, _, _ = run(coins=["SUI"])
    b, _, _ = run(coins=["SUI"])
    assert a[0]["episode_id"] == b[0]["episode_id"]
    assert a[0]["episode_id"].startswith("zone:SUI:")


def test_the_level_is_the_band_low_because_the_stop_comes_from_it():
    intents, _, _ = run()
    for i in intents:
        assert i["level_price"] == i["zone_lo"]
        assert i["suggested_stop"] < i["level_price"]


def test_entry_stop_and_target_are_ordered():
    intents, _, _ = run()
    for i in intents:
        assert i["suggested_stop"] < i["suggested_entry"] < i["suggested_tp"]


def test_the_origin_says_this_came_from_a_band():
    """review_flags keys off level_origin, and a band is not a flipped
    resistance or a pivot."""
    intents, _, _ = run()
    assert {i["level_origin"] for i in intents} == {"demand_zone"}


# ── failures are per coin ──────────────────────────────────────────────────

def test_one_coin_failing_does_not_lose_the_others():
    uni = [{"symbol": c} for c in sorted(BANDS)]
    intents, rejected, _ = ZP.plans_by_coin(
        FakeClient(fail={"SUI", "BCH"}), uni, min_rr=FLOOR)
    assert intents, "a venue error on two coins emptied the pass"
    assert {"SUI", "BCH"} & {r[0] for r in rejected}
    assert any("bars unavailable" in r[2] for r in rejected)


def test_an_unknown_coin_is_a_rejection_not_a_crash():
    intents, rejected, _ = ZP.plans_by_coin(
        FakeClient(), [{"symbol": "NOSUCHCOIN"}], min_rr=FLOOR)
    assert intents == []
    assert rejected and "no bars" in rejected[0][2]


def test_one_fetch_per_coin():
    """The scan it replaces made about 93 venue calls a pass."""
    c = FakeClient()
    ZP.plans_by_coin(c, [{"symbol": x} for x in sorted(BANDS)], min_rr=FLOOR)
    assert c.calls == len(BANDS), c.calls


def test_a_live_mid_overrides_the_last_bar_close():
    """The forming bar's close is stale between passes; the mid is not."""
    uni = [{"symbol": "SUI"}]
    far = BARS["SUI"][-1]["c"] * 10
    _, rejected, found = ZP.plans_by_coin(FakeClient(), uni, min_rr=FLOOR,
                                          mids={"SUI": far})
    assert "SUI" not in found, "a mid 10x away should put every band out of reach"


# ── flags a human should see ───────────────────────────────────────────────

def test_a_thin_stop_is_flagged_not_hidden():
    z = Z.Zone("X", lo=100.0, hi=100.9, base_i=1, bos_i=2)
    t = Z.Zone("X", lo=120.0, hi=121.0, base_i=3, bos_i=4, kind=Z.SUPPLY)
    p = Z.Plan(zone=z, target=t, entry=100.45, stop=99.75, tp=120.0,
               immediate=True)
    assert any("stop only" in f for f in ZP.review_flags(p, 100.0, 1000.0))


def test_the_flags_do_not_invent_level_statistics():
    """The level flags read near_sigma and level_hold_rate through a
    defaulting helper, which for a band produced "level held only 0% of its 0
    tests" -- a statistic about nothing."""
    z = Z.Zone("X", lo=100.0, hi=103.0, base_i=1, bos_i=2)
    t = Z.Zone("X", lo=120.0, hi=121.0, base_i=3, bos_i=4, kind=Z.SUPPLY)
    p = Z.Plan(zone=z, target=t, entry=101.5, stop=99.75, tp=120.0,
               immediate=True)
    flags = " ".join(ZP.review_flags(p, 100.0, 1000.0))
    assert "hold" not in flags and "sigma" not in flags


def test_price_in_the_lower_half_of_the_band_is_flagged():
    z = Z.Zone("X", lo=100.0, hi=104.0, base_i=1, bos_i=2)
    t = Z.Zone("X", lo=120.0, hi=121.0, base_i=3, bos_i=4, kind=Z.SUPPLY)
    p = Z.Plan(zone=z, target=t, entry=100.5, stop=99.75, tp=120.0,
               immediate=True)
    assert any("LOWER half" in f for f in ZP.review_flags(p, 100.0, 1000.0))


def test_an_empty_pass_says_so_rather_than_printing_a_header():
    assert "no band in reach" in ZP.render_candidates({}, FLOOR)

"""core/protect.py: every venue position carries one correctly sized stop.

Scenarios from the build brief covered here: stop missing, stop wrong size,
position larger than expected, unknown position at startup.
"""

from __future__ import annotations

import unittest

import pytest

from core import execute, protect, safety
from core.meta import MetaStore, TradeMeta
from tests.fake_venue import FakeHL, build_venue


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    monkeypatch.setattr(execute, "SETTLE_SEC", 0.0)
    monkeypatch.setattr(safety, "FLATTEN_PAUSE_SEC", 0.0)
    safety.reset_errors()
    yield
    safety.reset_errors()


@pytest.fixture
def store(tmp_path):
    return MetaStore(tmp_path / "meta.json")


def planned(stop=90.0, tp=0.0):
    return TradeMeta(reason="level", planned_stop=stop, planned_tp=tp,
                     opened_at="2026-10-17T00:00:00Z")


# ── the stop gets placed, and sized to the whole position ──────────────────

def test_a_missing_stop_is_placed_at_the_planned_price(store):
    fake = FakeHL(positions={"HYPE": 2.0}, mid=100.0)
    store.put("HYPE", planned(stop=90.0))
    res = protect.run_once(build_venue(fake), store)
    assert [r.action for r in res] == ["placed"]
    stops = [o for o in fake.resting if o["orderType"].startswith("Stop")]
    assert len(stops) == 1
    assert float(stops[0]["triggerPx"]) == pytest.approx(90.0)
    assert float(stops[0]["sz"]) == pytest.approx(2.0)


def test_a_correct_stop_is_left_alone(store):
    fake = FakeHL(positions={"HYPE": 2.0}, mid=100.0)
    store.put("HYPE", planned(stop=90.0))
    v = build_venue(fake)
    protect.run_once(v, store)
    sent_after_first = len(fake.sent)
    res = protect.run_once(v, store)
    assert [r.action for r in res] == ["ok"]
    assert len(fake.sent) == sent_after_first, "it replaced a correct stop"


def test_a_wrong_sized_stop_is_replaced_for_the_full_position(store):
    """REVIEW.md finding 6's other half: a stop covering less than the position
    leaves the remainder unstopped when it fires."""
    fake = FakeHL(positions={"HYPE": 2.0}, mid=100.0)
    fake.rest_protection("HYPE", 1.2, stop_px=90.0)      # covers 1.2 of 2.0
    store.put("HYPE", planned(stop=90.0))
    res = protect.run_once(build_venue(fake), store)
    assert [r.action for r in res] == ["replaced"]
    stops = [o for o in fake.resting if o["orderType"].startswith("Stop")]
    assert len(stops) == 1 and float(stops[0]["sz"]) == pytest.approx(2.0)


def test_a_position_that_grew_gets_a_bigger_stop(store):
    """The case the old reconcile had no branch for at all."""
    fake = FakeHL(positions={"HYPE": 1.2}, mid=100.0)
    store.put("HYPE", planned(stop=90.0))
    v = build_venue(fake)
    protect.run_once(v, store)
    fake.pos["HYPE"] = 2.0                               # the remainder filled
    res = protect.run_once(v, store)
    assert [r.action for r in res] == ["replaced"]
    stops = [o for o in fake.resting if o["orderType"].startswith("Stop")]
    assert len(stops) == 1 and float(stops[0]["sz"]) == pytest.approx(2.0)


def test_a_partly_closed_position_gets_a_smaller_stop(store):
    fake = FakeHL(positions={"HYPE": 2.0}, mid=100.0)
    store.put("HYPE", planned(stop=90.0))
    v = build_venue(fake)
    protect.run_once(v, store)
    fake.pos["HYPE"] = 1.0
    protect.run_once(v, store)
    stops = [o for o in fake.resting if o["orderType"].startswith("Stop")]
    assert len(stops) == 1 and float(stops[0]["sz"]) == pytest.approx(1.0)


def test_duplicate_stops_are_reduced_to_exactly_one(store):
    fake = FakeHL(positions={"HYPE": 2.0}, mid=100.0)
    fake.rest_protection("HYPE", 2.0, stop_px=90.0)
    fake.rest_protection("HYPE", 2.0, stop_px=90.0)
    store.put("HYPE", planned(stop=90.0))
    protect.run_once(build_venue(fake), store)
    stops = [o for o in fake.resting if o["orderType"].startswith("Stop")]
    assert len(stops) == 1


def test_a_short_gets_its_stop_above_the_mark(store):
    fake = FakeHL(positions={"HYPE": -2.0}, mid=100.0)
    store.put("HYPE", planned(stop=110.0))
    res = protect.run_once(build_venue(fake), store)
    assert [r.action for r in res] == ["placed"]
    stop = next(o for o in fake.resting if o["orderType"].startswith("Stop"))
    assert float(stop["triggerPx"]) == pytest.approx(110.0)
    assert stop["side"] == "B", "closing a short buys"


# ── place before cancel ────────────────────────────────────────────────────

def test_the_new_stop_rests_before_the_old_one_is_cancelled(store):
    """The approved ordering. No sequence of failures may leave a live
    position with no stop at all."""
    fake = FakeHL(positions={"HYPE": 2.0}, mid=100.0)
    fake.rest_protection("HYPE", 1.0, stop_px=90.0)
    store.put("HYPE", planned(stop=90.0))
    old_oid = fake.resting[0]["oid"]

    order_of_events = []
    real_order, real_cancel = fake.order, fake.cancel

    def spy_order(*a, **k):
        order_of_events.append("place")
        return real_order(*a, **k)

    def spy_cancel(name, oid):
        order_of_events.append(f"cancel:{oid}")
        return real_cancel(name, oid)

    fake.order = spy_order          # type: ignore[method-assign]
    fake.cancel = spy_cancel        # type: ignore[method-assign]
    protect.run_once(build_venue(fake), store)
    assert order_of_events[0] == "place"
    assert f"cancel:{old_oid}" in order_of_events
    assert order_of_events.index("place") < \
        order_of_events.index(f"cancel:{old_oid}")


def test_a_failed_cancel_leaves_the_position_protected(store):
    """Two reduce-only stops cannot open anything, and the venue cancels both
    when the position closes. Not worth closing a protected position over."""
    fake = FakeHL(positions={"HYPE": 2.0}, mid=100.0)
    fake.rest_protection("HYPE", 1.0, stop_px=90.0)
    store.put("HYPE", planned(stop=90.0))
    fake.fail_cancel = 99
    res = protect.run_once(build_venue(fake), store)
    assert res[0].protected
    stops = [o for o in fake.resting if o["orderType"].startswith("Stop")]
    assert len(stops) == 2, "the stale one survived, as expected"
    assert any(float(s["sz"]) == pytest.approx(2.0) for s in stops)


# ── failure to protect closes the position ─────────────────────────────────

def test_a_position_that_cannot_be_protected_is_flattened(store, captured_alerts):
    fake = FakeHL(positions={"HYPE": 2.0}, mid=100.0)
    store.put("HYPE", planned(stop=90.0))
    real_order = fake.order

    def refuse_triggers(name, is_buy, sz, px, order_type, **k):
        if "trigger" in order_type:
            return {"status": "ok"}          # accepted, but nothing rests
        return real_order(name, is_buy, sz, px, order_type, **k)

    fake.order = refuse_triggers             # type: ignore[method-assign]
    res = protect.run_once(build_venue(fake), store)
    assert [r.action for r in res] == ["flattened"]
    assert fake.pos == {}
    assert any("could not be protected" in str(p) for p in captured_alerts)


def test_an_unreadable_order_book_does_not_flatten_a_position(store,
                                                              captured_alerts):
    """An unanswerable query is not evidence of an unprotected position. The
    2026-09-18 incident was a correctly protected position emergency-closed by
    a check that fired on success."""
    fake = FakeHL(positions={"HYPE": 2.0}, mid=100.0)
    store.put("HYPE", planned(stop=90.0))
    v = build_venue(fake)
    real = fake.frontend_open_orders
    fake.frontend_open_orders = lambda *a, **k: (_ for _ in ()).throw(
        Exception("429 Too Many Requests"))    # type: ignore[method-assign]
    res = protect.run_once(v, store)
    assert [r.action for r in res] == ["unknown"]
    assert fake.pos.get("HYPE") == pytest.approx(2.0), "it closed a position"
    assert any("could not be verified" in str(p) for p in captured_alerts)


# ── positions with no plan ─────────────────────────────────────────────────

def test_an_unknown_position_is_adopted_with_a_default_stop(store,
                                                            captured_alerts):
    """Rule 1. A C-1 orphan, or a position whose metadata a restart lost."""
    fake = FakeHL(positions={"ZRO": 50.0}, mid=2.0)
    assert store.get("ZRO") is None
    res = protect.run_once(build_venue(fake), store)
    assert [r.action for r in res] == ["placed"]
    stop = next(o for o in fake.resting if o["orderType"].startswith("Stop"))
    expected = 2.0 * (1 - protect.MAX_RISK_PCT / 100.0)
    assert float(stop["triggerPx"]) == pytest.approx(expected, rel=0.01)
    assert float(stop["sz"]) == pytest.approx(50.0)
    assert any("no recorded plan" in str(p) for p in captured_alerts)
    assert store.get("ZRO").reason == "adopted", "recorded, so next cycle is quiet"


def test_a_planned_stop_on_the_wrong_side_is_not_used(store):
    """A stop at or above the mark on a long is a market order waiting to
    fire. A stale plan after a gap is exactly how one arrives."""
    fake = FakeHL(positions={"HYPE": 2.0}, mid=100.0)
    store.put("HYPE", planned(stop=105.0))   # above the mark, on a long
    res = protect.run_once(build_venue(fake), store)
    stop = next(o for o in fake.resting if o["orderType"].startswith("Stop"))
    assert float(stop["triggerPx"]) < 100.0
    assert "planned stop unusable" in res[0].detail


def test_a_stop_that_would_round_across_the_mark_is_refused(store,
                                                            captured_alerts):
    """BTC prices allow one decimal, so a 3% stop below a mark of 1e-9 rounds
    UP to 0.1 -- above the mark, where a long's stop fires instantly.

    Rule 1: a position that cannot be given a default stop is flattened, not
    carried unstopped.
    """
    fake = FakeHL(positions={"BTC": 0.5}, mid=1e-9)
    res = protect.run_once(build_venue(fake), store)
    assert res[0].action == "flattened", res[0].detail
    assert "no usable stop price" in res[0].detail
    assert any("could not be protected" in str(p) for p in captured_alerts)
    # The flatten then fails too, and says so: at a mark of 1e-9 no limit
    # price is representable for BTC either, so no order can be sent. The
    # premise of this scenario makes the position both unprotectable and
    # uncloseable, and the operator is told both times rather than once.
    assert fake.pos == {"BTC": 0.5}
    assert any("Flatten did not finish" in str(p) for p in captured_alerts)


def test_the_side_check_runs_on_the_rounded_price_for_both_directions():
    """Directly, because this is the check that was missing."""
    from core.venue import Position
    fake = FakeHL(positions={}, mid=100.0)
    v = build_venue(fake)
    long_pos = Position("BTC", 0.5, 100.0, 1e-9, 0.0)
    px, why = protect.desired_stop(v, long_pos, None)
    assert px == 0.0 and "no valid representation" in why
    short_pos = Position("HYPE", -2.0, 100.0, 100.0, 0.0)
    px, _ = protect.desired_stop(v, short_pos, None)
    assert px > 100.0, "a short's stop belongs above the mark"


# ── take profit ────────────────────────────────────────────────────────────

def test_a_planned_tp_is_placed_alongside_the_stop(store):
    fake = FakeHL(positions={"HYPE": 2.0}, mid=100.0)
    store.put("HYPE", planned(stop=90.0, tp=115.0))
    protect.run_once(build_venue(fake), store)
    tps = [o for o in fake.resting if o["orderType"].startswith("Take Profit")]
    assert len(tps) == 1 and float(tps[0]["sz"]) == pytest.approx(2.0)


def test_a_tp_already_through_is_not_placed(store):
    fake = FakeHL(positions={"HYPE": 2.0}, mid=100.0)
    store.put("HYPE", planned(stop=90.0, tp=95.0))   # below the mark on a long
    protect.run_once(build_venue(fake), store)
    assert not [o for o in fake.resting
                if o["orderType"].startswith("Take Profit")]


def test_a_missing_tp_never_flattens(store):
    fake = FakeHL(positions={"HYPE": 2.0}, mid=100.0)
    store.put("HYPE", planned(stop=90.0, tp=115.0))
    v = build_venue(fake)
    protect.run_once(v, store)
    fake.resting = [o for o in fake.resting
                    if not o["orderType"].startswith("Take Profit")]
    res = protect.run_once(v, store)
    assert res[0].protected and fake.pos.get("HYPE") == pytest.approx(2.0)


# ── the cycle ──────────────────────────────────────────────────────────────

def test_a_flat_account_is_a_quiet_cycle(store):
    fake = FakeHL(positions={}, mid=100.0)
    assert protect.run_once(build_venue(fake), store) == []
    assert fake.sent == []


def test_an_unreadable_venue_does_nothing_at_all(store):
    fake = FakeHL(positions={"HYPE": 2.0}, mid=100.0)
    v = build_venue(fake)
    fake.fail_info_always = True
    assert protect.run_once(v, store) == []
    assert fake.sent == []


def test_every_position_is_protected_not_just_the_first(store):
    fake = FakeHL(positions={"HYPE": 2.0, "ZRO": 50.0, "NIL": 1000.0}, mid=10.0)
    res = protect.run_once(build_venue(fake), store)
    assert {r.coin for r in res} == {"HYPE", "ZRO", "NIL"}
    assert all(r.protected for r in res)
    stops = [o for o in fake.resting if o["orderType"].startswith("Stop")]
    assert {s["coin"] for s in stops} == {"HYPE", "ZRO", "NIL"}


def test_metadata_for_a_closed_position_is_dropped(store):
    fake = FakeHL(positions={"HYPE": 2.0}, mid=100.0)
    store.put("HYPE", planned(stop=90.0))
    v = build_venue(fake)
    protect.run_once(v, store)
    fake.pos.clear()
    protect.run_once(v, store)
    assert store.get("HYPE") is None, "a stale plan must not reach a later trade"


def test_metadata_is_not_a_position_record(store):
    """Metadata for a coin the venue does not hold protects nothing and is
    simply dropped -- it can never make the system think it holds something."""
    fake = FakeHL(positions={}, mid=100.0)
    store.put("GHOST", planned(stop=90.0))
    assert protect.run_once(build_venue(fake), store) == []
    assert store.get("GHOST") is None
    assert fake.sent == []


# ── resting bids ───────────────────────────────────────────────────────────

def test_a_stale_bid_is_pulled(store, monkeypatch):
    fake = FakeHL(positions={}, mid=100.0)
    v = build_venue(fake)
    execute.rest_bid(v, "HYPE", usd=200.0, px=95.0)
    assert len(fake.resting) == 1
    monkeypatch.setattr(protect, "BID_TTL_HOURS", 0.0)   # everything is stale
    protect.run_once(v, store)
    assert fake.resting == [], "an expired bid must not stay at the venue"


def test_a_fresh_bid_is_left_alone(store):
    fake = FakeHL(positions={}, mid=100.0)
    v = build_venue(fake)
    execute.rest_bid(v, "HYPE", usd=200.0, px=95.0)
    protect.run_once(v, store)
    assert len(fake.resting) == 1


def test_a_stop_is_never_mistaken_for_an_expired_bid(store, monkeypatch):
    """expire_bids must only touch orders that can OPEN a position."""
    fake = FakeHL(positions={"HYPE": 2.0}, mid=100.0)
    store.put("HYPE", planned(stop=90.0))
    v = build_venue(fake)
    protect.run_once(v, store)
    monkeypatch.setattr(protect, "BID_TTL_HOURS", 0.0)
    protect.run_once(v, store)
    stops = [o for o in fake.resting if o["orderType"].startswith("Stop")]
    assert len(stops) == 1, "the protective stop was cancelled as a stale bid"


def test_a_plan_survives_while_its_bid_is_still_resting(store):
    """The plan for an unfilled bid is the plan its stop will need the moment
    it fills, so it must not be tidied away as a closed position."""
    fake = FakeHL(positions={}, mid=100.0)
    v = build_venue(fake)
    store.put("HYPE", planned(stop=90.0, tp=115.0))
    execute.rest_bid(v, "HYPE", usd=200.0, px=95.0)
    protect.run_once(v, store)
    assert store.get("HYPE") is not None, "the plan was dropped under the bid"


def test_a_bid_that_fills_gets_its_planned_stop_on_the_next_cycle(store):
    """The whole resting-bid lifecycle, with no `pending` anywhere."""
    fake = FakeHL(positions={}, mid=100.0)
    v = build_venue(fake)
    store.put("HYPE", planned(stop=90.0))
    execute.rest_bid(v, "HYPE", usd=200.0, px=95.0)
    protect.run_once(v, store)                       # nothing to protect yet
    fake.pos["HYPE"] = 2.1                           # the tap fills it
    fake.resting = [o for o in fake.resting if o["reduceOnly"]]
    res = protect.run_once(v, store)
    assert [r.action for r in res] == ["placed"]
    stop = next(o for o in fake.resting if o["orderType"].startswith("Stop"))
    assert float(stop["triggerPx"]) == pytest.approx(90.0)
    assert float(stop["sz"]) == pytest.approx(2.1), "sized to what filled"


def test_a_bid_stale_by_the_venues_own_clock_is_pulled(store):
    """Age comes from the order's venue timestamp, not from anything stored."""
    fake = FakeHL(positions={}, mid=100.0)
    v = build_venue(fake)
    fake.now_ms -= int((protect.BID_TTL_HOURS + 1) * 3_600_000)
    execute.rest_bid(v, "HYPE", usd=200.0, px=95.0)
    assert len(fake.resting) == 1
    protect.run_once(v, store)
    assert fake.resting == []


class TheLoopCannotOpenAPosition(unittest.TestCase):
    """The guarantee the installed plist claims. Asserted, not asserted-by-prose.

    The plist this one was derived from said "It CANNOT place an order", which
    was true of the reconcile loop it described and false of this one: the
    protection loop places stops and can market-close a position. What must
    stay true is the narrower claim -- it cannot OPEN one.
    """

    def test_the_loop_module_imports_nothing_that_can_enter(self):
        import inspect
        import tools.protect_loop as L
        src = inspect.getsource(L)
        for forbidden in ("execute.enter", "rest_bid", "send_limit",
                          "size_from_stop", "live_levels_trader"):
            self.assertNotIn(forbidden, src,
                             f"protect_loop reaches {forbidden}")

    def test_protect_never_sends_a_non_reduce_only_order(self):
        """Over a full cycle on an unknown position, every order it sends
        must be reduce-only."""
        fake = FakeHL(positions={"ZRO": 50.0}, mid=2.0)
        protect.run_once(build_venue(fake), MetaStore())
        self.assertTrue(fake.sent, "nothing was sent; the test proves nothing")
        for o in fake.sent:
            self.assertTrue(o["reduce_only"],
                            f"protect sent a non-reduce-only order: {o}")

    def test_the_plist_does_not_repeat_the_false_claim(self):
        import pathlib
        txt = pathlib.Path(
            "deploy/launchd/com.cuansniffer.protect.plist").read_text()
        self.assertNotIn("must never be able to\n       place an order", txt)
        self.assertNotIn("It CANNOT place an order", txt)
        self.assertIn("MARKET-CLOSES", txt,
                      "the plist must say it can close a position")
        self.assertIn("cannot OPEN one", txt,
                      "the plist must state the guarantee that IS true")


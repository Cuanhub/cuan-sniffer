"""core/venue.py against the recorded Hyperliquid payloads.

The first three tests are the regressions for REVIEW.md findings 1 and 15. They
assert properties of the RECORDED data, so they fail if anyone reintroduces
response parsing or re-derives size from a request.
"""

from __future__ import annotations

import pytest

from core.venue import Order, Venue, _f
from tests.fake_venue import (FakeHL, Transport, build_venue, load,
                              order_envelope, order_record)


# ── what the recorded payloads actually say ────────────────────────────────

def test_sz_is_the_remaining_size_not_the_filled_size():
    """REVIEW.md finding 1. A filled order reports sz=0; a cancelled one
    reports the full size. Any code that reads `sz` as "how much filled" has
    these exactly backwards.
    """
    filled = order_record("filled")["order"]
    assert float(filled["sz"]) == 0.0
    assert float(filled["origSz"]) > 0.0

    for rejected in ("perpMarginRejected", "iocCancelRejected",
                     "reduceOnlyRejected"):
        o = order_record(rejected)["order"]
        assert float(o["sz"]) == float(o["origSz"]), (
            f"{rejected} is a ZERO-fill reject; sz == origSz")


def test_no_recorded_order_carries_totalsz_or_avgpx():
    """The two fields the deleted parser read. They do not exist."""
    for row in load("order_status_by_inner_status.json"):
        assert "totalSz" not in row["order"]
        assert "avgPx" not in row["order"]


def test_the_orderstatus_envelope_cannot_distinguish_fill_from_reject():
    """Why reading the envelope was wrong: it says "order" for every inner
    status, so a zero-fill reject and a full fill are indistinguishable there.
    """
    statuses = {row["_inner_status"]: row["payload"].get("status")
                for row in load("order_status_envelopes.json")}
    assert len(statuses) >= 9, "expected the full recorded status vocabulary"
    assert set(statuses.values()) == {"order"}
    assert "perpMarginRejected" in statuses and "filled" in statuses


def test_unified_account_reports_zero_perp_account_value():
    """REVIEW.md finding 15: 0.0 is the real reading here, so equity must not
    come from marginSummary."""
    flat = load("clearinghouse_state_flat.json")
    assert flat["marginSummary"]["accountValue"] == "0.0"
    assert load("abstraction_state.json")["state"] == "unifiedAccount"
    usdc = next(b for b in load("spot_user_state.json")["balances"]
                if b["coin"] == "USDC")
    assert float(usdc["total"]) > 0.0


# ── reads ──────────────────────────────────────────────────────────────────

def test_positions_reports_what_the_venue_holds():
    v = build_venue(FakeHL(positions={"HYPE": 2.5}, mid=40.0))
    pos = v.positions()
    assert pos is not None and set(pos) == {"HYPE"}
    p = pos["HYPE"]
    assert p.szi == 2.5 and p.size == 2.5 and p.is_long
    assert p.mark_px == pytest.approx(40.0)


def test_a_short_is_reported_signed():
    v = build_venue(FakeHL(positions={"HYPE": -2.5}, mid=40.0))
    p = v.positions()["HYPE"]
    assert p.szi == -2.5 and p.size == 2.5 and not p.is_long


def test_an_unreadable_venue_is_none_never_flat():
    """Reporting an unreadable venue as flat is what orphans a position."""
    fake = FakeHL(positions={"HYPE": 2.0})
    fake.fail_info = 99
    v = build_venue(fake)
    assert v.positions() is None
    assert v.open_orders("HYPE") is None
    assert v.equity() is None


def test_open_orders_classifies_a_real_recorded_stop():
    fake = FakeHL(positions={"NIL": 1331.0})
    fake.rest_protection("NIL", 1331.0, stop_px=0.075, tp_px=0.095)
    v = build_venue(fake)
    orders = v.open_orders("NIL")
    assert orders is not None and len(orders) == 2
    stop = next(o for o in orders if o.is_stop)
    tp = next(o for o in orders if not o.is_stop)
    assert stop.is_protective and tp.is_protective
    assert stop.trigger_px == pytest.approx(0.075)
    assert stop.size == pytest.approx(1331.0)


def test_a_plain_resting_order_is_not_protective():
    """REVIEW.md finding 3. The old has_protection() returned True for exactly
    this input, because any order for the coin counted."""
    fake = FakeHL(positions={"HYPE": 2.0})
    fake.rest_plain_order("HYPE", 2.0, px=99.0)
    v = build_venue(fake)
    orders = v.open_orders("HYPE")
    assert orders is not None and len(orders) == 1
    assert orders[0].is_protective is False


def test_open_orders_filters_by_coin():
    fake = FakeHL()
    fake.rest_protection("NIL", 10.0, stop_px=0.07)
    fake.rest_protection("HYPE", 2.0, stop_px=90.0)
    v = build_venue(fake)
    assert [o.coin for o in v.open_orders("HYPE")] == ["HYPE"]
    assert len(v.open_orders()) == 2


def test_equity_comes_from_spot_usdc_which_already_marks_to_market():
    """`usdc_total` is the equity figure on its own -- it already contains the
    open position's unrealised PnL. There is deliberately no property that
    adds the two, because that double-counts. See core.venue.Equity.
    """
    fake = FakeHL(positions={"HYPE": 2.0}, usdc=720.17,
                  unrealized={"HYPE": -12.5})
    v = build_venue(fake)
    eq = v.equity()
    assert eq is not None
    assert eq.usdc_total == pytest.approx(720.17)
    assert eq.unrealized_pnl == pytest.approx(-12.5)
    assert not hasattr(eq, "with_unrealized")


def test_mark_price_needs_no_price_feed():
    """The kill switch prices from Position.mark_px, so a dead or throttled
    all_mids cannot block a flatten. REVIEW.md finding 10."""
    fake = FakeHL(positions={"HYPE": 2.0}, mid=40.0)
    v = build_venue(fake)
    pos = v.positions()
    fake.fail_info = 99                      # price feed now unreachable
    assert v.mid("HYPE") is None
    assert pos["HYPE"].mark_px == pytest.approx(40.0)


# ── rounding ───────────────────────────────────────────────────────────────

def test_size_rounds_down_to_venue_precision():
    v = build_venue(FakeHL())
    coin = next(a["name"] for a in load("meta.json")["universe"]
                if a["szDecimals"] == 2)
    assert v.round_size(coin, 1.2399) == pytest.approx(1.23)
    assert v.round_size(coin, 0.001) == 0.0
    assert v.round_size(coin, -5) == 0.0


@pytest.mark.parametrize("px", [1.23456, 0.0897123, 9.99999, 45678.9,
                                 0.00012345, 2.0014, 0.16819])
def test_price_rounding_never_crosses_the_input(px):
    """A stop rounded "down" must never land above its level, and a stop
    rounded "up" must never land below it -- at every step, not just the last.

    0.0 is a refusal, not a price: some prices have no representation at a
    coin's precision, and substituting the nearest increment would move a stop
    arbitrarily far. Every coin in the universe is checked, because the
    precision varies from 0 to 5 decimals across it.
    """
    v = build_venue(FakeHL())
    for coin in [a["name"] for a in load("meta.json")["universe"]]:
        down = v.round_price(coin, px, "down")
        up = v.round_price(coin, px, "up")
        assert down == 0.0 or down <= px, (coin, px, down)
        assert up == 0.0 or up >= px, (coin, px, up)


def test_an_unrepresentable_price_is_refused_not_substituted():
    """BTC allows one decimal place, so 0.00012345 has no representation there.

    Rounding DOWN collapses to zero, and zero is returned as a refusal -- the
    code this replaces substituted the smallest increment (0.1) instead, which
    is a stop three orders of magnitude from its level.

    Rounding UP to 0.1 is a real answer, not a substitution: it satisfies
    "never below the input". It is still a useless stop, and what rejects it is
    `core.protect`'s side-relation check against the mark price, not the
    rounder. The rounder's job ends at representability.
    """
    v = build_venue(FakeHL())
    btc_decimals = next(a["szDecimals"] for a in load("meta.json")["universe"]
                        if a["name"] == "BTC")
    assert 6 - btc_decimals == 1, "precondition: BTC prices allow 1 decimal"
    assert v.round_price("BTC", 0.00012345, "down") == 0.0
    assert v.round_price("BTC", 0.00012345, "up") == pytest.approx(0.1)
    assert v.round_price("BTC", 104532.7, "down") > 0


def test_price_rounding_edges():
    v = build_venue(FakeHL())
    coin = next(a["name"] for a in load("meta.json")["universe"]
                if a["szDecimals"] == 2)
    assert v.round_price(coin, 0.0) == 0.0
    assert v.round_price(coin, -1.0) == 0.0
    assert v.round_price(coin, 1.23456) > 0


def test_a_cloid_is_always_unique_and_valid():
    from hyperliquid.utils.types import Cloid
    v = build_venue(FakeHL())
    seen = [v.new_cloid().to_raw() for _ in range(50)]
    assert len(set(seen)) == 50
    for raw in seen:
        Cloid.from_str(raw)                  # raises if malformed


# ── writes ─────────────────────────────────────────────────────────────────

def test_a_failed_send_returns_an_envelope_instead_of_raising():
    """Callers must not need try/except around every send; they re-read the
    venue to find out what happened."""
    fake = FakeHL()
    fake.fail_order = 1
    v = build_venue(fake)
    resp = v.send_ioc("HYPE", True, 1.0, 100.0, reduce_only=False)
    assert isinstance(resp, dict) and resp["status"] == "err"


def test_nothing_reads_an_order_response_body():
    """Rule 2, asserted structurally.

    The fake returns a bare {"status": "ok"} with no statuses, no oid and no
    sizes. If any code path starts depending on a response body, it will stop
    working against this fake and this test's partner cases will fail -- which
    is the signal that response parsing has crept back in.
    """
    fake = FakeHL(positions={})
    v = build_venue(fake)
    resp = v.send_ioc("HYPE", True, 1.0, 100.0, reduce_only=False)
    assert resp == {"status": "ok"}
    assert fake.pos.get("HYPE") == pytest.approx(1.0), (
        "the position moved, which is the only thing a caller may observe")


def test_cancel_reports_whether_the_venue_removed_the_order():
    fake = FakeHL(positions={"NIL": 10.0})
    fake.rest_protection("NIL", 10.0, stop_px=0.07)
    v = build_venue(fake)
    oid = v.open_orders("NIL")[0].oid
    assert v.cancel("NIL", oid)["status"] == "ok"
    assert v.open_orders("NIL") == []
    assert v.cancel("NIL", oid)["status"] == "err"


def test_a_trigger_order_becomes_readable_on_the_book():
    """Confirmation is by observation: send, then read."""
    fake = FakeHL(positions={"HYPE": 2.0})
    v = build_venue(fake)
    assert v.open_orders("HYPE") == []
    v.send_trigger("HYPE", is_buy=False, size=2.0, trigger_px=90.0, is_stop=True)
    orders = v.open_orders("HYPE")
    assert len(orders) == 1 and orders[0].is_protective and orders[0].is_stop


def test_meta_recovers_after_a_startup_rate_limit():
    fake = FakeHL()
    fake.fail_info = 1
    v = object.__new__(Venue)
    v.debug = False
    v.address = "0x" + "11" * 20
    v.info = fake
    v.exchange = fake
    v._sz_decimals = {}
    v._cloid_seq = 0
    assert v.load_meta() is False            # the 429 window
    assert v.load_meta() is True             # retried on demand


def test_f_helper_rejects_junk():
    assert _f("1.5") == 1.5
    assert _f(None) is None
    assert _f("") is None
    assert _f({}) is None


def test_order_is_protective_requires_both_flags():
    base = dict(coin="X", oid=1, size=1.0, is_buy=False, trigger_px=1.0,
                is_stop=True)
    assert Order(reduce_only=True, is_trigger=True, **base).is_protective
    assert not Order(reduce_only=True, is_trigger=False, **base).is_protective
    assert not Order(reduce_only=False, is_trigger=True, **base).is_protective

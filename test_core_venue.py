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


def test_the_test_double_sets_every_attribute_the_real_init_does():
    """Guard against the silent-degradation trap.

    `load_meta` catches Exception so a venue outage degrades gracefully. That
    also means a missing attribute on a hand-built Venue is swallowed, leaving
    szDecimals=3 for every coin -- wrong sizing on every order, with nothing
    said. Adding a field to __init__ and forgetting it in build_venue must
    fail here, not quietly mis-size trades.
    """
    import ast
    import inspect
    src = inspect.getsource(Venue.__init__)
    assigned = {n.attr for n in ast.walk(ast.parse(src.strip()))
                if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store)}
    built = build_venue(FakeHL())
    missing = {a for a in assigned if not hasattr(built, a)}
    assert not missing, f"build_venue is missing {sorted(missing)}"


# ── a failed read must name its root cause ─────────────────────────────────
# Thirty minutes of reads failed on DNS on 2026-10-19 while the log looked
# like a rate limit, because the message was truncated one character into
# "NameResolutionError". requests nests the real failure -- ConnectionError
# -> MaxRetryError -> NameResolutionError -- and only the OUTERMOST name is
# on the exception. The remedies differ: a 429 wants the brake, a dead
# resolver in a long-lived process wants a restart, so the log has to tell
# them apart.

def test_the_innermost_cause_is_reported():
    from core.venue import _root_cause
    inner = OSError("getaddrinfo failed")
    middle = RuntimeError("max retries")
    middle.__cause__ = inner
    outer = ConnectionError("pool error")
    outer.__cause__ = middle
    assert _root_cause(outer) == "OSError"


def test_an_unchained_exception_is_its_own_cause():
    from core.venue import _root_cause
    assert _root_cause(ValueError("x")) == "ValueError"


def test_implicit_context_counts_too():
    """`raise X` inside an `except Y` sets __context__, not __cause__."""
    from core.venue import _root_cause
    try:
        try:
            raise OSError("dns")
        except OSError:
            raise ConnectionError("wrapped")
    except ConnectionError as e:
        assert _root_cause(e) == "OSError"


def test_a_cycle_does_not_hang():
    from core.venue import _root_cause
    a, b = ValueError("a"), ValueError("b")
    a.__cause__ = b
    b.__cause__ = a
    assert _root_cause(a) == "ValueError"


def test_a_real_dns_failure_unwraps_past_the_outer_type():
    """Not a mock: the actual shape requests produces."""
    import requests
    from core.venue import _root_cause
    try:
        requests.post("https://api-does-not-exist.invalid/info", timeout=5)
    except Exception as e:
        assert _root_cause(e) != type(e).__name__, (
            "the root cause is still the outer type -- not unwrapped")
    else:
        pytest.skip("the invalid host resolved; no DNS failure to read")


def test_the_read_handler_prints_the_root_cause(capsys):
    """End to end through _get, which is what writes the log line."""
    v = build_venue(FakeHL())

    def boom():
        try:
            raise OSError("getaddrinfo failed")
        except OSError:
            raise ConnectionError("HTTPSConnectionPool(...): Max retries")

    assert v._get("user_state", boom) is None
    out = capsys.readouterr().out
    assert "OSError" in out, out
    assert "ConnectionError" in out, out
    assert "<-" in out, out


def test_an_unchained_error_is_not_named_twice(capsys):
    """"Transport: Transport: ..." is noise on the one line that has to be
    readable in a hurry."""
    v = build_venue(FakeHL())

    def boom():
        raise ValueError("plain")

    assert v._get("user_state", boom) is None
    out = capsys.readouterr().out
    assert out.count("ValueError") == 1, out


# ── schema drift must read as UNREADABLE, never as empty ───────────────────
#
# The outer container check was not enough. A dict with no `assetPositions`
# iterated [] and returned {} -- indistinguishable from flat -- and "flat" is
# acted on hard. The three consequences, each verified below at the level that
# would have suffered them:
#
#   flatten_all  prints FLAT and returns flat=True with positions live
#   preflight    counts 0 held and opens MAX_CONCURRENT on top of them,
#                which is the 2026-09-18 runaway with a new cause
#   protect      calls _forget_closed and deletes every plan
#
# A present-but-wrong-TYPE value was a second bug in the other direction: it
# RAISED AttributeError out of positions() and equity(), through flatten_all's
# loop and through preflight.
#
# Verified safe before writing the guard: a flat account really does send the
# key. Live, while flat, and clearinghouse_state_flat.json both carry
# "assetPositions": []. If it did not, requiring it would make every read
# unreadable -- which now also trips the restart watchdog, so over-triggering
# costs more than it used to.

GOOD_ORDER = {"coin": "SOL", "oid": 1, "sz": "1", "side": "A",
              "orderType": "Stop Market", "triggerPx": "100",
              "limitPx": "100", "timestamp": 1, "reduceOnly": True,
              "isTrigger": True}


def _drifted_positions(fake, payload):
    fake.user_state = lambda *a, **k: payload


@pytest.mark.parametrize("payload,why", [
    ({"marginSummary": {}}, "key missing"),
    ({"assetPositions": None}, "None"),
    ({"assetPositions": {"SOL": 1}}, "dict not list"),
    ({"assetPositions": "none"}, "string"),
    ({"assetPositions": ["junk", 7]}, "no row is a dict"),
])
def test_drifted_user_state_reads_as_unreadable(payload, why):
    fake = FakeHL(positions={"SOL": 1.76})
    _drifted_positions(fake, payload)
    assert build_venue(fake).positions() is None, why


def test_a_genuinely_flat_account_still_reads_as_flat():
    """The guard must not misread flat, or every loop restarts every 20 min."""
    fake = FakeHL(positions={})
    assert build_venue(fake).positions() == {}


def test_the_recorded_flat_payload_carries_the_key():
    """The evidence the guard rests on, pinned so it cannot rot."""
    flat = load("clearinghouse_state_flat.json")
    assert "assetPositions" in flat
    assert flat["assetPositions"] == []


def test_a_zero_size_row_is_flat_not_drift():
    """Skipping |szi| <= FLAT_EPSILON is legitimate and happens around a
    close, so "non-empty but nothing parsed" could not be the rule."""
    fake = FakeHL(positions={})
    _drifted_positions(fake, {"assetPositions": [
        {"position": {"coin": "SOL", "szi": "0.0"}}]})
    assert build_venue(fake).positions() == {}


# ── orders ─────────────────────────────────────────────────────────────────

def test_orders_that_cannot_be_read_are_unreadable_not_empty():
    """A renamed `oid` alone would skip every row and return []."""
    fake = FakeHL()
    fake.frontend_open_orders = lambda *a, **k: [
        {**GOOD_ORDER, "oid": None, "order_id": 1}]
    assert build_venue(fake).open_orders() is None


def test_junk_order_rows_are_unreadable():
    fake = FakeHL()
    fake.frontend_open_orders = lambda *a, **k: ["nope"]
    assert build_venue(fake).open_orders() is None


def test_genuinely_no_orders_is_still_empty():
    fake = FakeHL()
    fake.frontend_open_orders = lambda *a, **k: []
    assert build_venue(fake).open_orders() == []


def test_a_coin_filter_matching_nothing_is_still_empty():
    """`open_orders("ETH")` returning [] while SOL rests is correct and
    common, so the filtered call cannot use the drift rule."""
    fake = FakeHL()
    fake.frontend_open_orders = lambda *a, **k: [GOOD_ORDER]
    v = build_venue(fake)
    assert v.open_orders("ETH") == []
    assert len(v.open_orders("SOL") or []) == 1


# ── equity ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("payload", [
    {"tokenToAvailableAfterMaintenance": {}},
    {"balances": {"USDC": "100"}},
    {"balances": "100"},
    {"balances": ["junk"]},
])
def test_drifted_spot_state_reads_as_unreadable(payload):
    """A missing `balances` already returned None. A wrong-TYPED one RAISED."""
    fake = FakeHL()
    fake.spot_user_state = lambda *a, **k: payload
    assert build_venue(fake).equity() is None


# ── and nothing raises, in either direction ────────────────────────────────

@pytest.mark.parametrize("payload", [
    {"assetPositions": {"SOL": 1}}, {"assetPositions": "x"},
    {"assetPositions": ["junk"]}, {},
])
def test_no_drifted_shape_raises_out_of_positions(payload):
    """`flatten_all` calls this inside its loop and `preflight` calls it
    unguarded, so an exception here reaches the kill switch."""
    fake = FakeHL(positions={"SOL": 1.0})
    _drifted_positions(fake, payload)
    build_venue(fake).positions()          # must not raise

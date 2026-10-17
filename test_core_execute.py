"""core/execute.py: the scenarios REVIEW.md finding 1 got wrong.

The scenarios from the build brief that belong to this module: fill, partial
fill, zero-fill IOC, timeout-after-fill, timeout-no-fill, restart mid-trade.

Note what these tests never do: they never hand the code a response payload.
Every outcome is driven by what the fake venue HOLDS, which is the point.
"""

from __future__ import annotations

import pytest

from core import execute
from core.meta import MetaStore, TradeMeta
from tests.fake_venue import FakeHL, build_venue


@pytest.fixture(autouse=True)
def no_settle_sleep(monkeypatch):
    monkeypatch.setattr(execute, "SETTLE_SEC", 0.0)


# ── entries ────────────────────────────────────────────────────────────────

def test_a_full_fill_is_the_whole_delta():
    fake = FakeHL(mid=100.0)
    out = execute.enter(build_venue(fake), "HYPE", usd=200.0)
    assert out.filled and out.filled_size == pytest.approx(2.0)
    assert out.notional == pytest.approx(200.0, rel=0.02)


def test_a_partial_fill_reports_only_what_filled():
    fake = FakeHL(mid=100.0)
    fake.fill_fraction = 0.6
    out = execute.enter(build_venue(fake), "HYPE", usd=200.0)
    assert out.filled and out.filled_size == pytest.approx(1.2)


def test_a_zero_fill_ioc_is_not_a_fill():
    """The real venue calls this iocCancelRejected or perpMarginRejected. The
    position does not move, so there is nothing to misread."""
    fake = FakeHL(mid=100.0)
    fake.order_fills = False
    out = execute.enter(build_venue(fake), "HYPE", usd=200.0)
    assert not out.filled and out.filled_size == 0.0
    assert not out.unknown and out.note == "no_fill"


def test_timeout_after_fill_is_reported_as_the_fill_it_was():
    """REVIEW.md finding 1, the headline case. The venue filled; the client saw
    an exception. The old code reported a reject and released the slot, leaving
    a live unprotected position untracked for up to 450s.
    """
    fake = FakeHL(mid=100.0)
    fake.raise_after_fill = True             # fills, THEN the request raises
    out = execute.enter(build_venue(fake), "HYPE", usd=200.0)
    assert out.filled, "a timed-out order that FILLED was reported as no fill"
    assert out.filled_size == pytest.approx(2.0)
    assert fake.pos["HYPE"] == pytest.approx(2.0)


def test_timeout_with_no_fill_is_not_a_phantom_position():
    """The other half. The old code adopted the venue's order record and, for
    every zero-fill reject status, reported a FULL fill at the limit price.
    """
    fake = FakeHL(mid=100.0)
    fake.fail_order = 1                      # never reaches the venue
    out = execute.enter(build_venue(fake), "HYPE", usd=200.0)
    assert not out.filled, "a never-placed order became a fill"
    assert "HYPE" not in fake.pos


def test_an_unreadable_venue_after_sending_is_unknown_not_a_reject():
    fake = FakeHL(mid=100.0)
    v = build_venue(fake)

    sent = {"n": 0}
    real = fake.order

    def order_then_blind(*a, **k):
        sent["n"] += 1
        r = real(*a, **k)
        fake.fail_info = 99                  # venue goes dark after the send
        return r

    fake.order = order_then_blind            # type: ignore[method-assign]
    out = execute.enter(v, "HYPE", usd=200.0)
    assert sent["n"] == 1
    assert out.unknown and not out.filled, (
        "an order was sent and the result is unknown; it is neither a fill nor "
        "a reject, and a caller must not round it to either")


def test_entry_refuses_rather_than_guessing_precision():
    fake = FakeHL(mid=100.0)
    v = build_venue(fake)
    v._sz_decimals = {}                      # as if meta never loaded
    fake.fail_info = 99
    out = execute.enter(v, "HYPE", usd=200.0)
    assert not out.filled and out.note == "no_asset_metadata"
    assert fake.sent == []


def test_entry_refuses_a_dust_size():
    fake = FakeHL(mid=100.0)
    out = execute.enter(build_venue(fake), "BTC", usd=0.0001)
    assert not out.filled and out.note == "size_rounds_to_zero"
    assert fake.sent == []


def test_entry_refuses_without_a_price():
    fake = FakeHL(mid=100.0)
    v = build_venue(fake)
    fake.fail_info = 99
    out = execute.enter(v, "HYPE", usd=200.0)
    assert not out.filled and out.note == "no_price"
    assert fake.sent == []


def test_entry_refuses_a_non_positive_notional():
    fake = FakeHL(mid=100.0)
    assert execute.enter(build_venue(fake), "HYPE", usd=0.0).note == \
        "non_positive_size"
    assert fake.sent == []


# ── exits ──────────────────────────────────────────────────────────────────

def test_close_takes_side_and_size_from_the_venue():
    fake = FakeHL(positions={"HYPE": 2.0}, mid=100.0)
    out = execute.close(build_venue(fake), "HYPE")
    assert out.filled and out.filled_size == pytest.approx(2.0)
    assert "HYPE" not in fake.pos
    assert fake.sent[-1]["is_buy"] is False and fake.sent[-1]["reduce_only"]


def test_close_works_on_a_short_without_being_told():
    """Every emergency call site used to pass side="LONG" unconditionally, so
    flatten refused to close a short at all."""
    fake = FakeHL(positions={"HYPE": -2.0}, mid=100.0)
    out = execute.close(build_venue(fake), "HYPE")
    assert out.filled and out.filled_size == pytest.approx(2.0)
    assert "HYPE" not in fake.pos
    assert fake.sent[-1]["is_buy"] is True


def test_close_on_a_position_nothing_opened():
    """A C-1 orphan. No metadata, no planned stop, and it still closes."""
    fake = FakeHL(positions={"ZRO": 50.0}, mid=2.0)
    assert MetaStore().get("ZRO") is None, "precondition: nothing is recorded"
    out = execute.close(build_venue(fake), "ZRO")
    assert out.filled and "ZRO" not in fake.pos


def test_a_partial_close_reports_the_part_that_closed():
    fake = FakeHL(positions={"HYPE": 2.0}, mid=100.0)
    fake.fill_fraction = 0.5
    out = execute.close(build_venue(fake), "HYPE")
    assert out.filled_size == pytest.approx(1.0)
    assert fake.pos["HYPE"] == pytest.approx(1.0), "half is still open"


def test_a_full_close_still_reports_a_price():
    """Once the position is gone there is no mark left to read, so a complete
    close would otherwise report a notional of zero."""
    fake = FakeHL(positions={"HYPE": 2.0}, mid=100.0)
    out = execute.close(build_venue(fake), "HYPE")
    assert out.avg_px == pytest.approx(100.0)
    assert out.notional == pytest.approx(200.0)


def test_close_of_a_flat_coin_sends_nothing():
    fake = FakeHL(positions={}, mid=100.0)
    out = execute.close(build_venue(fake), "HYPE")
    assert not out.filled and out.note == "already_flat"
    assert fake.sent == []


def test_close_cannot_send_an_order_that_would_add():
    """Size is clamped to what the venue holds, so an over-sized request
    cannot flip the position."""
    fake = FakeHL(positions={"HYPE": 2.0}, mid=100.0)
    execute.close(build_venue(fake), "HYPE", size=500.0)
    assert fake.sent[-1]["sz"] == pytest.approx(2.0)


def test_close_reports_unknown_when_the_venue_goes_dark_first():
    fake = FakeHL(positions={"HYPE": 2.0}, mid=100.0)
    v = build_venue(fake)
    fake.fail_info = 99
    out = execute.close(v, "HYPE")
    assert out.unknown and fake.sent == []


# ── restart mid-trade ──────────────────────────────────────────────────────

def test_restart_mid_trade_recovers_from_the_venue(tmp_path):
    """REVIEW.md finding 8. The process dies between sending an entry and
    recording it. On restart there is no metadata at all -- and the position is
    still fully described, because the venue describes it.
    """
    fake = FakeHL(mid=100.0)
    v = build_venue(fake)
    execute.enter(v, "HYPE", usd=200.0)      # filled, nothing recorded

    store = MetaStore(tmp_path / "meta.json")   # a fresh process
    assert store.get("HYPE") is None
    pos = v.positions()
    assert pos is not None and pos["HYPE"].size == pytest.approx(2.0), (
        "the venue knows the size, so it can never be missing")

    # And metadata written later does not change what we hold.
    store.put("HYPE", TradeMeta(reason="level", planned_stop=95.0,
                                planned_tp=115.0, opened_at="now"))
    assert store.get("HYPE").planned_stop == 95.0
    assert v.positions()["HYPE"].size == pytest.approx(2.0)

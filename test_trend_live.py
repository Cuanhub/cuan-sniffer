"""tools/trend_live.py and core/books.py: the trend book trades live, owns its
ten coins while armed, and changes nothing it cannot read. Fake venue, fake
candles; nothing here reaches the network."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

import tools.live_levels_trader as TR
from core import books, execute, protect, safety, streak
from core.meta import MetaStore, TradeMeta
from research.backtest.experiments import trend_daily as T
from research.trend import paper as P
from tests.fake_venue import FakeHL, build_venue
from tools import liveness
from tools import trend_live as L

DAY = P.DAY_MS
T0 = T.day_ms("2026-01-01")
MID = 1.0
CAP = 300.0


def closes(days=260, C=10, seed=11):
    rng = np.random.default_rng(seed)
    return 100.0 * np.exp(np.cumsum(rng.normal(0, 0.03, (days, C)), axis=0))


class FakeRO:
    """Daily candles up to `now`, every mid at MID, and the funding the test
    puts in `funding`."""

    def __init__(self, close):
        self.close, self.funding, self.broken = close, [], ""

    def get_candles(self, coin, interval, start_ms, end_ms):
        if self.broken == "candles":
            raise RuntimeError("500")
        j = P.COINS.index(coin)
        n = min(len(self.close), int((end_ms - T0) // DAY) + 1)
        return [{"t": T0 + i * DAY, "o": 1, "h": 1, "l": 1,
                 "c": float(self.close[i, j]), "v": 1, "n": 1} for i in range(n)]

    def get_all_mids(self):
        return {c: MID for c in P.COINS}

    def _post(self, payload, label):
        assert payload["type"] == "userFunding"
        return [r for r in self.funding
                if payload["startTime"] <= r["time"] <= payload["endTime"]]


def now_for(i, minutes=5):
    return T0 + (i + 1) * DAY + minutes * 60_000


@pytest.fixture(autouse=True)
def armed(monkeypatch):
    monkeypatch.setattr(execute, "SETTLE_SEC", 0.0)
    monkeypatch.setattr(safety, "FLATTEN_PAUSE_SEC", 0.0)
    monkeypatch.setenv("TREND_LIVE_ARMED", "true")


class Rig:
    def __init__(self, tmp: Path):
        self.fake = FakeHL(positions={}, mid=MID, usdc=1000.0)
        self.fake.fills_override = []
        self.ro = FakeRO(closes())
        self.store = MetaStore(tmp / "meta.json")
        self.halt = safety.HaltState(tmp / "halt.json")
        self.v = build_venue(self.fake)
        self.dir = tmp / "live"

    def run(self, i, minutes=5, capital=CAP, **kw):
        return L.run_pass(self.v, self.ro, now_for(i, minutes), capital, self.dir,
                          self.store, self.halt, **kw)

    def iocs(self):
        return [r for r in self.fake.sent if "limit" in r["type"]]

    def stops(self, coin):
        return [o for o in self.fake.resting
                if o["coin"] == coin and o["orderType"].startswith("Stop")]


@pytest.fixture
def rig(tmp_path):
    return Rig(tmp_path)


# ── the pass ──────────────────────────────────────────────────────────────

def test_the_first_pass_opens_the_book_at_its_targets(rig):
    row = rig.run(200)
    opened = 0
    for c, w in row["target"].items():
        usd = abs(w) * CAP
        if usd < L.MIN_TRADE_USD:
            assert c not in rig.fake.pos
            continue
        opened += 1
        assert np.sign(rig.fake.pos[c]) == np.sign(w), c
        assert abs(abs(rig.fake.pos[c]) * MID - usd) <= 1.0 + 1e-9, c
    assert opened >= 5
    assert row["gross"] == pytest.approx(sum(abs(w) for w in row["held_after"].values()))


def test_it_acts_once_per_utc_day(rig):
    assert rig.run(200) is not None
    sent = len(rig.iocs())
    assert rig.run(200, minutes=65) is None, "a second pass on the same day"
    assert len(rig.iocs()) == sent
    assert rig.run(201) is not None
    assert [r["day"] for r in P.read_ledger(rig.dir)] == [T.iso(T0 + 201 * DAY),
                                                          T.iso(T0 + 202 * DAY)]


def test_closes_and_reductions_go_before_anything_opens(rig):
    target = rig.run(200, dry_run=True)["target"]
    wrong = [c for c, w in target.items() if abs(w) * CAP >= 30][:3]
    for c in wrong:                       # held on the wrong side
        rig.fake.pos[c] = -np.sign(target[c]) * round(abs(target[c]) * CAP / MID)
    rig.run(200)
    reduce_only = [r["reduce_only"] for r in rig.iocs()]
    assert True in reduce_only and False in reduce_only
    assert reduce_only.index(False) > max(i for i, r in enumerate(reduce_only) if r), \
        "an order opened before every close had gone"
    for c in wrong:
        assert np.sign(rig.fake.pos[c]) == np.sign(target[c]), c


def test_a_coin_the_band_book_holds_is_left_alone(rig):
    rig.store.put("BTC", TradeMeta(reason="level", planned_stop=0.5,
                                   opened_at="2026-01-01T00:00:00Z"))
    rig.fake.rest_plain_order("ETH", 20.0, 0.9)          # a band bid resting
    row = rig.run(200)
    assert set(row["blocked"]) == {"BTC", "ETH"}
    assert not [r for r in rig.fake.sent if r["coin"] in ("BTC", "ETH")]
    assert any(o["coin"] == "ETH" and not o.get("reduceOnly") for o in rig.fake.resting)


def test_the_plan_skips_what_the_venue_would_refuse_but_never_a_close():
    c = P.COINS
    held, new = np.zeros(10), np.zeros(10)
    new[0] = 0.02                         # $6: under the minimum
    held[1], new[1] = 0.02, -0.2          # flip: close the dust, open $60 short
    held[2], new[2] = 0.2, 0.0            # close
    held[3], new[3] = 0.2, 0.25           # add $15
    held[4], new[4] = 0.2, 0.22           # $6 more: under the minimum
    held[5], new[5] = 0.3, 0.1            # reduce $60
    legs, skipped = L.plan_legs(held, new, {x: 1.0 for x in c}, 300.0, {})
    assert [(leg.coin, leg.kind) for leg in legs] == [
        (c[1], "close"), (c[2], "close"), (c[5], "reduce"), (c[1], "open"), (c[3], "add")]
    assert {s["coin"] for s in skipped} == {c[0], c[4]}
    assert legs[2].size == pytest.approx(60.0)
    assert legs[3].usd == pytest.approx(60.0) and not legs[3].is_buy
    assert legs[0].is_buy is False        # closing a long sells


# ── the disaster stop ─────────────────────────────────────────────────────

def test_every_position_gets_one_stop_beyond_its_mark(rig):
    rig.run(200)
    assert rig.fake.pos
    for c, szi in rig.fake.pos.items():
        stops = rig.stops(c)
        assert len(stops) == 1, c
        px = float(stops[0]["triggerPx"])
        assert (px < MID) if szi > 0 else (px > MID), c
        assert abs(px - MID) == pytest.approx(MID * L.STOP_PCT / 100, abs=0.06)
        assert float(stops[0]["sz"]) == pytest.approx(abs(szi))


def test_a_coin_that_went_flat_keeps_no_stop_and_is_reported(rig):
    rig.run(200)
    gone = sorted(rig.fake.pos)[0]
    rig.fake.pos.pop(gone)                # stopped out, its stop left behind
    assert rig.stops(gone)
    rig.run(200, minutes=65)              # the hourly check
    assert rig.stops(gone) == [], "a stale stop is left on a flat coin"
    assert gone in rig.run(201)["stopped_out"]


def test_the_hourly_run_restores_a_missing_stop_and_trades_nothing(rig):
    rig.run(200)
    coin = sorted(rig.fake.pos)[0]
    rig.fake.resting = [o for o in rig.fake.resting if o["coin"] != coin]
    sent = len(rig.iocs())
    assert rig.run(200, minutes=65) is None
    assert len(rig.iocs()) == sent
    assert len(rig.stops(coin)) == 1


# ── what stops a pass ─────────────────────────────────────────────────────

def test_a_halt_stops_the_pass_and_leaves_the_day_open(rig):
    rig.halt.set(safety.AUTO, "test")
    with pytest.raises(L.Blocked):
        rig.run(200)
    assert rig.fake.sent == []
    rig.halt.clear()
    assert rig.run(200) is not None


@pytest.mark.parametrize("broken", ["venue", "candles"])
def test_an_unreadable_venue_sends_nothing(rig, broken):
    if broken == "venue":
        rig.fake.fail_info_always = True
    else:
        rig.ro.broken = "candles"
    with pytest.raises(L.Unreadable):
        rig.run(200)
    assert rig.fake.sent == []
    assert not (rig.dir / L.STATE_NAME).exists()


def test_capital_above_equity_refuses_the_first_pass(rig):
    rig.fake.usdc = 100.0
    with pytest.raises(L.Blocked, match="exceeds account equity"):
        rig.run(200)
    assert rig.fake.sent == []


def test_the_capital_is_fixed_for_a_run(rig):
    rig.run(200)
    with pytest.raises(L.Blocked, match="fixed for a run"):
        rig.run(201, capital=CAP + 100)


def test_arming_needs_the_switch_and_a_capital(monkeypatch):
    monkeypatch.setenv("TREND_LIVE_ARMED", "false")
    assert "TREND_LIVE_ARMED" in L.arm_refusal(CAP, dry_run=False)
    assert "TREND_CAPITAL_USD" in L.arm_refusal(0.0, dry_run=True)
    with mock.patch.object(L.config_audit, "dead_keys", return_value=[]):
        assert L.arm_refusal(CAP, dry_run=True) == ""
        monkeypatch.setenv("TREND_LIVE_ARMED", "true")
        assert L.arm_refusal(CAP, dry_run=False) == ""


def test_a_dry_run_sends_and_writes_nothing(rig):
    row = rig.run(200, dry_run=True)
    assert row["dry_run"] and row["legs"]
    assert rig.fake.sent == []
    assert not rig.dir.exists()


# ── accounting and the kill rule ──────────────────────────────────────────

def fill(coin, t, pnl=0.0, fee=0.0, tid=0, side="A", px=1.0, sz=1.0):
    return {"coin": coin, "time": t, "closedPnl": str(pnl), "fee": str(fee),
            "tid": tid, "side": side, "px": str(px), "sz": str(sz)}


def test_the_book_is_valued_from_the_venues_fills_and_funding_counted_once(rig):
    t1 = now_for(200) + 1_000
    rig.fake.fills_override = [fill("BTC", t1, fee=0.05, tid=1),
                               fill("ETH", t1, fee=0.05, tid=2),
                               fill("ARB", t1, pnl=99.0, fee=1.0, tid=3)]  # not this book's
    paid_at = now_for(200)                # inside the first pass's window
    rig.ro.funding = [{"time": paid_at, "delta": {"coin": "BTC", "usdc": "-0.02"}},
                      {"time": paid_at, "delta": {"coin": "ARB", "usdc": "-5"}}]
    rig.fake.unrealized = {c: 0.1 for c in P.COINS}
    a = rig.run(200)["accounting"]
    held = len(rig.fake.pos)
    assert a["value"] == pytest.approx(CAP - 0.10 - 0.02 + 0.1 * held)

    t2 = now_for(201) + 1_000
    rig.fake.fills_override.append(fill("SOL", t2, pnl=2.0, fee=0.03, tid=4))
    rig.ro.funding.append({"time": paid_at, "delta": {"coin": "BTC", "usdc": "-0.02"}})
    a = rig.run(201)["accounting"]
    held = len(rig.fake.pos)
    assert a["value"] == pytest.approx(CAP + 2.0 - 0.13 - 0.02 + 0.1 * held), \
        "a fill or a funding payment was counted twice"


def test_k1_closes_the_book_and_stops_it(rig):
    rig.run(200)
    held = [c for c in rig.fake.pos]
    rig.fake.unrealized = {c: -(0.31 * CAP) / len(held) - 1.0 for c in P.COINS}
    row = rig.run(201)
    assert row["killed"] and "K1" in row["killed"]["reason"]
    assert not set(rig.fake.pos) & set(P.COINS), "the book was not closed"
    with pytest.raises(L.Blocked, match="stopped"):
        rig.run(202)


def test_status_reads_the_run_against_its_rules():
    def row(day, value, peak, funding, ex=True):
        return {"day": T.iso(T0 + day * DAY), "capital": CAP, "legs": [{"filled": 1.0}],
                "accounting": {"value": value, "peak": peak, "funding": funding,
                               "drawdown_pct": (value / peak - 1) * 100},
                "executions": {"BTC": {"notional": 100.0, "fees": 0.045,
                                       "adverse_bps": 6.0}} if ex else {}}
    rows = [row(0, 300, 300, -0.01), row(1, 310, 310, -0.01), row(2, 200, 310, -0.01)]
    st = L.status(rows, T0 + 2 * DAY)
    assert st["worst_drawdown_pct"] == pytest.approx(-35.48, abs=0.01)
    assert st["kill"]["K1_drawdown"]
    assert st["mean_adverse_bps"] == pytest.approx(6.0)
    assert st["mean_fee_bps"] == pytest.approx(4.5)
    assert st["review_on_day_90"]["R2_cost"]
    assert st["funding_paid_pct_year"] == pytest.approx(0.03 / CAP / 2 * 365 * 100)
    assert L.status([], T0) == {"started": False}


# ── who owns a coin ───────────────────────────────────────────────────────

def test_the_trend_coins_are_the_backtests_coins():
    assert books.TREND_COINS == frozenset(T.COINS) == frozenset(L.COINS)


def test_unarmed_nothing_is_reserved(monkeypatch):
    monkeypatch.setenv("TREND_LIVE_ARMED", "false")
    assert books.band_excludes() == frozenset()
    assert not books.trend_owns("BTC", None)


def test_armed_a_band_plan_keeps_a_coin_the_bands():
    assert books.band_excludes() == books.TREND_COINS
    assert books.trend_owns("btc", None)
    assert not books.trend_owns("BTC", TradeMeta(reason="level", planned_stop=1.0))
    assert not books.trend_owns("ARB", None)


def test_protect_leaves_the_trend_books_positions_to_it(tmp_path):
    fake = FakeHL(positions={"BTC": 2.0, "ETH": 1.0, "ARB": 3.0}, mid=100.0)
    store = MetaStore(tmp_path / "meta.json")
    store.put("ETH", TradeMeta(reason="level", planned_stop=90.0,
                               opened_at="2026-10-17T00:00:00Z"))
    res = {r.coin: r.action for r in protect.run_once(build_venue(fake), store)}
    assert "BTC" not in res, "protect managed a trend position"
    assert store.get("BTC") is None, "a trend position was adopted"
    assert not [o for o in fake.resting if o["coin"] == "BTC"]
    assert res["ETH"] == "placed", "a band trade in a trend coin lost its stop"
    assert res["ARB"] == "placed"


def test_unarmed_protect_adopts_as_before(tmp_path, monkeypatch):
    monkeypatch.setenv("TREND_LIVE_ARMED", "false")
    fake = FakeHL(positions={"BTC": 2.0}, mid=100.0)
    res = protect.run_once(build_venue(fake), MetaStore(tmp_path / "meta.json"))
    assert [r.action for r in res] == ["placed"]


def test_the_band_trader_does_not_count_the_trend_books_slots(tmp_path):
    fake = FakeHL(positions={"BTC": 1.0, "ETH": 1.0, "SOL": 1.0, "XRP": 1.0}, mid=100.0)
    fake.fills_override = []
    v = build_venue(fake)
    state = {"opened_today": 0}
    halt = safety.HaltState(tmp_path / "halt.json")
    with mock.patch.object(safety.HaltState, "active", return_value=None), \
         mock.patch.object(safety, "enforce_limits", return_value=[]):
        assert TR.preflight(v, state, halt, None) is None
        held, bids = TR.slots_in_use(v)
        assert held == {}
        with mock.patch.dict("os.environ", {"TREND_LIVE_ARMED": "false"}):
            assert "holding 4" in (TR.preflight(v, state, halt, None) or "")


def test_the_band_trader_announces_no_trend_close():
    now = int(time.time() * 1000)
    f = [{"coin": "BTC", "time": now - 5 * 3_600_000, "startPosition": "0", "side": "B",
          "sz": "1", "closedPnl": "0", "fee": "0.1", "tid": 1},
         {"coin": "BTC", "time": now - 4 * 3_600_000, "startPosition": "1", "side": "A",
          "sz": "1", "closedPnl": "-2", "fee": "0.1", "tid": 2}]
    fake = FakeHL()
    fake.fills_override = f
    with mock.patch.object(TR, "save_state"), mock.patch.object(TR.notify, "send") as send:
        state = {"announced_close_ms": now - 6 * 3_600_000}
        assert TR.announce_closes(build_venue(fake), state) == 0
        assert not send.called
        with mock.patch.dict("os.environ", {"TREND_LIVE_ARMED": "false"}):
            state = {"announced_close_ms": now - 6 * 3_600_000}
            assert TR.announce_closes(build_venue(fake), state) == 1, \
                "the same trade unarmed is announced; the test proves nothing"


def test_a_trend_coin_cannot_be_bid_by_hand():
    assert TR.rest_one_bid(None, None, None, [], "BTC", "100", 1000.0, 1000.0) == 1


def test_the_trend_books_losses_are_not_the_bands_streak():
    now = int(time.time() * 1000)
    fills = []
    for i in range(3):
        closed = now - 60_000 - (2 - i) * 3_600_000
        fills += [{"coin": "BTC", "time": closed - 1000, "side": "B", "sz": "1",
                   "startPosition": "0", "closedPnl": "0", "fee": "0", "tid": 2 * i},
                  {"coin": "BTC", "time": closed, "side": "A", "sz": "1",
                   "startPosition": "1", "closedPnl": "-1", "fee": "0", "tid": 2 * i + 1}]
    fake = FakeHL()
    fake.fills_override = fills
    v = build_venue(fake)
    assert streak.current(v, now_ms=now) is not None
    assert streak.current(v, now_ms=now, exclude=books.TREND_COINS) is None


def test_the_watchdog_watches_the_trend_pass_only_while_armed(monkeypatch):
    assert "trendlive" in liveness.services()
    monkeypatch.setenv("TREND_LIVE_ARMED", "false")
    assert "trendlive" not in liveness.services()


def test_the_ledger_is_plain_json(rig):
    rig.run(200)
    line = (rig.dir / P.LEDGER_NAME).read_text().splitlines()[0]
    assert json.loads(line)["capital"] == CAP

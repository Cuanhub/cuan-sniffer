"""
Tests for the ARMED path of tools/live_levels_trader.

This file exists because the armed path had no coverage at all, and that is
the direct cause of both bugs that reached live money on 2026-09-18:

  1. execute_one, reconcile and preflight were defined after the
     `if __name__ == "__main__"` guard, so main() raised NameError the first
     time it was armed. Dry runs never call them, so nothing failed earlier.
  2. execute_one read success from a "filled_size_coin" attribute FillResult
     has never had. getattr returned 0, every real fill scored as a rejection,
     state["owned"] stayed empty, MAX_CONCURRENT never engaged, and one pass
     opened 12 positions across $5,454 of notional on $775 of equity with no
     stops on any of them.

Both were reachable only with real orders going out. The fix for the code is
in the module; the fix for the process is here -- a fake backend that returns
the real FillResult shape, so the armed path can be exercised without a venue.
"""

import shutil
import unittest
from unittest import mock
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import tools.live_levels_trader as T
import inspect
import json
import tempfile
import time

from core import execute, safety
from core.meta import MetaStore
from tests.fake_venue import FakeHL, build_venue


class FakeCandles:
    """1h bars that satisfy the trigger: price dipped to the level, then held
    above it. Tests that want the trigger to fail override `held`."""

    def __init__(self, level=100.0, held=True, touched=True):
        self.level, self.held, self.touched = level, held, touched

    def get_candles(self, coin, interval, start, end):
        lv = self.level
        bars = []
        for i in range(40):
            # Closes must vary or log_sigma is 0 and the trigger has no scale
            # to measure the zone in. They must also sit CLOSE to the level in
            # sigma terms, since holding means holding in the zone, not merely
            # somewhere above it.
            # When the level was never reached, the whole series sits higher:
            # price that has not come down is also not holding in the zone, so
            # either rejection is the correct one.
            # 1.002 puts the last closed bar ~0.77 sigma above the level.
            # It was 1.004 (~1.02 sigma), which fitted under the old 2.0
            # ceiling and no longer fits the 1.0 one -- the fixture has to
            # hold in the zone the trigger actually accepts.
            base = 1.002 if self.touched else 1.035
            c = lv * (base + 0.004 * ((i % 5) - 2))
            low = lv * 0.999 if (self.touched and i == 35) else c * 0.998
            bars.append({"t": i * 3600_000, "o": c, "h": c * 1.002, "l": low, "c": c})
        if not self.held:
            bars[-2]["c"] = lv * 0.98
            bars[-3]["c"] = lv * 0.98
        return bars


@dataclass
class FakeFill:
    """Mirrors execution_backend.FillResult -- the real contract."""
    filled: bool
    fill_price: float
    fill_size_usd: float
    slippage_bps: float = 0.0
    reject_reason: str = ""
    fill_ratio: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)


class FakeBackend:
    def __init__(self, equity=1000.0, protection_ok=True, fill=True, mid=None,
                 exit_fills=True, protection_oids=False, realised_pnl=0.0,
                 bid_ok=True, venue_max_lev=10):
        self.equity = equity
        self.mid = mid
        self.protection_ok = protection_ok
        self.fill = fill
        self.entries = []
        self.protections = []
        self.exits = []
        self.cancels = []
        self.bids = []
        self.bid_ok = bid_ok
        self.leverage_set = []
        self.signed = None
        self.venue_max_lev = venue_max_lev
        self.pnl_queries = []
        self.realised_pnl = realised_pnl   # None models a failed query
        self.exit_fills = exit_fills
        self.protection_oids = protection_oids
        self.venue = {}          # coin -> notional
        self.exit_partial = False  # True = order lands, venue still holds size
        self.info = self

    # -- venue reads -------------------------------------------------------
    def get_margin_summary(self):
        return (self.equity, 0.0, self.equity)

    def _get_user_state_address(self):
        return "0xfake"

    def mid(self, coin):
        # default: sitting right at the planned entry
        return self.mid if self.mid is not None else 100.0

    def user_state(self, addr):
        return {"assetPositions": [
            {"position": {"coin": c, "szi": "1", "positionValue": str(v)}}
            for c, v in self.venue.items()]}

    def _get_venue_signed_size(self, coin):
        # `signed` lets a test model a PARTIAL close: the coin is still held,
        # but at a smaller size than the module believes.
        if coin not in self.venue:
            return 0.0
        return self.signed if self.signed is not None else 1.0

    # -- venue writes ------------------------------------------------------
    def execute_entry(self, coin, side, price, size_usd, confidence):
        self.entries.append(coin)
        if not self.fill:
            return FakeFill(filled=False, fill_price=0.0, fill_size_usd=0.0,
                            reject_reason="no_fill")
        self.venue[coin] = size_usd
        return FakeFill(filled=True, fill_price=price, fill_size_usd=size_usd,
                        fill_ratio=1.0)

    def place_native_protection(self, coin, side, size_coin, stop_price, tp_price):
        self.protections.append(coin)
        if not self.protection_ok:
            if self.protection_oids:
                # The shape that actually needs cleaning up: the venue ACCEPTED
                # an order and then described it back as something we did not
                # ask for, so a live OID exists despite the failed status.
                return {"status": "partial", "error": "verify_fail:not_reduce_only",
                        "stop_order_id": "s1", "tp_order_id": ""}
            return {"status": "missing", "error": "rejected"}
        # The real backend returns "protected" on success, never "ok".
        # The fake said "ok" and so agreed with the bug instead of catching it.
        return {"status": "protected", "stop_order_id": "s1", "tp_order_id": "t1"}

    def get_realised_pnl(self, coin, since_ms):
        self.pnl_queries.append((coin, since_ms))
        return self.realised_pnl

    def place_limit_entry(self, coin, side, price, size_usd):
        self.bids.append((coin, price, size_usd))
        if not self.bid_ok:
            return {"oid": "", "error": "rejected", "price": 0.0, "size_coin": 0.0}
        return {"oid": "B1", "price": price, "size_coin": size_usd / price,
                "error": ""}

    def max_leverage_for(self, coin):
        return self.venue_max_lev

    def set_leverage(self, coin, leverage, is_cross=True):
        self.leverage_set.append((coin, leverage))
        return True

    def cancel_order(self, coin, oid):
        self.cancels.append((coin, oid))
        return True

    def execute_exit(self, coin, side, price, size_usd, reason):
        self.exits.append(coin)
        if not self.exit_fills:
            return FakeFill(filled=False, fill_price=0.0, fill_size_usd=0.0,
                            reject_reason="exit_rejected")
        self.venue.pop(coin, None)
        return FakeFill(filled=True, fill_price=price, fill_size_usd=size_usd)

    def flatten(self, coin, side="LONG", reason="flatten", attempts=3):
        """Mirrors LiveExecutionBackend.flatten's contract.

        `exit_partial` models the 2026-10-17 case the real one exists for: the
        order goes through, the venue still holds size, and the caller must
        NOT be told it is flat. The old code sent one execute_exit and read
        `filled`, which is True for a partial, so a half-closed position
        passed the check and the slot was released anyway.
        """
        self.exits.append(coin)
        notional = self.venue.get(coin, 0.0)
        if not self.exit_fills:
            return {"flat": False, "attempts_used": attempts,
                    "closed_usd": 0.0, "detail": "exit_rejected"}
        if getattr(self, "exit_partial", False):
            return {"flat": False, "attempts_used": attempts,
                    "closed_usd": notional * 0.5,
                    "detail": f"STILL OPEN: venue size {self.signed}"}
        self.venue.pop(coin, None)
        return {"flat": True, "attempts_used": 1, "closed_usd": notional,
                "detail": "filled 100%"}


def episode(coin, entry=100.0, stop=95.0, tp=115.0):
    return {"coin": coin, "episode_id": f"{coin}_1_{entry}", "level_price": entry,
            "suggested_entry": entry, "suggested_stop": stop, "suggested_tp": tp,
            "suggested_rr": 3.0, "level_hold_rate": 1.0, "level_touches": 10,
            "near_sigma": 7.0, "level_origin": "swing_low"}




def _halt():
    """A HaltState in a throwaway file, so a test cannot halt the real bot."""
    return safety.HaltState(Path(tempfile.mkdtemp()) / "halt.json")


def _eqlog():
    return safety.EquityLog(Path(tempfile.mkdtemp()) / "eq.json")


def _venue(mid=100.0, **kw):
    """A real core.venue.Venue over the recorded-payload fake.

    Replaces the old FakeBackend, which mimicked LiveExecutionBackend's reply
    shapes. Nothing reads a reply shape any more, so the double only has to
    hold positions and orders.
    """
    fake = FakeHL(mid=mid, usdc=kw.pop("usdc", 1000.0), **kw)
    execute.SETTLE_SEC = 0.0
    safety.FLATTEN_PAUSE_SEC = 0.0
    return build_venue(fake)


class ArmedPathTest(unittest.TestCase):
    """The armed entry path, over core/.

    Rewritten for the execution rewrite, intent-for-intent. The old harness
    drove a FakeBackend that mimicked LiveExecutionBackend's response shapes;
    this one drives the recorded-payload fake, so what the tests assert about
    is what the venue HOLDS rather than what a reply said.
    """

    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp(prefix="armed_path_"))
        self._orig = (T.STATE_PATH, T.TRADE_LOG, T._RO_CLIENT,
                      execute.SETTLE_SEC, safety.FLATTEN_PAUSE_SEC)
        T.STATE_PATH = self._tmp / "state.json"
        T.TRADE_LOG = self._tmp / "trades.csv"
        T._RO_CLIENT = FakeCandles()
        execute.SETTLE_SEC = 0.0
        safety.FLATTEN_PAUSE_SEC = 0.0
        self.state = {"opened_today": 0, "day": "d"}
        self.meta = MetaStore(self._tmp / "meta.json")
        self.fake = FakeHL(mid=100.0, usdc=1000.0)
        self.venue = build_venue(self.fake)
        self.sent = []
        self._orig_send = T.notify.send
        T.notify.send = lambda text, **kw: (self.sent.append(text), True)[1]

    def tearDown(self):
        T.notify.send = self._orig_send
        (T.STATE_PATH, T.TRADE_LOG, T._RO_CLIENT, execute.SETTLE_SEC,
         safety.FLATTEN_PAUSE_SEC) = self._orig
        shutil.rmtree(self._tmp, ignore_errors=True)

    def fire(self, coin="HYPE", **kw):
        return T.execute_one(self.venue, self.meta, episode(coin, **kw),
                             1000.0, 1000.0, True, self.state)

    def stops(self, coin):
        return [o for o in self.fake.resting
                if o["coin"] == coin and o["orderType"].startswith("Stop")]

    # -- the entry lands, and carries a stop ------------------------------
    def test_successful_entry_is_recorded_and_protected(self):
        self.assertTrue(self.fire())
        # MARGIN_PCT_PER_TRADE of equity, levered FLAT_LEVERAGE times, at a
        # $100 mid -- not the whole $1000 of equity.
        expected = (1000.0 * T.MARGIN_PCT_PER_TRADE / 100.0
                    * T.FLAT_LEVERAGE / 100.0)
        self.assertAlmostEqual(self.fake.pos["HYPE"], expected, places=6)
        self.assertIsNotNone(self.meta.get("HYPE"), "the plan must be recorded")
        self.assertEqual(len(self.stops("HYPE")), 1,
                         "protection must follow every entry")

    def test_the_stop_is_sized_to_what_actually_filled(self):
        """A partial fill used to be protected for the size the code BELIEVED
        it filled, with no way to correct it afterwards."""
        self.fake.fill_fraction = 0.6
        self.assertTrue(self.fire())
        self.assertAlmostEqual(float(self.stops("HYPE")[0]["sz"]),
                               self.fake.pos["HYPE"], places=6)

    def test_concurrency_cannot_be_uncapped_by_a_parsing_bug(self):
        """The 2026-09-18 runaway, structurally prevented.

        The cap was counted from state["owned"]; one parsing bug emptied it
        and a single pass opened twelve positions with no stops. There is now
        nothing to parse and nothing to empty -- the cap reads the venue.
        """
        code = "\n".join(l.split("#")[0] for l in
                          inspect.getsource(T.main).splitlines())
        # The pass counts through slots_in_use, which reads both from the venue.
        self.assertIn("slots_in_use(venue)", code)
        reader = inspect.getsource(T.slots_in_use)
        self.assertIn("venue.positions()", reader)
        self.assertIn("venue.open_orders()", reader)
        self.assertNotIn('state.get("owned"', code)
        self.assertNotIn('state["owned"]', code)
        self.assertNotIn('state["pending"]', code)

    def test_rejected_entry_drops_the_plan(self):
        self.fake.order_fills = False
        out = self.fire()
        self.assertFalse(out)
        self.assertIsNone(self.meta.get("HYPE"),
                          "a plan with no position behind it must not linger")
        self.assertEqual(self.stops("HYPE"), [])

    def test_an_unknown_result_keeps_the_plan_for_the_protection_loop(self):
        """Neither a fill nor a reject. The old code recorded a reject and
        released the slot, leaving a live unprotected position untracked."""
        real = self.fake.order

        def order_then_blind(*a, **k):
            r = real(*a, **k)
            self.fake.fail_info_always = True
            return r

        self.fake.order = order_then_blind
        out = self.fire()
        self.assertTrue(out.order_sent)
        self.assertIsNotNone(self.meta.get("HYPE"),
                             "the plan must survive so the loop can use it")
        self.assertTrue(any("result unknown" in s for s in self.sent))

    # -- protection failure closes the position ---------------------------
    def test_protection_failure_closes_the_position(self):
        real = self.fake.order

        def refuse_triggers(name, is_buy, sz, px, order_type, **k):
            if "trigger" in order_type:
                return {"status": "ok"}          # accepted, nothing rests
            return real(name, is_buy, sz, px, order_type, **k)

        self.fake.order = refuse_triggers
        self.fire()
        self.assertEqual(self.fake.pos, {},
                         "an unprotected position is not the trade")
        self.assertTrue(any("could not be protected" in s for s in self.sent))

    # -- what the caller is told ------------------------------------------
    def test_a_filled_entry_reports_an_order(self):
        self.assertTrue(self.fire().order_sent)

    def test_a_rejected_entry_still_reports_an_order(self):
        self.fake.order_fills = False
        self.assertTrue(self.fire().order_sent,
                        "an order reached the venue even though it did not fill")

    def test_dry_run_sends_nothing(self):
        T.execute_one(self.venue, self.meta, episode("HYPE"), 1000.0, 1000.0,
                      False, self.state)
        self.assertEqual(self.fake.sent, [])
        self.assertEqual(self.fake.pos, {})

    def test_leverage_is_set_on_the_venue_before_the_order(self):
        """Hyperliquid enforces leverage per asset: a 5x-sized order into a
        coin set to 3x is refused for margin."""
        seen = []
        self.fake.update_leverage = lambda lev, name, is_cross=True: (
            seen.append((name, lev)), {"status": "ok"})[1]
        self.fire()
        self.assertTrue(seen, "leverage was never set")
        self.assertEqual(seen[0][0], "HYPE")

    def test_an_open_position_is_announced_with_its_protection(self):
        self.fire()
        blob = " ".join(self.sent)
        self.assertIn("HYPE", blob)
        self.assertTrue(any("OPENED" in s or "stop" in s for s in self.sent))

    # -- the watchboard ---------------------------------------------------
    def test_a_fired_trigger_is_not_on_the_watchboard(self):
        board = []
        T.execute_one(self.venue, self.meta, episode("HYPE"), 1000.0, 1000.0,
                      True, self.state, watchboard=board)
        self.assertEqual(board, [], "a trade that fired is not a near miss")

    # -- bar caching ------------------------------------------------------
    def test_no_cache_still_fetches(self):
        self.assertTrue(self.fire().order_sent)




class EntryDriftTest(unittest.TestCase):
    """The UNI trade: planned entry 7.50, filled 8.8125, stop distance 5.66%
    -> 19.7%, reward/risk 3.27 -> 0.04, risk $23 -> $80. execute_entry sends a
    marketable IOC at the CURRENT price and ignores the limit it is passed, so
    nothing downstream noticed the plan no longer applied."""

    def setUp(self):
        self._tmp = Path("/tmp/_test_live_drift")
        if self._tmp.exists():
            shutil.rmtree(self._tmp)
        self._tmp.mkdir(parents=True)
        self._o = (T.STATE_PATH, T.TRADE_LOG, T._RO_CLIENT)
        T.STATE_PATH = self._tmp / "s.json"
        T.TRADE_LOG = self._tmp / "t.csv"
        T._RO_CLIENT = FakeCandles()
        self.state = {"opened_today": 0, "day": "d"}
        self.meta = MetaStore(self._tmp / "meta.json")

    def tearDown(self):
        T.STATE_PATH, T.TRADE_LOG, T._RO_CLIENT = self._o
        if self._tmp.exists():
            shutil.rmtree(self._tmp)

    @staticmethod
    def orders_sent(venue):
        """Entry orders the venue was asked for -- triggers are protection."""
        return [o for o in venue.exchange.sent if "trigger" not in o["type"]]

    def test_refuses_to_fire_when_price_drifted_from_the_plan(self):
        b = _venue(mid=117.5)          # +17.5%, the UNI case
        ok = T.execute_one(b, self.meta, episode("UNI"), 1000.0, 1000.0, True, self.state)
        self.assertFalse(ok)
        self.assertEqual(self.orders_sent(b), [],
                         "no order may be sent when the plan no longer applies")

    def test_fires_when_price_is_at_the_level(self):
        b = _venue(mid=100.0)
        self.assertTrue(T.execute_one(b, self.meta, episode("AAA"), 1000.0, 1000.0, True, self.state))
        self.assertEqual([o["coin"] for o in self.orders_sent(b)], ["AAA"])

    def test_refuses_when_price_is_already_through_the_stop(self):
        b = _venue(mid=94.0)           # below the 95 stop
        self.assertFalse(T.execute_one(b, self.meta, episode("AAA"), 1000.0, 1000.0, True, self.state))
        self.assertEqual(self.orders_sent(b), [])

    def test_refuses_when_live_reward_risk_has_collapsed(self):
        b = _venue(mid=100.9)          # inside drift, but rr now poor
        self.assertFalse(T.execute_one(b, self.meta, episode("AAA", 100.0, 95.0, 101.0),
                                       1000.0, 1000.0, True, self.state))
        self.assertEqual(self.orders_sent(b), [])

    def test_size_comes_from_the_live_price_not_the_plan(self):
        """The UNI lesson: sizing off the plan while filling at market is what
        made that position carry 3.5x its intended risk.

        Asserted against the ORDER now rather than a state record: the size
        that matters is the one the venue was asked for, and the position it
        produced -- `state["owned"]` no longer exists to read it back from.
        """
        b = _venue(mid=101.0)          # within drift
        T.execute_one(b, self.meta, episode("AAA"), 1000.0, 1000.0, True,
                      self.state)
        sent = self.orders_sent(b)
        self.assertEqual(len(sent), 1)
        notional = 1000.0 * T.MARGIN_PCT_PER_TRADE / 100.0 * T.FLAT_LEVERAGE
        # Through the venue's OWN rounding, not round(). round_size uses
        # ROUND_DOWN deliberately -- a position must never size UP -- so a
        # nearest-rounded expectation differs by up to one tick. This passed
        # on tolerance while the precision default was 3 (a 0.005 gap at
        # places=2) and failed at the coin's real 2 decimals (0.01). The claim
        # under test is "from the LIVE price, not the plan", so the live price
        # is what belongs in the expectation; the precision is the venue's.
        want = b.round_size("AAA", notional / 101.0)
        self.assertEqual(sent[0]["sz"], want,
                         msg="size must come from the LIVE price")
        assert want != b.round_size("AAA", notional / episode("AAA")["suggested_entry"]), \
            "the planned and live prices round to the same size; this test " \
            "cannot tell them apart"


class DailyCapTest(unittest.TestCase):
    """MAX_NEW_PER_DAY was 0, and 0 means the rail does not exist.

    preflight guards it with `if MAX_NEW_PER_DAY and ...`, so the disabled
    state is silent -- the startup banner prints "unlimited" and nothing else
    ever mentions it.
    """

    def state(self, opened):
        return {"owned": {}, "opened_today": opened, "day": "2026-09-18",
                "realised_today": 0.0, "halted": False, "halt_reason": ""}

    def test_cap_is_enabled(self):
        self.assertGreater(T.MAX_NEW_PER_DAY, 0,
                           "0 disables the daily cap entirely")

    def test_refuses_once_the_cap_is_reached(self):
        reason = T.preflight(_venue(), self.state(T.MAX_NEW_PER_DAY),
                             _halt(), _eqlog())
        self.assertIsNotNone(reason)
        self.assertIn("today", reason)

    def test_allows_trading_below_the_cap(self):
        self.assertIsNone(T.preflight(_venue(),
                                      self.state(T.MAX_NEW_PER_DAY - 1),
                                      _halt(), _eqlog()))


def gated(coin="AAA", level=100.0, near=7.0, rest=20, overhead=7.0, rr=3.0):
    """A gated episode as call.find_candidates emits it."""
    return {"coin": coin, "level_price": level, "episode_id": f"{coin}-1H_1_{level}",
            "near_sigma": near, "level_bars_since_last_touch": rest,
            "overhead_sigma": overhead, "suggested_rr": rr,
            "suggested_entry": level, "suggested_stop": level * 0.95,
            "suggested_tp": level * 1.15, "level_hold_rate": 1.0,
            "level_touches": 10, "level_origin": "swing_low"}


class OutcomeTest(unittest.TestCase):
    """venue_sanity must run after every ORDER, not every candidate."""

    def test_truthiness_tracks_opened(self):
        self.assertTrue(T.Outcome(opened=True, order_sent=True))
        self.assertFalse(T.Outcome(opened=False, order_sent=True))
        self.assertFalse(T.Outcome())


class BackendStartTest(unittest.TestCase):
    """Info.__init__ fetches perp metadata, so construction hits the network.

    A 429 there raises before any of our code runs, and twice on 2026-09-19 it
    killed the process at startup. launchd restarts it, but a crash loop
    against a rate limit is the worst way to wait: every restart is another
    request, which is what caused the limit.
    """

    def test_a_clean_start_does_not_retry(self):
        calls = []
        b = T.start_venue(lambda debug: calls.append(1) or "venue",
                            sleep=lambda s: None)
        self.assertEqual(b, "venue")
        self.assertEqual(len(calls), 1)

    def test_a_transient_failure_is_retried(self):
        calls = []

        def flaky(debug):
            calls.append(1)
            if len(calls) < 3:
                raise RuntimeError("429 rate limited")
            return "venue"

        self.assertEqual(T.start_venue(flaky, sleep=lambda s: None), "venue")
        self.assertEqual(len(calls), 3)

    def test_a_persistent_failure_raises(self):
        # Coming up without a backend would be an armed trader that cannot
        # trade and does not say so.
        def dead(debug):
            raise RuntimeError("429 rate limited")

        with self.assertRaises(RuntimeError):
            T.start_venue(dead, sleep=lambda s: None)

    def test_a_stop_during_backoff_gives_up_quietly(self):
        def dead(debug):
            raise RuntimeError("429")

        T._stop = True
        try:
            self.assertIsNone(T.start_venue(dead, sleep=lambda s: None))
        finally:
            T._stop = False


class DayRolloverTest(unittest.TestCase):
    """The daily counters reset once, at startup, and never again.

    This trader runs for days under launchd. Across midnight, opened_today
    never cleared -- so after MAX_NEW_PER_DAY fills it would stop opening
    anything until someone restarted it -- and realised_today never cleared,
    so the daily loss stop began summing losses across days and would halt on
    a drawdown that was never a single day's worth.
    """

    def setUp(self):
        self._tmp = Path("/tmp/_test_rollover")
        if self._tmp.exists():
            shutil.rmtree(self._tmp)
        self._tmp.mkdir(parents=True)
        self._orig = T.STATE_PATH
        T.STATE_PATH = self._tmp / "state.json"

    def tearDown(self):
        T.STATE_PATH = self._orig
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_a_new_day_clears_the_counters(self):
        """roll_day owns the daily COUNT and nothing else now.

        It used to reset realised_today as well, and that field is gone: the
        daily loss limit reads equity against a day-start reading recorded by
        core.safety.EquityLog, so there is no running total for a date change
        to clear. The counter that still matters is opened_today.
        """
        st = {"day": "2020-01-01", "opened_today": 5}
        self.assertTrue(T.roll_day(st))
        self.assertEqual(st["opened_today"], 0)
        self.assertNotEqual(st["day"], "2020-01-01")

    def test_the_same_day_is_left_alone(self):
        from datetime import datetime, timezone
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        st = {"day": today, "opened_today": 3, "realised_today": -40.0,
              "owned": {}, "halted": False, "halt_reason": ""}
        self.assertFalse(T.roll_day(st))
        self.assertEqual(st["opened_today"], 3, "must not reset mid-day")
        self.assertEqual(st["realised_today"], -40.0)

    def test_a_halt_survives_the_rollover(self):
        # A halt is a human-cleared condition; midnight is not a human.
        st = {"day": "2020-01-01", "opened_today": 5, "realised_today": -80.0,
              "owned": {}, "halted": True, "halt_reason": "venue mismatch"}
        T.roll_day(st)
        self.assertTrue(st["halted"])
        self.assertEqual(st["halt_reason"], "venue mismatch")

    def test_the_universe_refreshes_on_the_day_roll(self):
        # "top N perps by 24h volume, refreshed daily" -- the trader picked it
        # once at startup and ran on it for the life of the process, which
        # under launchd is days.
        src = Path("tools/live_levels_trader.py").read_text()
        loop = src.index("while not _stop:")
        window = src[loop:loop + 900]
        self.assertIn("roll_day(state)", window)
        self.assertIn("select_universe", window,
                      "a frozen universe trades names that dropped out for "
                      "illiquidity and never sees the ones that came in")

    def test_the_loop_rolls_the_day_not_just_startup(self):
        src = Path("tools/live_levels_trader.py").read_text()
        loop = src.index("while not _stop:")
        # 400 rather than 200: drain_control_queue now runs first, so that a
        # halt queued from Telegram applies on THIS pass. The point of the
        # assertion is that roll_day is called by the loop and not only at
        # startup, which a window this size still pins.
        self.assertIn("roll_day(state)", src[loop:loop + 400],
                      "a process that runs for days must roll the day itself")


class StateFileTest(unittest.TestCase):
    """A corrupt state file is not a new account.

    load_state used to fall through to a fresh empty dict, which is the worst
    available answer: `halted` comes back False, clearing a halt no human
    cleared; `owned` comes back empty, forgetting live positions; `pending`
    comes back empty, orphaning resting bids at the venue. save_state was
    also non-atomic, and this process is SIGKILLed routinely -- so an
    interrupted write produced exactly that corrupt file.
    """

    def setUp(self):
        self._tmp = Path("/tmp/_test_statefile")
        if self._tmp.exists():
            shutil.rmtree(self._tmp)
        self._tmp.mkdir(parents=True)
        self._orig = T.STATE_PATH
        T.STATE_PATH = self._tmp / "state.json"

    def tearDown(self):
        T.STATE_PATH = self._orig
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_no_file_is_a_first_run(self):
        st = T.load_state()
        self.assertNotIn("owned", st, "a fresh state must not claim to know "
                                      "what the account holds")
        self.assertNotIn("pending", st)
        # No watchlist: it went with the level scan. A band is in reach
        # this pass or it is not, and one price has not reached has a resting
        # bid at the venue rather than an entry in this file.
        self.assertNotIn("watching", st)
        # The halt is not in here either: it lives in core.safety, so a
        # corrupt or missing state file cannot clear one.
        self.assertNotIn("halted", st)

    def test_a_corrupt_file_refuses_to_start(self):
        T.STATE_PATH.write_text('{"owned": {"AAA": ')      # truncated write
        with self.assertRaises(SystemExit):
            T.load_state()

    def test_a_non_object_file_refuses_to_start(self):
        T.STATE_PATH.write_text('["not", "an", "object"]')
        with self.assertRaises(SystemExit):
            T.load_state()

    def test_a_halt_survives_a_restart(self):
        """The requirement is unchanged; its home moved.

        A halt used to live in the state file, which is why a corrupt or
        truncated write could clear one. It is its own small file now, so a
        halt survives a restart independently of anything the strategy
        writes -- and the strategy's state file cannot clear it.
        """
        d = Path(tempfile.mkdtemp())
        safety.HaltState(d / "halt.json").set(safety.MANUAL, "venue mismatch")
        reread = safety.HaltState(d / "halt.json").active()   # a new process
        self.assertIsNotNone(reread)
        self.assertEqual(reread["reason"], "venue mismatch")
        self.assertEqual(reread["kind"], "manual")

    def test_the_strategy_state_cannot_express_a_halt(self):
        T.save_state({"watching": {}, "halted": True, "halt_reason": "nope"})
        self.assertNotIn("halted", T.load_state(),
                         "a stale flag here reads like a live halt")

    def test_the_write_is_atomic(self):
        # tmp-then-replace, so a reader sees the old file or the new one,
        # never a fragment.
        T.save_state({"owned": {}, "halted": False})
        self.assertFalse((self._tmp / "state.json.tmp").exists(),
                         "the temp file must be renamed away, not left behind")
        self.assertTrue(T.STATE_PATH.exists())


class FlatSizingTest(unittest.TestCase):
    """Fixed margin, fixed leverage: the position is the same every time.

    MARGIN_PCT_PER_TRADE of equity is committed as margin and levered
    FLAT_LEVERAGE times. 3% at 5x on a $1000 account is $30 of margin
    controlling $150 of notional -- not $5000, which is what equity x leverage
    would mean and would put the whole account behind one trade.
    """

    EQ = 1000.0

    def size(self, entry, stop, vmax=None):
        return T.size_from_stop(self.EQ, entry, stop, available=self.EQ,
                                venue_max_leverage=vmax)

    def expected(self, lev=None):
        lev = T.FLAT_LEVERAGE if lev is None else lev
        return self.EQ * T.MARGIN_PCT_PER_TRADE / 100.0 * lev

    def test_margin_is_a_share_of_equity_not_the_whole_of_it(self):
        d = self.size(100.0, 99.0)
        self.assertAlmostEqual(d["margin_usd"],
                               self.EQ * T.MARGIN_PCT_PER_TRADE / 100.0)
        self.assertAlmostEqual(d["size_usd"], self.expected())
        self.assertLess(d["size_usd"], self.EQ,
                        "a test-sized position must be smaller than the account")

    def test_size_is_the_same_whatever_the_stop(self):
        tight = self.size(100.0, 99.0)      # 1% stop
        wide = self.size(100.0, 87.0)       # 13% stop
        self.assertAlmostEqual(tight["size_usd"], wide["size_usd"])

    def test_risk_moves_with_the_stop_and_is_reported(self):
        tight = self.size(100.0, 99.0)
        wide = self.size(100.0, 87.0)
        self.assertLess(tight["risk_usd"], wide["risk_usd"])
        self.assertAlmostEqual(wide["risk_usd"] / tight["risk_usd"], 13.0, places=1)

    def test_the_venue_ceiling_is_respected(self):
        # 130 of the venue's 234 perps cap at 3x, PONS and CASHCAT among them.
        d = self.size(100.0, 99.0, vmax=3)
        self.assertAlmostEqual(d["size_usd"], self.expected(3))
        self.assertEqual(d["capped"], 1.0)

    def test_a_generous_venue_ceiling_does_not_raise_the_flat_rate(self):
        d = self.size(100.0, 99.0, vmax=40)
        self.assertAlmostEqual(d["size_usd"], self.expected())
        self.assertEqual(d["capped"], 0.0)

    def test_an_inverted_stop_sizes_to_nothing(self):
        self.assertEqual(self.size(100.0, 100.0)["size_usd"], 0.0)
        self.assertEqual(self.size(100.0, 101.0)["size_usd"], 0.0)


class ModuleShapeTest(unittest.TestCase):
    def test_every_function_main_calls_is_defined_before_the_guard(self):
        """Regression for the NameError on the first arming attempt: functions
        appended after `if __name__ == "__main__"` are invisible to main()."""
        src = Path("tools/live_levels_trader.py").read_text()
        guard = src.index('if __name__ == "__main__"')
        for fn in ("execute_one", "preflight", "rest_one_bid", "start_venue",
                   "place_resting_bid", "size_from_stop",
                   "drain_control_queue"):
            self.assertLess(src.index(f"def {fn}("), guard,
                            f"{fn} is defined after the __main__ guard")

    def test_runtime_paths_are_redirected_under_test(self):
        """conftest must keep every test off the live state and trade log.

        reconcile() saves state and appends a trade row as a matter of
        course, so a test that calls it with a hand-built dict writes to
        research/levels/data/ unless something has redirected the paths.
        Seventy fabricated rows reached the real-money audit trail that way.
        """
        for path in (T.STATE_PATH, T.TRADE_LOG):
            self.assertNotIn("research/levels/data", str(path),
                             "conftest's isolated_runtime_files is not applying")


if __name__ == "__main__":
    unittest.main()

# ---------------------------------------------------------------------------
# Retired with the execution/safety rewrite. Each class below tested a
# function that no longer exists; the behaviour it protected is covered where
# the behaviour now lives, and nothing was dropped silently:
#
#   VenueSanityTest
#       -> protect.run_once adopting an unknown position (test_core_protect: unknown_position_is_adopted)
#   RealisedPnlTest
#       -> equity-based limits (test_core_safety: daily_loss, drawdown) -- realised PnL no longer decides anything
#   HaltAlertTest
#       -> safety.halt (test_core_safety: halting_flattens, alerts_once, auto_clears)
#   RestingBidTest
#       -> execute.rest_bid + protect.expire_bids (test_core_execute, test_core_protect bid tests)
#   AccountModelTest
#       -> venue.equity (test_core_venue: spot_usdc, unified_account_reports_zero)
#   ReProtectionFailureIsFatal
#       -> protect.ensure_stop resizing every cycle (test_core_protect: grew/partly_closed)
#   CancelFailuresAreVisible
#       -> protect.ensure_stop + record_error (test_core_protect: failed_cancel)
#
# One class, BackendStartTest, was KEPT and retargeted at start_venue: the
# retry-on-429-at-startup it protects has explicit evidence behind it (twice
# on 2026-09-19 a rate limit killed the process outright at startup), and the
# first draft of this rewrite had dropped it.
# ---------------------------------------------------------------------------


class RetiredStateKeysAreStripped(unittest.TestCase):
    """A dead key left in the state file is misleading, not harmless.

    `halted: true` sitting in _live_state.json after the halt moved into
    core.safety reads like a live halt to anyone inspecting the file, and
    `owned` reads like a position record on a system whose whole point is
    that it has none.
    """

    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp())
        self._orig = T.STATE_PATH
        T.STATE_PATH = self._tmp / "state.json"

    def tearDown(self):
        T.STATE_PATH = self._orig
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_position_and_safety_keys_are_dropped_on_load(self):
        T.STATE_PATH.write_text(json.dumps({
            "watching": {"a": 1}, "day": "2026-10-17", "opened_today": 2,
            "owned": {"HYPE": {"entry": 1.0}}, "pending": {"ZRO": {}},
            "halted": True, "halt_kind": "manual", "halt_reason": "old",
            "realised_today": -12.0, "equity_samples": [["t", 1.0]],
            "equity_peak": 900.0, "loss_streak": 3, "forced": {"X": "t"}}))
        st = T.load_state()
        for dead in ("owned", "pending", "halted", "halt_kind", "halt_reason",
                     "realised_today", "equity_samples", "equity_peak",
                     "loss_streak", "forced"):
            self.assertNotIn(dead, st, f"{dead} survived the load")
        # and the strategy state it legitimately owns is untouched
        self.assertNotIn("watching", st, "the watchlist is retired")
        self.assertEqual(st["opened_today"], 2)

    def test_a_corrupt_file_is_still_fatal(self):
        """Stripping dead keys must not become 'repair whatever is there'."""
        T.STATE_PATH.write_text('{"watching": ')
        with self.assertRaises(SystemExit):
            T.load_state()


class UniverseLoadRetries(unittest.TestCase):
    """The trader's half of the 2026-10-22 crash loop.

    `select_universe` at startup was unguarded and raised VenueFetchError
    straight out of main() 58 times -- a raw traceback to stderr and a dead
    process each time, restarted by launchd at its 60s ThrottleInterval.
    77KB of identical tracebacks for "DNS is down". The in-loop refresh has
    always been guarded and keeps the previous universe; at startup there is
    no previous one, so the choice is retry or exit.
    """

    def test_a_transient_failure_is_retried(self):
        waits, calls = [], {"n": 0}

        def flaky(ro):
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError("metaAndAssetCtxs failed after 5 retries")
            return [{"symbol": "SOL"}]

        with mock.patch.object(T, "select_universe", flaky):
            got = T.start_universe(object(), sleep=waits.append)
        self.assertEqual(got, [{"symbol": "SOL"}])
        self.assertEqual(calls["n"], 3)
        self.assertGreater(len(waits), 0, "it did not back off")

    def test_it_returns_None_rather_than_raising(self):
        """A raise here is what produced the tracebacks. main() turns None
        into a clean non-zero exit, which is what launchd restarts on."""
        with mock.patch.object(T, "select_universe",
                               lambda ro: (_ for _ in ()).throw(
                                   RuntimeError("NameResolutionError"))):
            got = T.start_universe(object(), sleep=lambda s: None)
        self.assertIsNone(got)

    def test_an_empty_universe_is_a_failure_not_a_universe(self):
        """Trading an empty universe is a silent no-op forever."""
        with mock.patch.object(T, "select_universe", lambda ro: []):
            got = T.start_universe(object(), sleep=lambda s: None)
        self.assertIsNone(got)

    def test_a_first_attempt_success_does_not_wait(self):
        waits = []
        with mock.patch.object(T, "select_universe",
                               lambda ro: [{"symbol": "SOL"}]):
            T.start_universe(object(), sleep=waits.append)
        self.assertEqual(waits, [])

    def test_main_exits_cleanly_when_the_universe_never_loads(self):
        """Non-zero, so KeepAlive SuccessfulExit=false restarts it -- and no
        traceback, which is the difference from the 58 in the err log."""
        src = inspect.getsource(T.main)
        self.assertIn("universe = start_universe(ro)", src)
        self.assertIn("if universe is None:", src)
        self.assertIn("return 1", src)


class AFilledBidAnnouncesItself(unittest.TestCase):
    """The bug the operator reported: no alert when a position is placed.

    The "OPENED" notification lives in execute_one, which is the IMMEDIATE
    entry path -- price already inside the band. The band strategy almost
    never takes that one: on a live 45-coin pass 19 of 21 plans were bids
    waiting. So every position this system actually opened arrived in
    silence, and XRP 135 @ 1.2967 was found open by hand.
    """

    def setUp(self):
        self.store = MetaStore(Path(tempfile.mkdtemp()) / "m.json")
        from core.meta import TradeMeta
        self.store.put("XRP", TradeMeta(
            reason="resting bid at 1.2967", planned_stop=1.2678,
            planned_tp=1.3573, opened_at="2026-10-27T00:00:00Z",
            pending_episode=json.dumps({"coin": "XRP", "episode_id": "z:1"})))

    def venue(self, protected: bool = True):
        """XRP as it was actually found: filled at 1.2967, mark 1.3206."""
        fake = FakeHL(positions={"XRP": 135.0}, mid=1.3206,
                      unrealized={"XRP": 3.23}, entries={"XRP": 1.2967})
        if protected:
            fake.rest_protection("XRP", 135.0, 1.2678, 1.3573)
        return fake, build_venue(fake)

    def test_a_fill_sends_an_alert(self):
        fake, v = self.venue()
        with mock.patch.object(T.notify, "send") as send:
            T.persist_filled_bids(v, self.store)
        self.assertTrue(send.called, "a filled bid sent no alert")
        body = " ".join(str(a) for a in send.call_args[0])
        self.assertIn("XRP", body)
        self.assertIn("135", body)

    def test_the_alert_carries_what_a_chart_check_needs(self):
        fake, v = self.venue()
        with mock.patch.object(T.notify, "send") as send:
            T.persist_filled_bids(v, self.store)
        body = " ".join(str(a) for a in send.call_args[0])
        for want in ("1.2967", "1.2678", "1.3573", "risk", "rr"):
            self.assertIn(want, body, f"{want} missing from: {body}")

    def test_the_numbers_are_the_entry_s_not_the_mark_s(self):
        """Risk is measured from where the fill happened. Measuring it from
        the mark understates a position already in profit and overstates one
        already underwater -- the two cases where it matters most."""
        fake, v = self.venue()
        with mock.patch.object(T.notify, "send") as send:
            T.persist_filled_bids(v, self.store)
        body = " ".join(str(a) for a in send.call_args[0])
        risk = abs(1.2967 - 1.2678) * 135
        self.assertIn(f"risk ${risk:.2f}", body)
        self.assertIn(f"{(1.2967 - 1.2678) / 1.2967 * 100:.2f}% away", body)

    def test_the_alert_reads_the_venue_not_the_plan(self):
        """What is resting is the only thing worth alerting on. A stop moved
        or re-placed at a different price must show the resting price."""
        fake, v = self.venue(protected=False)
        fake.rest_protection("XRP", 135.0, 1.2500, 1.4000)
        with mock.patch.object(T.notify, "send") as send:
            T.persist_filled_bids(v, self.store)
        body = " ".join(str(a) for a in send.call_args[0])
        self.assertIn("1.25", body)
        self.assertIn("1.4", body)
        self.assertNotIn("1.2678", body)

    def test_a_position_with_no_resting_stop_says_so_loudly(self):
        """The plan having a stop is not the same as one being at the venue,
        and that difference is the whole point of this alert."""
        fake, v = self.venue(protected=False)
        with mock.patch.object(T.notify, "send") as send:
            T.persist_filled_bids(v, self.store)
        body = " ".join(str(a) for a in send.call_args[0])
        self.assertIn("PLANNED ONLY", body)

    def test_an_unreadable_book_does_not_claim_the_stop_is_missing(self):
        """A 429 on the order book is not evidence of an unprotected
        position. This is the mistake that cost NEAR $5.44 in protect.py."""
        fake, v = self.venue(protected=False)
        with mock.patch.object(type(v), "open_orders", return_value=None), \
             mock.patch.object(T.notify, "send") as send:
            T.persist_filled_bids(v, self.store)
        self.assertTrue(send.called, "the fill itself still deserves an alert")
        body = " ".join(str(a) for a in send.call_args[0])
        self.assertNotIn("PLANNED ONLY", body)
        self.assertIn("1.2678", body)      # falls back to the plan's stop

    def test_it_fires_once_not_every_pass(self):
        """pending_episode is cleared, and this is the only place that
        clears it -- so the second pass must be silent."""
        fake, v = self.venue()
        with mock.patch.object(T.notify, "send") as send:
            T.persist_filled_bids(v, self.store)
            first = send.call_count
            T.persist_filled_bids(v, self.store)
            self.assertEqual(send.call_count, first,
                             "the alert repeated on the next pass")

    def test_an_alert_failure_never_breaks_the_pass(self):
        fake, v = self.venue()
        with mock.patch.object(T.notify, "send",
                               side_effect=RuntimeError("telegram down")):
            T.persist_filled_bids(v, self.store)      # must not raise


class AClosedTradeAnnouncesItself(unittest.TestCase):
    """The other half of the silence: a stop fires at the venue and no code
    of ours runs, so seven consecutive stop-outs cost $20.31 in the dark."""

    def setUp(self):
        self.state = {}
        # The fake filters fills by `time >= start`, and the venue asks for
        # the last STREAK_LOOKBACK_HOURS, so fixture times have to be real
        # recent ones. An earlier version of this used 1_000_000 and every
        # fill was filtered out before the code under test ever saw it.
        self.now = int(time.time() * 1000)

    def ago(self, hours):
        return self.now - int(hours * 3_600_000)

    @staticmethod
    def fill(coin, t, start, side, sz, pnl=0.0, fee=0.0, tid=1):
        return {"coin": coin, "time": t, "startPosition": str(start),
                "side": side, "sz": str(sz), "closedPnl": str(pnl),
                "fee": str(fee), "tid": tid}

    def venue_with(self, fills):
        fake = FakeHL()
        fake.fills_override = fills
        return fake, build_venue(fake)

    def round_trip(self, coin="XRP", open_h=5.0, close_h=4.0, pnl=-2.80):
        """One round trip that opened `open_h` hours ago and closed
        `close_h` hours ago."""
        return [self.fill(coin, self.ago(open_h), 0, "B", 135, 0.0, 0.06, 1),
                self.fill(coin, self.ago(close_h), 135, "A", 135, pnl, 0.06, 2)]

    def test_the_first_pass_announces_nothing_and_stamps_the_mark(self):
        """A deploy must not fire two weeks of history at the operator."""
        fake, v = self.venue_with(self.round_trip())
        with mock.patch.object(T, "save_state"), \
             mock.patch.object(T.notify, "send") as send:
            n = T.announce_closes(v, self.state)
        self.assertEqual(n, 0)
        self.assertFalse(send.called, "a first run announced history")
        self.assertEqual(self.state["announced_close_ms"], self.ago(4.0))

    def test_a_close_after_the_mark_is_announced_once(self):
        self.state["announced_close_ms"] = self.ago(6.0)
        fake, v = self.venue_with(self.round_trip())
        with mock.patch.object(T, "save_state"), \
             mock.patch.object(T.notify, "send") as send:
            self.assertEqual(T.announce_closes(v, self.state), 1)
            body = " ".join(str(a) for a in send.call_args[0])
            # The second pass sees the same fills and must stay quiet.
            self.assertEqual(T.announce_closes(v, self.state), 0)
        self.assertIn("XRP", body)
        self.assertIn("-2.92", body)      # pnl net of BOTH fees
        self.assertIn("1.0h", body)

    def test_a_win_and_a_loss_do_not_read_the_same(self):
        for pnl, want in ((-2.80, "🔴"), (+5.40, "🎯")):
            self.state["announced_close_ms"] = self.ago(6.0)
            fake, v = self.venue_with(self.round_trip(pnl=pnl))
            with mock.patch.object(T, "save_state"), \
                 mock.patch.object(T.notify, "send") as send:
                T.announce_closes(v, self.state)
            body = " ".join(str(a) for a in send.call_args[0])
            self.assertIn(want, body, f"pnl {pnl} rendered as: {body}")

    def test_an_unreadable_venue_does_not_move_the_mark(self):
        """A 429 must not skip the next close's alert."""
        self.state["announced_close_ms"] = self.ago(6.0)
        fake, v = self.venue_with(self.round_trip())
        fake.fail_info_always = True
        with mock.patch.object(T, "save_state"), \
             mock.patch.object(T.notify, "send") as send:
            self.assertEqual(T.announce_closes(v, self.state), 0)
        self.assertFalse(send.called)
        self.assertEqual(self.state["announced_close_ms"], self.ago(6.0))
        fake.fail_info_always = False
        with mock.patch.object(T, "save_state"), \
             mock.patch.object(T.notify, "send") as send:
            self.assertEqual(T.announce_closes(v, self.state), 1,
                             "the close missed during the outage never fired")

    def test_a_still_open_position_is_not_announced(self):
        self.state["announced_close_ms"] = self.ago(99.0)
        fake, v = self.venue_with([self.fill("XRP", self.ago(5.0), 0, "B", 135)])
        with mock.patch.object(T, "save_state"), \
             mock.patch.object(T.notify, "send") as send:
            self.assertEqual(T.announce_closes(v, self.state), 0)
        self.assertFalse(send.called, "an open position was called closed")

    def test_every_one_of_several_closes_is_announced(self):
        self.state["announced_close_ms"] = self.ago(99.0)
        fills = (self.round_trip("XRP", 9.0, 8.0, -2.80)
                 + self.round_trip("SOL", 7.0, 6.0, +4.10)
                 + self.round_trip("WLD", 5.0, 4.0, -1.20))
        fake, v = self.venue_with(fills)
        with mock.patch.object(T, "save_state"), \
             mock.patch.object(T.notify, "send") as send:
            self.assertEqual(T.announce_closes(v, self.state), 3)
        named = " ".join(str(c) for c in send.call_args_list)
        for coin in ("XRP", "SOL", "WLD"):
            self.assertIn(coin, named)
        self.assertEqual(self.state["announced_close_ms"], self.ago(4.0))

    def test_an_alert_failure_never_breaks_the_pass(self):
        self.state["announced_close_ms"] = self.ago(6.0)
        fake, v = self.venue_with(self.round_trip())
        with mock.patch.object(T, "save_state"), \
             mock.patch.object(T.notify, "send",
                               side_effect=RuntimeError("telegram down")):
            T.announce_closes(v, self.state)      # must not raise


class CloseAlertsAreArmedOnly(unittest.TestCase):
    def test_a_dry_run_does_not_consume_the_watermark(self):
        """Both processes share one state file. A dry run that advanced the
        mark would make the armed process miss the next real close."""
        src = inspect.getsource(T.main)
        call = src[:src.index("announce_closes(venue, state)")]
        self.assertRegex(call.rsplit("\n", 2)[-2].strip(), r"^if armed:$")


class BandsAreRecordedWhileBlocked(unittest.TestCase):
    """The pass is blocked most of the time (a full book), and a blocked pass
    used to record nothing: the band record sampled ~8% of hours."""

    def test_hourly_when_blocked(self):
        self.assertTrue(T.should_record_while_blocked(
            "holding 1 and bidding on 3, max 4", last_record=0.0, now=4000.0))
        self.assertFalse(T.should_record_while_blocked(
            "holding 1 and bidding on 3, max 4", last_record=1000.0, now=4000.0))

    def test_never_while_the_venue_is_unreadable(self):
        """Scanning 45 coins during a rate limit only deepens it."""
        self.assertFalse(T.should_record_while_blocked(
            "venue unreadable -- not opening blind", last_record=0.0, now=10_000.0))

    def test_the_blocked_branch_actually_records(self):
        src = inspect.getsource(T.main)
        blocked = src[src.index("not opening this pass"):src.index("give_up_if_unreadable")]
        self.assertIn("should_record_while_blocked", blocked)
        self.assertIn("scan_and_record", blocked)

    def test_one_scan_path_for_both(self):
        """The normal pass and the blocked branch share one function, so they
        cannot record different things."""
        src = inspect.getsource(T.main)
        self.assertEqual(src.count("zone_log.record_pass"), 0)
        self.assertGreaterEqual(src.count("scan_and_record("), 2)


class UnreadableBooksAreNotEmpty(unittest.TestCase):
    """2026-11-03, in a burst of 429s. An ETH bid was sent, its read-back
    failed, and the trader logged "no bid rested" and dropped the plan -- but
    the bid had rested. The next candidate's open-orders read failed too,
    `(resting or [])` counted that as no bids, and XRP rested as a fifth
    commitment against MAX_CONCURRENT 4."""

    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp(prefix="unreadable_"))
        self._orig = (T.STATE_PATH, T.TRADE_LOG, execute.SETTLE_SEC, T.notify.send)
        T.STATE_PATH = self._tmp / "state.json"
        T.TRADE_LOG = self._tmp / "trades.csv"
        execute.SETTLE_SEC = 0.0
        self.sent = []
        T.notify.send = lambda text, **kw: (self.sent.append(text), True)[1]
        self.meta = MetaStore(self._tmp / "meta.json")
        self.fake = FakeHL(mid=100.0, usdc=1000.0)
        self.venue = build_venue(self.fake)
        self.state = {"opened_today": 0, "day": "d"}

    def tearDown(self):
        T.STATE_PATH, T.TRADE_LOG, execute.SETTLE_SEC, T.notify.send = self._orig
        shutil.rmtree(self._tmp, ignore_errors=True)

    def bid(self, fail_reads=()):
        """place_resting_bid with the Nth open-orders read (1-based) failing
        for each N in fail_reads. Read 1 is rest_bid's `before`, read 2 its
        read-back after sending, read 3 the second check."""
        real = type(self.venue).open_orders
        n = [0]

        def flaky(v, coin="", urgent=False):
            n[0] += 1
            return None if n[0] in fail_reads else real(v, coin, urgent)

        with mock.patch.object(type(self.venue), "open_orders", flaky):
            return T.place_resting_bid(self.venue, self.meta,
                                       episode("HYPE", entry=95.0, stop=92.0, tp=104.0),
                                       1000.0, 1000.0, self.state)

    def resting_bids(self):
        return [o for o in self.fake.resting
                if o["coin"] == "HYPE" and not o.get("reduceOnly")]

    def test_a_normal_bid_rests_and_keeps_its_plan(self):
        self.assertEqual(self.bid(), 0)
        self.assertIsNotNone(self.meta.get("HYPE"))
        self.assertEqual(len(self.resting_bids()), 1)

    def test_a_bid_sent_blind_keeps_its_plan_and_stops_the_pass(self):
        rc = self.bid(fail_reads=(2, 3))
        self.assertEqual(rc, T.BID_UNKNOWN)
        self.assertEqual(len(self.resting_bids()), 1, "the bid did rest")
        row = self.meta.get("HYPE")
        self.assertIsNotNone(row, "a bid that may be live must keep its stop")
        self.assertAlmostEqual(row.planned_stop, 92.0)
        self.assertIn("last_entry_at", self.state, "it is a commitment: spacing applies")

    def test_a_bid_confirmed_on_a_second_read_counts_as_placed(self):
        self.assertEqual(self.bid(fail_reads=(2,)), 0)
        self.assertIsNotNone(self.meta.get("HYPE"))
        self.assertTrue(any("Bid resting on HYPE" in s for s in self.sent))

    def test_only_a_readable_empty_book_drops_the_plan(self):
        with mock.patch.object(self.fake, "rest_plain_order", lambda *a, **k: None):
            self.assertEqual(self.bid(), 1)
        self.assertIsNone(self.meta.get("HYPE"))

    def test_an_unreadable_book_is_not_an_empty_one(self):
        with mock.patch.object(type(self.venue), "open_orders", return_value=None):
            self.assertIsNone(T.slots_in_use(self.venue))
        with mock.patch.object(type(self.venue), "positions", return_value=None):
            self.assertIsNone(T.slots_in_use(self.venue))
        self.fake.rest_plain_order("SOL", 1.0, 90.0, is_buy=True)
        held, bids = T.slots_in_use(self.venue)
        self.assertEqual(bids, {"SOL"})

    def test_preflight_will_not_open_on_a_book_it_cannot_read(self):
        v = _venue()
        with mock.patch.object(type(v), "open_orders", return_value=None):
            reason = T.preflight(v, {"opened_today": 0, "day": "d"}, _halt(), _eqlog())
        self.assertIsNotNone(reason)
        self.assertIn("unreadable", reason)
        self.assertFalse(T.should_record_while_blocked(reason, 0.0, 10_000.0),
                         "a rate limit is no time to scan 45 coins")

    def test_the_pass_counts_slots_through_the_fail_closed_reader(self):
        strip = lambda f: "\n".join(l.split("#")[0]                  # noqa: E731
                                    for l in inspect.getsource(f).splitlines())
        main, pre = strip(T.main), strip(T.preflight)
        self.assertIn("slots_in_use(venue)", main)
        self.assertIn("rc == BID_UNKNOWN", main)
        self.assertNotIn("or [])", main + pre)

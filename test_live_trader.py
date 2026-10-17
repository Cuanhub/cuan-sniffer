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
        self.assertIn("venue.positions()", code)
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
    def test_a_skipped_candidate_reports_no_order(self):
        T._RO_CLIENT = FakeCandles(held=False)
        self.assertFalse(self.fire().order_sent)

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

    def test_fetch_failure_is_recorded_not_just_printed(self):
        class Boom:
            def get_candles(self, *a, **k):
                raise RuntimeError("rate limited")

        T._RO_CLIENT = Boom()
        self.fire()
        rows = T.TRADE_LOG.read_text() if T.TRADE_LOG.exists() else ""
        self.assertIn("fetch_failed", rows,
                      "a candidate lost to a rate limit must leave a row")

    # -- the watchboard ---------------------------------------------------
    def test_a_near_miss_is_recorded_on_the_watchboard(self):
        T._RO_CLIENT = FakeCandles(held=False)
        board = []
        T.execute_one(self.venue, self.meta, episode("HYPE"), 1000.0, 1000.0,
                      True, self.state, watchboard=board)
        self.assertTrue(board, "a near miss must reach the board")
        self.assertEqual(board[0]["coin"], "HYPE")

    def test_a_fired_trigger_is_not_on_the_watchboard(self):
        board = []
        T.execute_one(self.venue, self.meta, episode("HYPE"), 1000.0, 1000.0,
                      True, self.state, watchboard=board)
        self.assertEqual(board, [], "a trade that fired is not a near miss")

    # -- bar caching ------------------------------------------------------
    def test_1h_bars_are_fetched_once_per_coin_per_pass(self):
        calls = []
        inner = FakeCandles()

        class Counting:
            def get_candles(self, coin, interval, start, end):
                calls.append(coin)
                return inner.get_candles(coin, interval, start, end)

        T._RO_CLIENT = Counting()
        cache = {}
        for _ in range(3):
            T.execute_one(self.venue, self.meta, episode("HYPE"), 1000.0,
                          1000.0, True, self.state, bars_cache=cache)
        self.assertEqual(len(calls), 1, f"fetched {len(calls)} times")

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
        self.assertAlmostEqual(sent[0]["sz"], round(notional / 101.0, 2),
                               places=2,
                               msg="size must come from the LIVE price")


class TriggerTest(unittest.TestCase):
    """The user's rule: price came down to the level and has held above it for
    two hours. Arrival alone is not enough -- every setup they marked "maybe"
    was one price was sitting on at that moment."""

    def setUp(self):
        self._tmp = Path("/tmp/_test_trigger")
        if self._tmp.exists():
            shutil.rmtree(self._tmp)
        self._tmp.mkdir(parents=True)
        self._o = (T.STATE_PATH, T.TRADE_LOG, T._RO_CLIENT)
        T.STATE_PATH = self._tmp / "s.json"
        T.TRADE_LOG = self._tmp / "t.csv"
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

    def test_fires_after_the_level_has_held(self):
        T._RO_CLIENT = FakeCandles(held=True, touched=True)
        b = _venue(mid=100.0)
        self.assertTrue(T.execute_one(b, self.meta, episode("AAA"), 1000.0, 1000.0, True, self.state))

    def test_waits_while_the_level_is_breaking(self):
        T._RO_CLIENT = FakeCandles(held=False, touched=True)
        b = _venue(mid=100.0)
        self.assertFalse(T.execute_one(b, self.meta, episode("AAA"), 1000.0, 1000.0, True, self.state))
        self.assertEqual([o for o in b.exchange.sent
                          if "trigger" not in o["type"]], [],
                         "must not buy a level that is closing below itself")

    def test_waits_when_price_never_reached_the_level(self):
        T._RO_CLIENT = FakeCandles(held=True, touched=False)
        b = _venue(mid=100.0)
        self.assertFalse(T.execute_one(b, self.meta, episode("AAA"), 1000.0, 1000.0, True, self.state))

    def test_hold_ignores_the_still_forming_bar(self):
        """A live candle is not a hold. Counting it would let a wick that is
        about to be filled stand in for two hours of support."""
        from research.levels.trigger import level_has_held
        lv = 100.0
        # 1.002 keeps the last closed bar inside TRIGGER_MAX_ABOVE_SIGMA;
        # this test is about the forming bar, not about the ceiling.
        bars = [{"t": i * 3600_000, "o": lv * 1.002, "h": lv * 1.006,
                 "l": lv * 0.999 if i == 30 else lv * 1.001,
                 "c": lv * (1.002 + 0.004 * ((i % 5) - 2))} for i in range(40)]
        bars[-1]["c"] = lv * 0.90        # forming bar collapsing -- must be ignored
        fired, _ = level_has_held(bars, lv)
        self.assertTrue(fired)
        bars[-2]["c"] = lv * 0.90        # a CLOSED bar below the level must block
        self.assertFalse(level_has_held(bars, lv)[0])


class ReviewFlagTest(unittest.TestCase):
    """Trades that pass the rails but a human should still look at. These are
    surfaced, never blocked -- the filters are fitted to 18 labels and their
    edges are exactly where they should be questioned."""

    def test_flags_notional_far_above_equity(self):
        ep = episode("BNB", 741.0, 733.0, 757.0)
        flags = T.review_flags(ep, 2118.0, 768.0)
        self.assertTrue(any("equity" in f for f in flags), flags)

    def test_flags_a_stop_tight_enough_to_gap_through(self):
        ep = episode("BNB", 741.457, 733.386, 757.75)   # 1.09% stop
        self.assertTrue(any("gap" in f for f in T.review_flags(ep, 100.0, 768.0)))

    def test_flags_a_level_with_a_poor_hold_record(self):
        ep = episode("AAA") | {"level_hold_rate": 0.55, "level_touches": 9}
        self.assertTrue(any("held only" in f for f in T.review_flags(ep, 100.0, 768.0)))

    def test_flags_the_flipped_resistance_category(self):
        ep = episode("AAA") | {"level_origin": "flipped_resistance"}
        self.assertTrue(any("flipped" in f for f in T.review_flags(ep, 100.0, 768.0)))

    def test_a_clean_trade_raises_nothing(self):
        ep = episode("AAA", 100.0, 90.0, 140.0) | {
            "level_hold_rate": 0.95, "level_touches": 12,
            "near_sigma": 8.0, "suggested_rr": 4.0, "level_origin": "swing_low"}
        self.assertEqual(T.review_flags(ep, 300.0, 768.0), [])


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


class WatchlistTest(unittest.TestCase):
    """The quality bar and the entry trigger describe the same level at two
    different moments, and contradict each other at one.

    Depth wants price >=5 sigma ABOVE the level; the trigger wants it within
    2 sigma. Rest wants the level untested for >=9 bars; the trigger wants it
    touched inside 12 hours. Checked together on one candidate in one pass
    they are an empty set, and the trader proved it: 3041 consecutive
    not_triggered decisions, the trigger never once firing.
    """

    def test_depth_and_rest_are_admission_only(self):
        shallow = gated(near=0.5, rest=1)
        self.assertIsNotNone(T.passes_quality(shallow, watched=False))
        self.assertIsNone(T.passes_quality(shallow, watched=True))

    def test_entry_criteria_still_apply_while_watched(self):
        # These stay true and stay meaningful with price sitting on the level.
        # The rr floor correctly refused all three USELESS triggers at 0.45-0.63.
        self.assertIsNotNone(T.passes_quality(gated(near=0.5, rr=0.45), watched=True))
        self.assertIsNotNone(T.passes_quality(gated(near=0.5, overhead=1.0), watched=True))
        self.assertIsNotNone(T.passes_quality(gated(near=0.5, overhead=""), watched=True))

    def test_price_returning_to_an_admitted_level_still_qualifies(self):
        orig = T._find_candidates
        watching = {}
        try:
            # pass 1: price far above -- clears the full bar, gets admitted
            T._find_candidates = lambda c, u, **kw: [gated(near=7.0, rest=20)]
            intents, _ = T.build_intents(None, None, watching)
            self.assertEqual(len(intents), 1)
            self.assertIn("AAA|100.0", watching)

            # pass 2: price has come back to the level -- depth and rest now
            # fail, which is exactly what being ready to enter looks like
            T._find_candidates = lambda c, u, **kw: [gated(near=0.4, rest=1)]
            intents, rejected = T.build_intents(None, None, watching)
            self.assertEqual(len(intents), 1, "a watched level must survive the pullback")
            self.assertTrue(intents[0]["_watched"])
            self.assertEqual(rejected, [])
        finally:
            T._find_candidates = orig

    def test_requalifying_restarts_the_ttl_clock(self):
        # A level price is still far above keeps qualifying pass after pass.
        # Ageing it out underneath that would drop it for no reason; the
        # window that matters starts when it stops clearing on its own.
        from datetime import datetime, timedelta, timezone
        stale = (datetime.now(timezone.utc)
                 - timedelta(hours=T.WATCH_TTL_HOURS - 1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        watching = {"AAA|100.0": {"coin": "AAA", "level_price": 100.0,
                                  "admitted_at": stale}}
        orig = T._find_candidates
        try:
            T._find_candidates = lambda c, u, **kw: [gated(near=7.0, rest=20)]
            T.build_intents(None, None, watching)
        finally:
            T._find_candidates = orig
        self.assertNotEqual(watching["AAA|100.0"]["admitted_at"], stale,
                            "re-qualifying must restart the clock")
        self.assertEqual(T.prune_watchlist(watching), 0)

    def test_an_unadmitted_shallow_level_is_still_rejected(self):
        orig = T._find_candidates
        try:
            T._find_candidates = lambda c, u, **kw: [gated(near=0.4, rest=1)]
            intents, rejected = T.build_intents(None, None, {})
            self.assertEqual(intents, [])
            self.assertIn("depth", rejected[0][2])
        finally:
            T._find_candidates = orig

    def test_watchlist_expires(self):
        from datetime import datetime, timedelta, timezone
        stale = (datetime.now(timezone.utc)
                 - timedelta(hours=T.WATCH_TTL_HOURS + 1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        watching = {"AAA|100.0": {"admitted_at": stale},
                    "BBB|200.0": {"admitted_at": T.now_iso()}}
        self.assertEqual(T.prune_watchlist(watching), 1)
        self.assertNotIn("AAA|100.0", watching)
        self.assertIn("BBB|200.0", watching)

    def test_seeding_replays_admissions_from_the_trade_log(self):
        # A dry_run/planned row is written only for a setup that cleared the
        # full bar, so the log is an accurate register of what qualified.
        from datetime import datetime, timedelta, timezone
        old_ts = (datetime.now(timezone.utc)
                  - timedelta(hours=T.WATCH_TTL_HOURS + 2)).strftime("%Y-%m-%dT%H:%M:%SZ")
        T.log_trade({"ts_utc": T.now_iso(), "action": "dry_run", "coin": "AAA",
                     "level_price": "100.0", "result": "planned"})
        T.log_trade({"ts_utc": T.now_iso(), "action": "skip", "coin": "BBB",
                     "level_price": "200.0", "result": "not_triggered"})
        T.log_trade({"ts_utc": old_ts, "action": "dry_run", "coin": "CCC",
                     "level_price": "300.0", "result": "planned"})
        watching = {}
        added = T.seed_watchlist_from_log(watching)
        self.assertEqual(added, 1)
        self.assertIn("AAA|100.0", watching)
        self.assertNotIn("BBB|200.0", watching, "a skip is not an admission")
        self.assertNotIn("CCC|300.0", watching, "outside the TTL")

    def test_seeding_is_idempotent(self):
        T.log_trade({"ts_utc": T.now_iso(), "action": "dry_run", "coin": "AAA",
                     "level_price": "100.0", "result": "planned"})
        watching = {}
        self.assertEqual(T.seed_watchlist_from_log(watching), 1)
        self.assertEqual(T.seed_watchlist_from_log(watching), 0)

    def test_seeding_without_a_log_is_harmless(self):
        self.assertEqual(T.seed_watchlist_from_log({}), 0)

    def test_unparseable_admission_time_expires(self):
        watching = {"AAA|100.0": {"admitted_at": "garbage"}}
        self.assertEqual(T.prune_watchlist(watching), 1)


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


class HoldRuleSelectionTest(unittest.TestCase):
    """execute_one must pick the hold rule from the level's origin."""

    def test_the_two_rules_are_different(self):
        self.assertNotEqual(T.TRIGGER_HOLD_HOURS, T.TRIGGER_HOLD_HOURS_SWING)
        self.assertLess(T.TRIGGER_HOLD_HOURS_SWING, T.TRIGGER_HOLD_HOURS)

    def test_origin_selects_the_hold_window(self):
        seen = []
        orig = T.level_has_held
        T.level_has_held = lambda bars, lvl, hold_hours=None, **kw: (
            seen.append(hold_hours), (False, "not_triggered"))[1]
        tmp = Path("/tmp/_test_holdrule")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        orig_paths = (T.STATE_PATH, T.TRADE_LOG, T._RO_CLIENT)
        T.STATE_PATH, T.TRADE_LOG = tmp / "s.json", tmp / "t.csv"
        T._RO_CLIENT = FakeCandles()
        st = {"owned": {}, "pending": {}, "opened_today": 0, "day": "d",
              "realised_today": 0.0, "halted": False, "halt_reason": ""}
        try:
            for origin, expected in (("flipped_resistance", T.TRIGGER_HOLD_HOURS),
                                     ("swing_low", T.TRIGGER_HOLD_HOURS_SWING)):
                ep = dict(episode("AAA"), level_origin=origin)
                T.execute_one(_venue(), MetaStore(Path(tempfile.mkdtemp()) / "m.json"),
                      ep, 1000.0, 1000.0, True, st)
                self.assertEqual(seen[-1], expected, origin)
        finally:
            T.level_has_held = orig
            T.STATE_PATH, T.TRADE_LOG, T._RO_CLIENT = orig_paths
            shutil.rmtree(tmp, ignore_errors=True)


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
        self.assertEqual(st["watching"], {})
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
                   "build_intents", "size_from_stop", "drain_control_queue"):
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
        self.assertEqual(st["watching"], {"a": 1})
        self.assertEqual(st["opened_today"], 2)

    def test_a_corrupt_file_is_still_fatal(self):
        """Stripping dead keys must not become 'repair whatever is there'."""
        T.STATE_PATH.write_text('{"watching": ')
        with self.assertRaises(SystemExit):
            T.load_state()

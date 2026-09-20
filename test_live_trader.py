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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import tools.live_levels_trader as T


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
            base = 1.004 if self.touched else 1.035
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
                 bid_ok=True):
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
        self.pnl_queries = []
        self.realised_pnl = realised_pnl   # None models a failed query
        self.exit_fills = exit_fills
        self.protection_oids = protection_oids
        self.venue = {}          # coin -> notional
        self.info = self

    # -- venue reads -------------------------------------------------------
    def get_margin_summary(self):
        return (self.equity, 0.0, self.equity)

    def _get_user_state_address(self):
        return "0xfake"

    def get_mid_price(self, coin):
        # default: sitting right at the planned entry
        return self.mid if self.mid is not None else 100.0

    def user_state(self, addr):
        return {"assetPositions": [
            {"position": {"coin": c, "szi": "1", "positionValue": str(v)}}
            for c, v in self.venue.items()]}

    def _get_venue_signed_size(self, coin):
        return 1.0 if coin in self.venue else 0.0

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


def episode(coin, entry=100.0, stop=95.0, tp=115.0):
    return {"coin": coin, "episode_id": f"{coin}_1_{entry}", "level_price": entry,
            "suggested_entry": entry, "suggested_stop": stop, "suggested_tp": tp,
            "suggested_rr": 3.0, "level_hold_rate": 1.0, "level_touches": 10,
            "near_sigma": 7.0, "level_origin": "swing_low"}


class ArmedPathTest(unittest.TestCase):
    def setUp(self):
        self._tmp = Path("/tmp/_test_live_trader")
        if self._tmp.exists():
            shutil.rmtree(self._tmp)
        self._tmp.mkdir(parents=True)
        self._orig_state, self._orig_log, self._orig_ro = T.STATE_PATH, T.TRADE_LOG, T._RO_CLIENT
        T.STATE_PATH = self._tmp / "state.json"
        T.TRADE_LOG = self._tmp / "trades.csv"
        T._RO_CLIENT = FakeCandles()
        self.state = {"owned": {}, "opened_today": 0, "day": "d",
                      "realised_today": 0.0, "halted": False, "halt_reason": ""}
        self.sent = []
        self._orig_send = T.notify.send
        T.notify.send = lambda text, **kw: (self.sent.append(text), True)[1]

    def tearDown(self):
        T.notify.send = self._orig_send
        T.STATE_PATH, T.TRADE_LOG, T._RO_CLIENT = self._orig_state, self._orig_log, self._orig_ro
        if self._tmp.exists():
            shutil.rmtree(self._tmp)

    def test_successful_entry_is_recorded_and_protected(self):
        b = FakeBackend()
        ok = T.execute_one(b, episode("AAA"), 1000.0, 1000.0, True, self.state)
        self.assertTrue(ok)
        self.assertIn("AAA", self.state["owned"])
        self.assertTrue(self.state["owned"]["AAA"]["confirmed"])
        self.assertEqual(b.protections, ["AAA"], "protection must follow every entry")

    def test_concurrency_holds_even_if_the_fill_parse_breaks(self):
        """The 2026-09-18 runaway in miniature. The slot is claimed before the
        order is sent, so a backend that fills without reporting it usefully
        can no longer uncap the loop."""
        class WeirdBackend(FakeBackend):
            def execute_entry(self, coin, side, price, size_usd, confidence):
                self.entries.append(coin)
                self.venue[coin] = size_usd
                # filled, but with the shape the old code guessed at
                return FakeFill(filled=True, fill_price=price, fill_size_usd=size_usd)
        b = WeirdBackend()
        for coin in ("AAA", "BBB", "CCC", "DDD"):
            if len(self.state["owned"]) >= T.MAX_CONCURRENT:
                break
            T.execute_one(b, episode(coin), 1000.0, 1000.0, True, self.state)
        self.assertLessEqual(len(b.entries), T.MAX_CONCURRENT,
                             f"sent {len(b.entries)} orders against a cap of {T.MAX_CONCURRENT}")

    def test_rejected_entry_releases_the_slot(self):
        b = FakeBackend(fill=False)
        ok = T.execute_one(b, episode("AAA"), 1000.0, 1000.0, True, self.state)
        self.assertFalse(ok)
        self.assertNotIn("AAA", self.state["owned"], "a rejected entry must not hold a slot")

    def test_protection_failure_closes_the_position(self):
        b = FakeBackend(protection_ok=False)
        ok = T.execute_one(b, episode("AAA"), 1000.0, 1000.0, True, self.state)
        self.assertFalse(ok)
        self.assertEqual(b.exits, ["AAA"], "an unprotected position must be closed")
        self.assertNotIn("AAA", self.state["owned"])

    def test_protection_failure_cancels_what_the_venue_accepted(self):
        # A contradicted leg may not be reduce-only, and a non-reduce-only
        # order resting on a flat account opens a position instead of closing
        # one -- in a coin this module has just stopped tracking.
        b = FakeBackend(protection_ok=False, protection_oids=True)
        T.execute_one(b, episode("AAA"), 1000.0, 1000.0, True, self.state)
        self.assertEqual(b.cancels, [("AAA", "s1")])

    def test_failed_emergency_exit_halts_and_keeps_the_slot(self):
        # The close can be rejected. Releasing the slot anyway would discard
        # the only record of a position that is open and unprotected.
        b = FakeBackend(protection_ok=False, exit_fills=False)
        ok = T.execute_one(b, episode("AAA"), 1000.0, 1000.0, True, self.state)
        self.assertFalse(ok)
        self.assertTrue(self.state["halted"], "an unprotected open position must halt")
        self.assertIn("AAA", self.state["owned"])

    def test_successful_emergency_exit_releases_the_slot(self):
        b = FakeBackend(protection_ok=False)
        T.execute_one(b, episode("AAA"), 1000.0, 1000.0, True, self.state)
        self.assertFalse(self.state.get("halted", False))
        self.assertNotIn("AAA", self.state["owned"])

    def test_emergency_exit_accrues_its_cost(self):
        # An open/close round trip costs fees and spread. If that never
        # reaches realised_today, a protection failure that repeats every
        # pass can churn all day without the daily loss stop noticing.
        b = FakeBackend(protection_ok=False, realised_pnl=-3.5)
        T.execute_one(b, episode("AAA"), 1000.0, 1000.0, True, self.state)
        self.assertAlmostEqual(self.state["realised_today"], -3.5)

    def test_fetch_failure_is_recorded_not_just_printed(self):
        # Every other drop reason writes a row; this one only printed, so a
        # candidate lost to a rate limit left no trace in the durable record.
        class Boom:
            def get_candles(self, *a, **kw):
                raise RuntimeError("candleSnapshot:AAA:1h failed: 429 rate limit")

        orig, T._RO_CLIENT = T._RO_CLIENT, Boom()
        try:
            b = FakeBackend()
            ok = T.execute_one(b, episode("AAA"), 1000.0, 1000.0, True, self.state)
        finally:
            T._RO_CLIENT = orig
        self.assertFalse(ok)
        self.assertEqual(b.entries, [], "nothing may be sent without a trigger check")
        rows = T.TRADE_LOG.read_text()
        self.assertIn("fetch_failed", rows)
        self.assertIn("429", rows)

    def test_a_skipped_candidate_reports_no_order(self):
        # venue_sanity keys off this. A dry candidate must not cost a
        # user_state call -- that is what truncated 22 passes.
        b = FakeBackend()
        out = T.execute_one(b, episode("AAA"), 1000.0, 1000.0, False, self.state)
        self.assertFalse(out.order_sent)

    def test_a_filled_entry_reports_an_order(self):
        b = FakeBackend()
        out = T.execute_one(b, episode("AAA"), 1000.0, 1000.0, True, self.state)
        self.assertTrue(out.order_sent)
        self.assertTrue(out.opened)

    def test_a_rejected_entry_still_reports_an_order(self):
        # The order reached the venue; whether it filled is a separate
        # question, and venue_sanity must still run.
        b = FakeBackend(fill=False)
        out = T.execute_one(b, episode("AAA"), 1000.0, 1000.0, True, self.state)
        self.assertTrue(out.order_sent)
        self.assertFalse(out.opened)

    def test_1h_bars_are_fetched_once_per_coin_per_pass(self):
        # One coin carries several levels; 141 candidates came from ~30 coins,
        # so this refetched the same series ~5x. Rate limits truncate passes,
        # so the waste was not free.
        calls = []
        real = T._RO_CLIENT

        class Counting:
            def get_candles(self, coin, interval, start, end):
                calls.append(coin)
                return real.get_candles(coin, interval, start, end)

        T._RO_CLIENT = Counting()
        try:
            cache = {}
            for _ in range(4):
                T.cached_1h("AAA", cache)
            T.cached_1h("BBB", cache)
        finally:
            T._RO_CLIENT = real
        self.assertEqual(calls, ["AAA", "BBB"], "one fetch per coin per pass")

    def test_no_cache_still_fetches(self):
        # cache=None is the uncached path and must still work.
        self.assertTrue(len(T.cached_1h("AAA", None)) > 0)

    def test_an_open_position_is_announced_with_its_protection(self):
        # Everything needed to check the fill against a chart, because the
        # alternative is opening a laptop to find out what was bought.
        b = FakeBackend()
        T.execute_one(b, episode("AAA"), 1000.0, 1000.0, True, self.state)
        opened = [s for s in self.sent if "OPENED" in s]
        self.assertEqual(len(opened), 1)
        for token in ("AAA", "stop", "tp", "size", "rr", "level"):
            self.assertIn(token, opened[0])

    def test_a_near_miss_is_recorded_on_the_watchboard(self):
        b = FakeBackend()
        board = []
        T._RO_CLIENT = FakeCandles(level=100.0, held=False)
        T.execute_one(b, episode("AAA"), 1000.0, 1000.0, True, self.state,
                      watchboard=board)
        self.assertEqual(len(board), 1)
        self.assertEqual(board[0]["coin"], "AAA")
        self.assertIn("above", board[0])

    def test_a_fired_trigger_is_not_on_the_watchboard(self):
        b = FakeBackend()
        board = []
        T.execute_one(b, episode("AAA"), 1000.0, 1000.0, True, self.state,
                      watchboard=board)
        self.assertEqual(board, [], "only near-misses belong on the board")

    def test_dry_run_sends_nothing(self):
        b = FakeBackend()
        T.execute_one(b, episode("AAA"), 1000.0, 1000.0, False, self.state)
        self.assertEqual(b.entries, [])


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
        self.state = {"owned": {}, "opened_today": 0, "day": "d",
                      "realised_today": 0.0, "halted": False, "halt_reason": ""}

    def tearDown(self):
        T.STATE_PATH, T.TRADE_LOG, T._RO_CLIENT = self._o
        if self._tmp.exists():
            shutil.rmtree(self._tmp)

    def test_refuses_to_fire_when_price_drifted_from_the_plan(self):
        b = FakeBackend(mid=117.5)          # +17.5%, the UNI case
        ok = T.execute_one(b, episode("UNI"), 1000.0, 1000.0, True, self.state)
        self.assertFalse(ok)
        self.assertEqual(b.entries, [], "no order may be sent when the plan no longer applies")

    def test_fires_when_price_is_at_the_level(self):
        b = FakeBackend(mid=100.0)
        self.assertTrue(T.execute_one(b, episode("AAA"), 1000.0, 1000.0, True, self.state))
        self.assertEqual(b.entries, ["AAA"])

    def test_refuses_when_price_is_already_through_the_stop(self):
        b = FakeBackend(mid=94.0)           # below the 95 stop
        self.assertFalse(T.execute_one(b, episode("AAA"), 1000.0, 1000.0, True, self.state))
        self.assertEqual(b.entries, [])

    def test_refuses_when_live_reward_risk_has_collapsed(self):
        b = FakeBackend(mid=100.9)          # inside drift, but rr now poor
        self.assertFalse(T.execute_one(b, episode("AAA", 100.0, 95.0, 101.0),
                                       1000.0, 1000.0, True, self.state))
        self.assertEqual(b.entries, [])

    def test_size_comes_from_the_live_price_not_the_plan(self):
        b = FakeBackend(mid=101.0)          # within drift
        T.execute_one(b, episode("AAA"), 1000.0, 1000.0, True, self.state)
        rec = self.state["owned"]["AAA"]
        self.assertAlmostEqual(rec["entry"], 101.0, places=6)


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
        self.state = {"owned": {}, "opened_today": 0, "day": "d",
                      "realised_today": 0.0, "halted": False, "halt_reason": ""}

    def tearDown(self):
        T.STATE_PATH, T.TRADE_LOG, T._RO_CLIENT = self._o
        if self._tmp.exists():
            shutil.rmtree(self._tmp)

    def test_fires_after_the_level_has_held(self):
        T._RO_CLIENT = FakeCandles(held=True, touched=True)
        b = FakeBackend(mid=100.0)
        self.assertTrue(T.execute_one(b, episode("AAA"), 1000.0, 1000.0, True, self.state))

    def test_waits_while_the_level_is_breaking(self):
        T._RO_CLIENT = FakeCandles(held=False, touched=True)
        b = FakeBackend(mid=100.0)
        self.assertFalse(T.execute_one(b, episode("AAA"), 1000.0, 1000.0, True, self.state))
        self.assertEqual(b.entries, [], "must not buy a level that is closing below itself")

    def test_waits_when_price_never_reached_the_level(self):
        T._RO_CLIENT = FakeCandles(held=True, touched=False)
        b = FakeBackend(mid=100.0)
        self.assertFalse(T.execute_one(b, episode("AAA"), 1000.0, 1000.0, True, self.state))

    def test_hold_ignores_the_still_forming_bar(self):
        """A live candle is not a hold. Counting it would let a wick that is
        about to be filled stand in for two hours of support."""
        from research.levels.trigger import level_has_held
        lv = 100.0
        bars = [{"t": i * 3600_000, "o": lv * 1.004, "h": lv * 1.006,
                 "l": lv * 0.999 if i == 30 else lv * 1.002,
                 "c": lv * (1.004 + 0.004 * ((i % 5) - 2))} for i in range(40)]
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


class VenueSanityTest(unittest.TestCase):
    def setUp(self):
        self.state = {"owned": {}, "halted": False}

    def test_flags_positions_the_module_did_not_open(self):
        b = FakeBackend()
        b.venue = {"ZZZ": 100.0}
        r, fatal = T.venue_sanity(b, self.state, 1000.0)
        self.assertIn("ZZZ", r or "")
        self.assertTrue(fatal)

    def test_flags_breaching_the_concurrency_cap(self):
        b = FakeBackend()
        b.venue = {f"C{i}": 10.0 for i in range(T.MAX_CONCURRENT + 1)}
        self.state["owned"] = {c: {} for c in b.venue}
        r, fatal = T.venue_sanity(b, self.state, 1000.0)
        self.assertIn("cap", (r or "").lower())
        self.assertTrue(fatal)

    def test_flags_notional_above_the_leverage_ceiling(self):
        b = FakeBackend()
        b.venue = {"AAA": 50_000.0}
        self.state["owned"] = {"AAA": {}}
        msg, fatal = T.venue_sanity(b, self.state, 1000.0)
        self.assertIn("exceeds", msg or "")
        self.assertTrue(fatal)

    def test_a_query_failure_is_transient_not_fatal(self):
        """A 429 put the bot into a permanent halt on its second live pass.
        Failing to ask is not the same as getting a bad answer."""
        class Broken(FakeBackend):
            def user_state(self, addr):
                raise RuntimeError("429 rate limited")
        r, fatal = T.venue_sanity(Broken(), self.state, 1000.0)
        self.assertIn("429", r)
        self.assertFalse(fatal, "a transient query failure must not halt trading")

    def test_quiet_when_everything_agrees(self):
        b = FakeBackend()
        b.venue = {"AAA": 100.0}
        self.state["owned"] = {"AAA": {}}
        self.assertEqual(T.venue_sanity(b, self.state, 1000.0), (None, False))


class RealisedPnlTest(unittest.TestCase):
    """The daily loss stop was inert for every trade this module ever made.

    realised_today was initialised to 0.0, reset to 0.0 each day, and read
    once by preflight -- and never written in between. preflight therefore
    asked whether 0.0 <= -(15% of equity), which is never true. reconcile is
    the only place a close is observed, so it is where the number has to come
    from.
    """

    def state_holding(self, coin="AAA", entry=100.0, stop=95.0, size_coin=2.0):
        return {"owned": {coin: {"episode_id": "e1", "entry": entry, "stop": stop,
                                 "size_coin": size_coin, "size_usd": entry * size_coin,
                                 "opened_at": "2026-09-18T07:00:00Z", "confirmed": True}},
                "opened_today": 1, "day": "2026-09-18", "realised_today": 0.0,
                "halted": False, "halt_reason": ""}

    def test_iso_round_trips_to_venue_milliseconds(self):
        self.assertEqual(T.iso_to_ms("2026-09-18T07:00:00Z"), 1789714800000)
        self.assertIsNone(T.iso_to_ms("not a timestamp"))

    def test_closed_position_accrues_its_loss(self):
        b = FakeBackend(realised_pnl=-42.5)
        b.venue = {}                       # flat at the venue
        st = self.state_holding()
        T.reconcile(b, st)
        self.assertNotIn("AAA", st["owned"])
        self.assertAlmostEqual(st["realised_today"], -42.5)
        self.assertEqual(b.pnl_queries, [("AAA", 1789714800000)])

    def test_a_close_is_announced(self):
        sent = []
        orig = T.notify.send
        T.notify.send = lambda text, **kw: (sent.append(text), True)[1]
        try:
            b = FakeBackend(realised_pnl=-42.5)
            T.reconcile(b, self.state_holding())
        finally:
            T.notify.send = orig
        self.assertEqual(len(sent), 1)
        self.assertIn("CLOSED", sent[0])
        self.assertIn("AAA", sent[0])

    def test_closed_position_accrues_its_win(self):
        b = FakeBackend(realised_pnl=18.0)
        st = self.state_holding()
        T.reconcile(b, st)
        self.assertAlmostEqual(st["realised_today"], 18.0)

    def test_unmeasurable_close_is_charged_as_a_stop_out(self):
        # Reading a failed query as 0.0 is the bug this replaces. (100-95)*2.
        b = FakeBackend(realised_pnl=None)
        st = self.state_holding()
        T.reconcile(b, st)
        self.assertAlmostEqual(st["realised_today"], -10.0)

    def test_losses_accumulate_across_closes(self):
        st = self.state_holding()
        T.reconcile(FakeBackend(realised_pnl=-30.0), st)
        st["owned"]["BBB"] = {"episode_id": "e2", "entry": 100.0, "stop": 95.0,
                              "size_coin": 2.0, "size_usd": 200.0,
                              "opened_at": "2026-09-18T08:00:00Z", "confirmed": True}
        T.reconcile(FakeBackend(realised_pnl=-30.0), st)
        self.assertAlmostEqual(st["realised_today"], -60.0)

    def test_daily_loss_stop_now_fires(self):
        # The payoff. At $1000 equity the 15% stop trips below -$150.
        b = FakeBackend(equity=1000.0, realised_pnl=-160.0)
        st = self.state_holding()
        T.reconcile(b, st)
        reason = T.preflight(b, st)
        self.assertIsNotNone(reason)
        self.assertIn("daily loss stop", reason)
        self.assertTrue(st["halted"])

    def test_daily_loss_stop_leaves_a_normal_day_alone(self):
        b = FakeBackend(equity=1000.0, realised_pnl=-20.0)
        st = self.state_holding()
        T.reconcile(b, st)
        self.assertIsNone(T.preflight(b, st))
        self.assertFalse(st["halted"])

    def test_open_position_is_not_accrued(self):
        b = FakeBackend(realised_pnl=-99.0)
        b.venue = {"AAA": 200.0}           # still open
        st = self.state_holding()
        T.reconcile(b, st)
        self.assertIn("AAA", st["owned"])
        self.assertEqual(st["realised_today"], 0.0)
        self.assertEqual(b.pnl_queries, [])


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
        reason = T.preflight(FakeBackend(equity=1000.0),
                             self.state(T.MAX_NEW_PER_DAY))
        self.assertIsNotNone(reason)
        self.assertIn("today", reason)

    def test_allows_trading_below_the_cap(self):
        self.assertIsNone(T.preflight(FakeBackend(equity=1000.0),
                                      self.state(T.MAX_NEW_PER_DAY - 1)))


class HaltAlertTest(unittest.TestCase):
    """A halt is the one condition this module cannot leave on its own.

    preflight refuses on the persisted flag until a human clears it with
    --resume, so a halt nobody is told about is a bot that has silently
    stopped trading -- indistinguishable, from the outside, from a quiet
    market.
    """

    def setUp(self):
        self._tmp = Path("/tmp/_test_halt_alert")
        if self._tmp.exists():
            shutil.rmtree(self._tmp)
        self._tmp.mkdir(parents=True)
        self._orig_state = T.STATE_PATH
        T.STATE_PATH = self._tmp / "state.json"
        self.sent = []
        self._orig_send = T.notify.send
        T.notify.send = lambda text, **kw: (self.sent.append((text, kw)), True)[1]

    def tearDown(self):
        T.notify.send = self._orig_send
        T.STATE_PATH = self._orig_state
        if self._tmp.exists():
            shutil.rmtree(self._tmp)

    def state(self):
        return {"owned": {}, "opened_today": 0, "day": "2026-09-18",
                "realised_today": 0.0, "halted": False, "halt_reason": ""}

    def test_halt_sets_the_flag_and_alerts(self):
        st = self.state()
        returned = T.halt(st, "venue holds ['UNI'] which this module did not open")
        self.assertTrue(st["halted"])
        self.assertIn("did not open", st["halt_reason"])
        self.assertEqual(returned, st["halt_reason"])
        self.assertEqual(len(self.sent), 1)
        self.assertIn("did not open", self.sent[0][0])

    def test_a_persisting_halt_is_announced_once(self):
        # The flag persists and the trader polls every 15 minutes. Without the
        # transition check this is 96 identical messages a day, which is the
        # same as none.
        st = self.state()
        T.halt(st, "first reason")
        T.halt(st, "second reason")
        self.assertEqual(len(self.sent), 1)
        self.assertEqual(st["halt_reason"], "second reason")

    def test_no_halt_path_bypasses_halt(self):
        # Structural: the next person to add a halt should not have to know
        # that alerting exists for their halt to be heard.
        import inspect
        body = inspect.getsource(T)
        self.assertEqual(
            body.count('state["halted"] = True'), 1,
            "a halt that sets the flag directly is a halt nobody is told about")


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
            T._find_candidates = lambda c, u: [gated(near=7.0, rest=20)]
            intents, _ = T.build_intents(None, None, watching)
            self.assertEqual(len(intents), 1)
            self.assertIn("AAA|100.0", watching)

            # pass 2: price has come back to the level -- depth and rest now
            # fail, which is exactly what being ready to enter looks like
            T._find_candidates = lambda c, u: [gated(near=0.4, rest=1)]
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
            T._find_candidates = lambda c, u: [gated(near=7.0, rest=20)]
            T.build_intents(None, None, watching)
        finally:
            T._find_candidates = orig
        self.assertNotEqual(watching["AAA|100.0"]["admitted_at"], stale,
                            "re-qualifying must restart the clock")
        self.assertEqual(T.prune_watchlist(watching), 0)

    def test_an_unadmitted_shallow_level_is_still_rejected(self):
        orig = T._find_candidates
        try:
            T._find_candidates = lambda c, u: [gated(near=0.4, rest=1)]
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
        b = T.start_backend(lambda debug: calls.append(1) or "backend",
                            sleep=lambda s: None)
        self.assertEqual(b, "backend")
        self.assertEqual(len(calls), 1)

    def test_a_transient_failure_is_retried(self):
        calls = []

        def flaky(debug):
            calls.append(1)
            if len(calls) < 3:
                raise RuntimeError("429 rate limited")
            return "backend"

        self.assertEqual(T.start_backend(flaky, sleep=lambda s: None), "backend")
        self.assertEqual(len(calls), 3)

    def test_a_persistent_failure_raises(self):
        # Coming up without a backend would be an armed trader that cannot
        # trade and does not say so.
        def dead(debug):
            raise RuntimeError("429 rate limited")

        with self.assertRaises(RuntimeError):
            T.start_backend(dead, sleep=lambda s: None)

    def test_a_stop_during_backoff_gives_up_quietly(self):
        def dead(debug):
            raise RuntimeError("429")

        T._stop = True
        try:
            self.assertIsNone(T.start_backend(dead, sleep=lambda s: None))
        finally:
            T._stop = False


class RestingBidTest(unittest.TestCase):
    """A resting bid is the only path that can open a position unattended.

    execute_entry sends a marketable IOC, so a fill is known instantly. A bid
    is filled by the market whenever the market feels like it, and this module
    only finds out on its next pass -- so the fill has to become a KNOWN,
    protected position before venue_sanity can meet it as an unknown one and
    halt.
    """

    def setUp(self):
        self._tmp = Path("/tmp/_test_resting_bid")
        if self._tmp.exists():
            shutil.rmtree(self._tmp)
        self._tmp.mkdir(parents=True)
        self._orig = (T.STATE_PATH, T.TRADE_LOG, T.notify.send)
        T.STATE_PATH = self._tmp / "state.json"
        T.TRADE_LOG = self._tmp / "trades.csv"
        self.sent = []
        T.notify.send = lambda text, **kw: (self.sent.append(text), True)[1]
        self.state = {"owned": {}, "pending": {}, "opened_today": 0, "day": "d",
                      "realised_today": 0.0, "halted": False, "halt_reason": ""}

    def tearDown(self):
        T.STATE_PATH, T.TRADE_LOG, T.notify.send = self._orig
        if self._tmp.exists():
            shutil.rmtree(self._tmp)

    def pending(self, coin="AAA", price=100.0, hours_ago=0):
        from datetime import datetime, timedelta, timezone
        ts = (datetime.now(timezone.utc)
              - timedelta(hours=hours_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")
        return {coin: {"oid": "B1", "price": price, "size_coin": 2.0,
                       "stop": 95.0, "tp": 115.0, "episode_id": "e1",
                       "placed_at": ts}}

    def test_a_fill_becomes_a_known_protected_position(self):
        b = FakeBackend()
        b.venue = {"AAA": 200.0}                 # the bid filled
        self.state["pending"] = self.pending()
        T.reconcile_pending(b, self.state)
        self.assertIn("AAA", self.state["owned"], "must be known before venue_sanity")
        self.assertNotIn("AAA", self.state["pending"])
        self.assertEqual(b.protections, ["AAA"], "a fill must be protected at once")
        self.assertEqual(self.state["opened_today"], 1)
        self.assertTrue(any("OPENED" in s for s in self.sent))

    def test_a_fill_that_cannot_be_protected_is_closed(self):
        b = FakeBackend(protection_ok=False)
        b.venue = {"AAA": 200.0}
        self.state["pending"] = self.pending()
        T.reconcile_pending(b, self.state)
        self.assertEqual(b.exits, ["AAA"], "an unprotected fill is not the trade")
        self.assertNotIn("AAA", self.state["owned"])

    def test_a_resting_bid_counts_toward_the_concurrency_cap(self):
        # A bid is exposure the account has already committed to; it just has
        # not filled yet. Counting only `owned` would let the bot open its
        # full allowance alongside bids that then fill on top of it.
        b = FakeBackend(equity=1000.0)
        st = {"owned": {}, "pending": {f"C{i}": {} for i in range(T.MAX_CONCURRENT)},
              "opened_today": 0, "day": "2026-09-19", "realised_today": 0.0,
              "halted": False, "halt_reason": ""}
        why = T.preflight(b, st)
        self.assertIsNotNone(why)
        self.assertIn("bidding", why)

    def test_an_unfilled_bid_is_left_alone_inside_its_ttl(self):
        b = FakeBackend()
        b.venue = {}
        self.state["pending"] = self.pending(hours_ago=1)
        T.reconcile_pending(b, self.state)
        self.assertIn("AAA", self.state["pending"])
        self.assertEqual(b.cancels, [])

    def test_an_expired_bid_is_cancelled(self):
        # A bid left sitting is an opinion nobody is holding any more.
        b = FakeBackend()
        b.venue = {}
        self.state["pending"] = self.pending(hours_ago=T.LIMIT_ENTRY_TTL_HOURS + 1)
        T.reconcile_pending(b, self.state)
        self.assertEqual(b.cancels, [("AAA", "B1")])
        self.assertNotIn("AAA", self.state["pending"])


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
                T.execute_one(FakeBackend(), ep, 1000.0, 1000.0, True, st)
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
        st = {"day": "2020-01-01", "opened_today": 5, "realised_today": -80.0,
              "owned": {}, "halted": False, "halt_reason": ""}
        self.assertTrue(T.roll_day(st))
        self.assertEqual(st["opened_today"], 0)
        self.assertEqual(st["realised_today"], 0.0)

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
        self.assertIn("roll_day(state)", src[loop:loop + 200],
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
        self.assertEqual(st["owned"], {})
        self.assertFalse(st["halted"])

    def test_a_corrupt_file_refuses_to_start(self):
        T.STATE_PATH.write_text('{"owned": {"AAA": ')      # truncated write
        with self.assertRaises(SystemExit):
            T.load_state()

    def test_a_non_object_file_refuses_to_start(self):
        T.STATE_PATH.write_text('["not", "an", "object"]')
        with self.assertRaises(SystemExit):
            T.load_state()

    def test_a_halt_survives_a_round_trip(self):
        T.save_state({"owned": {"AAA": {}}, "halted": True,
                      "halt_reason": "venue mismatch", "pending": {}})
        st = T.load_state()
        self.assertTrue(st["halted"])
        self.assertEqual(st["halt_reason"], "venue mismatch")
        self.assertIn("AAA", st["owned"])

    def test_the_write_is_atomic(self):
        # tmp-then-replace, so a reader sees the old file or the new one,
        # never a fragment.
        T.save_state({"owned": {}, "halted": False})
        self.assertFalse((self._tmp / "state.json.tmp").exists(),
                         "the temp file must be renamed away, not left behind")
        self.assertTrue(T.STATE_PATH.exists())


class AccountModelTest(unittest.TestCase):
    """Never guess the account model.

    A failed query used to fall through to `standard`, which on this account
    reads marginSummary.accountValue -- $70.88 against a real $786.25, an 11x
    under-read. Every position would size off that and fall under
    MIN_NOTIONAL, so the trader would stop trading and look like a quiet
    market. The query fails on a 429, which happens routinely.
    """

    class Info:
        def __init__(self, boom=False):
            self.boom = boom

        def query_user_abstraction_state(self, addr):
            if self.boom:
                raise RuntimeError("429 rate limited")
            return "unifiedAccount"

        def spot_user_state(self, addr):
            return {"balances": [{"coin": "USDC", "total": "786.25", "hold": "0.0"}]}

    def test_a_unified_account_reads_spot_usdc(self):
        acc = T.account_equity(self.Info(), "0xabc")
        self.assertEqual(acc["model"], "unified")
        self.assertAlmostEqual(acc["equity"], 786.25)

    def test_an_unknown_model_refuses_rather_than_guessing(self):
        with self.assertRaises(RuntimeError):
            T.account_equity(self.Info(boom=True), "0xabc")


class ModuleShapeTest(unittest.TestCase):
    def test_every_function_main_calls_is_defined_before_the_guard(self):
        """Regression for the NameError on the first arming attempt: functions
        appended after `if __name__ == "__main__"` are invisible to main()."""
        src = Path("tools/live_levels_trader.py").read_text()
        guard = src.index('if __name__ == "__main__"')
        for fn in ("execute_one", "reconcile", "preflight", "venue_sanity",
                   "build_intents", "size_from_stop"):
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

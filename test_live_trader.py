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
                 exit_fills=True, protection_oids=False, realised_pnl=0.0):
        self.equity = equity
        self.mid = mid
        self.protection_ok = protection_ok
        self.fill = fill
        self.entries = []
        self.protections = []
        self.exits = []
        self.cancels = []
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

    def tearDown(self):
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

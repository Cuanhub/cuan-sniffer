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
    def __init__(self, equity=1000.0, protection_ok=True, fill=True, mid=None):
        self.equity = equity
        self.mid = mid
        self.protection_ok = protection_ok
        self.fill = fill
        self.entries = []
        self.protections = []
        self.exits = []
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
            return {"status": "missing", "error": "rejected"}
        # The real backend returns "protected" on success, never "ok".
        # The fake said "ok" and so agreed with the bug instead of catching it.
        return {"status": "protected", "stop_order_id": "s1", "tp_order_id": "t1"}

    def execute_exit(self, coin, side, price, size_usd, reason):
        self.exits.append(coin)
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
        self.assertIn("ZZZ", T.venue_sanity(b, self.state, 1000.0) or "")

    def test_flags_breaching_the_concurrency_cap(self):
        b = FakeBackend()
        b.venue = {f"C{i}": 10.0 for i in range(T.MAX_CONCURRENT + 1)}
        self.state["owned"] = {c: {} for c in b.venue}
        self.assertIn("cap", (T.venue_sanity(b, self.state, 1000.0) or "").lower())

    def test_flags_notional_above_the_leverage_ceiling(self):
        b = FakeBackend()
        b.venue = {"AAA": 50_000.0}
        self.state["owned"] = {"AAA": {}}
        msg = T.venue_sanity(b, self.state, 1000.0) or ""
        self.assertIn("exceeds", msg)

    def test_quiet_when_everything_agrees(self):
        b = FakeBackend()
        b.venue = {"AAA": 100.0}
        self.state["owned"] = {"AAA": {}}
        self.assertIsNone(T.venue_sanity(b, self.state, 1000.0))


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


if __name__ == "__main__":
    unittest.main()

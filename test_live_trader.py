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

import json
import shutil
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import tools.live_levels_trader as T


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
    def __init__(self, equity=1000.0, protection_ok=True, fill=True):
        self.equity = equity
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
            return {"status": "failed", "error": "rejected"}
        return {"status": "ok", "stop_order_id": "s1", "tp_order_id": "t1"}

    def execute_exit(self, coin, side, price, size_usd, reason):
        self.exits.append(coin)
        self.venue.pop(coin, None)
        return FakeFill(filled=True, fill_price=price, fill_size_usd=size_usd)


def episode(coin, entry=100.0, stop=95.0, tp=115.0):
    return {"coin": coin, "episode_id": f"{coin}_1_{entry}", "level_price": entry,
            "suggested_entry": entry, "suggested_stop": stop, "suggested_tp": tp,
            "suggested_rr": 3.0, "level_hold_rate": 1.0, "level_touches": 10}


class ArmedPathTest(unittest.TestCase):
    def setUp(self):
        self._tmp = Path("/tmp/_test_live_trader")
        if self._tmp.exists():
            shutil.rmtree(self._tmp)
        self._tmp.mkdir(parents=True)
        self._orig_state, self._orig_log = T.STATE_PATH, T.TRADE_LOG
        T.STATE_PATH = self._tmp / "state.json"
        T.TRADE_LOG = self._tmp / "trades.csv"
        self.state = {"owned": {}, "opened_today": 0, "day": "d",
                      "realised_today": 0.0, "halted": False, "halt_reason": ""}

    def tearDown(self):
        T.STATE_PATH, T.TRADE_LOG = self._orig_state, self._orig_log
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

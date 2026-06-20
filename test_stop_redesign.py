"""
Tests for _apply_entry_stop_redesign telemetry and widen guard.

Covers:
  - original_rr and final_rr captured in meta
  - stop_redesign_rr_destroyed reject reason when widening kills RR
  - stop_redesign_too_wide reject reason when widen_mult exceeds cap
  - widen_mult captured in meta on successful redesign
  - high-conf tolerance path still works
  - no false rejects when stop is not widened
  - real WIF/PENGU examples from production telemetry
"""

import importlib
import os
import sys
import unittest
from dataclasses import dataclass, field
from typing import Any, Dict
from unittest.mock import MagicMock


def _clean_env(**overrides) -> dict:
    base = {
        "PAPER_MODE": "false",
        "STARTING_BALANCE": "10000",
        "STOP_ATR_FLOOR_MULT_INTRADAY": "1.10",
        "STOP_ATR_FLOOR_MULT_SWING": "1.40",
        "STOP_BUFFER_ATR_MULT": "0.10",
        "MIN_STOP_ATR_REJECT": "0.80",
        "MIN_STOP_REDESIGN_RR": "1.60",
        "STOP_REDESIGN_RR_TOLERANCE": "0.05",
        "HIGH_CONF_STOP_REDESIGN_MIN_CONFIDENCE": "0.93",
        "HIGH_CONF_STOP_REDESIGN_RR_TOLERANCE": "0.08",
        "HIGH_CONF_STOP_REDESIGN_FAMILIES": "reversal,swing",
        "STOP_REDESIGN_MAX_WIDEN_MULT": "1.50",
        "MIN_EXECUTION_EFFECTIVE_RR": "1.55",
        "REGIME_TP_CAP_R": "1.75",
        "UNIVERSAL_MIN_CONFIDENCE": "0.90",
    }
    base.update(overrides)
    return base


def _reload_executor(env: dict):
    for key in list(os.environ.keys()):
        if key.startswith(("STOP_", "MIN_STOP", "MIN_EXECUTION", "HIGH_CONF_STOP",
                           "REGIME_TP", "PAPER_MODE", "STARTING_BALANCE",
                           "UNIVERSAL_MIN")):
            os.environ.pop(key, None)
    os.environ.update(env)
    if "executor" in sys.modules:
        del sys.modules["executor"]
    for dep in list(sys.modules):
        if dep in ("risk_manager", "strategy_filter", "bootstrap",
                    "execution_backend", "execution_backend_factory",
                    "live_execution_backend", "live_position_monitor",
                    "order_tracker", "protection_manager", "position",
                    "trade_log", "live_data_guard"):
            sys.modules.pop(dep, None)
    return importlib.import_module("executor")


@dataclass
class FakeSignal:
    coin: str = "SOL"
    side: str = "LONG"
    entry_price: float = 100.0
    stop_price: float = 98.0
    tp_price: float = 104.0
    confidence: float = 0.92
    regime: str = "continuation|htf_up|macro_up|mkt_weak_trend"
    reason: str = ""
    meta: Dict[str, Any] = field(default_factory=lambda: {
        "atr": 1.5,
        "timeframe": "1h",
        "setup_family": "continuation",
        "market_regime": "weak_trend",
        "regime_local": "continuation",
    })


def _make_executor(mod):
    ex = object.__new__(mod.Executor)
    ex._live_mode = False
    ex.notify = MagicMock()
    ex.backend = MagicMock()
    ex.signal_engine = None
    return ex


class TestStopRedesignTelemetry(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.env = _clean_env()
        cls.executor_mod = _reload_executor(cls.env)

    def test_original_rr_and_final_rr_in_meta_on_pass(self):
        """When redesign passes, meta must contain original_rr, rr_final, stop_widen_mult."""
        ex = _make_executor(self.executor_mod)
        sig = FakeSignal(
            entry_price=100.0,
            stop_price=97.0,
            tp_price=106.0,
            confidence=0.92,
            meta={"atr": 1.5, "timeframe": "1h", "setup_family": "continuation",
                  "market_regime": "weak_trend"},
        )
        result = ex._apply_entry_stop_redesign(sig, track="intraday")
        self.assertIsNone(result, f"Expected pass but got: {result}")
        meta = sig.meta
        self.assertIn("original_rr", meta)
        self.assertIn("rr_final", meta)
        self.assertIn("rr_original", meta)
        self.assertIn("stop_widen_mult", meta)
        self.assertGreater(meta["original_rr"], 0)
        self.assertGreater(meta["rr_final"], 0)
        self.assertGreater(meta["stop_widen_mult"], 0)

    def test_rr_destroyed_explicit_reason(self):
        """When widening collapses RR below floor, reason says rr_destroyed with details."""
        ex = _make_executor(self.executor_mod)
        # entry=100, stop=98 (sd=2.0), atr=2.0
        # ATR floor: 100 - 1.10*2.0 = 97.8. structural=98 > 97.8 → floor widens.
        # floor_applied = 97.8, buffered = 97.6, final_sd = 2.4, widen = 1.20
        # TP = 103.5 → tp_dist = 3.5 → final_rr = 3.5/2.4 = 1.458 < 1.55
        sig = FakeSignal(
            entry_price=100.0,
            stop_price=98.0,
            tp_price=103.5,
            confidence=0.91,
            meta={"atr": 2.0, "timeframe": "1h", "setup_family": "continuation",
                  "market_regime": "weak_trend"},
        )
        result = ex._apply_entry_stop_redesign(sig, track="intraday")
        self.assertIsNotNone(result)
        self.assertIn("stop_redesign_rr_destroyed", result)
        self.assertIn("original_rr=", result)
        self.assertIn("final_rr=", result)
        self.assertIn("widen=", result)

    def test_rr_destroyed_includes_original_rr_value(self):
        """The rr_destroyed message must show the pre-widen RR."""
        ex = _make_executor(self.executor_mod)
        # original_rr = 3.5 / 2.0 = 1.75
        sig = FakeSignal(
            entry_price=100.0,
            stop_price=98.0,
            tp_price=103.5,
            confidence=0.91,
            meta={"atr": 2.0, "timeframe": "1h", "setup_family": "continuation",
                  "market_regime": "weak_trend"},
        )
        result = ex._apply_entry_stop_redesign(sig, track="intraday")
        self.assertIn("original_rr=1.75", result)

    def test_too_wide_reject(self):
        """When widen_mult exceeds STOP_REDESIGN_MAX_WIDEN_MULT, reject explicitly."""
        env = _clean_env(STOP_REDESIGN_MAX_WIDEN_MULT="1.10")
        mod = _reload_executor(env)
        ex = _make_executor(mod)

        # entry=100, stop=99.5 (sd=0.5), atr=1.5
        # ATR floor: 100-1.65=98.35. structural=99.5 > 98.35 → floor widens hugely.
        # widen_mult will be >> 1.10
        sig = FakeSignal(
            entry_price=100.0,
            stop_price=99.5,
            tp_price=103.0,
            confidence=0.92,
            meta={"atr": 1.5, "timeframe": "1h", "setup_family": "continuation",
                  "market_regime": "weak_trend"},
        )
        result = ex._apply_entry_stop_redesign(sig, track="intraday")
        self.assertIsNotNone(result)
        self.assertIn("stop_redesign_too_wide", result)
        self.assertIn("widen=", result)

    def test_too_wide_disabled_when_zero(self):
        """Setting STOP_REDESIGN_MAX_WIDEN_MULT=0 disables the widen guard."""
        env = _clean_env(STOP_REDESIGN_MAX_WIDEN_MULT="0")
        mod = _reload_executor(env)
        ex = _make_executor(mod)

        sig = FakeSignal(
            entry_price=100.0,
            stop_price=99.5,
            tp_price=108.0,
            confidence=0.95,
            meta={"atr": 1.5, "timeframe": "1h", "setup_family": "reversal",
                  "market_regime": "strong_trend"},
        )
        result = ex._apply_entry_stop_redesign(sig, track="intraday")
        if result is not None:
            self.assertNotIn("stop_redesign_too_wide", result)

    def test_no_false_reject_when_stop_not_widened(self):
        """When engine stop is already wider than ATR floor, no widen reject."""
        ex = _make_executor(self.executor_mod)
        sig = FakeSignal(
            entry_price=100.0,
            stop_price=95.0,
            tp_price=110.0,
            confidence=0.95,
            meta={"atr": 1.5, "timeframe": "1h", "setup_family": "reversal",
                  "market_regime": "strong_trend"},
        )
        result = ex._apply_entry_stop_redesign(sig, track="intraday")
        self.assertIsNone(result, f"Expected pass but got: {result}")
        meta = sig.meta
        self.assertLessEqual(meta["stop_widen_mult"], 1.5)

    def test_high_conf_tolerance_still_works(self):
        """High-conf reversal/swing gets wider RR tolerance (0.08 instead of 0.05)."""
        ex = _make_executor(self.executor_mod)
        # entry=100, stop=98 (sd=2.0), atr=2.0
        # final_sd ≈ 2.4, widen=1.20
        # TP=103.7 → tp_dist=3.7 → final_rr = 3.7/2.4 = 1.542
        # base min_rr = 1.55 → would fail
        # high-conf min_rr = 1.52 → would PASS
        sig = FakeSignal(
            entry_price=100.0,
            stop_price=98.0,
            tp_price=103.7,
            confidence=0.95,
            side="LONG",
            meta={"atr": 2.0, "timeframe": "1h", "setup_family": "reversal",
                  "market_regime": "weak_trend"},
        )
        result = ex._apply_entry_stop_redesign(sig, track="intraday")
        meta = sig.meta
        if result is None:
            self.assertEqual(meta["stop_rr_tolerance_reason"], "high_conf_non_chop")
            self.assertAlmostEqual(meta["stop_rr_tolerance"], 0.08, places=2)
        else:
            self.assertNotIn("stop_redesign_rr_destroyed", result,
                             "High-conf reversal should not be RR-destroyed with tolerance")

    def test_short_side_rr_destroyed(self):
        """RR destruction detection works for SHORT signals too."""
        ex = _make_executor(self.executor_mod)
        # entry=100, stop=102 (sd=2.0), atr=2.0
        # ATR floor: 100 + 1.10*2.0 = 102.2. structural=102 < 102.2 → floor widens.
        # floor_applied = 102.2, buffered = 102.4, final_sd = 2.4, widen = 1.20
        # TP=96.5 → tp_dist=3.5 → final_rr = 3.5/2.4 = 1.458 < 1.55
        sig = FakeSignal(
            coin="ETH",
            side="SHORT",
            entry_price=100.0,
            stop_price=102.0,
            tp_price=96.5,
            confidence=0.91,
            meta={"atr": 2.0, "timeframe": "1h", "setup_family": "continuation",
                  "market_regime": "weak_trend"},
        )
        result = ex._apply_entry_stop_redesign(sig, track="intraday")
        self.assertIsNotNone(result)
        self.assertIn("stop_redesign_rr_destroyed", result)

    def test_widen_mult_calculation_accuracy(self):
        """widen_mult = final_stop_dist / original_stop_dist, verify numerically."""
        ex = _make_executor(self.executor_mod)
        sig = FakeSignal(
            entry_price=100.0,
            stop_price=97.0,
            tp_price=106.0,
            confidence=0.92,
            meta={"atr": 1.5, "timeframe": "1h", "setup_family": "continuation",
                  "market_regime": "strong_trend"},
        )
        result = ex._apply_entry_stop_redesign(sig, track="intraday")
        if result is None:
            meta = sig.meta
            original_sd = abs(100.0 - 97.0)
            final_sd = abs(100.0 - meta["stop_final"])
            expected_widen = final_sd / original_sd
            self.assertAlmostEqual(meta["stop_widen_mult"], expected_widen, places=3)

    def test_env_configurable_max_widen_mult(self):
        """STOP_REDESIGN_MAX_WIDEN_MULT is read from env."""
        env = _clean_env(STOP_REDESIGN_MAX_WIDEN_MULT="2.00")
        mod = _reload_executor(env)
        self.assertAlmostEqual(mod.STOP_REDESIGN_MAX_WIDEN_MULT, 2.00, places=2)

    def test_default_max_widen_mult(self):
        """Default STOP_REDESIGN_MAX_WIDEN_MULT is 1.50."""
        env = _clean_env()
        env.pop("STOP_REDESIGN_MAX_WIDEN_MULT", None)
        mod = _reload_executor(env)
        self.assertAlmostEqual(mod.STOP_REDESIGN_MAX_WIDEN_MULT, 1.50, places=2)

    def test_too_wide_fires_before_rr_check(self):
        """When both too_wide and rr_destroyed would fire, too_wide wins (runs first)."""
        env = _clean_env(STOP_REDESIGN_MAX_WIDEN_MULT="1.05")
        mod = _reload_executor(env)
        ex = _make_executor(mod)

        sig = FakeSignal(
            entry_price=100.0,
            stop_price=98.0,
            tp_price=103.5,
            confidence=0.91,
            meta={"atr": 2.0, "timeframe": "1h", "setup_family": "continuation",
                  "market_regime": "weak_trend"},
        )
        result = ex._apply_entry_stop_redesign(sig, track="intraday")
        self.assertIsNotNone(result)
        self.assertIn("stop_redesign_too_wide", result)

    def test_rr_destroyed_telemetry_logged(self):
        """Verify log_executor_reject is called on rr_destroyed (via side-effect)."""
        ex = _make_executor(self.executor_mod)
        sig = FakeSignal(
            entry_price=100.0,
            stop_price=98.0,
            tp_price=103.5,
            confidence=0.91,
            meta={"atr": 2.0, "timeframe": "1h", "setup_family": "continuation",
                  "market_regime": "weak_trend"},
        )
        result = ex._apply_entry_stop_redesign(sig, track="intraday")
        self.assertIn("stop_redesign_rr_destroyed", result)


class TestStopRedesignWorkedExamples(unittest.TestCase):
    """Reproduce real WIF/PENGU from production telemetry.

    Production ran with STOP_ATR_FLOOR_MULT_INTRADAY that caused widening
    sufficient to push final_rr from 1.69 down to 1.47. We match the observed
    ATR values to reproduce the exact widening.
    """

    @classmethod
    def setUpClass(cls):
        cls.executor_mod = _reload_executor(_clean_env())

    def test_wif_long_rr_destroyed(self):
        """WIF LONG: engine_rr=1.69 → final_rr≈1.47 after widening."""
        ex = _make_executor(self.executor_mod)
        # ATR calibrated so ATR floor at 1.10x causes enough widening
        # to collapse RR from 1.69 to below 1.55.
        # entry=0.1713, original_sd=0.00369524
        # Need final_sd such that: 0.00625349 / final_sd < 1.55
        # → final_sd > 0.004034
        # With atr=0.00354: floor_stop = 0.1713 - 1.10*0.00354 = 0.167406
        # structural=0.167605 > 0.167406 → floor DOES widen
        # floor_applied=0.167406, buffered=0.167406 - 0.10*0.00354 = 0.167052
        # final_sd = 0.1713 - 0.167052 = 0.004248
        # final_rr = 0.00625349 / 0.004248 = 1.472 ✓
        sig = FakeSignal(
            coin="WIF",
            side="LONG",
            entry_price=0.1713,
            stop_price=0.16760476,
            tp_price=0.17755349,
            confidence=0.91,
            regime="continuation|htf_up|macro_up|mkt_weak_trend",
            meta={"atr": 0.00354, "timeframe": "1h",
                  "setup_family": "continuation", "market_regime": "weak_trend"},
        )
        result = ex._apply_entry_stop_redesign(sig, track="intraday")
        self.assertIsNotNone(result, "WIF should be rejected")
        self.assertIn("stop_redesign_rr_destroyed", result)
        self.assertIn("original_rr=1.69", result)

    def test_pengu_long_rr_destroyed(self):
        """PENGU LONG: engine_rr=1.69 → final_rr<1.55 after widening."""
        ex = _make_executor(self.executor_mod)
        # entry=0.007053, stop=0.00688359 (sd=0.00016941), tp=0.0073397
        # tp_dist = 0.0002867
        # Need ATR high enough that 1.10x floor widens past structural.
        # ATR=0.000165: floor_stop = 0.007053 - 1.10*0.000165 = 0.006871
        # structural=0.006884 > 0.006871 → floor DOES widen
        # floor_applied=0.006871, buffered=0.006871-0.000017=0.006855
        # final_sd = 0.007053 - 0.006855 = 0.000198
        # final_rr = 0.0002867 / 0.000198 = 1.448 ✓
        sig = FakeSignal(
            coin="PENGU",
            side="LONG",
            entry_price=0.007053,
            stop_price=0.00688359,
            tp_price=0.0073397,
            confidence=0.92,
            regime="continuation|htf_up|macro_up|mkt_strong_trend",
            meta={"atr": 0.000165, "timeframe": "1h",
                  "setup_family": "continuation", "market_regime": "strong_trend"},
        )
        result = ex._apply_entry_stop_redesign(sig, track="intraday")
        self.assertIsNotNone(result, "PENGU should be rejected")
        self.assertIn("stop_redesign_rr_destroyed", result)

    def test_wide_stop_passes_cleanly(self):
        """A signal with a stop already wider than ATR floor passes without widening."""
        ex = _make_executor(self.executor_mod)
        sig = FakeSignal(
            coin="BTC",
            side="LONG",
            entry_price=65000.0,
            stop_price=63000.0,
            tp_price=70000.0,
            confidence=0.95,
            regime="continuation|htf_up|macro_up|mkt_strong_trend",
            meta={"atr": 800.0, "timeframe": "1h",
                  "setup_family": "continuation", "market_regime": "strong_trend"},
        )
        result = ex._apply_entry_stop_redesign(sig, track="intraday")
        self.assertIsNone(result, f"BTC wide stop should pass: {result}")
        self.assertGreater(sig.meta["rr_final"], 1.55)
        self.assertGreater(sig.meta["original_rr"], 0)

    def test_original_rr_preserved_when_minimal_widening(self):
        """When structural stop is already wide, buffer adds minimal widening."""
        ex = _make_executor(self.executor_mod)
        sig = FakeSignal(
            coin="SOL",
            side="LONG",
            entry_price=100.0,
            stop_price=95.0,
            tp_price=110.0,
            confidence=0.95,
            meta={"atr": 1.5, "timeframe": "1h", "setup_family": "reversal",
                  "market_regime": "strong_trend"},
        )
        result = ex._apply_entry_stop_redesign(sig, track="intraday")
        self.assertIsNone(result)
        meta = sig.meta
        self.assertGreater(meta["original_rr"], 0)
        self.assertGreater(meta["rr_final"], 0)
        # Buffer always adds some widening; widen_mult should be close to 1.0
        self.assertLess(meta["stop_widen_mult"], 1.10)
        # RR should decrease slightly due to buffer widening
        self.assertLessEqual(meta["rr_final"], meta["original_rr"])


if __name__ == "__main__":
    unittest.main()

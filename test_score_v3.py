"""
Tests for score_v3 shadow scoring model.

Covers: bounds, factor weights, caps, missing fields, recipe tag,
live execution isolation, and analyzer robustness.
"""

import os
import sys
import unittest

from score_v3 import compute_shadow_score_v3


class TestScoreV3Bounds(unittest.TestCase):

    def test_empty_context_bounded(self):
        r = compute_shadow_score_v3({})
        self.assertGreaterEqual(r["score_v3"], 0.0)
        self.assertLessEqual(r["score_v3"], 1.0)

    def test_none_context_bounded(self):
        r = compute_shadow_score_v3(None)
        self.assertGreaterEqual(r["score_v3"], 0.0)

    def test_max_positive_capped_at_1(self):
        ctx = {
            "stop_method": "ob", "fvg_bull": True, "ob_bull": True,
            "macro_regime": "chop", "htf_regime": "down", "market_regime": "chop",
            "session": "ny_open", "symbol": "SOL",
        }
        r = compute_shadow_score_v3(ctx)
        self.assertLessEqual(r["score_v3"], 1.0)

    def test_max_negative_floored_at_0(self):
        ctx = {
            "choch_bear": True, "sweep_bull": True,
            "macro_regime": "up", "htf_regime": "up", "market_regime": "strong_trend",
            "session": "ny_pm", "symbol": "ETH",
        }
        r = compute_shadow_score_v3(ctx)
        self.assertGreaterEqual(r["score_v3"], 0.0)

    def test_version_string(self):
        r = compute_shadow_score_v3({})
        self.assertEqual(r["score_v3_version"], "v3_2026_06_factor_shadow")


class TestScoreV3Structure(unittest.TestCase):

    def test_ob_stop_increases(self):
        base = compute_shadow_score_v3({"stop_method": "atr"})
        ob = compute_shadow_score_v3({"stop_method": "ob"})
        self.assertGreater(ob["score_v3"], base["score_v3"])

    def test_fvg_increases(self):
        base = compute_shadow_score_v3({})
        fvg = compute_shadow_score_v3({"fvg_bull": True})
        self.assertGreater(fvg["score_v3"], base["score_v3"])

    def test_choch_decreases(self):
        base = compute_shadow_score_v3({})
        choch = compute_shadow_score_v3({"choch_bear": True})
        self.assertLess(choch["score_v3"], base["score_v3"])

    def test_sweep_without_fvg_decreases(self):
        base = compute_shadow_score_v3({})
        sweep = compute_shadow_score_v3({"sweep_bull": True})
        self.assertLess(sweep["score_v3"], base["score_v3"])

    def test_sweep_with_fvg_no_penalty(self):
        base = compute_shadow_score_v3({"fvg_bull": True})
        both = compute_shadow_score_v3({"fvg_bull": True, "sweep_bull": True})
        self.assertEqual(both["score_v3"], base["score_v3"])


class TestScoreV3Session(unittest.TestCase):

    def test_ny_open_increases(self):
        base = compute_shadow_score_v3({"session": "dead_zone"})
        ny = compute_shadow_score_v3({"session": "ny_open"})
        self.assertGreater(ny["score_v3"], base["score_v3"])

    def test_ny_pm_decreases(self):
        base = compute_shadow_score_v3({"session": "dead_zone"})
        nypm = compute_shadow_score_v3({"session": "ny_pm"})
        self.assertLess(nypm["score_v3"], base["score_v3"])


class TestScoreV3Symbol(unittest.TestCase):

    def test_preferred_increases(self):
        base = compute_shadow_score_v3({"symbol": "UNKNOWN"})
        pref = compute_shadow_score_v3({"symbol": "SOL"})
        self.assertGreater(pref["score_v3"], base["score_v3"])

    def test_negative_decreases(self):
        base = compute_shadow_score_v3({"symbol": "UNKNOWN"})
        neg = compute_shadow_score_v3({"symbol": "ETH"})
        self.assertLess(neg["score_v3"], base["score_v3"])

    def test_eth_capped_at_065(self):
        ctx = {"symbol": "ETH", "fvg_bull": True, "stop_method": "ob", "session": "ny_open"}
        r = compute_shadow_score_v3(ctx)
        self.assertLessEqual(r["score_v3"], 0.65)

    def test_zec_capped(self):
        ctx = {"symbol": "ZEC", "fvg_bull": True, "session": "ny_open"}
        r = compute_shadow_score_v3(ctx)
        self.assertLessEqual(r["score_v3"], 0.65)

    def test_bnb_capped(self):
        ctx = {"symbol": "BNB", "fvg_bull": True, "session": "ny_open", "stop_method": "ob"}
        r = compute_shadow_score_v3(ctx)
        self.assertLessEqual(r["score_v3"], 0.65)


class TestScoreV3Recipe(unittest.TestCase):

    def test_full_recipe_tag_applied(self):
        ctx = {
            "setup_family": "continuation",
            "fvg_bull": True,
            "stop_method": "ob",
            "session": "ny_open",
            "symbol": "SOL",
            "macro_regime": "chop",
            "market_regime": "chop",
        }
        r = compute_shadow_score_v3(ctx)
        self.assertIn("v3_full_recipe", r["score_v3_tags"])
        self.assertGreaterEqual(r["score_v3"], 0.70)

    def test_recipe_not_applied_bad_session(self):
        ctx = {
            "setup_family": "continuation",
            "fvg_bull": True,
            "session": "ny_pm",
            "symbol": "SOL",
        }
        r = compute_shadow_score_v3(ctx)
        self.assertNotIn("v3_full_recipe", r["score_v3_tags"])

    def test_recipe_not_applied_negative_symbol(self):
        ctx = {
            "setup_family": "continuation",
            "fvg_bull": True,
            "session": "ny_open",
            "symbol": "ETH",
        }
        r = compute_shadow_score_v3(ctx)
        self.assertNotIn("v3_full_recipe", r["score_v3_tags"])

    def test_recipe_not_applied_no_fvg(self):
        ctx = {
            "setup_family": "continuation",
            "session": "ny_open",
            "symbol": "SOL",
            "stop_method": "ob",
        }
        r = compute_shadow_score_v3(ctx)
        self.assertNotIn("v3_full_recipe", r["score_v3_tags"])


class TestScoreV3MissingFields(unittest.TestCase):

    def test_missing_fields_tagged(self):
        r = compute_shadow_score_v3({})
        self.assertIn("missing_symbol", r["score_v3_tags"])
        self.assertIn("missing_session", r["score_v3_tags"])

    def test_partial_context_no_crash(self):
        r = compute_shadow_score_v3({"symbol": "SOL"})
        self.assertGreaterEqual(r["score_v3"], 0.0)


class TestScoreV3Isolation(unittest.TestCase):

    def test_does_not_mutate_input(self):
        ctx = {"symbol": "SOL", "session": "ny_open", "fvg_bull": True}
        original = dict(ctx)
        compute_shadow_score_v3(ctx)
        self.assertEqual(ctx, original)

    def test_not_referenced_by_risk_manager(self):
        import importlib
        for mod in list(sys.modules):
            if "risk_manager" in mod:
                del sys.modules[mod]
        import risk_manager
        import inspect
        src = inspect.getsource(risk_manager.RiskManager.check_signal)
        self.assertNotIn("score_v3", src)

    def test_not_referenced_by_executor_accept(self):
        import importlib
        for mod in list(sys.modules):
            if mod == "executor":
                del sys.modules[mod]
            if mod.startswith("executor_modules"):
                del sys.modules[mod]
        import executor
        import inspect
        src = inspect.getsource(executor.Executor._on_signal_inner)
        self.assertNotIn("score_v3", src)


if __name__ == "__main__":
    unittest.main()

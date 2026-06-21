"""
Tests for score_v2 shadow scoring model.

Covers:
  - Bounded output [0.0, 1.0]
  - Factor weights: predictive vs anti-predictive
  - Caps: reversal, CHoCH, negative symbols
  - Missing fields: no crash, neutral default
  - Tag auditing
  - Live execution unchanged (score_v2 never gates trades)
  - Analysis script handles missing files
  - Recipe monitor identification
"""

import importlib
import os
import sys
import unittest
import tempfile
from unittest.mock import MagicMock

from score_v2 import compute_shadow_score_v2


class TestScoreV2Bounds(unittest.TestCase):

    def test_empty_context_returns_bounded(self):
        result = compute_shadow_score_v2({})
        self.assertGreaterEqual(result["score_v2"], 0.0)
        self.assertLessEqual(result["score_v2"], 1.0)

    def test_max_positive_context_capped_at_1(self):
        ctx = {
            "setup_family": "continuation",
            "fvg_bull": True,
            "ob_bull": True,
            "stop_method": "ob",
            "session": "ny_open",
            "macro_regime": "chop",
            "market_regime": "chop",
            "symbol": "SOL",
            "score": 0.85,
        }
        result = compute_shadow_score_v2(ctx)
        self.assertLessEqual(result["score_v2"], 1.0)

    def test_max_negative_context_floored_at_0(self):
        ctx = {
            "setup_family": "reversal",
            "choch_bear": True,
            "sweep_bull": True,
            "session": "ny_pm",
            "symbol": "ETH",
            "macro_regime": "up",
            "market_regime": "strong_trend",
            "score": 0.95,
        }
        result = compute_shadow_score_v2(ctx)
        self.assertGreaterEqual(result["score_v2"], 0.0)

    def test_version_present(self):
        result = compute_shadow_score_v2({})
        self.assertEqual(result["score_v2_version"], "v2_2026_06_factor_shadow")


class TestScoreV2Factors(unittest.TestCase):

    def test_continuation_fvg_ny_beats_reversal_sweep_nypm(self):
        good = compute_shadow_score_v2({
            "setup_family": "continuation",
            "fvg_bull": True,
            "session": "ny_open",
            "symbol": "SOL",
        })
        bad = compute_shadow_score_v2({
            "setup_family": "reversal",
            "sweep_bull": True,
            "session": "ny_pm",
            "symbol": "ETH",
        })
        self.assertGreater(good["score_v2"], bad["score_v2"],
                           f"Good ({good['score_v2']}) should beat bad ({bad['score_v2']})")

    def test_fvg_adds_score(self):
        without = compute_shadow_score_v2({"setup_family": "continuation"})
        with_fvg = compute_shadow_score_v2({"setup_family": "continuation", "fvg_bull": True})
        self.assertGreater(with_fvg["score_v2"], without["score_v2"])

    def test_ob_stop_adds_score(self):
        without = compute_shadow_score_v2({"setup_family": "continuation", "stop_method": "atr"})
        with_ob = compute_shadow_score_v2({"setup_family": "continuation", "stop_method": "ob"})
        self.assertGreater(with_ob["score_v2"], without["score_v2"])

    def test_reversal_penalized(self):
        cont = compute_shadow_score_v2({"setup_family": "continuation"})
        rev = compute_shadow_score_v2({"setup_family": "reversal"})
        self.assertGreater(cont["score_v2"], rev["score_v2"])

    def test_choch_penalized(self):
        without = compute_shadow_score_v2({"setup_family": "continuation"})
        with_choch = compute_shadow_score_v2({"setup_family": "continuation", "choch_bear": True})
        self.assertGreater(without["score_v2"], with_choch["score_v2"])

    def test_preferred_symbol_bonus(self):
        generic = compute_shadow_score_v2({"setup_family": "continuation", "symbol": "UNKNOWN"})
        preferred = compute_shadow_score_v2({"setup_family": "continuation", "symbol": "SOL"})
        self.assertGreater(preferred["score_v2"], generic["score_v2"])


class TestScoreV2Caps(unittest.TestCase):

    def test_eth_capped_at_075(self):
        ctx = {
            "setup_family": "continuation",
            "fvg_bull": True,
            "session": "ny_open",
            "symbol": "ETH",
        }
        result = compute_shadow_score_v2(ctx)
        self.assertLessEqual(result["score_v2"], 0.75)

    def test_zec_capped_at_075(self):
        ctx = {
            "setup_family": "continuation",
            "fvg_bull": True,
            "session": "ny_open",
            "symbol": "ZEC",
            "stop_method": "ob",
        }
        result = compute_shadow_score_v2(ctx)
        self.assertLessEqual(result["score_v2"], 0.75)

    def test_choch_without_fvg_capped_at_060(self):
        ctx = {
            "setup_family": "continuation",
            "choch_bear": True,
            "session": "ny_open",
            "symbol": "SOL",
        }
        result = compute_shadow_score_v2(ctx)
        self.assertLessEqual(result["score_v2"], 0.60)

    def test_reversal_capped_at_070(self):
        ctx = {
            "setup_family": "reversal",
            "ob_bull": True,
            "session": "asia_open",
            "symbol": "SOL",
        }
        result = compute_shadow_score_v2(ctx)
        self.assertLessEqual(result["score_v2"], 0.70)

    def test_reversal_exception_fvg_ny_open(self):
        ctx = {
            "setup_family": "reversal",
            "fvg_bull": True,
            "session": "ny_open",
            "symbol": "SOL",
        }
        result = compute_shadow_score_v2(ctx)
        # This should NOT be capped at 0.70 (exception applies)
        self.assertNotIn("cap_reversal_070", result["score_v2_tags"])


class TestScoreV2MissingFields(unittest.TestCase):

    def test_none_context(self):
        result = compute_shadow_score_v2(None)
        self.assertGreaterEqual(result["score_v2"], 0.0)
        self.assertIn("missing_symbol", result["score_v2_tags"])

    def test_empty_dict(self):
        result = compute_shadow_score_v2({})
        self.assertIn("missing_symbol", result["score_v2_tags"])
        self.assertIn("missing_setup_family", result["score_v2_tags"])

    def test_partial_context_no_crash(self):
        result = compute_shadow_score_v2({"symbol": "SOL"})
        self.assertGreaterEqual(result["score_v2"], 0.0)


class TestScoreV2Tags(unittest.TestCase):

    def test_positive_tags_present(self):
        ctx = {
            "setup_family": "continuation",
            "fvg_bull": True,
            "session": "ny_open",
        }
        result = compute_shadow_score_v2(ctx)
        self.assertIn("+continuation", result["score_v2_tags"])
        self.assertIn("+fvg", result["score_v2_tags"])
        self.assertIn("+ny_open", result["score_v2_tags"])

    def test_negative_tags_present(self):
        ctx = {
            "setup_family": "reversal",
            "session": "ny_pm",
            "symbol": "ETH",
        }
        result = compute_shadow_score_v2(ctx)
        self.assertIn("-reversal", result["score_v2_tags"])
        self.assertIn("-session_ny_pm", result["score_v2_tags"])
        self.assertIn("-negative_symbol", result["score_v2_tags"])

    def test_synergy_tag(self):
        ctx = {
            "setup_family": "continuation",
            "fvg_bull": True,
            "session": "ny_open",
            "symbol": "SOL",
        }
        result = compute_shadow_score_v2(ctx)
        self.assertIn("+synergy_cont_fvg_ny", result["score_v2_tags"])

    def test_reason_string_has_factors(self):
        ctx = {"setup_family": "continuation", "fvg_bull": True}
        result = compute_shadow_score_v2(ctx)
        self.assertIn("+continuation", result["score_v2_reason"])
        self.assertIn("+fvg", result["score_v2_reason"])


class TestLiveExecutionUnchanged(unittest.TestCase):
    """Score v2 must never affect live trade decisions."""

    def test_score_v2_not_read_by_risk_manager(self):
        """RiskManager.check_signal uses signal.confidence, not score_v2."""
        # Reload risk_manager fresh
        for mod in list(sys.modules):
            if mod in ("risk_manager",):
                del sys.modules[mod]
        os.environ["UNIVERSAL_MIN_CONFIDENCE"] = "0.90"
        os.environ["STARTING_BALANCE"] = "10000"
        import risk_manager as rm

        # check_signal reads signal.confidence, not meta["score_v2"]
        import inspect
        src = inspect.getsource(rm.RiskManager.check_signal)
        self.assertNotIn("score_v2", src)

    def test_score_v2_does_not_mutate_signal_confidence(self):
        """compute_shadow_score_v2 never writes to the input dict."""
        ctx = {"confidence": 0.75, "score": 0.75, "setup_family": "continuation"}
        original_conf = ctx["confidence"]
        compute_shadow_score_v2(ctx)
        self.assertEqual(ctx["confidence"], original_conf)
        self.assertNotIn("score_v2", ctx)


class TestRecipeMonitor(unittest.TestCase):

    def test_full_recipe_identified(self):
        ctx = {
            "setup_family": "continuation",
            "fvg_bull": True,
            "session": "ny_open",
            "symbol": "SOL",
            "stop_method": "ob",
            "macro_regime": "chop",
            "market_regime": "chop",
            "score": 0.80,
        }
        result = compute_shadow_score_v2(ctx)
        # Full recipe should score very high
        self.assertGreater(result["score_v2"], 0.90)
        # Should have multiple synergy tags
        synergy_tags = [t for t in result["score_v2_tags"] if t.startswith("+synergy")]
        self.assertGreaterEqual(len(synergy_tags), 2)

    def test_anti_recipe_scores_low(self):
        ctx = {
            "setup_family": "reversal",
            "choch_bear": True,
            "sweep_bull": True,
            "session": "ny_pm",
            "symbol": "ETH",
            "macro_regime": "up",
            "market_regime": "strong_trend",
            "score": 0.95,
        }
        result = compute_shadow_score_v2(ctx)
        self.assertLess(result["score_v2"], 0.30)


class TestAnalysisScript(unittest.TestCase):

    def test_handles_missing_files(self):
        """tools/research/analyze_shadow_score_v2.py should not crash on missing files."""
        import subprocess
        script = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "tools",
            "research",
            "analyze_shadow_score_v2.py",
        )
        result = subprocess.run(
            [sys.executable, script],
            capture_output=True, text=True, timeout=10,
            cwd=os.path.dirname(os.path.abspath(__file__)) or ".",
            env={
                **os.environ,
                "SHADOW_RESEARCH_CANDIDATES_PATH": "/tmp/nonexistent_shadow_candidates.csv",
                "SHADOW_RESEARCH_EXECUTIONS_PATH": "/tmp/nonexistent_shadow_executions.csv",
                "SHADOW_RESEARCH_OUTCOMES_PATH": "/tmp/nonexistent_shadow_outcomes.csv",
            },
        )
        self.assertEqual(result.returncode, 0, f"Script crashed: {result.stderr}")
        self.assertIn("shadow_research_candidates.csv", result.stdout)


if __name__ == "__main__":
    unittest.main()

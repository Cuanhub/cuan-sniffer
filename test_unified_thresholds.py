"""
test_unified_thresholds.py — Verify the Unified Threshold Framework loads correctly.

Ensures every confidence gate defaults to UNIVERSAL_MIN_CONFIDENCE, all score
pre-filters are at 0.64, RR gates are consolidated, and feature flags are set
for the post-candle-fix live-test phase.

Run: pytest test_unified_thresholds.py -v
"""

import importlib
import os
import sys
import types
import unittest
from unittest.mock import patch


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clean_env(**overrides) -> dict:
    """Return a minimal env dict with only the keys supplied."""
    base = {k: v for k, v in os.environ.items() if not _is_threshold_key(k)}
    base.update(overrides)
    return base


def _is_threshold_key(k: str) -> bool:
    THRESHOLD_KEYS = {
        "UNIVERSAL_MIN_CONFIDENCE",
        "SWING_MIN_CONFIDENCE",
        "SMC_4H_MIN_CONFIDENCE",
        "WEAK_TREND_MIN_CONFIDENCE",
        "CHOP_REVERSAL_MIN_CONFIDENCE",
        "MIN_SIGNAL_CONFIDENCE",
        "REGIME_SCORE_THRESHOLD_STRONG",
        "REGIME_SCORE_THRESHOLD_WEAK",
        "REGIME_SCORE_THRESHOLD_CHOP",
        "MIN_STOP_REDESIGN_RR",
        "MIN_EXECUTION_EFFECTIVE_RR",
        "REGIME_TP_CAP_R",
        "STOP_REDESIGN_RR_TOLERANCE",
        "SMC_ENABLE_4H_LIVE",
        "HARD_BLOCK_CONTINUATION",
        "HARD_BLOCKED_TIMEFRAMES",
    }
    return k in THRESHOLD_KEYS


def _reload_module(name: str, env: dict):
    """Re-import a module with a patched environment."""
    with patch.dict(os.environ, env, clear=True):
        if name in sys.modules:
            del sys.modules[name]
        mod = importlib.import_module(name)
    return mod


# ── UNIVERSAL_MIN_CONFIDENCE ───────────────────────────────────────────────────

class TestUniversalMinConfidence(unittest.TestCase):

    def test_default_is_0_90(self):
        """Default UNIVERSAL_MIN_CONFIDENCE must be 0.90 without any env var."""
        mod = _reload_module("signal_engine", _clean_env())
        self.assertAlmostEqual(mod.UNIVERSAL_MIN_CONFIDENCE, 0.90, places=4)

    def test_env_override_respected(self):
        """UNIVERSAL_MIN_CONFIDENCE must honour the env var."""
        mod = _reload_module("signal_engine", _clean_env(UNIVERSAL_MIN_CONFIDENCE="0.92"))
        self.assertAlmostEqual(mod.UNIVERSAL_MIN_CONFIDENCE, 0.92, places=4)

    def test_executor_default_is_0_90(self):
        mod = _reload_module("executor", _clean_env())
        self.assertAlmostEqual(mod.UNIVERSAL_MIN_CONFIDENCE, 0.90, places=4)

    def test_risk_manager_default_is_0_90(self):
        mod = _reload_module("risk_manager", _clean_env())
        self.assertAlmostEqual(mod.UNIVERSAL_MIN_CONFIDENCE, 0.90, places=4)


# ── Per-gate confidence defaults ──────────────────────────────────────────────

class TestConfidenceGateDefaults(unittest.TestCase):
    """Every per-gate confidence variable must fall back to UNIVERSAL_MIN_CONFIDENCE."""

    def setUp(self):
        # Load executor with only UNIVERSAL_MIN_CONFIDENCE set
        self.executor = _reload_module(
            "executor",
            _clean_env(UNIVERSAL_MIN_CONFIDENCE="0.90"),
        )
        self.engine = _reload_module(
            "signal_engine",
            _clean_env(UNIVERSAL_MIN_CONFIDENCE="0.90"),
        )
        self.risk = _reload_module(
            "risk_manager",
            _clean_env(UNIVERSAL_MIN_CONFIDENCE="0.90"),
        )

    def test_swing_min_confidence_defaults_to_universal(self):
        self.assertAlmostEqual(self.executor.SWING_MIN_CONFIDENCE, 0.90, places=4)

    def test_smc_4h_min_confidence_defaults_to_universal(self):
        self.assertAlmostEqual(self.engine.SMC_4H_MIN_CONFIDENCE, 0.90, places=4)

    def test_weak_trend_min_confidence_defaults_to_universal(self):
        self.assertAlmostEqual(self.executor.WEAK_TREND_MIN_CONFIDENCE, 0.90, places=4)

    def test_chop_reversal_min_confidence_defaults_to_universal(self):
        self.assertAlmostEqual(self.executor.CHOP_REVERSAL_MIN_CONFIDENCE, 0.90, places=4)

    def test_min_signal_confidence_defaults_to_universal(self):
        self.assertAlmostEqual(self.risk.MIN_SIGNAL_CONFIDENCE, 0.90, places=4)

    def test_per_gate_override_takes_precedence(self):
        """An explicit per-gate env var must win over UNIVERSAL_MIN_CONFIDENCE."""
        mod = _reload_module(
            "executor",
            _clean_env(UNIVERSAL_MIN_CONFIDENCE="0.90", SWING_MIN_CONFIDENCE="0.95"),
        )
        self.assertAlmostEqual(mod.SWING_MIN_CONFIDENCE, 0.95, places=4)

    def test_universal_propagates_when_per_gate_absent(self):
        """When universal is raised, un-overridden per-gate values must follow."""
        mod = _reload_module(
            "executor",
            _clean_env(UNIVERSAL_MIN_CONFIDENCE="0.93"),
        )
        self.assertAlmostEqual(mod.SWING_MIN_CONFIDENCE, 0.93, places=4)
        self.assertAlmostEqual(mod.WEAK_TREND_MIN_CONFIDENCE, 0.93, places=4)
        self.assertAlmostEqual(mod.CHOP_REVERSAL_MIN_CONFIDENCE, 0.93, places=4)


# ── Score pre-filter thresholds ───────────────────────────────────────────────

class TestScoreThresholds(unittest.TestCase):
    """All REGIME_SCORE_THRESHOLD_* values must default to 0.64."""

    def setUp(self):
        self.engine = _reload_module("signal_engine", _clean_env())

    def test_strong_threshold(self):
        self.assertAlmostEqual(self.engine.REGIME_SCORE_THRESHOLD_STRONG, 0.64, places=4)

    def test_weak_threshold(self):
        self.assertAlmostEqual(self.engine.REGIME_SCORE_THRESHOLD_WEAK, 0.64, places=4)

    def test_chop_threshold(self):
        self.assertAlmostEqual(self.engine.REGIME_SCORE_THRESHOLD_CHOP, 0.64, places=4)


# ── RR gates ──────────────────────────────────────────────────────────────────

class TestRRGates(unittest.TestCase):

    def setUp(self):
        self.engine = _reload_module("signal_engine", _clean_env())
        self.executor = _reload_module("executor", _clean_env())

    def test_engine_min_rr_floor(self):
        """Engine-level RR floor must read from MIN_STOP_REDESIGN_RR, default 1.60."""
        self.assertAlmostEqual(self.engine.ENGINE_MIN_RR_FLOOR, 1.60, places=4)

    def test_executor_min_stop_redesign_rr(self):
        self.assertAlmostEqual(self.executor.MIN_STOP_REDESIGN_RR, 1.60, places=4)

    def test_executor_min_execution_effective_rr(self):
        self.assertAlmostEqual(self.executor.MIN_EXECUTION_EFFECTIVE_RR, 1.55, places=4)

    def test_regime_tp_cap_default_exceeds_execution_rr_floor(self):
        self.assertAlmostEqual(self.executor.REGIME_TP_CAP_R, 1.75, places=4)
        self.assertGreater(self.executor.REGIME_TP_CAP_R, self.executor.MIN_EXECUTION_EFFECTIVE_RR)

    def test_rr_gates_env_override(self):
        env = _clean_env(MIN_STOP_REDESIGN_RR="1.70", MIN_EXECUTION_EFFECTIVE_RR="1.65")
        engine = _reload_module("signal_engine", env)
        executor = _reload_module("executor", env)
        self.assertAlmostEqual(engine.ENGINE_MIN_RR_FLOOR, 1.70, places=4)
        self.assertAlmostEqual(executor.MIN_STOP_REDESIGN_RR, 1.70, places=4)
        self.assertAlmostEqual(executor.MIN_EXECUTION_EFFECTIVE_RR, 1.65, places=4)

    def test_regime_tp_cap_env_override(self):
        executor = _reload_module("executor", _clean_env(REGIME_TP_CAP_R="1.90"))
        self.assertAlmostEqual(executor.REGIME_TP_CAP_R, 1.90, places=4)


# ── Feature flags ─────────────────────────────────────────────────────────────

class TestFeatureFlags(unittest.TestCase):

    def setUp(self):
        self.engine = _reload_module("signal_engine", _clean_env())
        self.executor = _reload_module("executor", _clean_env())

    def test_4h_enabled_by_default(self):
        """SMC_ENABLE_4H_LIVE must default to True post-candle-fix."""
        self.assertTrue(self.engine.SMC_ENABLE_4H_LIVE)

    def test_hard_block_continuation_disabled_by_default(self):
        """HARD_BLOCK_CONTINUATION must default to False post-candle-fix."""
        self.assertFalse(self.executor.HARD_BLOCK_CONTINUATION)

    def test_hard_blocked_timeframes_empty_by_default(self):
        """HARD_BLOCKED_TIMEFRAMES must default to empty set (4h no longer blocked)."""
        self.assertEqual(self.executor.HARD_BLOCKED_TIMEFRAMES, set())

    def test_4h_can_be_disabled_via_env(self):
        mod = _reload_module("signal_engine", _clean_env(SMC_ENABLE_4H_LIVE="false"))
        self.assertFalse(mod.SMC_ENABLE_4H_LIVE)

    def test_hard_block_continuation_can_be_enabled_via_env(self):
        mod = _reload_module("executor", _clean_env(HARD_BLOCK_CONTINUATION="true"))
        self.assertTrue(mod.HARD_BLOCK_CONTINUATION)


# ── validate_thresholds() ─────────────────────────────────────────────────────

class TestValidateThresholds(unittest.TestCase):

    def _run_validate(self, env_extra: dict = None) -> tuple[bool, str]:
        """Run validate_thresholds() and capture stdout + return value."""
        import io, contextlib
        env = _clean_env(UNIVERSAL_MIN_CONFIDENCE="0.90")
        if env_extra:
            env.update(env_extra)
        agent = _reload_module("agent", env)
        buf = io.StringIO()
        with patch.dict(os.environ, env, clear=True):
            with contextlib.redirect_stdout(buf):
                result = agent.validate_thresholds()
        return result, buf.getvalue()

    def test_aligned_returns_true(self):
        """validate_thresholds() must return True when all gates match universal."""
        ok, _ = self._run_validate()
        self.assertTrue(ok)

    def test_aligned_no_warning(self):
        """No [THRESHOLD WARNING] when all gates are aligned."""
        _, output = self._run_validate()
        self.assertNotIn("[THRESHOLD WARNING]", output)

    def test_diverged_returns_false(self):
        """validate_thresholds() must return False when a gate diverges."""
        ok, _ = self._run_validate({"SWING_MIN_CONFIDENCE": "0.85"})
        self.assertFalse(ok)

    def test_diverged_prints_warning(self):
        """validate_thresholds() must print [THRESHOLD WARNING] for diverged gates."""
        _, output = self._run_validate({"SWING_MIN_CONFIDENCE": "0.85"})
        self.assertIn("[THRESHOLD WARNING]", output)

    def test_summary_always_printed(self):
        """validate_thresholds() must always print the threshold summary header."""
        _, output = self._run_validate()
        self.assertIn("THRESHOLD SUMMARY", output)
        self.assertIn("UNIVERSAL_MIN_CONFIDENCE", output)
        self.assertIn("REGIME_TP_CAP_R", output)

    def test_regime_tp_cap_must_exceed_execution_rr_floor(self):
        ok, output = self._run_validate({
            "REGIME_TP_CAP_R": "1.50",
            "MIN_EXECUTION_EFFECTIVE_RR": "1.55",
        })
        self.assertFalse(ok)
        self.assertIn("REGIME_TP_CAP_R", output)
        self.assertIn("must be >", output)


# ── Consistency: engine RR floor == executor redesign RR ──────────────────────

class TestCrossLayerConsistency(unittest.TestCase):
    """Engine and executor RR thresholds must be consistent."""

    def test_engine_and_executor_redesign_rr_match(self):
        env = _clean_env()
        engine = _reload_module("signal_engine", env)
        executor = _reload_module("executor", env)
        self.assertAlmostEqual(
            engine.ENGINE_MIN_RR_FLOOR,
            executor.MIN_STOP_REDESIGN_RR,
            places=4,
            msg="ENGINE_MIN_RR_FLOOR and MIN_STOP_REDESIGN_RR must read from the same env var",
        )

    def test_execution_rr_below_redesign_rr(self):
        """MIN_EXECUTION_EFFECTIVE_RR must be <= MIN_STOP_REDESIGN_RR (tolerance layer)."""
        executor = _reload_module("executor", _clean_env())
        self.assertLessEqual(executor.MIN_EXECUTION_EFFECTIVE_RR, executor.MIN_STOP_REDESIGN_RR)

    def test_regime_tp_cap_above_execution_rr_floor(self):
        """Capped weak/chop TP must remain executable after the RR guard."""
        executor = _reload_module("executor", _clean_env())
        self.assertGreater(executor.REGIME_TP_CAP_R, executor.MIN_EXECUTION_EFFECTIVE_RR)

    def test_universal_confidence_consistent_across_modules(self):
        env = _clean_env(UNIVERSAL_MIN_CONFIDENCE="0.91")
        engine = _reload_module("signal_engine", env)
        executor = _reload_module("executor", env)
        risk = _reload_module("risk_manager", env)
        self.assertAlmostEqual(engine.UNIVERSAL_MIN_CONFIDENCE, 0.91, places=4)
        self.assertAlmostEqual(executor.UNIVERSAL_MIN_CONFIDENCE, 0.91, places=4)
        self.assertAlmostEqual(risk.UNIVERSAL_MIN_CONFIDENCE, 0.91, places=4)


if __name__ == "__main__":
    unittest.main(verbosity=2)

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
        "MIN_SIGNAL_CONFIDENCE",
        "MIN_SIGNAL_SCORE",
        "REGIME_SCORE_THRESHOLD_STRONG",
        "REGIME_SCORE_THRESHOLD_WEAK",
        "REGIME_SCORE_THRESHOLD_CHOP",
        "MIN_STOP_REDESIGN_RR",
        "MIN_EXECUTION_EFFECTIVE_RR",
        "REGIME_TP_CAP_R",
        "STOP_REDESIGN_RR_TOLERANCE",
        "SMC_ENABLE_4H_LIVE",
        "LIVE_ELIGIBILITY_MODEL",
        "HARD_BLOCK_CONTINUATION",
        "HARD_BLOCK_CHOP",
        "BLOCK_CONTINUATION_IN_WEAK_TREND",
        "BLOCK_REVERSAL_IN_WEAK_TREND",
        "BLOCK_CONTINUATION_IN_CHOP",
        "BLOCK_REVERSAL_AGAINST_DUAL_TREND",
        "HARD_BLOCKED_TIMEFRAMES",
    }
    return k in THRESHOLD_KEYS


def _reload_module(name: str, env: dict):
    """Re-import a module with a patched environment."""
    with patch.dict(os.environ, env, clear=True):
        if name in sys.modules:
            del sys.modules[name]
        for dep in list(sys.modules):
            if dep.startswith("executor_modules"):
                del sys.modules[dep]
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

    def test_live_eligibility_model_defaults_to_v3(self):
        self.assertEqual(self.engine.LIVE_ELIGIBILITY_MODEL, "v3")

    def test_live_eligibility_model_can_rollback_to_v1(self):
        mod = _reload_module("signal_engine", _clean_env(LIVE_ELIGIBILITY_MODEL="v1"))
        self.assertEqual(mod.LIVE_ELIGIBILITY_MODEL, "v1")

    def test_live_eligibility_model_invalid_falls_back_to_v3(self):
        mod = _reload_module("signal_engine", _clean_env(LIVE_ELIGIBILITY_MODEL="bad"))
        self.assertEqual(mod.LIVE_ELIGIBILITY_MODEL, "v3")

    def test_live_eligibility_model_can_select_v3c(self):
        mod = _reload_module("signal_engine", _clean_env(LIVE_ELIGIBILITY_MODEL="v3c"))
        self.assertEqual(mod.LIVE_ELIGIBILITY_MODEL, "v3c")

    def test_v3_eligibility_threshold_and_htf_guard(self):
        engine = self.engine.AdaptiveSignalEngine(debug=False)
        self.assertEqual(engine._v3_eligibility_reject_reason(0.80, "down"), "")
        self.assertIn("v3_score_below_threshold", engine._v3_eligibility_reject_reason(0.79, "down"))
        self.assertEqual(engine._v3_eligibility_reject_reason(0.95, "up"), "v3_htf_up_block")
        self.assertEqual(engine._v3_eligibility_reject_reason(None, "down"), "v3_score_missing_for_live_model")
        self.assertEqual(engine._v3_eligibility_reject_reason("bad", "down"), "v3_score_missing_for_live_model")

    def test_v3_htf_up_block_applies_to_long_too(self):
        # Reverted 2026-08-25: a carve-out here used to let LONG-in-htf=up
        # signals through (justified by a 249-signal/5-day shadow sample
        # claiming PF 1.5-3.4). A real full-pipeline replay (real
        # generate_signal(), 60 days, all 8 traded coins) showed the 60
        # LONG signals that carve-out admitted had WR=10.0%, PF=0.163,
        # TotalR=-44.69R. htf==up must block BOTH sides, not just SHORT.
        engine = self.engine.AdaptiveSignalEngine(debug=False)
        self.assertEqual(engine._v3_eligibility_reject_reason(0.95, "up", "LONG"), "v3_htf_up_block")
        self.assertEqual(engine._v3_eligibility_reject_reason(0.95, "up", "SHORT"), "v3_htf_up_block")
        self.assertEqual(engine._v3_eligibility_reject_reason(0.95, "down", "LONG"), "")

    def test_active_quality_stamp_defaults_to_v3(self):
        engine = self.engine.AdaptiveSignalEngine(debug=False)
        meta = {"score_v3": 0.83}
        confidence = engine._stamp_active_quality(
            meta,
            score_v1=0.66,
            confidence_v1=0.66,
            threshold_v1=0.64,
        )
        self.assertAlmostEqual(confidence, 0.83)
        self.assertEqual(meta["active_quality_model"], "v3")
        self.assertAlmostEqual(meta["active_quality_score"], 0.83)
        self.assertAlmostEqual(meta["confidence_v1"], 0.66)

    def test_v3c_eligibility_threshold_and_htf_guard(self):
        mod = _reload_module("signal_engine", _clean_env(LIVE_ELIGIBILITY_MODEL="v3c"))
        engine = mod.AdaptiveSignalEngine(debug=False)
        thresh = mod.LIVE_V3C_ELIGIBILITY_THRESHOLD
        self.assertEqual(engine._v3_eligibility_reject_reason(thresh, "down", threshold=thresh), "")
        self.assertIn(
            "v3_score_below_threshold",
            engine._v3_eligibility_reject_reason(thresh - 0.01, "down", threshold=thresh),
        )
        # htf=up block is model-independent -- applies under v3c too.
        self.assertEqual(engine._v3_eligibility_reject_reason(0.95, "up", "LONG", threshold=thresh), "v3_htf_up_block")

    def test_active_quality_stamp_rolls_to_v3c(self):
        mod = _reload_module("signal_engine", _clean_env(LIVE_ELIGIBILITY_MODEL="v3c"))
        engine = mod.AdaptiveSignalEngine(debug=False)
        meta = {"score_v3c": 0.62}
        confidence = engine._stamp_active_quality(
            meta,
            score_v1=0.66,
            confidence_v1=0.66,
            threshold_v1=0.64,
        )
        self.assertAlmostEqual(confidence, 0.62)
        self.assertEqual(meta["active_quality_model"], "v3c")
        self.assertAlmostEqual(meta["active_quality_score"], 0.62)
        self.assertEqual(meta["active_quality_score_source"], "score_v3c")

    def test_active_quality_stamp_rolls_back_to_v1(self):
        mod = _reload_module("signal_engine", _clean_env(LIVE_ELIGIBILITY_MODEL="v1"))
        engine = mod.AdaptiveSignalEngine(debug=False)
        meta = {"score_v3": 0.91}
        confidence = engine._stamp_active_quality(
            meta,
            score_v1=0.67,
            confidence_v1=0.67,
            threshold_v1=0.64,
        )
        self.assertAlmostEqual(confidence, 0.67)
        self.assertEqual(meta["active_quality_model"], "v1")
        self.assertAlmostEqual(meta["active_quality_score"], 0.67)
        self.assertAlmostEqual(meta["active_quality_threshold"], 0.64)

    def test_generate_swing_signal_stamps_active_quality_after_meta_exists(self):
        import inspect

        src = inspect.getsource(self.engine.AdaptiveSignalEngine.generate_swing_signal)
        meta_idx = src.index("meta = {")
        stamp_idx = src.index("self._stamp_active_quality")
        self.assertLess(meta_idx, stamp_idx)

    def test_signal_payload_shape_stays_executor_compatible(self):
        from dataclasses import fields

        field_names = [field.name for field in fields(self.engine.Signal)]
        self.assertEqual(
            field_names,
            [
                "coin",
                "side",
                "entry_price",
                "stop_price",
                "tp_price",
                "confidence",
                "regime",
                "reason",
                "meta",
            ],
        )

    def test_score_telemetry_still_emits_v2_and_v3(self):
        from signal_engine_modules.score_adapter import compute_all_shadow_scores

        meta = {
            "symbol": "SOL",
            "setup_family": "continuation",
            "session": "ny_open",
            "market_regime": "chop",
            "macro_regime": "chop",
            "htf_regime": "down",
            "fvg_bull": True,
            "stop_method": "ob",
        }
        original = dict(meta)
        fields = compute_all_shadow_scores(meta, "SOL", "LONG", debug=False)
        self.assertEqual(meta, original)
        self.assertIn("score_v2", fields)
        self.assertIn("score_v3", fields)
        self.assertIn("score_v2_reason", fields)
        self.assertIn("score_v3_reason", fields)

    def test_hard_block_continuation_disabled_by_default(self):
        """HARD_BLOCK_CONTINUATION must default to False post-candle-fix."""
        self.assertFalse(self.executor.HARD_BLOCK_CONTINUATION)

    def test_hard_blocked_timeframes_empty_by_default(self):
        """HARD_BLOCKED_TIMEFRAMES must default to empty set (4h no longer blocked)."""
        self.assertEqual(self.executor.HARD_BLOCKED_TIMEFRAMES, set())

    def test_executor_quality_score_resolves_to_active_v3(self):
        sig = types.SimpleNamespace(
            confidence=0.86,
            meta={
                "active_quality_model": "v3",
                "active_quality_score": 0.86,
                "total_score": 0.10,
            },
        )
        self.assertAlmostEqual(self.executor.Executor._signal_total_score(sig), 0.86)

    def test_executor_quality_score_rolls_back_to_v1_total_score(self):
        sig = types.SimpleNamespace(
            confidence=0.86,
            meta={
                "active_quality_model": "v1",
                "active_quality_score": 0.86,
                "total_score": 0.10,
            },
        )
        self.assertAlmostEqual(self.executor.Executor._signal_total_score(sig), 0.10)

    def test_4h_can_be_disabled_via_env(self):
        mod = _reload_module("signal_engine", _clean_env(SMC_ENABLE_4H_LIVE="false"))
        self.assertFalse(mod.SMC_ENABLE_4H_LIVE)

    def test_hard_block_continuation_can_be_enabled_via_env(self):
        mod = _reload_module("executor", _clean_env(HARD_BLOCK_CONTINUATION="true"))
        self.assertTrue(mod.HARD_BLOCK_CONTINUATION)


class TestWeakTrendRegimeProtection(unittest.TestCase):

    def test_weak_trend_blocks_default_on(self):
        filters = _reload_module("executor_modules.session_filters", _clean_env())
        self.assertTrue(filters.BLOCK_CONTINUATION_IN_WEAK_TREND)
        self.assertTrue(filters.BLOCK_REVERSAL_IN_WEAK_TREND)

    def test_regime_block_rejects_v3_loss_lanes_by_default(self):
        filters = _reload_module("executor_modules.session_filters", _clean_env())

        base = {
            "coin": "TEST",
            "side": "SHORT",
            "market_regime": "weak_trend",
            "confidence": 0.93,
            "htf_regime": "down",
            "macro_regime": "down",
            "timeframe": "1h",
        }

        blocked, reason = filters.evaluate_regime_block(
            **base,
            setup_family="continuation",
        )
        self.assertTrue(blocked)
        self.assertEqual(reason, "market_regime_block:continuation_in_weak_trend")

        blocked, reason = filters.evaluate_regime_block(
            **base,
            setup_family="reversal",
        )
        self.assertTrue(blocked)
        self.assertEqual(reason, "market_regime_block:reversal_in_weak_trend")

        swing_base = dict(base, timeframe="4h")
        blocked, reason = filters.evaluate_regime_block(
            **swing_base,
            setup_family="swing",
        )
        self.assertTrue(blocked)
        self.assertEqual(reason, "market_regime_block:reversal_in_weak_trend")

    def test_chop_reversal_still_allowed_when_chop_hard_block_disabled(self):
        filters = _reload_module(
            "executor_modules.session_filters",
            _clean_env(HARD_BLOCK_CHOP="false"),
        )
        blocked, reason = filters.evaluate_regime_block(
            coin="TEST",
            side="SHORT",
            setup_family="reversal",
            market_regime="chop",
            confidence=0.93,
            htf_regime="down",
            macro_regime="down",
            timeframe="1h",
        )
        self.assertFalse(blocked)
        self.assertEqual(reason, "")

    def test_executor_market_regime_uses_local_mkt_before_htf_or_macro(self):
        executor = _reload_module("executor", _clean_env())

        signal = types.SimpleNamespace(
            regime="reversal|htf_chop|macro_chop|mkt_weak_trend",
            meta={
                "market_regime": "weak_trend",
                "regime_htf_1h": "chop",
                "regime_macro_4h": "chop",
            },
        )
        self.assertEqual(executor.Executor._signal_market_regime(signal), "weak_trend")

        signal = types.SimpleNamespace(
            regime="reversal|htf_down|macro_chop|mkt_weak_trend",
            meta={
                "market_regime": "weak_trend",
                "regime_htf_1h": "down",
                "regime_macro_4h": "chop",
            },
        )
        self.assertEqual(executor.Executor._signal_market_regime(signal), "weak_trend")

    def test_executor_market_regime_falls_back_to_mkt_token(self):
        executor = _reload_module("executor", _clean_env())
        signal = types.SimpleNamespace(
            regime="reversal|htf_chop|macro_chop|mkt_weak_trend",
            meta={
                "regime_htf_1h": "chop",
                "regime_macro_4h": "chop",
            },
        )
        self.assertEqual(executor.Executor._signal_market_regime(signal), "weak_trend")


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
        self.assertIn("LIVE_ELIGIBILITY_MODEL", output)

    def test_regime_tp_cap_must_exceed_execution_rr_floor(self):
        ok, output = self._run_validate({
            "REGIME_TP_CAP_R": "1.50",
            "MIN_EXECUTION_EFFECTIVE_RR": "1.55",
        })
        self.assertFalse(ok)
        self.assertIn("REGIME_TP_CAP_R", output)
        self.assertIn("must be >", output)

    def test_live_eligibility_model_invalid_warns(self):
        ok, output = self._run_validate({"LIVE_ELIGIBILITY_MODEL": "v2"})
        self.assertFalse(ok)
        self.assertIn("LIVE_ELIGIBILITY_MODEL", output)
        self.assertIn("supported values: v1, v3, v3c", output)

    def test_live_eligibility_model_v3c_not_warned(self):
        ok, output = self._run_validate({"LIVE_ELIGIBILITY_MODEL": "v3c"})
        self.assertTrue(ok)
        self.assertNotIn("supported values", output)


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

    def test_risk_manager_v3_does_not_reject_legacy_total_score(self):
        env = _clean_env(
            UNIVERSAL_MIN_CONFIDENCE="0.90",
            MIN_SIGNAL_CONFIDENCE="0.90",
            MIN_SIGNAL_SCORE="0.90",
        )
        risk_mod = _reload_module("risk_manager", env)
        risk = risk_mod.RiskManager(strategy_filter=None)
        sig = types.SimpleNamespace(
            coin="SUI",
            entry_price=100.0,
            stop_price=99.0,
            confidence=0.85,
            meta={
                "timeframe": "15m",
                "setup_family": "continuation",
                "market_regime": "weak_trend",
                "total_score": 0.10,
                "effective_threshold": 0.90,
                "active_quality_model": "v3",
                "active_quality_score": 0.85,
                "active_quality_threshold": 0.80,
            },
        )
        decision = risk.check_signal(sig)
        self.assertTrue(decision.approved, decision.reason)
        self.assertIn("quality=v3:0.85", decision.reason)

    def test_risk_manager_v1_keeps_legacy_score_floor(self):
        env = _clean_env(
            UNIVERSAL_MIN_CONFIDENCE="0.90",
            MIN_SIGNAL_CONFIDENCE="0.90",
            MIN_SIGNAL_SCORE="0.90",
        )
        risk_mod = _reload_module("risk_manager", env)
        risk = risk_mod.RiskManager(strategy_filter=None)
        sig = types.SimpleNamespace(
            coin="SUI",
            entry_price=100.0,
            stop_price=99.0,
            confidence=0.95,
            meta={
                "timeframe": "15m",
                "setup_family": "continuation",
                "market_regime": "weak_trend",
                "total_score": 0.10,
                "effective_threshold": 0.90,
                "active_quality_model": "v1",
                "active_quality_score": 0.10,
            },
        )
        decision = risk.check_signal(sig)
        self.assertFalse(decision.approved)
        self.assertIn("score 0.10 < threshold 0.90", decision.reason)


if __name__ == "__main__":
    unittest.main(verbosity=2)

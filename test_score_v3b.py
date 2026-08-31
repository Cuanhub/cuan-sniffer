"""
Tests for score_v3b shadow scoring model (regime-scoring redesign of
score_v3's CONTEXT block — see score_v3b.py docstring for the empirical
basis).

Covers: bounds, the redesigned regime/alignment logic specifically,
missing fields, and live execution isolation (this must never gate a
real trade — it's shadow-only, collecting data toward a future
promotion decision, same as score_v2/score_v3 were before v3 went live).
"""

import inspect
import sys
import unittest

from score_v3b import compute_shadow_score_v3b


class TestScoreV3bBounds(unittest.TestCase):

    def test_empty_context_bounded(self):
        r = compute_shadow_score_v3b({})
        self.assertGreaterEqual(r["score_v3b"], 0.0)
        self.assertLessEqual(r["score_v3b"], 1.0)

    def test_none_context_bounded(self):
        r = compute_shadow_score_v3b(None)
        self.assertGreaterEqual(r["score_v3b"], 0.0)

    def test_does_not_mutate_input(self):
        ctx = {"side": "LONG", "market_regime": "strong_trend"}
        original = dict(ctx)
        compute_shadow_score_v3b(ctx)
        self.assertEqual(ctx, original)

    def test_version_string(self):
        r = compute_shadow_score_v3b({})
        self.assertEqual(r["score_v3b_version"], "v3b_2026_08_25_regime_redesign")


class TestRegimeScoring(unittest.TestCase):
    """The core redesign: market_regime as primary signal, direction-aware
    reversal-alignment penalty. Locks in the specific empirically-derived
    behavior so a future edit can't silently drift back toward the old
    (backwards) logic without a test noticing."""

    def _base(self, **kw):
        ctx = {"side": "LONG", "setup_family": "continuation", "market_regime": "",
               "htf_regime": "", "macro_regime": ""}
        ctx.update(kw)
        return ctx

    def test_strong_trend_rewarded(self):
        r = compute_shadow_score_v3b(self._base(market_regime="strong_trend"))
        self.assertIn("+mkt_strong_trend", r["score_v3b_tags"])

    def test_chop_penalized_harder_for_continuation(self):
        cont = compute_shadow_score_v3b(self._base(setup_family="continuation", market_regime="chop"))
        rev = compute_shadow_score_v3b(self._base(setup_family="reversal", market_regime="chop"))
        self.assertIn("-mkt_chop_continuation", cont["score_v3b_tags"])
        self.assertIn("-mkt_chop", rev["score_v3b_tags"])
        self.assertNotIn("-mkt_chop_continuation", rev["score_v3b_tags"])
        # continuation-in-chop penalty must be strictly larger (data: PF 0.06
        # vs reversal-in-chop PF 0.83-0.86) -- not an arbitrary equal split.
        self.assertLess(cont["score_v3b"], rev["score_v3b"])

    def test_weak_trend_mild_penalty(self):
        r = compute_shadow_score_v3b(self._base(market_regime="weak_trend"))
        self.assertIn("-mkt_weak_trend", r["score_v3b_tags"])

    def test_reversal_aligned_long_penalized(self):
        # LONG reversal already confirmed by both htf+macro as "up" is a
        # late/lagging signal, not a genuine reversal -- empirically bad.
        r = compute_shadow_score_v3b(self._base(
            side="LONG", setup_family="reversal", htf_regime="up", macro_regime="up",
        ))
        self.assertIn("-reversal_late_aligned", r["score_v3b_tags"])

    def test_reversal_aligned_short_penalized(self):
        r = compute_shadow_score_v3b(self._base(
            side="SHORT", setup_family="reversal", htf_regime="down", macro_regime="down",
        ))
        self.assertIn("-reversal_late_aligned", r["score_v3b_tags"])

    def test_reversal_opposed_not_penalized(self):
        # A LONG reversal against a down htf/macro is a genuine counter-trend
        # call -- exactly what a reversal setup is for. Must not be penalized
        # by the alignment term (a separate, unrelated dual-trend hard block
        # already exists for this at the executor level).
        r = compute_shadow_score_v3b(self._base(
            side="LONG", setup_family="reversal", htf_regime="down", macro_regime="down",
        ))
        self.assertNotIn("-reversal_late_aligned", r["score_v3b_tags"])

    def test_continuation_alignment_deliberately_unscored(self):
        # The LONG (n=20, PF=1.33) vs SHORT (n=94, PF=0.66) split for
        # aligned continuation was inconsistent -- deliberately left
        # unscored rather than asserting a direction from a noisy signal.
        r = compute_shadow_score_v3b(self._base(
            side="LONG", setup_family="continuation", htf_regime="up", macro_regime="up",
        ))
        self.assertFalse(any("reversal_late_aligned" in t for t in r["score_v3b_tags"]))


class TestLiveExecutionIsolation(unittest.TestCase):
    """Shadow-only, same guarantee score_v2/score_v3 carry: must never be
    read by anything that gates a real order."""

    def test_not_referenced_by_risk_manager(self):
        for mod in list(sys.modules):
            if "risk_manager" in mod:
                del sys.modules[mod]
        import risk_manager
        src = inspect.getsource(risk_manager.RiskManager.check_signal)
        self.assertNotIn("score_v3b", src)
        self.assertNotIn("score_v3", src)

    def test_not_referenced_by_executor_accept(self):
        for mod in list(sys.modules):
            if mod == "executor":
                del sys.modules[mod]
            if mod.startswith("executor_modules"):
                del sys.modules[mod]
        import executor
        src = inspect.getsource(executor.Executor._on_signal_inner)
        self.assertNotIn("score_v3b", src)
        self.assertNotIn("score_v3", src)

    def test_not_referenced_by_v3_eligibility_gate(self):
        import signal_engine
        src = inspect.getsource(signal_engine.AdaptiveSignalEngine._v3_eligibility_reject_reason)
        self.assertNotIn("score_v3b", src)


if __name__ == "__main__":
    unittest.main()

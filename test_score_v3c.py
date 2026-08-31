"""
Tests for score_v3c — the side-conditioned rescore that is the LIVE
eligibility model when LIVE_ELIGIBILITY_MODEL=v3c (see score_v3c.py
docstring for the empirical basis and score_v3's docstring for why it
was replaced).

Covers: bounds, the side-conditioned CONTEXT/STRUCTURE logic specifically
(locking in the empirically-derived direction so a future edit can't
silently drift back), and live-gating wiring (opposite of score_v3b's
isolation guarantee -- v3c MUST be reachable from the eligibility gate
and the active-quality stamp, since it's the thing gating real trades).
"""

import importlib
import inspect
import os
import sys
import unittest
from unittest.mock import patch

from score_v3c import compute_shadow_score_v3c


class TestScoreV3cBounds(unittest.TestCase):

    def test_empty_context_bounded(self):
        r = compute_shadow_score_v3c({})
        self.assertGreaterEqual(r["score_v3c"], 0.0)
        self.assertLessEqual(r["score_v3c"], 1.0)

    def test_none_context_bounded(self):
        r = compute_shadow_score_v3c(None)
        self.assertGreaterEqual(r["score_v3c"], 0.0)

    def test_does_not_mutate_input(self):
        ctx = {"side": "SHORT", "market_regime": "chop"}
        original = dict(ctx)
        compute_shadow_score_v3c(ctx)
        self.assertEqual(ctx, original)

    def test_version_string(self):
        r = compute_shadow_score_v3c({})
        self.assertEqual(r["score_v3c_version"], "v3c_2026_08_29_eqh_eql_only")


class TestSideConditionedContext(unittest.TestCase):
    """The core redesign: score_v3 applied one flat weight per regime tag
    regardless of side, canceling out opposite-signed real effects. These
    lock in the specific empirically-derived, side-dependent direction."""

    def _base(self, **kw):
        ctx = {"side": "LONG", "market_regime": "", "htf_regime": "", "macro_regime": ""}
        ctx.update(kw)
        return ctx

    def test_macro_chop_rewarded_much_more_for_short(self):
        # SHORT+macro_chop PF=2.14 (n=40) vs LONG+macro_chop PF=0.82 (n=23).
        short_r = compute_shadow_score_v3c(self._base(side="SHORT", macro_regime="chop"))
        long_r = compute_shadow_score_v3c(self._base(side="LONG", macro_regime="chop"))
        self.assertIn("+macro_chop_short", short_r["score_v3c_tags"])
        self.assertIn("+macro_chop_long", long_r["score_v3c_tags"])
        self.assertGreater(short_r["score_v3c"] - 0.50, long_r["score_v3c"] - 0.50)

    def test_htf_down_penalized_for_short_trend_aligned_trap(self):
        # SHORT+htf_down (trend-aligned) PF=0.427 (n=92) vs not-aligned
        # PF=1.209 (n=95) -- a late-entry/chasing trap, must be penalized.
        r = compute_shadow_score_v3c(self._base(side="SHORT", htf_regime="down"))
        self.assertIn("-htf_down_short_aligned", r["score_v3c_tags"])
        self.assertLess(r["score_v3c"], 0.50)

    def test_htf_down_still_rewarded_for_long(self):
        # LONG+htf_down is rare (n=2 in the diagnosis) -- too thin to
        # override, left at score_v3's original direction.
        r = compute_shadow_score_v3c(self._base(side="LONG", htf_regime="down"))
        self.assertIn("+htf_down_long", r["score_v3c_tags"])
        self.assertGreater(r["score_v3c"], 0.50)

    def test_macro_up_penalized_for_short(self):
        r = compute_shadow_score_v3c(self._base(side="SHORT", macro_regime="up"))
        self.assertIn("-macro_up_short", r["score_v3c_tags"])
        self.assertLess(r["score_v3c"], 0.50)

    def test_macro_up_not_penalized_for_long(self):
        # score_v3 penalized macro_up equally for both sides (-0.06); for
        # LONG this was empirically backwards (PF 0.961 vs 0.895 for
        # macro!=up, n=293/25) -- removed, not flipped to a bonus.
        r = compute_shadow_score_v3c(self._base(side="LONG", macro_regime="up"))
        self.assertNotIn("-macro_up_short", r["score_v3c_tags"])
        self.assertEqual(r["score_v3c"], 0.50)


class TestTrendConfirmedStructure(unittest.TestCase):
    """EQH/EQL modifiers, added 2026-08-28 after reproducing on two
    non-overlapping out-of-sample windows (see score_v3c.py docstring for
    the exact numbers). A BOS-bull+htf-up bonus was tried alongside these
    but removed 2026-08-29: htf=="up" is unconditionally hard-blocked at
    the live eligibility gate regardless of score, so it could never
    affect a real trade, and a full-pipeline replay of the underlying
    carve-out (real generate_signal(), real outcomes) failed out-of-sample
    (window 1 PF=1.165 n=186, window 2/OOS PF=0.875 n=134 -- net-negative,
    blended AvgR=+0.026R, essentially breakeven). bos_bull/bos_bear must
    stay unscored in this model; only test that here, plus EQH/EQL."""

    def _base(self, **kw):
        ctx = {"side": "LONG", "htf_regime": "", "macro_regime": "", "market_regime": ""}
        ctx.update(kw)
        return ctx

    def test_bos_bull_never_scored(self):
        # Removed 2026-08-29 -- see class docstring. Locks in that a
        # future edit can't silently reintroduce this without deliberate
        # re-validation.
        r = compute_shadow_score_v3c(self._base(bos_bull=True, htf_regime="up"))
        self.assertFalse(any("bos_bull" in t for t in r["score_v3c_tags"]))
        self.assertEqual(r["score_v3c"], compute_shadow_score_v3c(self._base(htf_regime="up"))["score_v3c"])

    def test_bos_bear_never_scored(self):
        # No positive edge found for bos_bear in either out-of-sample
        # window, even trend-aligned (htf=down) -- deliberately unscored.
        r = compute_shadow_score_v3c(self._base(side="SHORT", bos_bear=True, htf_regime="down"))
        self.assertFalse(any("bos_bear" in t for t in r["score_v3c_tags"]))

    def test_eq_high_rewarded_long_penalized_short(self):
        # Reproduced finding: EQH cluster precedes upside more often than
        # the naive "equal highs = bearish liquidity grab" reading -- the
        # opposite of textbook SMC. Rewarded for LONG, penalized for SHORT.
        long_r = compute_shadow_score_v3c(self._base(side="LONG", eq_high=True))
        short_r = compute_shadow_score_v3c(self._base(side="SHORT", eq_high=True))
        self.assertIn("+eq_high_long", long_r["score_v3c_tags"])
        self.assertIn("-eq_high_short", short_r["score_v3c_tags"])

    def test_eq_low_rewarded_long_penalized_short_smaller_weight(self):
        # Directionally consistent both windows but unstable magnitude --
        # weighted at roughly half of eq_high's confidence.
        long_r = compute_shadow_score_v3c(self._base(side="LONG", eq_low=True))
        short_r = compute_shadow_score_v3c(self._base(side="SHORT", eq_low=True))
        self.assertIn("+eq_low_long", long_r["score_v3c_tags"])
        self.assertIn("-eq_low_short", short_r["score_v3c_tags"])
        eq_high_effect = compute_shadow_score_v3c(self._base(side="LONG", eq_high=True))["score_v3c"] - 0.50
        eq_low_effect = long_r["score_v3c"] - 0.50
        self.assertLess(eq_low_effect, eq_high_effect)


class TestStructureReweighted(unittest.TestCase):

    def test_ob_trigger_inverted_to_penalty(self):
        # score_v3 rewarded +ob (+0.04); empirically inverted (PF 0.79 with
        # vs 1.37 without, n=414/91) -- must now be a penalty.
        r = compute_shadow_score_v3c({"side": "LONG", "ob_bull": True})
        self.assertIn("-ob", r["score_v3c_tags"])
        self.assertLess(r["score_v3c"], 0.50)

    def test_ob_stop_weight_reduced_not_dominant(self):
        # score_v3's single largest weight (+0.14) had near-zero/slightly
        # negative real edge -- cut to +0.02, must no longer dominate.
        r = compute_shadow_score_v3c({"side": "LONG", "stop_method": "ob_structural"})
        self.assertIn("+ob_stop", r["score_v3c_tags"])
        self.assertLess(r["score_v3c"] - 0.50, 0.14)

    def test_choch_still_penalized(self):
        # Confirmed correctly signed in the diagnosis -- unchanged.
        r = compute_shadow_score_v3c({"side": "LONG", "choch_bull": True})
        self.assertIn("-choch", r["score_v3c_tags"])


class TestLiveGatingWiring(unittest.TestCase):
    """v3c is the LIVE model when LIVE_ELIGIBILITY_MODEL=v3c -- opposite
    guarantee of score_v3b's isolation tests. It MUST be reachable from
    the eligibility gate and the active-quality stamp."""

    def test_stamp_active_quality_reads_score_v3c(self):
        for mod in list(sys.modules):
            if mod == "signal_engine":
                del sys.modules[mod]
        with patch.dict(os.environ, {"LIVE_ELIGIBILITY_MODEL": "v3c"}):
            import signal_engine
            importlib.reload(signal_engine)
            src = inspect.getsource(signal_engine.AdaptiveSignalEngine._stamp_active_quality)
            self.assertIn("score_v3c", src)

    def test_eligibility_gate_supports_v3c_field_selection(self):
        import signal_engine
        src = inspect.getsource(signal_engine.AdaptiveSignalEngine.generate_signal)
        self.assertIn('"v3", "v3c"', src.replace("'", '"'))
        src_swing = inspect.getsource(signal_engine.AdaptiveSignalEngine.generate_swing_signal)
        self.assertIn('"v3", "v3c"', src_swing.replace("'", '"'))

    def test_htf_up_block_is_model_independent(self):
        # The htf=up hard block was validated independently of which score
        # model is active (real full-pipeline replay); it must still apply
        # under v3c.
        import signal_engine
        engine = signal_engine.AdaptiveSignalEngine(debug=False)
        self.assertEqual(
            engine._v3_eligibility_reject_reason(0.99, "up", "LONG", threshold=0.1),
            "v3_htf_up_block",
        )


if __name__ == "__main__":
    unittest.main()

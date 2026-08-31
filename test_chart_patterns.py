import unittest

import pandas as pd

from chart_patterns import add_chart_pattern_features, _fallback_pivots, _is_monotonic
from signal_engine import AdaptiveSignalEngine


def _base_frame(n=90, price=100.0, trend_up=False, trend_down=False):
    rows = []
    for i in range(n):
        rows.append({
            "time": pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(hours=i),
            "open": price - 0.1,
            "high": price + 0.5,
            "low": price - 0.5,
            "close": price,
            "volume": 1000.0,
            "atr_14": 1.0,
            "body_pct": 0.40,
            "trend_up": bool(trend_up),
            "trend_down": bool(trend_down),
            "swing_high": False,
            "swing_low": False,
            "vol_spike": False,
        })
    return pd.DataFrame(rows)


class TestChartPatterns(unittest.TestCase):
    def test_confirmed_double_bottom_marks_long_reversal(self):
        df = _base_frame(trend_down=True)

        for i in range(35):
            p = 120.0 - i * 0.75
            df.loc[i, ["open", "high", "low", "close"]] = [p + 0.1, p + 0.6, p - 0.6, p]
        for i in range(35, 66):
            p = 92.0 + 4.0 * (1 - abs(i - 50) / 15)
            df.loc[i, ["open", "high", "low", "close"]] = [p - 0.1, p + 0.5, p - 0.5, p]
        for i in range(66, 89):
            p = 92.0 + (i - 66) * 0.18
            df.loc[i, ["open", "high", "low", "close"]] = [p - 0.1, p + 0.5, p - 0.5, p]

        df.loc[35, ["open", "high", "low", "close", "swing_low"]] = [91.0, 92.0, 90.0, 90.8, True]
        df.loc[50, ["open", "high", "low", "close", "swing_high"]] = [95.5, 96.4, 95.0, 96.0, True]
        df.loc[65, ["open", "high", "low", "close", "swing_low"]] = [91.0, 92.0, 90.2, 90.8, True]
        df.loc[88, ["open", "high", "low", "close"]] = [95.0, 95.5, 94.8, 95.2]
        df.loc[89, ["open", "high", "low", "close", "volume", "body_pct", "vol_spike"]] = [
            96.5, 98.8, 96.4, 98.5, 2000.0, 0.70, True,
        ]

        out = add_chart_pattern_features(df)
        row = out.iloc[-1]

        self.assertEqual(row["chart_pattern"], "double_bottom")
        self.assertEqual(row["chart_pattern_family"], "reversal")
        self.assertEqual(row["chart_pattern_side"], "LONG")
        self.assertTrue(bool(row["chart_pattern_confirmed"]))
        self.assertGreater(float(row["chart_pattern_score"]), 0.0)

    def test_confirmed_ascending_triangle_marks_long_continuation(self):
        df = _base_frame(trend_up=True)
        for i in range(len(df)):
            p = 80.0 + i * 0.15
            df.loc[i, ["open", "high", "low", "close"]] = [p - 0.1, p + 0.4, p - 0.4, p]

        for idx, high in [(55, 105.0), (65, 105.2), (75, 104.9)]:
            df.loc[idx, ["high", "close", "swing_high"]] = [high, high - 0.5, True]
        for idx, low in [(58, 97.0), (68, 99.0), (78, 101.5)]:
            df.loc[idx, ["low", "close", "swing_low"]] = [low, low + 0.5, True]
        df.loc[88, ["open", "high", "low", "close"]] = [103.5, 104.0, 103.0, 103.6]
        df.loc[89, ["open", "high", "low", "close", "volume", "body_pct", "vol_spike"]] = [
            105.6, 107.0, 105.4, 106.7, 2200.0, 0.75, True,
        ]

        out = add_chart_pattern_features(df)
        row = out.iloc[-1]

        self.assertEqual(row["chart_pattern"], "ascending_triangle")
        self.assertEqual(row["chart_pattern_family"], "continuation")
        self.assertEqual(row["chart_pattern_side"], "LONG")
        self.assertTrue(bool(row["chart_pattern_confirmed"]))
        self.assertGreater(float(row["chart_pattern_score"]), 0.0)


class TestSignalEngineChartPatternScoring(unittest.TestCase):
    def test_aligned_confirmed_pattern_adds_confluence_score(self):
        engine = AdaptiveSignalEngine(debug=False)
        score, notes = engine._score_chart_pattern(
            {
                "chart_pattern": "double_bottom",
                "chart_pattern_family": "reversal",
                "chart_pattern_side": "LONG",
                "chart_pattern_score": 0.075,
                "chart_pattern_confirmed": True,
                "chart_pattern_volume_confirmed": True,
                "chart_pattern_candle_confirmed": True,
            },
            "LONG",
            "reversal",
        )

        self.assertGreater(score, 0.0)
        self.assertIn("chart_pattern_double_bottom_confirmed", notes)

    def test_conflicting_confirmed_pattern_penalizes_opposite_side(self):
        engine = AdaptiveSignalEngine(debug=False)
        score, notes = engine._score_chart_pattern(
            {
                "chart_pattern": "double_top",
                "chart_pattern_family": "reversal",
                "chart_pattern_side": "SHORT",
                "chart_pattern_score": 0.075,
                "chart_pattern_confirmed": True,
                "chart_pattern_volume_confirmed": True,
                "chart_pattern_candle_confirmed": True,
            },
            "LONG",
            "reversal",
        )

        self.assertLess(score, 0.0)
        self.assertIn("chart_pattern_conflict_double_top_short", notes)

    def test_signal_governance_passes_independent_thesis(self):
        engine = AdaptiveSignalEngine(debug=False)
        ok, reason, summary, notes = engine._govern_signal_thesis(
            score=0.74,
            notes=[
                "continuation_bull_trigger",
                "htf_up_aligned",
                "session_vwap_trend_strength_0.014",
            ],
            triggers={"bos_bull": True},
            side="LONG",
            setup_family="continuation",
        )

        self.assertTrue(ok)
        self.assertEqual(reason, "")
        self.assertEqual(summary["governance_reason"], "pass")
        self.assertIn("structure", summary["edge_buckets"])
        self.assertIn("regime", summary["edge_buckets"])
        self.assertIn("price_location", summary["edge_buckets"])
        self.assertIn("signal_governance_pass", notes)

    def test_signal_governance_blocks_thin_cosmetic_stack(self):
        engine = AdaptiveSignalEngine(debug=False)
        ok, reason, summary, notes = engine._govern_signal_thesis(
            score=0.74,
            notes=["continuation_bull_trigger", "strong_body_0.71"],
            triggers={"bos_bull": True},
            side="LONG",
            setup_family="continuation",
        )

        self.assertFalse(ok)
        self.assertTrue(reason.startswith("signal_governance_block:thin_independent_thesis"))
        self.assertEqual(summary["independent_bucket_count"], 1)
        self.assertIn(reason, notes)


class TestFallbackPivotAmplitudeFilter(unittest.TestCase):
    """_fallback_pivots must apply the same ATR amplitude filter
    detect_swings() uses (2026-08-25 fix) -- otherwise it re-admits noise
    exactly when the stricter primary method correctly found nothing."""

    def test_low_amplitude_noise_filtered_out(self):
        # Flat, low-amplitude wiggle: high/low swing of ~0.2 against
        # atr_14=1.0 (min required amplitude = 0.5*1.0 = 0.5) -- should be
        # filtered out entirely, not treated as real pivots.
        n = 20
        rows = []
        for i in range(n):
            wiggle = 0.1 if i % 2 == 0 else -0.1
            rows.append({"high": 100.0 + wiggle + 0.05, "low": 100.0 + wiggle - 0.05,
                         "close": 100.0 + wiggle, "atr_14": 1.0})
        df = pd.DataFrame(rows)
        pivots = _fallback_pivots(df, "high", lookback=20)
        self.assertEqual(pivots, [])

    def test_real_amplitude_swing_still_detected(self):
        # A genuine swing high with amplitude well above 0.5*ATR must
        # still be found.
        n = 20
        rows = []
        for i in range(n):
            if i == 10:
                price = 110.0  # clear spike
            else:
                price = 100.0 - abs(i - 10) * 0.05
            rows.append({"high": price + 0.2, "low": price - 0.2, "close": price, "atr_14": 1.0})
        df = pd.DataFrame(rows)
        pivots = _fallback_pivots(df, "high", lookback=20)
        self.assertTrue(any(idx == 10 for idx, _ in pivots))

    def test_missing_atr_column_no_filter_no_crash(self):
        # Graceful degradation matches detect_swings()'s has_atr behavior:
        # no ATR column means no amplitude filter, not a crash.
        n = 10
        rows = [{"high": 100.0 + (0.1 if i == 5 else 0.0), "low": 99.9, "close": 100.0} for i in range(n)]
        df = pd.DataFrame(rows)
        pivots = _fallback_pivots(df, "high", lookback=10)
        self.assertIsInstance(pivots, list)


class TestTriangleMonotonicityFix(unittest.TestCase):
    """Triangle convergence must require every pivot to move the right
    way, not just the first-vs-last endpoints (2026-08-25 fix)."""

    def test_dip_then_recover_is_not_ascending(self):
        # 100 -> 80 -> 105: endpoints look "ascending" (105 > 100) but the
        # middle point breaks it -- must NOT count as a real trendline.
        self.assertFalse(_is_monotonic([100.0, 80.0, 105.0], "up"))

    def test_spike_then_fade_is_not_descending(self):
        self.assertFalse(_is_monotonic([100.0, 120.0, 95.0], "down"))

    def test_genuine_ascending_sequence_passes(self):
        self.assertTrue(_is_monotonic([90.0, 95.0, 100.0], "up"))

    def test_genuine_descending_sequence_passes(self):
        self.assertTrue(_is_monotonic([110.0, 105.0, 100.0], "down"))

    def test_flat_sequence_fails_both_directions(self):
        self.assertFalse(_is_monotonic([100.0, 100.0, 100.0], "up"))
        self.assertFalse(_is_monotonic([100.0, 100.0, 100.0], "down"))


class TestChartPatternPerfGate(unittest.TestCase):
    """CHART_PATTERN_SCORING_ENABLED=false previously did not skip the
    expensive computation (~94% of generate_signal's runtime) -- it only
    discarded the resulting score afterward via _score_chart_pattern's own
    guard. Fixed 2026-08-28: _build_feature_frame now skips the call
    entirely when disabled, relying on its existing default-column
    fill-in (flag_cols/text_cols_defaults) to keep the frame valid."""

    @staticmethod
    def _synthetic_ohlcv(n=260):
        base = pd.Timestamp("2026-01-01", tz="UTC")
        rows = []
        for i in range(n):
            price = 100.0 + (i % 20) * 0.3
            rows.append({
                "time": base + pd.Timedelta(hours=i),
                "open": price - 0.1, "high": price + 0.6,
                "low": price - 0.6, "close": price,
                "volume": 1000.0 + i,
            })
        return pd.DataFrame(rows)

    def test_disabled_skips_computation_entirely(self):
        import signal_engine
        call_count = {"n": 0}
        original = signal_engine.add_chart_pattern_features

        def counting_wrapper(df):
            call_count["n"] += 1
            return original(df)

        signal_engine.add_chart_pattern_features = counting_wrapper
        signal_engine.CHART_PATTERN_SCORING_ENABLED = False
        try:
            engine = AdaptiveSignalEngine(debug=False)
            feat = engine._build_feature_frame(self._synthetic_ohlcv(), min_len=200)
            self.assertFalse(feat.empty)
            self.assertEqual(call_count["n"], 0)
        finally:
            signal_engine.add_chart_pattern_features = original
            signal_engine.CHART_PATTERN_SCORING_ENABLED = True

    def test_disabled_still_fills_valid_default_columns(self):
        import signal_engine
        signal_engine.CHART_PATTERN_SCORING_ENABLED = False
        try:
            engine = AdaptiveSignalEngine(debug=False)
            feat = engine._build_feature_frame(self._synthetic_ohlcv(), min_len=200)
            self.assertFalse(feat.empty)
            row = feat.iloc[-1]
            self.assertEqual(row["chart_pattern"], "")
            self.assertEqual(float(row["chart_pattern_score"]), 0.0)
            self.assertFalse(bool(row["chart_pattern_confirmed"]))
        finally:
            signal_engine.CHART_PATTERN_SCORING_ENABLED = True

    def test_enabled_still_calls_computation(self):
        import signal_engine
        call_count = {"n": 0}
        original = signal_engine.add_chart_pattern_features

        def counting_wrapper(df):
            call_count["n"] += 1
            return original(df)

        signal_engine.add_chart_pattern_features = counting_wrapper
        signal_engine.CHART_PATTERN_SCORING_ENABLED = True
        try:
            engine = AdaptiveSignalEngine(debug=False)
            engine._build_feature_frame(self._synthetic_ohlcv(), min_len=200)
            self.assertEqual(call_count["n"], 1)
        finally:
            signal_engine.add_chart_pattern_features = original


if __name__ == "__main__":
    unittest.main(verbosity=2)

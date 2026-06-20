import unittest

import pandas as pd

from chart_patterns import add_chart_pattern_features
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


if __name__ == "__main__":
    unittest.main(verbosity=2)

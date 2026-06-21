"""
Tests for replay_shadow_trades.py replay logic.

Uses synthetic candles — no network access required.
"""

import sys
import os
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "tools" / "research"))
from replay_shadow_trades import (  # type: ignore[import-not-found]
    replay_trade, _parse_row, is_v3_full_recipe, generate_summary,
    load_shadow_trades,
)


def _candles_from(start_ms, n, bar_ms=3600000, base=100.0, move=0.5):
    """Generate n synthetic candles starting at start_ms."""
    candles = []
    for i in range(n):
        o = base + i * move * 0.1
        h = o + move
        l = o - move * 0.5
        c = o + move * 0.3
        candles.append({
            "time": start_ms + i * bar_ms,
            "open": round(o, 4),
            "high": round(h, 4),
            "low": round(l, 4),
            "close": round(c, 4),
            "volume": 1000.0,
        })
    return candles


class TestReplayLong(unittest.TestCase):

    def test_long_tp_hit(self):
        # Entry=100, TP=102, SL=99. Candle with high=103 should hit TP.
        candles = [
            {"time": 1000, "open": 100, "high": 100.5, "low": 99.5, "close": 100.2, "volume": 1},
            {"time": 2000, "open": 100.2, "high": 103, "low": 100, "close": 102.5, "volume": 1},
        ]
        r = replay_trade(entry=100, stop=99, tp=102, side="LONG",
                         candles=candles, signal_epoch_ms=500, max_hold_bars=24)
        self.assertIsNotNone(r)
        self.assertEqual(r["exit_reason"], "tp")
        self.assertAlmostEqual(r["realized_R"], 2.0, places=2)

    def test_long_sl_hit(self):
        candles = [
            {"time": 1000, "open": 100, "high": 100.5, "low": 98, "close": 98.5, "volume": 1},
        ]
        r = replay_trade(entry=100, stop=99, tp=103, side="LONG",
                         candles=candles, signal_epoch_ms=500, max_hold_bars=24)
        self.assertIsNotNone(r)
        self.assertEqual(r["exit_reason"], "stop")
        self.assertAlmostEqual(r["realized_R"], -1.0, places=2)


class TestReplayShort(unittest.TestCase):

    def test_short_tp_hit(self):
        candles = [
            {"time": 1000, "open": 100, "high": 100.5, "low": 97, "close": 97.5, "volume": 1},
        ]
        r = replay_trade(entry=100, stop=101, tp=98, side="SHORT",
                         candles=candles, signal_epoch_ms=500, max_hold_bars=24)
        self.assertIsNotNone(r)
        self.assertEqual(r["exit_reason"], "tp")
        self.assertAlmostEqual(r["realized_R"], 2.0, places=2)

    def test_short_sl_hit(self):
        candles = [
            {"time": 1000, "open": 100, "high": 102, "low": 99.5, "close": 101.5, "volume": 1},
        ]
        r = replay_trade(entry=100, stop=101, tp=97, side="SHORT",
                         candles=candles, signal_epoch_ms=500, max_hold_bars=24)
        self.assertIsNotNone(r)
        self.assertEqual(r["exit_reason"], "stop")
        self.assertAlmostEqual(r["realized_R"], -1.0, places=2)


class TestSameCandleConflict(unittest.TestCase):

    def test_sl_first_default(self):
        # Both TP and SL hit on same candle
        candles = [
            {"time": 1000, "open": 100, "high": 105, "low": 95, "close": 100, "volume": 1},
        ]
        r = replay_trade(entry=100, stop=96, tp=104, side="LONG",
                         candles=candles, signal_epoch_ms=500, max_hold_bars=24, tp_first=False)
        self.assertEqual(r["exit_reason"], "stop")
        self.assertTrue(r["same_candle_conflict"])

    def test_tp_first_override(self):
        candles = [
            {"time": 1000, "open": 100, "high": 105, "low": 95, "close": 100, "volume": 1},
        ]
        r = replay_trade(entry=100, stop=96, tp=104, side="LONG",
                         candles=candles, signal_epoch_ms=500, max_hold_bars=24, tp_first=True)
        self.assertEqual(r["exit_reason"], "tp")
        self.assertTrue(r["same_candle_conflict"])


class TestMaxHold(unittest.TestCase):

    def test_max_hold_closes_at_final(self):
        candles = _candles_from(1000, 5, base=100.0, move=0.1)
        # TP/SL far away so neither hits
        r = replay_trade(entry=100, stop=90, tp=120, side="LONG",
                         candles=candles, signal_epoch_ms=500, max_hold_bars=3)
        self.assertIsNotNone(r)
        self.assertEqual(r["exit_reason"], "max_hold")
        self.assertEqual(r["bars_held"], 3)


class TestInvalidInputs(unittest.TestCase):

    def test_zero_risk_rejected(self):
        candles = [{"time": 1000, "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1}]
        r = replay_trade(entry=100, stop=100, tp=105, side="LONG",
                         candles=candles, signal_epoch_ms=500)
        self.assertIsNone(r)

    def test_wrong_side_tp_rejected(self):
        candles = [{"time": 1000, "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1}]
        r = replay_trade(entry=100, stop=99, tp=98, side="LONG",
                         candles=candles, signal_epoch_ms=500)
        self.assertIsNone(r)

    def test_no_candles_after_signal(self):
        candles = [{"time": 500, "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1}]
        r = replay_trade(entry=100, stop=99, tp=102, side="LONG",
                         candles=candles, signal_epoch_ms=1000)
        self.assertIsNone(r)


class TestRealizedR(unittest.TestCase):

    def test_long_r_calculation(self):
        candles = [
            {"time": 1000, "open": 100, "high": 103, "low": 99.5, "close": 102, "volume": 1},
        ]
        r = replay_trade(entry=100, stop=98, tp=103, side="LONG",
                         candles=candles, signal_epoch_ms=500)
        # risk = 100 - 98 = 2. TP hit at 103. R = (103-100)/2 = 1.5
        self.assertAlmostEqual(r["realized_R"], 1.5, places=2)

    def test_short_r_calculation(self):
        candles = [
            {"time": 1000, "open": 100, "high": 100.5, "low": 96, "close": 97, "volume": 1},
        ]
        r = replay_trade(entry=100, stop=102, tp=97, side="SHORT",
                         candles=candles, signal_epoch_ms=500)
        # risk = 102 - 100 = 2. TP hit at 97. R = (100-97)/2 = 1.5
        self.assertAlmostEqual(r["realized_R"], 1.5, places=2)


class TestV3Recipe(unittest.TestCase):

    def test_full_recipe_detected(self):
        row = {
            "setup_family": "continuation",
            "score_v3": "0.85",
            "symbol": "SOL",
            "session": "ny_open",
            "score_v3_tags": "+fvg,+ob_stop,v3_full_recipe",
        }
        self.assertTrue(is_v3_full_recipe(row))

    def test_not_recipe_no_fvg(self):
        row = {
            "setup_family": "continuation",
            "score_v3": "0.85",
            "symbol": "SOL",
            "session": "ny_open",
            "score_v3_tags": "+ob_stop",
        }
        self.assertFalse(is_v3_full_recipe(row))

    def test_not_recipe_bad_symbol(self):
        row = {
            "setup_family": "continuation",
            "score_v3": "0.85",
            "symbol": "ETH",
            "session": "ny_open",
            "score_v3_tags": "+fvg",
        }
        self.assertFalse(is_v3_full_recipe(row))


class TestSummary(unittest.TestCase):

    def test_summary_with_results(self):
        results = [
            {"realized_R": 1.5, "timestamp": "2026-06-17", "setup_family": "continuation",
             "session": "ny_open", "symbol": "SOL", "market_regime": "chop",
             "score_v1": "0.8", "score_v2": "0.7", "score_v3": "0.85",
             "score_v3_tags": "v3_full_recipe"},
            {"realized_R": -1.0, "timestamp": "2026-06-18", "setup_family": "continuation",
             "session": "ny_pm", "symbol": "ETH", "market_regime": "weak_trend",
             "score_v1": "0.6", "score_v2": "0.4", "score_v3": "0.3",
             "score_v3_tags": ""},
        ]
        summary = generate_summary(results, 2)
        self.assertIn("Win rate:", summary)
        self.assertIn("PROMOTE_V3_LANE = NO", summary)

    def test_summary_empty(self):
        summary = generate_summary([], 1)
        self.assertIn("No trades replayed", summary)


class TestMissingFiles(unittest.TestCase):

    def test_load_missing_file(self):
        # Patch PROJECT_ROOT to prevent fallback to real files
        import replay_shadow_trades as mod
        orig_root = mod.PROJECT_ROOT
        mod.PROJECT_ROOT = Path("/tmp/nonexistent_project_root_test")
        try:
            rows = load_shadow_trades("/tmp/nonexistent_shadow_replay_test.csv")
            self.assertEqual(rows, [])
        finally:
            mod.PROJECT_ROOT = orig_root


class TestParseRow(unittest.TestCase):

    def test_valid_row(self):
        row = {"timestamp": "2026-06-17T12:00:00Z", "symbol": "SOL", "side": "LONG",
               "entry_price": "100", "stop_price": "98", "tp_price": "104"}
        p = _parse_row(row)
        self.assertIsNotNone(p)
        self.assertEqual(p["entry"], 100.0)

    def test_missing_required(self):
        row = {"symbol": "SOL", "side": "LONG"}
        p = _parse_row(row)
        self.assertIsNone(p)


if __name__ == "__main__":
    unittest.main()

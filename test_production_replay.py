"""Tests for production policy replay report logic."""

import io
import sys
import unittest
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "tools" / "research"))
from analyze_production_policy_replay import (  # type: ignore[import-not-found]
    _stats, _sf,
)


class TestStats(unittest.TestCase):

    def test_empty(self):
        s = _stats([])
        self.assertEqual(s["n"], 0)
        self.assertEqual(s["pf"], 0)

    def test_all_wins(self):
        s = _stats([1.5, 1.0, 2.0])
        self.assertEqual(s["n"], 3)
        self.assertAlmostEqual(s["wr"], 100.0)
        self.assertEqual(s["pf"], 999)

    def test_mixed(self):
        s = _stats([1.5, -1.0, 1.0, -1.0])
        self.assertEqual(s["n"], 4)
        self.assertAlmostEqual(s["wr"], 50.0)
        self.assertAlmostEqual(s["pf"], 2.5 / 2.0, places=3)

    def test_max_dd(self):
        # peak=1.0, cumulative hits -2.0, so drawdown from peak = 3.0
        s = _stats([1.0, -3.0, 1.0])
        self.assertAlmostEqual(s["mdd"], 3.0, places=2)

    def test_all_losses(self):
        s = _stats([-1.0, -1.0])
        self.assertAlmostEqual(s["pf"], 0.0)
        self.assertAlmostEqual(s["wr"], 0.0)

    def test_single_trade(self):
        s = _stats([1.5])
        self.assertEqual(s["n"], 1)
        self.assertAlmostEqual(s["wr"], 100.0)

    def test_median_even(self):
        s = _stats([1.0, 2.0])
        self.assertAlmostEqual(s["med"], 1.5)


class TestSafeFloat(unittest.TestCase):

    def test_valid(self):
        self.assertAlmostEqual(_sf("1.5"), 1.5)

    def test_empty(self):
        self.assertAlmostEqual(_sf(""), 0.0)

    def test_none(self):
        self.assertAlmostEqual(_sf(None), 0.0)

    def test_garbage(self):
        self.assertAlmostEqual(_sf("abc"), 0.0)


class TestPromotionCriteria(unittest.TestCase):
    """Validate promotion criteria logic using _stats."""

    def _evaluate_criteria(self, prod_rs, sym_map, day_map):
        s = _stats(prod_rs)
        prof_syms = sum(1 for vals in sym_map.values() if sum(vals) > 0)
        max_sym_pct = (max(sum(v) for v in sym_map.values()) / s["total"] * 100
                       if s["total"] > 0 and sym_map else 0)
        max_day_pct = (max(sum(v) for v in day_map.values()) / s["total"] * 100
                       if s["total"] > 0 and day_map else 0)
        days_pos = sum(1 for v in day_map.values() if sum(v) > 0)
        return {
            "n_pass": s["n"] >= 100,
            "pf_pass": s["pf"] > 1.50,
            "avg_pass": s["avg"] > 0.20,
            "mdd_pass": s["mdd"] < 15,
            "sym_pass": prof_syms >= 3,
            "max_sym_pass": max_sym_pct <= 40,
            "max_day_pass": max_day_pct <= 35,
            "days_pos_pass": days_pos / max(1, len(day_map)) >= 0.60 if day_map else False,
        }

    def test_fails_when_pf_below_threshold(self):
        rs = [-1.0] * 70 + [0.5] * 60
        syms = {"A": rs[:40], "B": rs[40:80], "C": rs[80:]}
        days = {f"day{i}": rs[i*10:(i+1)*10] for i in range(13)}
        c = self._evaluate_criteria(rs, syms, days)
        self.assertTrue(c["n_pass"])
        self.assertFalse(c["pf_pass"])

    def test_fails_when_sample_too_small(self):
        rs = [1.0] * 50
        syms = {"A": rs[:20], "B": rs[20:35], "C": rs[35:]}
        days = {f"day{i}": rs[i*5:(i+1)*5] for i in range(10)}
        c = self._evaluate_criteria(rs, syms, days)
        self.assertFalse(c["n_pass"])

    def test_passes_when_all_conditions_met(self):
        # 120 trades, ~60% WR, interleaved to keep MDD low
        import random
        random.seed(42)
        rs = [1.5] * 72 + [-1.0] * 48
        random.shuffle(rs)
        syms = {"A": rs[:30], "B": rs[30:60], "C": rs[60:90], "D": rs[90:]}
        days = {f"day{i}": rs[i*6:(i+1)*6] for i in range(20)}
        c = self._evaluate_criteria(rs, syms, days)
        self.assertTrue(c["n_pass"])
        self.assertTrue(c["pf_pass"])
        self.assertTrue(c["avg_pass"])
        self.assertTrue(c["mdd_pass"])
        self.assertTrue(c["sym_pass"])

    def test_fails_single_symbol_dominance(self):
        rs = [2.0] * 100 + [-1.0] * 50
        syms = {"A": [2.0] * 80, "B": [2.0] * 10, "C": [2.0] * 10 + [-1.0] * 50}
        days = {f"day{i}": rs[i*10:(i+1)*10] for i in range(15)}
        c = self._evaluate_criteria(rs, syms, days)
        # A has 160R out of 150R total = 107% > 40%
        self.assertFalse(c["max_sym_pass"])


class TestEngineOnlyWarning(unittest.TestCase):

    def test_warning_triggers(self):
        eng = _stats([1.5, 1.0, 1.0, -1.0])
        prod = _stats([0.5, -1.0, -1.0, 0.2])
        self.assertGreater(eng["pf"], 1.5)
        self.assertLess(prod["pf"], 1.5)

    def test_no_warning_when_both_pass(self):
        eng = _stats([2.0, 1.0, -1.0])
        prod = _stats([2.0, 1.0, -1.0])
        self.assertGreater(eng["pf"], 1.5)
        self.assertGreater(prod["pf"], 1.5)


class TestRejectAttribution(unittest.TestCase):

    def test_leaderboard_ranking(self):
        rejected = [
            {"reason": "market_regime_block:chop", "exit_r": 1.5},
            {"reason": "market_regime_block:chop", "exit_r": 1.0},
            {"reason": "stop_redesign_rr_destroyed", "exit_r": -1.0},
            {"reason": "stop_redesign_rr_destroyed", "exit_r": 0.5},
        ]
        reasons = defaultdict(list)
        for r in rejected:
            reasons[r["reason"]].append(r["exit_r"])
        ranked = sorted(reasons, key=lambda k: -sum(reasons[k]))
        self.assertEqual(ranked[0], "market_regime_block:chop")
        self.assertAlmostEqual(sum(reasons[ranked[0]]), 2.5)


class TestAnalyzerHandlesEmpty(unittest.TestCase):

    def test_no_crash_on_missing_file(self):
        import subprocess, os
        proj = str(Path(__file__).resolve().parent)
        result = subprocess.run(
            [sys.executable, os.path.join(proj, "tools/research/analyze_production_policy_replay.py"),
             "--csv", "/tmp/nonexistent_shadow_candidates.csv"],
            capture_output=True, text=True, timeout=10,
            cwd=proj,
            env={**os.environ},
        )
        self.assertEqual(result.returncode, 0, f"stderr: {result.stderr}")
        self.assertIn("No shadow_research_candidates", result.stdout)


if __name__ == "__main__":
    unittest.main()

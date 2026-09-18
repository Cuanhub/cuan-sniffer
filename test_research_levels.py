"""Unit tests for research/levels/ -- detection, gating, storage."""

import csv
import math
import shutil
import unittest
from pathlib import Path

from research.levels import storage
from research.levels.gate import evaluate
from research.levels.levels import (
    detect_levels, find_pivots, log_sigma, sigma_distance, Level,
)


def synth(n=300, base=100.0, step=0.0, noise=0.0):
    out = []
    for i in range(n):
        c = base + step * i + (noise if i % 7 == 0 else 0.0)
        out.append({"t": i * 14400000, "o": c, "h": c * 1.01, "l": c * 0.99, "c": c, "v": 100.0})
    return out


class TestSigma(unittest.TestCase):
    def test_zero_sigma_on_flat_series(self):
        self.assertEqual(log_sigma(synth(), len(synth()) - 1), 0.0)

    def test_sigma_distance_is_symmetric_in_magnitude(self):
        s = 0.05
        self.assertAlmostEqual(sigma_distance(110, 100, s), -sigma_distance(100, 110, s), places=9)

    def test_sigma_distance_guards_bad_input(self):
        self.assertEqual(sigma_distance(0, 100, 0.05), 0.0)
        self.assertEqual(sigma_distance(100, 100, 0.0), 0.0)


class TestPivots(unittest.TestCase):
    def test_finds_an_obvious_low(self):
        b = synth(60)
        b[30]["l"] = 50.0
        _, lows = find_pivots(b, window=3)
        self.assertIn(30, [i for i, _ in lows])

    def test_no_pivots_without_enough_bars(self):
        h, l = find_pivots(synth(4), window=3)
        self.assertEqual((h, l), ([], []))


class TestLevelQuality(unittest.TestCase):
    def test_hold_rate_is_not_always_one(self):
        """Regression: the first implementation judged a break inside the
        touching bar, which required a single candle to dip 1 sigma and close
        3 below. That never happens, so every level scored 100% and the metric
        was informationless."""
        import random
        random.seed(3)
        b = []
        px = 100.0
        for i in range(400):
            px *= math.exp(random.gauss(0, 0.02))
            b.append({"t": i * 14400000, "o": px, "h": px * 1.02, "l": px * 0.98,
                      "c": px, "v": 100.0})
        lv = detect_levels(b, kind="support")
        if lv:
            self.assertTrue(any(L.hold_rate < 0.999 for L in lv),
                            "every level scored a perfect hold rate")

    def test_level_requires_min_touches(self):
        self.assertTrue(all(L.touches >= 2 for L in detect_levels(synth(300), kind="support")))


class TestGate(unittest.TestCase):
    def test_returns_none_when_sigma_undefined(self):
        b = synth(200)
        self.assertIsNone(evaluate(b, 150, 100.0))

    def test_non_firing_evaluation_is_still_returned(self):
        """Near-misses must be recorded, not dropped -- they are what a
        labelling pass needs to locate the gate's real boundary."""
        import random
        random.seed(11)
        b = []
        px = 100.0
        for i in range(300):
            px *= math.exp(random.gauss(0, 0.01))
            b.append({"t": i * 14400000, "o": px, "h": px * 1.01, "l": px * 0.99,
                      "c": px, "v": 100.0})
        f = evaluate(b, 250, px * 0.9)
        self.assertIsNotNone(f)
        self.assertIn("gate_fires", f)
        for k in ("vol_ratio_recent_ref", "vol_trend_slope", "vol_down_up_ratio"):
            self.assertIn(k, f, "volume features must be captured even though ungated")


class TestStorage(unittest.TestCase):
    def setUp(self):
        self._orig = storage.DATA_DIR
        self._tmp = Path("/tmp/_test_levels_storage")
        if self._tmp.exists():
            shutil.rmtree(self._tmp)
        self._tmp.mkdir(parents=True)
        storage.DATA_DIR = self._tmp

    def tearDown(self):
        storage.DATA_DIR = self._orig
        if self._tmp.exists():
            shutil.rmtree(self._tmp)

    def _ep(self, eid="C_1_0.5"):
        return {k: "" for k in storage.EPISODE_FIELDNAMES} | {
            "episode_id": eid, "coin": "C", "level_price": 0.5}

    def test_episode_append_is_idempotent(self):
        self.assertTrue(storage.append_episode(self._ep()))
        self.assertFalse(storage.append_episode(self._ep()))

    def test_distinct_levels_on_one_bar_are_distinct_episodes(self):
        a = storage.make_episode_id("C", 1000, 0.5)
        b = storage.make_episode_id("C", 1000, 0.7)
        self.assertNotEqual(a, b)
        self.assertTrue(storage.append_episode(self._ep(a)))
        self.assertTrue(storage.append_episode(self._ep(b)))

    def test_widening_preserves_values_and_leaves_no_ragged_rows(self):
        path = self._tmp / "episodes.csv"
        narrow = storage.EPISODE_FIELDNAMES[:6]
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(narrow)
            w.writerow(["E1", "C", "t", "1", "0.5", "support"])
        storage.append_episode(self._ep("E2"))
        with open(path, newline="") as f:
            rows = list(csv.reader(f))
        width = len(storage.EPISODE_FIELDNAMES)
        self.assertTrue(all(len(r) == width for r in rows), f"ragged: {[len(r) for r in rows]}")
        self.assertEqual(rows[1][:6], ["E1", "C", "t", "1", "0.5", "support"])

    def test_labels_are_separate_from_outcomes(self):
        storage.append_episode(self._ep("E9"))
        self.assertTrue(storage.append_label({
            "episode_id": "E9", "labelled_at_utc": "t", "verdict": "take",
            "confidence": "", "note": "n"}))
        self.assertFalse(storage.append_label({
            "episode_id": "E9", "labelled_at_utc": "t", "verdict": "skip",
            "confidence": "", "note": ""}))
        self.assertEqual(storage.resolved_ids(), set())


if __name__ == "__main__":
    unittest.main()

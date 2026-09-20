"""Unit tests for research/levels/ -- detection, gating, storage."""

import csv
import math
import shutil
import unittest
from pathlib import Path

from research.levels import storage
from research.levels.gate import evaluate
from research.levels.levels import (
    detect_levels, find_pivots, log_sigma, sigma_distance,
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


class SameLevelLabelTest(unittest.TestCase):
    """episode_id carries the BAR, so one level gets a new id every bar.

    storage dedupes on episode_id, which cannot see that two rows describe
    the same level. On 2026-09-19 USELESS 1H 0.25817 was held as both a take
    and a skip a day apart, and the filter is validated on "keeps 8 of 11
    takes and admits 0 of 5 skips" -- a level in both halves makes that
    number unanswerable. Relabelling is legitimate; doing it silently is not.
    """

    def setUp(self):
        from research.levels import call
        self.call = call
        self._orig = call._labelled
        call._labelled = lambda: {
            "USELESS-1H_1000_0.25817": {"episode_id": "USELESS-1H_1000_0.25817",
                                        "verdict": "take", "labelled_at_utc": "T1",
                                        "note": "flipped resistance"},
            "BTC-4H_1000_50000": {"episode_id": "BTC-4H_1000_50000",
                                  "verdict": "skip", "labelled_at_utc": "T1", "note": ""},
        }

    def tearDown(self):
        self.call._labelled = self._orig

    def test_same_level_different_bar_is_found(self):
        hits = self.call._same_level_labels("USELESS-1H_9999_0.25817")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["verdict"], "take")

    def test_the_same_episode_is_not_its_own_prior(self):
        self.assertEqual(self.call._same_level_labels("USELESS-1H_1000_0.25817"), [])

    def test_a_different_level_is_not_matched(self):
        self.assertEqual(self.call._same_level_labels("USELESS-1H_9999_0.24"), [])

    def test_a_different_timeframe_is_not_matched(self):
        self.assertEqual(self.call._same_level_labels("USELESS-4H_9999_0.25817"), [])

    def test_a_malformed_id_is_harmless(self):
        self.assertEqual(self.call._same_level_labels("nonsense"), [])


class HoldRuleTest(unittest.TestCase):
    """The 2h hold belongs to flipped resistance, not to ordinary support.

    Broken resistance becoming support is a claim, and holding a channel for
    two hours is the proof. Ordinary support taps and sends -- demanding two
    hours of sitting asks it to behave like the thing it is not, and by the
    time the hold completes price has run and MAX_ENTRY_DRIFT_PCT refuses the
    fill. That is the tap-and-go miss.
    """

    LEVEL = 100.0

    def tap_and_reclaim(self):
        """Price wanders above, dips through the level, then closes back over
        it on the very next candle. The last bar is the forming one that
        level_has_held always discards."""
        bars = []
        for i in range(40):
            c = 104.0 + (i % 5) * 0.4          # varied, so sigma is defined
            bars.append({"t": i, "o": c, "h": c + 0.5, "l": c - 0.5, "c": c})
        # the tap: low through the level, close below it
        bars.append({"t": 40, "o": 101.0, "h": 101.2, "l": 99.2, "c": 99.6})
        # the reclaim: closes back above
        bars.append({"t": 41, "o": 99.8, "h": 101.5, "l": 99.7, "c": 101.0})
        # forming bar, discarded by level_has_held
        bars.append({"t": 42, "o": 101.0, "h": 101.4, "l": 100.8, "c": 101.2})
        return bars

    def test_a_swing_low_fires_on_the_reclaim(self):
        from research.levels.trigger import level_has_held
        from research.levels.config import TRIGGER_HOLD_HOURS_SWING
        fired, why = level_has_held(self.tap_and_reclaim(), self.LEVEL,
                                    hold_hours=TRIGGER_HOLD_HOURS_SWING)
        self.assertTrue(fired, why)

    def test_the_same_bars_do_not_satisfy_a_flip(self):
        # The candle before the reclaim closed below the level, so a two-hour
        # channel has not formed -- which is exactly what a flip must prove.
        from research.levels.trigger import level_has_held
        from research.levels.config import TRIGGER_HOLD_HOURS
        fired, why = level_has_held(self.tap_and_reclaim(), self.LEVEL,
                                    hold_hours=TRIGGER_HOLD_HOURS)
        self.assertFalse(fired)
        self.assertIn("closed below", why)

    def test_a_swing_low_still_will_not_buy_under_the_level(self):
        from research.levels.trigger import level_has_held
        from research.levels.config import TRIGGER_HOLD_HOURS_SWING
        bars = self.tap_and_reclaim()
        bars[-2]["c"] = 99.0                    # never reclaimed
        fired, _ = level_has_held(bars, self.LEVEL,
                                  hold_hours=TRIGGER_HOLD_HOURS_SWING)
        self.assertFalse(fired)


class StructuralStopTest(unittest.TestCase):
    """The stop goes under the lows that formed the level, not a fixed sigma.

    STOP_SIGMA_BELOW=3.0 ignores the level's own shape. UNI 8.5254 is a double
    bottom with lows at 8.4516 -- invalidation 0.9% away -- and the fixed rule
    stopped it at 8.073, 5.9% below, which read a 3.3 reward/risk as 0.91 and
    had the floor reject it. The stop was manufacturing the rejection.
    """

    LEVEL = 100.0
    SIGMA = 0.02

    def bars_with_lows(self, *lows):
        """Filler well above the level, then bars wicking to each low."""
        out = [{"t": i, "o": 106.0, "h": 106.5, "l": 105.5, "c": 106.0}
               for i in range(30)]
        for j, lo in enumerate(lows):
            out.append({"t": 30 + j, "o": 101.0, "h": 101.5, "l": lo, "c": 101.0})
        return out

    def stop(self, *lows):
        from research.levels.levels import structural_stop
        b = self.bars_with_lows(*lows)
        return structural_stop(b, len(b) - 1, self.LEVEL, self.SIGMA)

    def test_it_sits_under_the_deepest_touch(self):
        s = self.stop(99.4, 99.6)
        self.assertLess(s, 99.4, "must clear the wick that made the level")
        self.assertGreater(s, 98.0, "but not a fixed 3 sigma below")

    def test_it_is_tighter_than_the_fixed_rule(self):
        import math
        from research.levels.config import STOP_SIGMA_BELOW
        fixed = self.LEVEL * math.exp(-STOP_SIGMA_BELOW * self.SIGMA)
        self.assertGreater(self.stop(99.4, 99.6), fixed)

    def test_a_low_outside_the_touch_band_is_not_a_touch(self):
        # A wick far below is not a test of THIS level, it is a break of it.
        # 90 against a level of 100 at sigma 0.02 is 5.3 sigma away, so there
        # is no touch to measure and the caller falls back to the fixed rule.
        self.assertIsNone(self.stop(90.0))

    def test_it_never_goes_wider_than_the_fixed_rule(self):
        # The widest a touch can sit is TOUCH_BAND_SIGMA below, plus the
        # buffer -- so the clamp is defensive rather than reachable. Assert
        # the guarantee anyway: this may only move the stop toward the
        # structure, never further away.
        import math
        from research.levels.config import STOP_SIGMA_BELOW, TOUCH_BAND_SIGMA
        fixed = self.LEVEL * math.exp(-STOP_SIGMA_BELOW * self.SIGMA)
        # just inside the band; exactly on it is a float coin-flip
        deepest = self.LEVEL * math.exp(-0.99 * TOUCH_BAND_SIGMA * self.SIGMA)
        s = self.stop(deepest)
        self.assertIsNotNone(s)
        self.assertGreaterEqual(s, fixed - 1e-9)

    def test_it_is_never_tighter_than_the_floor(self):
        # Touches landing exactly on the level would otherwise stop inches
        # away, at the price most likely to be wicked.
        import math
        from research.levels.config import STOP_MIN_SIGMA
        floor = self.LEVEL * math.exp(-STOP_MIN_SIGMA * self.SIGMA)
        self.assertLessEqual(self.stop(100.0, 100.0), floor + 1e-9)

    def test_no_touch_returns_none_so_the_caller_falls_back(self):
        from research.levels.levels import structural_stop
        b = [{"t": i, "o": 130.0, "h": 131.0, "l": 129.0, "c": 130.0} for i in range(40)]
        self.assertIsNone(structural_stop(b, len(b) - 1, self.LEVEL, self.SIGMA))

    def test_undefined_sigma_returns_none(self):
        from research.levels.levels import structural_stop
        self.assertIsNone(structural_stop(self.bars_with_lows(99.5), 30, self.LEVEL, 0.0))

    def test_it_lifts_reward_to_risk(self):
        # The point of the change, in one assertion.
        import math
        from research.levels.config import STOP_SIGMA_BELOW
        entry, tp = 100.4, 106.0
        fixed = self.LEVEL * math.exp(-STOP_SIGMA_BELOW * self.SIGMA)
        struct = self.stop(99.4, 99.6)
        rr_fixed = (tp - entry) / (entry - fixed)
        rr_struct = (tp - entry) / (entry - struct)
        self.assertGreater(rr_struct, rr_fixed * 1.5)

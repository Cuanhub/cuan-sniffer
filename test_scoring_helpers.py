"""
Tests for signal_engine_modules/scoring_helpers.py.

Covers the funding-threshold recalibration (2026-08-25): thresholds were
previously 12-100x higher than any real funding rate observed across all
8 currently-traded coins over 30 days (Hyperliquid fundingHistory), so the
scoring function had never fired even its lowest tier in practice.
"""

import unittest

from signal_engine_modules.scoring_helpers import (
    score_funding_context,
    FUNDING_MILD_THRESHOLD,
    FUNDING_HIGH_THRESHOLD,
    FUNDING_EXTREME_THRESHOLD,
)


class TestFundingThresholdsRecalibrated(unittest.TestCase):

    def test_thresholds_are_realistic_scale(self):
        # Must be well under the old thresholds (0.001/0.005/0.01) --
        # those were 12-100x too high for real Hyperliquid funding rates.
        self.assertLess(FUNDING_MILD_THRESHOLD, 0.0005)
        self.assertLess(FUNDING_HIGH_THRESHOLD, 0.001)
        self.assertLess(FUNDING_EXTREME_THRESHOLD, 0.001)
        self.assertLess(FUNDING_MILD_THRESHOLD, FUNDING_HIGH_THRESHOLD)
        self.assertLess(FUNDING_HIGH_THRESHOLD, FUNDING_EXTREME_THRESHOLD)

    def test_typical_real_funding_rate_now_activates_mild_tier(self):
        # A funding rate at roughly the observed p95-p99 range (well within
        # what actually occurs) must now produce a nonzero score -- this
        # NEVER happened under the old thresholds for any of the 8 traded
        # coins across a full month of real data.
        score, notes = score_funding_context(0.00003)  # ~p97, real-world scale
        self.assertNotEqual(score, 0.0)
        self.assertTrue(any("funding_pos" in n or "funding_neg" in n for n in notes))

    def test_median_real_funding_rate_stays_below_mild(self):
        # The typical/median observed rate (~0.000013) should NOT trigger
        # any bump -- only genuinely elevated readings should score.
        score, notes = score_funding_context(0.000013)
        self.assertEqual(score, 0.0)
        self.assertEqual(notes, [])

    def test_direction_contrarian_crowd_fade(self):
        # Positive funding (crowded long) must penalize LONG-favoring score
        # (negative contribution); negative funding (crowded short) must
        # be a positive contribution. This logic itself was already
        # correct -- only the thresholds changed.
        pos_score, _ = score_funding_context(0.0001)
        neg_score, _ = score_funding_context(-0.0001)
        self.assertLess(pos_score, 0)
        self.assertGreater(neg_score, 0)

    def test_extreme_tier_reachable_at_observed_max(self):
        # The actual 30-day observed max across all 8 coins was ~0.0000854.
        score, notes = score_funding_context(0.00009)
        self.assertAlmostEqual(abs(score), 0.15, places=6)
        self.assertTrue(any("extreme" in n for n in notes))


if __name__ == "__main__":
    unittest.main(verbosity=2)

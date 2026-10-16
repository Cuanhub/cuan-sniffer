"""Tests for the tool that decides which features are allowed to veto trades.

This file exists because of what happened on 2026-10-15. The production code
had 482 tests; the analysis code that decides WHAT production code to write
had none. A screening harness written that day carried three bugs at once:

  1. It ranked rows with `sorted(pairs)` on (feature, outcome) tuples. Python
     sorts lexicographically, so within tied feature values it ordered by the
     OUTCOME -- handing the top bucket the highest R and manufacturing
     monotonicity from any feature with ties. level_hold_rate, which is 1.0
     for 658 of 1649 episodes, came out "100% positive, +1.166R" that way and
     was briefly believed.
  2. Fixing that with a stable sort left ties in their ORIGINAL order, which
     was chronological, and the regime drifts hard across the sample. A
     CONSTANT feature still produced [+0.20, +0.57, +0.37, -0.17, -0.42].
  3. `float()` on the 'True'/'False' that outcomes.csv writes raised and was
     swallowed to None, so every boolean column read as missing.

A bad analysis is worse than a bad function: it ships a wrong rail. The two
tests that would have caught all three are the ones at the top of
HarnessCannotInventSignal -- a constant feature and a random feature must both
come out flat. They cost nothing and they are the whole point of this file.
"""

import math
import random
import unittest

import tools.screen_context as S


def rows(pairs, feat="basis_bps", rr=1.0, day="2026-10-01"):
    """[(feature, outcome_r)] -> the row shape the screen consumes."""
    return [{feat: f, "outcome_r": r, "rr": rr, "coin": "X",
             "at": f"{day}T00:00:{i:02d}Z"}
            for i, (f, r) in enumerate(pairs)]


class Parsing(unittest.TestCase):
    def test_numbers_parse(self):
        self.assertEqual(S._f({"a": "1.5"}, "a"), 1.5)
        self.assertEqual(S._f({"a": "-2"}, "a"), -2.0)

    def test_blank_and_missing_are_none(self):
        self.assertIsNone(S._f({"a": ""}, "a"))
        self.assertIsNone(S._f({"a": "   "}, "a"))
        self.assertIsNone(S._f({}, "a"))

    def test_junk_is_none_rather_than_raising(self):
        self.assertIsNone(S._f({"a": "abc"}, "a"))

    def test_a_missing_value_is_not_silently_a_zero(self):
        """None and 0.0 mean different things to a median split: one is
        absent, the other is a real reading at the bottom of the range."""
        self.assertIsNone(S._f({"a": ""}, "a"))
        self.assertEqual(S._f({"a": "0"}, "a"), 0.0)


class Ratios(unittest.TestCase):
    def test_basis_is_mark_against_oracle_in_bps(self):
        r = {"mark_px": 101.0, "oracle_px": 100.0}
        self.assertAlmostEqual(S.RATIOS["basis_bps"](r), 100.0)

    def test_basis_is_signed(self):
        self.assertLess(S.RATIOS["basis_bps"](
            {"mark_px": 99.0, "oracle_px": 100.0}), 0)

    def test_a_zero_denominator_is_none_not_an_exception(self):
        for r in ({"mark_px": 1.0, "oracle_px": 0.0},
                  {"mark_px": 1.0, "oracle_px": None}):
            self.assertIsNone(S.RATIOS["basis_bps"](r))

    def test_every_ratio_tolerates_a_missing_input(self):
        """main() catches TypeError/ZeroDivisionError/KeyError per ratio. If a
        ratio raises anything else the whole screen dies mid-run."""
        blank = {k: None for k in ("mark_px", "oracle_px", "funding_rate",
                                   "oi_notional_usd", "day_ntl_vlm",
                                   "impact_spread_bps", "price",
                                   "prev_day_px")}
        for name, fn in S.RATIOS.items():
            try:
                fn(blank)
            except (TypeError, ZeroDivisionError, KeyError):
                pass          # the three main() handles
            except Exception as e:       # noqa: BLE001 - that is the point
                self.fail(f"{name} raised {type(e).__name__}: {e}")


class HarnessCannotInventSignal(unittest.TestCase):
    """The two tests that would have caught the 2026-10-15 harness bugs."""

    def test_a_constant_feature_produces_no_gap(self):
        rng = random.Random(1)
        data = rows([(5.0, rng.gauss(0, 1)) for _ in range(400)])
        gap = S.split_mean_gap(data, "basis_bps")
        if gap is not None:
            self.assertAlmostEqual(gap[2], 0.0, delta=0.35,
                                   msg="a feature with one value cannot "
                                       "separate outcomes")

    def test_a_constant_feature_is_not_significant(self):
        rng = random.Random(2)
        data = rows([(5.0, rng.gauss(0, 1)) for _ in range(300)])
        p = S.permutation_p(data, "basis_bps", 400, random.Random(3))
        if p is not None:
            self.assertGreater(p, 0.05)

    def test_pure_noise_is_not_significant(self):
        rng = random.Random(4)
        data = rows([(rng.random(), rng.gauss(0, 1)) for _ in range(300)])
        p = S.permutation_p(data, "basis_bps", 600, random.Random(5))
        self.assertIsNotNone(p)
        self.assertGreater(p, 0.05, "noise came out significant")

    def test_ties_are_not_ordered_by_outcome(self):
        """The exact 2026-10-15 bug. Half the rows share one feature value;
        if ranking breaks ties by outcome, the top bucket fills with wins."""
        data = rows([(1.0, 1.0)] * 50 + [(1.0, -1.0)] * 50)
        table = S.quartile_table(data, "basis_bps")
        means = [m for _, _, m, _ in table]
        self.assertTrue(all(abs(m) < 0.9 for m in means),
                        f"ties leaked outcome ordering into the buckets: "
                        f"{means}")

    def test_a_real_gap_is_detected(self):
        """The harness must not be so conservative it finds nothing."""
        rng = random.Random(6)
        lo = [(0.0, rng.gauss(-0.6, 0.5)) for _ in range(150)]
        hi = [(1.0, rng.gauss(+0.6, 0.5)) for _ in range(150)]
        data = rows(lo + hi)
        p = S.permutation_p(data, "basis_bps", 800, random.Random(7))
        self.assertIsNotNone(p)
        self.assertLess(p, 0.05, "a planted half-sigma gap was missed")
        self.assertGreater(S.split_mean_gap(data, "basis_bps")[2], 0)


class Stratification(unittest.TestCase):
    def test_the_split_is_taken_within_rr_bands(self):
        """rr correlates with nearly everything here, so an unstratified
        split mostly rediscovers rr."""
        src = open("tools/screen_context.py").read()
        self.assertIn("terciles", src)
        self.assertIn("median", src)

    def test_rr_stratification_attenuates_but_does_NOT_remove_rr(self):
        """Pinned as a limitation, not an aspiration.

        Plant a feature that IS rr, with the outcome driven entirely by rr.
        Tercile stratification takes the median WITHIN each tercile, and a
        tercile is wide, so the split still separates by rr and the feature
        comes out significant. Measured: p = 0.0017.

        This is why the 2026-10-15 screen stopped relying on tercile
        stratification and shuffled outcomes within (day x rr-QUINTILE) cells
        instead. It is also why near_sigma, pivots_per_touch and
        bars_since_touch looked strong on R -- they track rr -- and had to be
        re-tested on P(hit target), where they came out at p = 0.95, 0.71 and
        0.69.

        If a future change makes this test fail by making p LARGER, the
        stratification got stronger and that is good news; update the note.
        """
        rng = random.Random(8)
        data = []
        for i in range(300):
            rr = rng.choice([0.5, 1.0, 2.0, 4.0])
            data.append({"basis_bps": rr, "rr": rr, "coin": "X",
                         "outcome_r": rr + rng.gauss(0, 0.3),
                         "at": f"2026-10-01T00:00:{i % 60:02d}Z"})
        p = S.permutation_p(data, "basis_bps", 600, random.Random(9))
        self.assertIsNotNone(p)
        self.assertLess(p, 0.05,
                        "rr stratification now removes an rr-collinear "
                        "feature -- better than when this was written, so "
                        "revisit the note above")


class Degenerate(unittest.TestCase):
    def test_too_few_rows_returns_none_rather_than_a_verdict(self):
        self.assertIsNone(S.split_mean_gap(rows([(1.0, 1.0)] * 4),
                                           "basis_bps"))
        self.assertIsNone(S.permutation_p(rows([(1.0, 1.0)] * 4),
                                          "basis_bps", 10, random.Random(0)))

    def test_rows_missing_the_feature_are_dropped_not_zeroed(self):
        data = rows([(1.0, 1.0)] * 30)
        for r in data[:15]:
            r["basis_bps"] = None
        gap = S.split_mean_gap(data, "basis_bps")
        if gap is not None:
            self.assertEqual(gap[3] + gap[4], 15)

    def test_rows_missing_rr_are_dropped(self):
        data = rows([(float(i), 1.0) for i in range(40)])
        for r in data:
            r["rr"] = None
        self.assertIsNone(S.split_mean_gap(data, "basis_bps"))

    def test_terciles_on_a_short_list(self):
        self.assertEqual(S.terciles([1.0, 2.0, 3.0]), (2.0, 3.0))

    def test_quartile_table_is_empty_below_eight_rows(self):
        self.assertEqual(S.quartile_table(rows([(1.0, 1.0)] * 7),
                                          "basis_bps"), [])

    def test_no_nan_reaches_a_verdict(self):
        data = rows([(float(i), 1.0) for i in range(40)])
        gap = S.split_mean_gap(data, "basis_bps")
        self.assertFalse(any(math.isnan(v) for v in gap[:3]))


class Determinism(unittest.TestCase):
    def test_the_same_seed_gives_the_same_p(self):
        """A screen whose answer moves between runs cannot be audited."""
        data = rows([(float(i % 7), (i % 3) - 1.0) for i in range(120)])
        a = S.permutation_p(data, "basis_bps", 300, random.Random(42))
        b = S.permutation_p(data, "basis_bps", 300, random.Random(42))
        self.assertEqual(a, b)

    def test_permutation_does_not_mutate_the_caller_rows(self):
        data = rows([(float(i), float(i % 5) - 2) for i in range(60)])
        before = [r["outcome_r"] for r in data]
        S.permutation_p(data, "basis_bps", 100, random.Random(11))
        self.assertEqual([r["outcome_r"] for r in data], before,
                         "the screen shuffled its own input")


if __name__ == "__main__":
    unittest.main()

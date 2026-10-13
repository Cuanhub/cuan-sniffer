"""Which hold a level must clear depends on how it BEHAVES, not its label.

level_origin marks only what the resistance detector happened to find, so a
level can be scored swing_low while plainly still capping price -- and a
swing_low is asked for the easy 1h hold rather than the 2h channel hold a flip
has to prove.

ARB 0.186655 was bought on 2026-10-13 having capped price for 66 of the last
120 bars, most recently 6 bars earlier, on a 1h hold. SOL 101.72 was the same
shape at 95 of 120, last 12 bars ago. In both cases the advisory flag said
"price is still flipping it" while the rule had already waved it through.

Deliberately NOT justified by outcomes. Hold rate, touch count and the capping
flag itself were each screened against the live book and all came out flat --
the capping flag at p = 0.74, hold rate at p = 0.99 with identical average P&L
either side of 70%. This corrects which rule a setup is measured by. It is a
consistency fix, not an edge claim.
"""

import inspect
import unittest

import tools.live_levels_trader as T
from research.levels import config as C


def ep(**over):
    e = {"level_origin": "swing_low", "capped_bars": 0, "bars_since_capped": -1}
    e.update(over)
    return e


class Classification(unittest.TestCase):
    def test_a_declared_flip_is_a_flip(self):
        self.assertTrue(T.behaves_like_flip(ep(level_origin="flipped_resistance")))

    def test_a_clean_swing_low_is_not(self):
        self.assertFalse(T.behaves_like_flip(ep()))

    def test_the_arb_shape_is_now_a_flip(self):
        """66 of 120 bars, last 6 bars ago -- bought on a 1h hold."""
        self.assertTrue(T.behaves_like_flip(
            ep(capped_bars=66, bars_since_capped=6)))

    def test_the_sol_shape_is_now_a_flip(self):
        """95 of 120 bars, last 12 ago, scored swing_low."""
        self.assertTrue(T.behaves_like_flip(
            ep(capped_bars=95, bars_since_capped=12)))

    def test_capping_long_ago_does_not_count(self):
        """A level that capped price months back is not still flipping."""
        self.assertFalse(T.behaves_like_flip(
            ep(capped_bars=90, bars_since_capped=C.FLIP_CAPPED_RECENT_BARS + 1)))

    def test_capping_briefly_does_not_count(self):
        self.assertFalse(T.behaves_like_flip(
            ep(capped_bars=C.FLIP_CAPPED_BARS_MIN - 1, bars_since_capped=2)))

    def test_never_capped_does_not_count(self):
        self.assertFalse(T.behaves_like_flip(
            ep(capped_bars=0, bars_since_capped=-1)))

    def test_the_boundaries(self):
        self.assertTrue(T.behaves_like_flip(
            ep(capped_bars=C.FLIP_CAPPED_BARS_MIN,
               bars_since_capped=C.FLIP_CAPPED_RECENT_BARS)))
        self.assertFalse(T.behaves_like_flip(
            ep(capped_bars=C.FLIP_CAPPED_BARS_MIN - 1,
               bars_since_capped=C.FLIP_CAPPED_RECENT_BARS)))

    def test_junk_values_do_not_raise(self):
        for bad in ({"capped_bars": None}, {"capped_bars": "x"},
                    {"bars_since_capped": None}, {}):
            T.behaves_like_flip(ep(**bad))


class OneDefinitionNotTwo(unittest.TestCase):
    """The thresholds lived hardcoded inside review_flags and now decide a
    rule, so the flag and the rule must read the same function -- the mistake
    the stop rule and the target rule each made by being written twice."""

    def test_the_hold_decision_uses_it(self):
        src = inspect.getsource(T.execute_one)
        self.assertIn("behaves_like_flip(ep)", src)
        self.assertNotIn('== "flipped_resistance"', src)

    def test_review_flags_uses_it(self):
        self.assertIn("behaves_like_flip(ep)", inspect.getsource(T.review_flags))

    def test_the_thresholds_are_not_hardcoded_anywhere(self):
        whole = inspect.getsource(T)
        self.assertNotIn("cap >= 25", whole)
        self.assertNotIn("since <= 48", whole)

    def test_a_behavioural_flip_gets_the_longer_hold(self):
        """The point of the change, asserted on the numbers rather than the
        code path."""
        self.assertGreater(C.TRIGGER_HOLD_HOURS, C.TRIGGER_HOLD_HOURS_SWING)
        arb = ep(capped_bars=66, bars_since_capped=6)
        hold = C.TRIGGER_HOLD_HOURS if T.behaves_like_flip(arb) else C.TRIGGER_HOLD_HOURS_SWING
        self.assertEqual(hold, C.TRIGGER_HOLD_HOURS)


if __name__ == "__main__":
    unittest.main()

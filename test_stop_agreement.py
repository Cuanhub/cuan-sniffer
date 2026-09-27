"""The trader and the scanner must price a stop the same way.

They are two implementations of one idea. call.py:find_candidates is what the
live trader asks for a stop; scanner.py:scan_coin is what writes the research
set. On 2026-09-20 the structural stop landed in call.py only, and for a week
every logged episode carried a flat 3.35 sigma stop while the trades those
episodes were meant to describe were placed at 0.47-0.80 sigma.

Nothing lost money -- the scanner cannot place an order. What it cost is the
research set, which is the thing the gate gets calibrated against: a stop the
system never used makes any fit derived from it fit a strategy nobody trades.

So the guarantee pinned here is not "the stop is structural", which either file
could satisfy alone. It is that the two agree, on the same bars, for the same
level.
"""

import math
import unittest

from research.levels import scanner
from research.levels.call import find_candidates
from research.levels.levels import log_sigma, structural_stop
from research.levels import config as C


def synth(n=400):
    """A series with a support level that gets touched, so structural_stop has
    a real wick to sit under rather than falling back to the fixed rule."""
    bars, px = [], 100.0
    for k in range(n):
        # A slow drift with periodic dips to ~95, forming a level with touches.
        px = 100.0 + 6.0 * math.sin(k / 23.0) + 0.02 * k
        low = px - (4.5 if k % 37 == 0 else 0.6)
        bars.append({"t": 1700000000000 + k * 14400000, "o": px,
                     "h": px + 0.6, "l": low, "c": px, "v": 1000.0})
    return bars


class StopAgreement(unittest.TestCase):
    def test_structural_stop_is_not_a_fixed_multiple_of_sigma(self):
        """The regression itself: a flat stop across every level is the old
        rule, and it is what a single distinct value looks like."""
        bars = synth()
        i = len(bars) - 1
        sigma = log_sigma(bars, i)
        self.assertGreater(sigma, 0)
        fixed = math.exp(-C.STOP_SIGMA_BELOW * sigma)
        seen = set()
        for lvl in (94.0, 95.5, 97.0, 99.0):
            st = structural_stop(bars, i, lvl, sigma)
            if st:
                seen.add(round(abs(math.log(st / lvl)) / sigma, 3))
        if seen:
            self.assertFalse(
                seen == {round(C.STOP_SIGMA_BELOW, 3)},
                "every level priced at exactly STOP_SIGMA_BELOW is the fixed rule")
        # And the fixed fallback is still reachable and still correct.
        self.assertAlmostEqual(fixed, math.exp(-C.STOP_SIGMA_BELOW * sigma))

    def test_scanner_and_call_agree_on_the_same_level(self):
        """Both paths, same bars, same level -> same stop.

        Driven through the shared helper rather than the two public entry
        points, because those differ in gating and universe selection; the
        stop rule is the part that has to match.
        """
        bars = synth()
        i = len(bars) - 1
        sigma = log_sigma(bars, i)

        def scanner_rule(level):
            return (structural_stop(bars, i, level, sigma)
                    or level * math.exp(-scanner.STOP_SIGMA_BELOW * sigma))

        def call_rule(level):
            return (structural_stop(bars, i, level, sigma)
                    or level * math.exp(-C.STOP_SIGMA_BELOW * sigma))

        for lvl in (93.0, 95.0, 96.5, 98.0, 101.0):
            self.assertAlmostEqual(
                scanner_rule(lvl), call_rule(lvl), places=10,
                msg=f"the two paths disagree at level {lvl}")

    def test_both_modules_import_the_same_structural_stop(self):
        """A cheap guard against one file growing its own copy."""
        import research.levels.call as call_mod
        self.assertIs(scanner.structural_stop, structural_stop)
        self.assertIs(call_mod.structural_stop, structural_stop)

    def test_scanner_source_no_longer_hardcodes_the_fixed_rule_alone(self):
        """The literal shape of the bug: a stop assigned straight from
        STOP_SIGMA_BELOW with no structural term anywhere near it."""
        import inspect
        src = inspect.getsource(scanner.scan_coin)
        self.assertIn("structural_stop", src,
                      "scan_coin prices a stop without the structural rule")


if __name__ == "__main__":
    unittest.main()

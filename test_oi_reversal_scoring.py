"""
Regression test for the OI-directional reversal-scoring fix (2026-08-25).

Standard institutional OI framework: falling OI (short-covering / long-
liquidation from the prior trend) confirms a genuine reversal; rising OI
means fresh positions are still piling into the prior trend, which argues
AGAINST a reversal. The code previously had this backwards for both LONG
and SHORT reversal setups (1h builder) and the swing reversal path,
bonusing rising OI as "oi_supports_reversal".

Uses source inspection rather than full pipeline mocking — the fix lives
inline inside AdaptiveSignalEngine.generate_signal, which needs a real
DataFrame/sentiment snapshot to invoke end-to-end. This locks in the
correct sign condition so a future edit can't silently revert it.
"""

import inspect
import re
import unittest

import signal_engine
from signal_engine_modules.scoring_helpers import score_oi_directional


class TestScoreOiDirectionalUnchanged(unittest.TestCase):
    """score_oi_directional itself is correct as a raw OI-momentum
    function — it was never the bug. The bug was in how callers
    interpreted its sign for reversal setups."""

    def test_rising_oi_positive(self):
        s, _ = score_oi_directional(0.10, 0.0, 0.0, 0.0, "LONG")
        self.assertGreater(s, 0)

    def test_falling_oi_negative(self):
        s, _ = score_oi_directional(-0.10, 0.0, 0.0, 0.0, "LONG")
        self.assertLess(s, 0)


class TestReversalOiDirectionFixed(unittest.TestCase):

    def _reversal_builder_source(self):
        return inspect.getsource(signal_engine.AdaptiveSignalEngine._build_reversal_signal)

    def test_no_bare_positive_oi_reversal_bonus(self):
        """The old, backwards condition must not be present anywhere:
        `if oi_score > 0:` immediately followed by an oi_supports_reversal
        bonus. A correct fix flips this to `oi_score < 0`."""
        src = self._reversal_builder_source()
        # Every "oi_supports_reversal" tag must be reachable only from an
        # `if oi_score < 0:` guard, never `if oi_score > 0:`. The bonus line
        # is written as `if oi_score X 0:\n    score += ...; notes.append(...)`
        # (semicolon-joined, same line) -- match that shape exactly, not a
        # loose pattern that could silently match zero times.
        matches = list(re.finditer(
            r'if oi_score ([<>]) 0:\s*\n\s*score \+= [^;\n]*;\s*notes\.append\("oi_supports_reversal"\)',
            src,
        ))
        self.assertEqual(len(matches), 2, "expected to find both reversal branches' oi bonus lines")
        for m in matches:
            self.assertEqual(
                m.group(1), "<",
                "oi_supports_reversal must trigger on FALLING OI (oi_score < 0), "
                "not rising OI -- rising OI during a prior trend argues against "
                "a reversal, it doesn't confirm one.",
            )

    def test_reversal_bonus_count_matches_both_sides(self):
        # Exactly two occurrences expected: LONG reversal branch, SHORT
        # reversal branch. If this count changes, the assumption behind
        # the regex-based check above needs re-verifying by hand.
        src = self._reversal_builder_source()
        self.assertEqual(src.count('notes.append("oi_supports_reversal")'), 2)

    def test_swing_reversal_negates_oi(self):
        src = inspect.getsource(signal_engine.AdaptiveSignalEngine)
        self.assertIn('swing_family == "reversal"', src)
        # The specific negation fix for the swing path.
        self.assertIn("oi_s = -oi_s", src)


if __name__ == "__main__":
    unittest.main()

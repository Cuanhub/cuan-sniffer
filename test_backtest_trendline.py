"""Trendline breaks must be computable at the time they are reported.

The decisive test is on REAL bars: recompute on a series cut off at bar t and
require the identical answer at t. If any value at t used a bar after t, the
cut series could not reproduce it.
"""

import json
import unittest
from dataclasses import replace
from pathlib import Path

from research.backtest import data as D
from research.backtest import trendline as TL

REAL = json.loads((Path(__file__).parent / "tests/fixtures/zones/bars_1h.json").read_text())


def bars_from_highs(highs, closes=None):
    closes = closes or [h - 1 for h in highs]
    return [{"t": k * 3_600_000, "o": c, "h": h, "l": min(c, h) - 2, "c": c}
            for k, (h, c) in enumerate(zip(highs, closes))]


class Causality(unittest.TestCase):
    def test_breaks_are_identical_on_a_series_cut_at_t(self):
        found = 0
        for coin in ("ETH", "HYPE", "FARTCOIN", "WLD", "AERO", "BCH"):
            bars = REAL[coin]
            full, lines = TL.breaks(bars)
            for t in range(150, len(bars), 3):
                cut, cut_lines = TL.breaks(bars[:t + 1])
                self.assertEqual(cut[t], full[t], f"{coin} t={t}")
                self.assertEqual(cut_lines[t], lines[t], f"{coin} t={t}")
                found += full[t]
        self.assertGreater(found, 10, "too few breaks checked to mean anything")

    def test_causal_for_every_robustness_variant_too(self):
        for p in (TL.TLParams(k=2, lookback=72), TL.TLParams(k=5, lookback=240),
                  TL.TLParams(buffer_atr=0.25), TL.TLParams(min_touches=3)):
            bars = REAL["NEAR"]
            full, _ = TL.breaks(bars, p)
            for t in range(200, len(bars), 11):
                self.assertEqual(TL.breaks(bars[:t + 1], p)[0][t], full[t], f"{p} t={t}")

    def test_a_swing_high_is_unusable_until_k_bars_later(self):
        bars = bars_from_highs(BASE)
        p = TL.TLParams(k=3, lookback=50, min_span=6)
        sh = TL.swing_highs(bars, 3)
        self.assertEqual(sh, [A, B])
        atr = [1.0] * len(bars)
        self.assertIsNone(TL.line_at(bars, sh, B + 3 - 1, p, atr),
                          "line drawn through B before B was confirmed")
        ln = TL.line_at(bars, sh, B + 3, p, atr)
        self.assertEqual((ln.a_i, ln.b_i), (A, B))


# A swing high at 4 (100) and a lower one at 10 (95); with k=3 the second is
# confirmed at bar 13. The line falls 5/6 per bar. Every case below is built
# from this so it is checked to produce a line first -- an earlier version
# put A at index 2, where k=3 cannot find a swing high at all, and three of
# its "no line" / "no break" tests passed because there was never a line.
BASE = [80, 81, 82, 83, 100, 92, 91, 90, 89, 88, 95, 87, 86, 85, 84, 83, 82,
        81, 80, 79, 78, 77, 76]
A, B = 4, 10
P = TL.TLParams(k=3, lookback=50, min_span=6)


def line_value(t):
    return 100 + (95 - 100) / 6 * (t - A)


class TheLine(unittest.TestCase):
    def test_the_base_series_has_a_line(self):
        bars = bars_from_highs(BASE)
        ln = TL.line_at(bars, TL.swing_highs(bars, 3), 15, P, [1.0] * len(bars))
        self.assertIsNotNone(ln)
        self.assertAlmostEqual(ln.at(15), line_value(15))

    def test_b_is_the_upper_envelope(self):
        """Of the lower highs after A, B is the one with the least negative
        slope, so no confirmed swing high since A is above the line."""
        highs = list(BASE)
        highs[16] = 88          # a third, steeper lower high
        bars = bars_from_highs(highs)
        sh = TL.swing_highs(bars, 3)
        self.assertIn(16, sh)
        ln = TL.line_at(bars, sh, 21, P, [1.0] * len(bars))
        self.assertEqual((ln.a_i, ln.b_i), (A, B))
        for j in sh:
            if j > ln.a_i:
                self.assertLessEqual(bars[j]["h"], ln.at(j) + 1e-12)

    def test_a_close_above_the_line_between_anchors_rejects_it(self):
        closes = [h - 1 for h in BASE]
        closes[7] = line_value(7) + 0.1       # pokes above, between A and B
        bars = bars_from_highs(BASE, closes)
        sh = TL.swing_highs(bars, 3)
        self.assertEqual(sh, [A, B], "the poke must not create a pivot")
        self.assertIsNone(TL.line_at(bars, sh, 14, P, [1.0] * len(bars)))

    def test_anchors_too_close_together_are_no_line(self):
        bars = bars_from_highs(BASE)
        sh = TL.swing_highs(bars, 3)
        self.assertIsNotNone(TL.line_at(bars, sh, 14, P, [1.0] * len(bars)))
        self.assertIsNone(TL.line_at(bars, sh, 14, replace(P, min_span=7),
                                     [1.0] * len(bars)))

    def test_three_touches_needs_a_third_swing_high_on_the_line(self):
        bars = bars_from_highs(BASE)
        sh = TL.swing_highs(bars, 3)
        self.assertIsNone(TL.line_at(bars, sh, 14, replace(P, min_touches=3),
                                     [1.0] * len(bars)))
        highs = list(BASE)
        highs[16] = line_value(16) - 0.1      # a third high, just under it
        bars = bars_from_highs(highs)
        sh = TL.swing_highs(bars, 3)
        ln = TL.line_at(bars, sh, 21, replace(P, min_touches=3), [1.0] * len(bars))
        self.assertIsNotNone(ln)
        self.assertEqual(ln.touches, 3)


class TheBreak(unittest.TestCase):
    def series(self, closes_at):
        highs, closes = list(BASE), [h - 1 for h in BASE]
        for t, c in closes_at.items():
            closes[t] = c
            highs[t] = max(highs[t], c + 0.05)
        return bars_from_highs(highs, closes)

    def test_no_break_without_a_close_above(self):
        br, _ = TL.breaks(bars_from_highs(BASE), P)
        self.assertFalse(any(br))

    def test_the_first_close_above_is_the_break(self):
        br, _ = TL.breaks(self.series({15: line_value(15) + 0.5}), P)
        self.assertEqual([i for i, x in enumerate(br) if x], [15])

    def test_a_second_close_above_the_same_line_is_not_a_new_break(self):
        br, _ = TL.breaks(self.series({14: line_value(14) + 0.5,
                                       16: line_value(16) + 0.5}), P)
        self.assertEqual([i for i, x in enumerate(br) if x], [14])

    def test_a_close_above_before_b_was_confirmed_means_no_break_later(self):
        """Price was already through the line before anyone could draw it,
        so when the line becomes drawable it is already broken."""
        control, _ = TL.breaks(self.series({14: line_value(14) + 0.3}), P)
        self.assertTrue(control[14], "control: a later close does break it")
        br, _ = TL.breaks(self.series({12: line_value(12) + 0.3,
                                       14: line_value(14) + 0.3}), P)
        self.assertFalse(any(br[13:]))

    def test_a_buffer_demands_more_than_a_touch_through(self):
        bars = self.series({15: line_value(15) + 0.05})
        self.assertTrue(TL.breaks(bars, P)[0][15])
        self.assertFalse(TL.breaks(bars, replace(P, buffer_atr=5.0))[0][15])


class FourHourMapping(unittest.TestCase):
    def test_a_4h_break_lands_on_the_1h_bar_that_closes_the_4h_bar(self):
        bars = [dict(b, t=1_700_006_400_000 + k * 3_600_000)   # a 4h boundary
                for k, b in enumerate(REAL["ETH"])]
        four = D.resample(bars, 4)
        br4, _ = TL.breaks(four, TL.TLParams(lookback=30))
        pos = {b["t"]: i for i, b in enumerate(bars)}
        for q, hit in enumerate(br4):
            if hit:
                i = pos[four[q]["t"] + 3 * 3_600_000]
                self.assertEqual(bars[i]["c"], four[q]["c"])
        self.assertTrue(any(br4))


class LastBreak(unittest.TestCase):
    def test_window_is_inclusive_of_t_and_bounded(self):
        br = [False] * 40
        br[10] = True
        self.assertEqual(TL.last_break_before(br, 10, 1), 0)    # t itself counts
        self.assertEqual(TL.last_break_before(br, 23, 24), 13)
        self.assertEqual(TL.last_break_before(br, 33, 24), 23)  # window edge: in
        self.assertIsNone(TL.last_break_before(br, 34, 24))     # one past: out
        self.assertIsNone(TL.last_break_before(br, 9, 24))      # before it: never


if __name__ == "__main__":
    unittest.main()

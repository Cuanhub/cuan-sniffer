"""Refusing a setup when the hour it fires in has already traded through the
stop.

entry_is_actionable already refused when price was at or below the stop -- but
it tested the MID, so a dip through the stop that bounced back before the pass
looked at it sailed through untouched. Across 53 live entries, eleven were
taken after the live bar had already traded past the stop. Not one won:

    live bar's low, toward the stop     n   win rate      net
    above the level                    15        47%   +$34.34
    0-50% of the way                   18        28%   +$ 3.82
    50-100%                             9        33%   -$ 1.63
    already past the stop              11         0%   -$27.37

level_has_held cannot catch it: it judges CLOSED bars and excludes the forming
one deliberately, so the level can be breaking at the moment of purchase. The
operator read this off the STRK chart as "no double bottom had been created
yet" -- STRK's entry hour had already traded halfway to its stop, and the trade
died in 95 minutes for -$6.89.

This is an extension of a rule the system already held, not a new filter. It
can only remove trades.
"""

import unittest

import tools.live_levels_trader as T
from research.levels import config as C


class Backend:
    def __init__(self, mid):
        self.mid = mid

    def get_mid_price(self, coin):
        return self.mid


def bars(low, n=40):
    """n closed bars well above everything, then a forming bar with this low."""
    out = [{"t": i * 3_600_000, "o": 100.0, "h": 101.0, "l": 99.5, "c": 100.0}
           for i in range(n)]
    out.append({"t": n * 3_600_000, "o": 100.0, "h": 100.5, "l": low, "c": 100.0})
    return out


class TheVeto(unittest.TestCase):
    LEVEL, STOP, TP, ENTRY, MID = 100.0, 98.0, 110.0, 100.35, 100.3

    def check(self, low):
        return T.entry_is_actionable(Backend(self.MID), "TEST", self.ENTRY,
                                     self.STOP, self.TP, level=self.LEVEL,
                                     bars_1h=bars(low))

    def test_a_bar_that_held_above_the_level_is_fine(self):
        ref, why = self.check(100.4)
        self.assertIsNotNone(ref)
        self.assertIsNone(why)

    def test_a_shallow_wick_into_the_level_is_fine(self):
        """0.5 of the way to the stop. Noise, not a break -- and the data says
        these are no worse than average."""
        ref, why = self.check(99.0)
        self.assertIsNotNone(ref)

    def test_a_bar_that_reached_the_stop_is_refused(self):
        ref, why = self.check(97.9)
        self.assertIsNone(ref)
        self.assertIn("way from the level", why)
        self.assertIn("stop", why)

    def test_the_boundary_is_the_stop_itself(self):
        just_above, _ = self.check(98.01)
        self.assertIsNotNone(just_above, "a bar stopping just short must pass")
        just_below, _ = self.check(97.99)
        self.assertIsNone(just_below, "a bar reaching the stop must not")

    def test_the_strk_shape_passes_at_the_shipped_default(self):
        """STRK's hour had traded 50% of the way to its stop, not past it. The
        default setting does NOT catch it -- 1.0 is the unarguable rule, and
        catching STRK needs a fitted threshold the operator can choose."""
        ref, _ = self.check(99.0)
        self.assertIsNotNone(ref)

    def test_tightening_the_threshold_does_catch_it(self):
        saved = T.ENTRY_MAX_BREAK_FRAC
        try:
            T.ENTRY_MAX_BREAK_FRAC = 0.5
            ref, why = self.check(98.9)      # 55% of the way to the stop
            self.assertIsNone(ref)
        finally:
            T.ENTRY_MAX_BREAK_FRAC = saved


class ItCannotMisfire(unittest.TestCase):
    LEVEL, STOP, TP, ENTRY, MID = 100.0, 98.0, 110.0, 100.35, 100.3

    def call(self, **kw):
        kw.setdefault("level", self.LEVEL)
        kw.setdefault("bars_1h", bars(97.0))
        return T.entry_is_actionable(Backend(self.MID), "TEST", self.ENTRY,
                                     self.STOP, self.TP, **kw)

    def test_no_bars_means_no_veto(self):
        """A dry run supplies none, and the check must not invent a refusal."""
        ref, why = self.call(bars_1h=None)
        self.assertIsNotNone(ref)

    def test_no_level_means_no_veto(self):
        ref, why = self.call(level=0.0)
        self.assertIsNotNone(ref)

    def test_a_zero_low_is_ignored_rather_than_dividing_oddly(self):
        ref, why = self.call(bars_1h=bars(0.0))
        self.assertIsNotNone(ref)

    def test_it_can_be_switched_off(self):
        saved = T.ENTRY_MAX_BREAK_FRAC
        try:
            T.ENTRY_MAX_BREAK_FRAC = 0.0
            ref, why = self.call()
            self.assertIsNotNone(ref)
        finally:
            T.ENTRY_MAX_BREAK_FRAC = saved

    def test_the_older_checks_still_fire_first(self):
        """The veto sits after the drift guard and the stop check, so a setup
        that fails one of those is still reported against THAT reason -- the
        new check must not take credit for a refusal it did not make."""
        # mid far below the planned entry: the drift guard owns this one.
        ref, why = T.entry_is_actionable(Backend(97.0), "TEST", self.ENTRY,
                                         self.STOP, self.TP, level=self.LEVEL,
                                         bars_1h=bars(97.0))
        self.assertIsNone(ref)
        self.assertIn("from the planned entry", why)
        # mid at the stop but within the drift guard: the stop check owns it.
        ref, why = T.entry_is_actionable(Backend(98.0), "TEST", 98.5,
                                         self.STOP, self.TP, level=self.LEVEL,
                                         bars_1h=bars(97.0))
        self.assertIsNone(ref)
        self.assertIn("at or below the stop", why)

    def test_it_only_ever_removes_trades(self):
        """Whatever the bar did, the veto can refuse but never approve
        something the existing checks rejected."""
        for low in (101.0, 100.0, 99.0, 98.0, 97.0, 90.0):
            with_bars, _ = self.call(bars_1h=bars(low))
            without, _ = self.call(bars_1h=None)
            if with_bars is not None:
                self.assertIsNotNone(without,
                                     "the veto approved something the base path refused")


if __name__ == "__main__":
    unittest.main()

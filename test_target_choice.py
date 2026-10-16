"""Which resistance rung the target aims at.

Both callers used to take the NEAREST rung above price. That quietly tied the
target to the admission test: the ceiling above a support barely moves, so the
closer a support sits to price the less headroom it has and the worse its
reward/risk reads, however good the support itself is. Every freshly formed
shelf was refused for it -- measured 2026-09-28, 41 levels across the live
universe, median 42% return on margin and median 7 touches on the rung they
should have been aiming at.

The property that makes this safe to ship is that it is STRICTLY ADDITIVE. When
no rung clears both gates the nearest is still returned, so the caller's filter
refuses the setup for exactly the reason it refuses it today. The change can
turn a rejection into a trade; it can never turn a trade into a rejection.
"""

import unittest

from research.levels import config as C
from research.levels.levels import Level, choose_target, resistance_ladder, sigma_distance


def R(price, touches=5, holds=5):
    return Level(price=price, kind="resistance", touches=touches, holds=holds)


class TargetChoice(unittest.TestCase):
    # entry/stop/sigma chosen so rung 1 fails both gates and rung 2 clears
    # them -- the XPL 0.10327 shape, in round numbers.
    ENTRY, STOP, SIGMA = 100.0, 98.0, 0.01

    def rungs(self, *prices):
        return [R(p) for p in prices]

    def oh(self, price):
        return sigma_distance(price, self.ENTRY, self.SIGMA)

    def rr(self, price):
        return (price - self.ENTRY) / (self.ENTRY - self.STOP)

    def choose(self, *prices):
        return choose_target(self.rungs(*prices), self.ENTRY, self.STOP, self.SIGMA)

    # -- the fix --------------------------------------------------------------

    def test_it_skips_a_rung_that_fails_headroom(self):
        near, far = 100.5, 106.0          # 0.5s vs ~5.8s of headroom
        self.assertLess(self.oh(near), C.TRADE_MIN_OVERHEAD_SIGMA)
        self.assertGreaterEqual(self.oh(far), C.TRADE_MIN_OVERHEAD_SIGMA)
        rung, ladder, qualified = self.choose(near, far)
        self.assertTrue(qualified)
        self.assertEqual(rung.price, far)
        self.assertEqual([L.price for L in ladder], [near, far])

    def test_it_skips_a_rung_that_fails_reward_risk(self):
        # Far enough in sigma, but too little gain against a 2.0 risk unit.
        near = self.ENTRY + (C.MIN_RR - 0.5) * (self.ENTRY - self.STOP)
        far = self.ENTRY + (C.MIN_RR + 2.0) * (self.ENTRY - self.STOP)
        self.assertLess(self.rr(near), C.MIN_RR)
        rung, _, qualified = self.choose(near, far)
        self.assertTrue(qualified)
        self.assertEqual(rung.price, far)

    def test_it_takes_the_nearest_rung_that_qualifies_not_the_furthest(self):
        """A nearer target is likelier to be reached; this is meant to unblock
        setups, not reach for the biggest payoff."""
        rung, _, qualified = self.choose(100.5, 106.0, 120.0)
        self.assertTrue(qualified)
        self.assertEqual(rung.price, 106.0)

    def test_rung_one_is_kept_when_it_already_qualifies(self):
        rung, _, qualified = self.choose(106.0, 120.0)
        self.assertTrue(qualified)
        self.assertEqual(rung.price, 106.0)

    # -- the safety property -------------------------------------------------

    def test_when_no_rung_qualifies_the_nearest_is_returned_unqualified(self):
        """So the caller's own gate refuses it for the reason it does today."""
        rung, ladder, qualified = self.choose(100.2, 100.4, 100.6)
        self.assertFalse(qualified)
        self.assertEqual(rung.price, 100.2)
        self.assertEqual(len(ladder), 3)

    def test_nothing_mapped_above_stays_nothing(self):
        rung, ladder, qualified = choose_target([], self.ENTRY, self.STOP, self.SIGMA)
        self.assertIsNone(rung)
        self.assertEqual(ladder, [])
        self.assertFalse(qualified)

    def test_rungs_at_or_below_the_entry_are_not_targets(self):
        rung, ladder, _ = choose_target(self.rungs(95.0, 100.0, 106.0),
                                        self.ENTRY, self.STOP, self.SIGMA)
        self.assertEqual([L.price for L in ladder], [106.0])
        self.assertEqual(rung.price, 106.0)

    def test_bad_geometry_falls_back_rather_than_dividing_by_zero(self):
        for entry, stop, sigma in ((100.0, 100.0, 0.01),     # no risk unit
                                   (100.0, 101.0, 0.01),     # stop above entry
                                   (100.0, 98.0, 0.0)):      # no sigma
            rung, ladder, qualified = choose_target(
                self.rungs(100.5, 106.0), entry, stop, sigma)
            self.assertFalse(qualified)
            self.assertEqual(rung.price, 100.5)

    def test_the_ladder_is_capped_and_ordered(self):
        got = resistance_ladder(self.rungs(110.0, 101.0, 130.0, 105.0, 120.0), 100.0)
        self.assertEqual(len(got), C.TARGET_MAX_RUNGS)
        self.assertEqual([L.price for L in got][:3], [101.0, 105.0, 110.0])


class MeasuredFromTheEntryNotThePrice(unittest.TestCase):
    """The AAVE case, 2026-10-01.

    The ladder used to filter on the CURRENT price while headroom and
    reward/risk were already measured from the entry -- one function asking two
    different questions. A level far below an extended price could then find no
    target at all: AAVE at 166.74 had five mapped resistances above a 142.84
    entry (147, 150, 156, 163, 167) and exactly one above the current price, so
    six of its eleven levels were refused for "no resistance mapped above".
    Two of those six were the operator's own stated buy zones.

    The rule was never "this level has nowhere to go". It was "price is
    currently too far above this level for us to measure where it could go",
    which is a different statement and the wrong one to gate on.
    """

    AAVE_RESISTANCES = [118.19, 120.93, 124.0, 129.57, 133.78, 138.03, 140.71,
                        147.15, 150.13, 156.19, 163.28, 166.97]
    ENTRY, STOP, SIGMA = 142.84, 139.0, 0.012

    def rungs(self):
        return [Level(price=p, kind="resistance", touches=5, holds=4)
                for p in self.AAVE_RESISTANCES]

    def test_a_deep_level_under_an_extended_price_still_finds_a_target(self):
        rung, ladder, qualified = choose_target(
            self.rungs(), self.ENTRY, self.STOP, self.SIGMA)
        self.assertIsNotNone(rung, "a level with five rungs above it found none")
        self.assertTrue(qualified)
        self.assertEqual([L.price for L in ladder], [147.15, 150.13, 156.19])

    def test_the_old_price_relative_rule_would_have_found_almost_nothing(self):
        """Pinned as the counter-example. If this stops being true the fix is
        harmless, but the reasoning above has gone stale."""
        px = 166.74
        old_ladder = [p for p in self.AAVE_RESISTANCES if p > px]
        new_ladder = [p for p in self.AAVE_RESISTANCES if p > self.ENTRY]
        self.assertEqual(len(old_ladder), 1)
        self.assertEqual(len(new_ladder), 5)

    def test_when_price_sits_at_the_entry_nothing_changes(self):
        """The drift guard keeps price within 1.5% of the entry when a trade
        actually fires, so at entry time the two rules coincide -- this only
        ever affected whether a level was VISIBLE while price was away."""
        at_entry = [p for p in self.AAVE_RESISTANCES if p > self.ENTRY * 1.005]
        rung, ladder, _ = choose_target(self.rungs(), self.ENTRY, self.STOP, self.SIGMA)
        self.assertEqual([L.price for L in ladder], at_entry[:C.TARGET_MAX_RUNGS])


class BothCallersAgree(unittest.TestCase):
    """The stop rule diverged between these two modules for a week. The target
    rule is the same shape of risk, so it is pinned the same way."""

    def test_scanner_and_call_import_the_one_chooser(self):
        import research.levels.scanner as sc
        import research.levels.call as cl
        for mod in (sc, cl):
            fn = getattr(mod, "choose_target", None)
            self.assertIsNotNone(fn, f"{mod.__name__} does not import choose_target")
            self.assertEqual(fn.__module__, "research.levels.levels")
            self.assertEqual(fn.__qualname__, "choose_target")

    def test_neither_module_measures_the_ladder_from_the_current_price(self):
        import inspect
        import research.levels.scanner as sc
        import research.levels.call as cl
        for fn in (sc.scan_coin, cl._candidates_for):
            src = inspect.getsource(fn)
            self.assertNotIn("choose_target(resistances, px", src,
                             f"{fn.__qualname__} still passes px to choose_target")

    def test_neither_module_picks_a_rung_by_hand(self):
        import inspect
        import research.levels.scanner as sc
        import research.levels.call as cl
        # _candidates_for, not find_candidates: the latter only fans out over
        # coins and timeframes and never touches a target.
        for fn in (sc.scan_coin, cl._candidates_for):
            src = inspect.getsource(fn)
            self.assertIn("choose_target", src,
                          f"{fn.__qualname__} chooses a target without the shared rule")
            self.assertNotIn("[:3]", src,
                             f"{fn.__qualname__} still slices a ladder by hand")


if __name__ == "__main__":
    unittest.main()

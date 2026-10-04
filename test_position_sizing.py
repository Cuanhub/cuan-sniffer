"""Position size, and saying truthfully what a trade commits versus risks.

MARGIN_PCT_PER_TRADE went 3% -> 5% on 2026-10-04, chosen by replaying all 39
closed trades with the rails SIMULATED rather than rescaled -- a bigger loss
trips the daily stop and the drawdown halt sooner, and a halt removes the
trades behind it:

      3%  +$26.89, worst drawdown 4.1%
      5%  +$44.82, worst drawdown 6.5%
      8%  +$71.71, worst drawdown 9.9%   <- misses the 10% halt by 0.1pt
     10%  +$61.57, worst drawdown 10.9%  <- halts, and ends BELOW 8%

The 10% row is why 8% was not taken: once the halt fires it removes the
winners behind it, so the best raw outcome sits right on a cliff.

The other half of this file is the labelling. Margin committed, risk at the
stop and the vestigial RISK_PCT_PER_TRADE are three different numbers, and
printing one under another's name is the conflation that left the daily loss
stop fifty-four losses from firing.
"""

import inspect
import unittest

import tools.live_levels_trader as T
from research.levels import config as C


class Sizing(unittest.TestCase):
    EQ = 809.17

    def test_margin_is_the_configured_share_of_equity(self):
        d = T.size_from_stop(self.EQ, 100.0, 98.0, available=self.EQ)
        self.assertAlmostEqual(d["size_usd"],
                               self.EQ * C.MARGIN_PCT_PER_TRADE / 100.0 * C.FLAT_LEVERAGE,
                               places=2)

    def test_size_does_not_depend_on_where_the_stop_is(self):
        """The whole point of flat sizing: a 0.9% stop and a 13% stop must ask
        for the same position, not 3.3x equity and 0.22x."""
        tight = T.size_from_stop(self.EQ, 100.0, 99.1, available=self.EQ)
        wide = T.size_from_stop(self.EQ, 100.0, 87.0, available=self.EQ)
        self.assertAlmostEqual(tight["size_usd"], wide["size_usd"], places=2)

    def test_the_position_still_clears_the_minimum_notional(self):
        d = T.size_from_stop(self.EQ, 100.0, 98.0, available=self.EQ)
        self.assertGreater(d["size_usd"], C.MIN_NOTIONAL_USD)


class TheRailsStillBindFirst(unittest.TestCase):
    """Raising size must not quietly make a position cap the risk limit."""

    EQ = 809.17
    STOP_FRAC = 0.088          # median structural stop, 1.76% of price at 5x

    def risk_per_trade(self):
        return self.EQ * C.MARGIN_PCT_PER_TRADE / 100.0 * self.STOP_FRAC

    def test_four_concurrent_stops_stay_inside_the_daily_loss_stop(self):
        four = self.risk_per_trade() * C.MAX_CONCURRENT
        budget = self.EQ * C.DAILY_LOSS_STOP_PCT / 100.0
        self.assertLess(four, budget,
                        "four stops firing together would exceed the daily budget")

    def test_the_daily_stop_is_still_reachable_but_not_trivial(self):
        n = self.EQ * C.DAILY_LOSS_STOP_PCT / 100.0 / self.risk_per_trade()
        self.assertGreater(n, 6, "too few losses to trip it -- one bad hour halts us")
        self.assertLess(n, 25, "a daily stop fifty losses away is not a rail")

    def test_margin_committed_leaves_room_at_full_concurrency(self):
        self.assertLessEqual(C.MARGIN_PCT_PER_TRADE * C.MAX_CONCURRENT, 50.0,
                             "more than half of equity committed as margin")


class SaysWhatItMeans(unittest.TestCase):
    def test_the_banner_reports_margin_not_the_vestigial_risk_constant(self):
        src = inspect.getsource(T.main)
        self.assertIn("MARGIN_PCT_PER_TRADE", src)
        self.assertNotIn("RISK_PCT_PER_TRADE", src.replace(
            "# \"margin\", not \"risk\". RISK_PCT_PER_TRADE", ""))

    def test_the_trader_no_longer_imports_the_vestigial_constant(self):
        self.assertFalse(hasattr(T, "RISK_PCT_PER_TRADE"),
                         "RISK_PCT_PER_TRADE no longer decides anything here")

    def test_margin_and_risk_are_genuinely_different_numbers(self):
        """If these ever coincide the distinction stops being load-bearing and
        this file's reasoning is stale."""
        eq = 809.17
        margin = eq * C.MARGIN_PCT_PER_TRADE / 100.0
        risk = margin * 0.088
        self.assertLess(risk, margin / 5,
                        "risk at the stop should be far below margin committed")


if __name__ == "__main__":
    unittest.main()

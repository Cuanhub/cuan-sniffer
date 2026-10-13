"""Bounding the account by drawdown rather than by trade count.

The old daily stop was calibrated against a risk-per-trade that was wrong by
about 11x. Its reasoning -- "at 3% risk a 10% stop halts after 3 losing
trades" -- was sound, but 3% is the MARGIN COMMITTED, not the amount at risk.
The structural stop sits a median 1.76% below the fill, so at 5x the real risk
is $2.11 against $797 of equity: 0.26%. Measured against the observed $2.20
average loss, 15% halted after FIFTY-FOUR consecutive losses. It was not a
rail, and the count cap was quietly doing all the work -- 433 blocked passes,
more than every other reason combined.

The drawdown-from-peak rail is new. The daily stop resets at 00:00 UTC, so a
slow bleed never trips it: the three days to 2026-09-30 lost $19.99 across 17
trades with no single day coming close. Nothing watched the trajectory.
"""

import unittest

import tools.live_levels_trader as T
from research.levels import config as C


class Backend:
    def __init__(self, equity, used=0.0):
        self.equity, self.used = equity, used

    def get_margin_summary(self):
        return (self.equity, self.used, self.equity - self.used)


class DrawdownRail(unittest.TestCase):
    def pre(self, equity, state=None):
        st = {"owned": {}, "pending": {}, "realised_today": 0.0}
        st.update(state or {})
        return T.preflight(Backend(equity), st), st

    # Reference changed on 2026-10-13: the highest of the last
    # ACCOUNT_DD_LOOKBACK_DAYS OPENING balances, not an all-time high-water
    # mark. The old mark was set on 09-27 and never came down, so the rail
    # ratcheted -- the longer the account sat below its best-ever day the less
    # room it had, and it fired twice in six days while flat to recovering.

    def test_a_fresh_state_cannot_halt_itself(self):
        """No history yet, so current equity is the reference."""
        why, st = self.pre(800.0)
        self.assertIsNone(why)
        self.assertFalse(st.get("halted"))

    def test_it_measures_against_the_best_recent_day_open(self):
        opens = [700.0, 800.0, 750.0]
        at_limit = 800.0 * (1 - C.ACCOUNT_DD_HALT_PCT / 100.0)
        why, st = self.pre(at_limit - 0.01, {"day_open_equity": opens})
        self.assertTrue(st["halted"])
        self.assertIn("account drawdown", why)
        self.assertIn("day-opens", why)

    def test_it_does_not_halt_just_inside_the_limit(self):
        opens = [800.0]
        inside = 800.0 * (1 - (C.ACCOUNT_DD_HALT_PCT - 0.5) / 100.0)
        why, st = self.pre(inside, {"day_open_equity": opens})
        self.assertIsNone(why)
        self.assertFalse(st.get("halted"))

    def test_a_stale_high_no_longer_ratchets(self):
        """The whole point of the change. 817.51 on 09-27 must not still be
        constraining the account two weeks later."""
        why, st = self.pre(733.0, {"equity_peak": 817.51,
                                   "day_open_equity": [752.87, 740.01, 752.87]})
        self.assertIsNone(why, "a stale all-time high is still constraining us")
        self.assertFalse(st.get("halted"))

    def test_the_peak_is_still_tracked_but_decides_nothing(self):
        why, st = self.pre(900.0, {"equity_peak": 817.51,
                                   "day_open_equity": [800.0]})
        self.assertIsNone(why)
        self.assertAlmostEqual(st["equity_peak"], 900.0)

    def test_a_fast_collapse_inside_the_window_halts(self):
        """What a trailing reference IS for."""
        opens = [800.0, 795.0, 790.0]
        why, st = self.pre(700.0, {"day_open_equity": opens})   # -12.5% vs 800
        self.assertTrue(st["halted"])

    def test_a_slow_grind_does_NOT_halt_and_that_is_the_trade(self):
        """Pinned as a limitation, not an aspiration. I claimed the opposite
        when I wrote this change: a trailing reference cannot catch a slow
        bleed, because the reference falls with the account. A steady
        -$6.66/day -- the rate that cost $19.99 over the three days to 09-30 --
        measures 0.81% on day one and 3.13% by day thirty. Only an all-time
        mark catches that shape, and an all-time mark is the ratchet being
        removed. The slow case is the operator's to spot in the daily report.
        """
        eq = 817.51
        opens = [eq]
        for day in range(1, 40):
            eq -= 6.66
            opens = (opens + [eq])[-C.ACCOUNT_DD_LOOKBACK_DAYS:]
            why, st = self.pre(eq, {"day_open_equity": opens})
            self.assertFalse(st.get("halted"),
                             f"halted on day {day} -- if a trailing window now "
                             f"catches a slow grind, the reasoning above is stale")

    def test_the_rail_can_be_switched_off(self):
        saved = T.ACCOUNT_DD_HALT_PCT
        try:
            T.ACCOUNT_DD_HALT_PCT = 0.0
            why, st = self.pre(100.0, {"day_open_equity": [817.51]})
            self.assertIsNone(why)
            self.assertFalse(st.get("halted"))
        finally:
            T.ACCOUNT_DD_HALT_PCT = saved

    def test_empty_or_junk_history_does_not_divide_by_zero(self):
        for hist in ([], [0.0], [None, 0.0]):
            why, st = self.pre(800.0, {"day_open_equity": hist})
            self.assertIsNone(why)

    def test_equity_is_recorded_for_the_next_day_roll(self):
        why, st = self.pre(800.0, {"day_open_equity": [800.0]})
        self.assertAlmostEqual(st["_equity_for_day_open"], 800.0)


class DailyStopIsNowARail(unittest.TestCase):
    def test_eighteen_average_losses_trip_it(self):
        eq, avg_loss = 797.52, 2.20
        n = int(eq * C.DAILY_LOSS_STOP_PCT / 100.0 / avg_loss)
        self.assertGreater(n, 10, "too tight to trade through normal variance")
        self.assertLess(n, 25, "a daily stop fifty losses away is not a rail")

    def test_it_fires_on_the_realised_total(self):
        st = {"owned": {}, "pending": {},
              "realised_today": -797.52 * C.DAILY_LOSS_STOP_PCT / 100.0 - 1.0,
              "equity_peak": 797.52}
        why = T.preflight(Backend(797.52), st)
        self.assertTrue(st["halted"])
        self.assertIn("daily loss stop", why)

    def test_the_worst_day_so_far_would_not_have_halted(self):
        st = {"owned": {}, "pending": {}, "realised_today": -12.44,
              "equity_peak": 817.51}
        why = T.preflight(Backend(805.07), st)
        self.assertIsNone(why)


class CountCapIsNowABackstop(unittest.TestCase):
    def test_drawdown_binds_before_the_count_does(self):
        """The whole point of the change: the count must not be what stops a
        healthy flat account from taking a setup."""
        eq, avg_loss = 797.52, 2.20
        losses_to_daily_stop = eq * C.DAILY_LOSS_STOP_PCT / 100.0 / avg_loss
        self.assertGreater(C.MAX_NEW_PER_DAY, losses_to_daily_stop,
                           "the count cap would still bind before drawdown")


if __name__ == "__main__":
    unittest.main()

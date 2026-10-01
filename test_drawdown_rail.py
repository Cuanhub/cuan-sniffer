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

    def test_the_peak_is_recorded_on_a_fresh_state(self):
        why, st = self.pre(800.0)
        self.assertIsNone(why)
        self.assertAlmostEqual(st["equity_peak"], 800.0)

    def test_the_peak_only_ever_rises(self):
        why, st = self.pre(790.0, {"equity_peak": 817.51})
        self.assertIsNone(why)
        self.assertAlmostEqual(st["equity_peak"], 817.51)

    def test_a_new_high_advances_it(self):
        why, st = self.pre(900.0, {"equity_peak": 817.51})
        self.assertIsNone(why)
        self.assertAlmostEqual(st["equity_peak"], 900.0)

    def test_it_halts_at_the_limit(self):
        peak = 817.51
        at_limit = peak * (1 - C.ACCOUNT_DD_HALT_PCT / 100.0)
        why, st = self.pre(at_limit - 0.01, {"equity_peak": peak})
        self.assertTrue(st["halted"])
        self.assertIn("account drawdown", why)

    def test_it_does_not_halt_just_inside_the_limit(self):
        peak = 817.51
        inside = peak * (1 - (C.ACCOUNT_DD_HALT_PCT - 0.5) / 100.0)
        why, st = self.pre(inside, {"equity_peak": peak})
        self.assertIsNone(why)
        self.assertFalse(st.get("halted"))

    def test_the_drawdown_we_have_actually_had_would_not_halt(self):
        """Worst through the first 26 trades was 3.25% of an $817.51 peak."""
        why, st = self.pre(817.51 * (1 - 0.0325), {"equity_peak": 817.51})
        self.assertIsNone(why)

    def test_the_slow_bleed_that_no_day_would_catch(self):
        """17 trades over three days for -$19.99, every day inside the daily
        stop. Stretch that to the limit and this rail is what stops it."""
        peak, eq = 817.51, 817.51
        halted_at = None
        for day in range(1, 40):
            eq -= 6.66                      # the observed daily bleed
            why, st = self.pre(eq, {"equity_peak": peak})
            if st.get("halted"):
                halted_at = day
                break
        self.assertIsNotNone(halted_at, "a persistent bleed must halt eventually")
        self.assertLess(halted_at, 30)
        self.assertGreater(eq, peak * 0.85, "it should fire well before -15%")

    def test_the_rail_can_be_switched_off(self):
        saved = T.ACCOUNT_DD_HALT_PCT
        try:
            T.ACCOUNT_DD_HALT_PCT = 0.0
            why, st = self.pre(100.0, {"equity_peak": 817.51})
            self.assertIsNone(why)
            self.assertFalse(st.get("halted"))
        finally:
            T.ACCOUNT_DD_HALT_PCT = saved

    def test_a_missing_peak_does_not_divide_by_zero(self):
        why, st = self.pre(800.0, {"equity_peak": 0.0})
        self.assertIsNone(why)


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

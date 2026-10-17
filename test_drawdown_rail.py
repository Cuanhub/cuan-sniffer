"""Bounding the account by drawdown rather than by trade count.

The old daily stop was calibrated against a risk-per-trade that was wrong by
about 11x. Its reasoning -- "at 3% risk a 10% stop halts after 3 losing
trades" -- was sound, but 3% is the MARGIN COMMITTED, not the amount at risk.
The structural stop sits a median 1.76% below the fill, so at 5x the real risk
is $2.11 against $797 of equity: 0.26%. Measured against the observed $2.20
average loss, 15% halted after FIFTY-FOUR consecutive losses. It was not a
rail, and the count cap was quietly doing all the work -- 433 blocked passes,
more than every other reason combined.

The drawdown rail itself has moved to test_rolling_drawdown.py, which covers
it on the rolling 24h window it uses since 2026-10-15. What stays here is the
DAILY stop and the count cap, and the relationship between them: the count cap
was quietly doing all the work and must not be what stops a healthy flat
account from taking a setup.
"""

import unittest

import tools.live_levels_trader as T
from research.levels import config as C


class Backend:
    def __init__(self, equity, used=0.0):
        self.equity, self.used = equity, used

    def get_margin_summary(self):
        return (self.equity, self.used, self.equity - self.used)


class DailyStopIsNowARail(unittest.TestCase):
    def test_eighteen_average_losses_trip_it(self):
        eq, avg_loss = 797.52, 2.20
        n = int(eq * C.DAILY_LOSS_STOP_PCT / 100.0 / avg_loss)
        self.assertGreater(n, 10, "too tight to trade through normal variance")
        self.assertLess(n, 25, "a daily stop fifty losses away is not a rail")

    def test_it_fires_on_the_realised_total(self):
        # No equity_samples, so the rolling drawdown reference is current
        # equity and the only rail that can fire is the daily stop.
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


class ConfigInvariants(unittest.TestCase):
    """Relationships between dials that have each been changed many times.

    None of these is obvious from any single constant's own comment, and all
    of them are the kind of thing that breaks quietly when one number moves.
    Checked in the 2026-10-17 pre-resume review.
    """

    def test_the_daily_stop_fires_before_the_drawdown_halt(self):
        """Otherwise the gentler, self-clearing rail never gets a turn."""
        self.assertLess(C.DAILY_LOSS_STOP_PCT, C.ACCOUNT_DD_HALT_PCT)

    def test_sizing_cannot_over_commit_the_account(self):
        committed = C.MARGIN_PCT_PER_TRADE * C.MAX_CONCURRENT
        self.assertLessEqual(committed, 100.0,
                             f"{committed:.0f}% of equity would be committed")

    def test_flat_leverage_is_inside_the_cap(self):
        self.assertLessEqual(C.FLAT_LEVERAGE, C.MAX_LEVERAGE)

    def test_retention_keeps_a_full_watch_lifecycle(self):
        """seed_watchlist_from_log replays dry_run rows inside
        WATCH_TTL_HOURS, so pruning inside that window would hide watches
        the trader still intends to use."""
        self.assertGreater(C.RETAIN_WAITING_DAYS * 24, C.WATCH_TTL_HOURS)

    def test_the_audit_trail_is_never_on_a_clock(self):
        for action in ("entry", "closed", "protection", "emergency_exit"):
            self.assertIn(action, C.RETAIN_FOREVER_ACTIONS)

    def test_entry_spacing_and_the_day_cap_do_not_contradict(self):
        """Either can bind; what matters is that neither is nonsense."""
        max_by_spacing = 24 * 60 / C.ENTRY_SPACING_MINUTES
        self.assertGreater(max_by_spacing, 1,
                           "spacing allows at most one trade a day")
        self.assertGreater(C.MAX_NEW_PER_DAY, 1)

    def test_the_vetoes_do_not_admit_nothing(self):
        """Measured on 781 candidates since 2026-10-10: basis<=0 keeps 73%,
        then pull/run<=0.60 keeps 32%, then rr>=2 keeps 26% -- about 29
        admissible a day against a 24-a-day spacing limit. If a threshold
        moves far enough that this inverts, the rails stop being rails and
        become an off switch."""
        self.assertGreaterEqual(C.ENTRY_MAX_PULL_OVER_RUN, 0.3,
                                "pull/run this tight rejected ~89% of "
                                "candidates when measured")
        self.assertLessEqual(C.MIN_RR, 3.0)

"""The drawdown rail on a rolling 24h window, and halts that release.

Replaces the day-open machinery (test_day_open_reading.py, deleted). That
design recorded one equity reading per UTC day and measured against the best
of the last three. Two things were wrong with it, and the second is why it was
changed on instruction:

  1. The reading was taken below preflight's early returns, so a halted
     trader -- which returns early on every pass -- took no readings at all.
     Across the 2026-10-13 halt the same frozen pre-halt figure was recorded
     as both the 10-14 and the 10-15 day-open.

  2. More fundamentally, a reference that only moves when a day rolls can
     strand the account. The 10-13 halt spent two days measured against a
     10-12 reference the account could no longer reach.

A rolling window cannot do either. Nothing older than the window survives in
it, so the reference follows the account within a day, and the halt it sets
releases itself.

Calibration, from the 3,193 equity readings logged across 23 days: a rolling
24h rail at 10% fires exactly once -- the 10-12/13 collapse, 10.85% -- and the
next worst day is 4.97%. Every limit from 5% to 10% fires on that same single
day; 4% fires on two.
"""

import unittest
from unittest import mock
import datetime as dt

import tools.live_levels_trader as T
from research.levels import config as C


class Backend:
    def __init__(self, equity, used=0.0):
        self.equity, self.used, self.reads = equity, used, 0

    def get_margin_summary(self):
        self.reads += 1
        if self.equity is None:
            return None
        return (self.equity, self.used, self.equity - self.used)


NOW = dt.datetime(2026, 10, 15, 12, 0, tzinfo=dt.timezone.utc)


def ago(hours, equity):
    """A sample `hours` before NOW."""
    return [(NOW - dt.timedelta(hours=hours)).isoformat().replace("+00:00", "Z"),
            equity]


class Clock:
    @staticmethod
    def now(tz=None):
        return NOW

    @staticmethod
    def fromisoformat(s):
        return dt.datetime.fromisoformat(s)

    @staticmethod
    def strftime(*a):
        raise AssertionError("unexpected")


def frozen():
    return mock.patch.object(T, "datetime", Clock)


class Window(unittest.TestCase):
    def test_old_samples_are_dropped(self):
        st = {"equity_samples": [ago(48, 900.0), ago(25, 850.0),
                                 ago(23, 800.0), ago(1, 700.0)]}
        with frozen():
            kept = T.prune_equity_samples(st, NOW)
        self.assertEqual(kept, [800.0, 700.0])

    def test_junk_samples_are_skipped_not_fatal(self):
        st = {"equity_samples": [ago(1, 800.0), ["not-a-date", 900.0],
                                 ["2026-10-15T11:00:00Z", "abc"], [], None,
                                 ago(2, 0.0)]}
        with frozen():
            kept = T.prune_equity_samples(st, NOW)
        self.assertEqual(kept, [800.0])

    def test_the_reference_cannot_outlive_the_window(self):
        """The ratchet, pinned. 817.51 from two weeks ago must be gone."""
        st = {"equity_samples": [ago(336, 817.51), ago(2, 710.0)]}
        with frozen():
            dd, ref = T.drawdown_now(st, 704.79)
        self.assertAlmostEqual(ref, 710.0)
        self.assertLess(dd, C.ACCOUNT_DD_HALT_PCT)


class Rail(unittest.TestCase):
    def pre(self, equity, state=None):
        st = {"owned": {}, "pending": {}, "realised_today": 0.0}
        st.update(state or {})
        with frozen():
            return T.preflight(Backend(equity), st), st

    def test_a_fresh_state_cannot_halt_itself(self):
        why, st = self.pre(800.0)
        self.assertIsNone(why)
        self.assertFalse(st.get("halted"))

    def test_a_fast_collapse_inside_the_window_halts(self):
        """What a trailing reference IS for -- the 10-12/13 shape."""
        why, st = self.pre(709.61, {"equity_samples": [ago(20, 807.90)]})
        self.assertTrue(st["halted"])
        self.assertEqual(st["halt_kind"], "drawdown")
        self.assertIn("account drawdown", why)
        self.assertIn("24h", why)

    def test_the_observed_worst_day_short_of_the_collapse_does_not_halt(self):
        """10-07, the next worst at 4.97%, must stay inside the rail."""
        why, st = self.pre(750.10, {"equity_samples": [ago(12, 789.33)]})
        self.assertIsNone(why)
        self.assertFalse(st.get("halted"))

    def test_a_high_outside_the_window_does_not_halt(self):
        """Same collapse, but the high is 25h old -- it has aged out."""
        why, st = self.pre(709.61, {"equity_samples": [ago(25, 807.90),
                                                       ago(2, 712.0)]})
        self.assertIsNone(why)
        self.assertFalse(st.get("halted"))

    def test_a_slow_grind_does_NOT_halt_and_that_is_the_trade(self):
        """Pinned as a limitation, not an aspiration, and now tighter than
        before: a 24h window is even blinder to a slow bleed than a 3-day one.
        A steady -$6.66/day -- the rate that cost $19.99 over the three days
        to 09-30 -- cannot reach 10% of a reference that falls with it. Only
        an all-time mark catches that shape, and an all-time mark is the
        ratchet being removed. LOSS_STREAK_TRIGGER is what sees this case.
        """
        eq = 817.51
        for day in range(1, 40):
            prev = eq
            eq -= 6.66
            why, st = self.pre(eq, {"equity_samples": [ago(23, prev)]})
            self.assertFalse(st.get("halted"),
                             f"halted on day {day} -- if a 24h window now "
                             f"catches a slow grind, the note above is stale")

    def test_the_rail_can_be_switched_off(self):
        saved = T.ACCOUNT_DD_HALT_PCT
        try:
            T.ACCOUNT_DD_HALT_PCT = 0.0
            why, st = self.pre(100.0, {"equity_samples": [ago(1, 817.51)]})
            self.assertIsNone(why)
            self.assertFalse(st.get("halted"))
        finally:
            T.ACCOUNT_DD_HALT_PCT = saved

    def test_empty_or_junk_history_does_not_divide_by_zero(self):
        for hist in ([], [[None, 0.0]], None):
            why, st = self.pre(800.0, {"equity_samples": hist})
            self.assertIsNone(why)

    def test_the_peak_is_still_tracked_but_decides_nothing(self):
        why, st = self.pre(900.0, {"equity_peak": 817.51})
        self.assertIsNone(why)
        self.assertAlmostEqual(st["equity_peak"], 900.0)


class SamplingHappensOnEveryPass(unittest.TestCase):
    """Every early return used to skip the reading. A halted trader returns
    early on every pass, which is how it went two days without one."""

    def pre(self, equity, state=None):
        st = {"owned": {}, "pending": {}, "realised_today": 0.0}
        st.update(state or {})
        with frozen(), mock.patch.object(T, "save_state"):
            return T.preflight(Backend(equity), st), st

    def test_a_halted_pass_still_samples(self):
        why, st = self.pre(704.79, {
            "halted": True, "halt_kind": "manual",
            "halt_reason": "set by hand"})
        self.assertIn("halted", why)
        self.assertEqual([e for _, e in st["equity_samples"]], [704.79])

    def test_a_fully_loaded_pass_still_samples(self):
        owned = {f"C{i}": {} for i in range(C.MAX_CONCURRENT)}
        why, st = self.pre(704.79, {"owned": owned})
        self.assertIn("max", why)
        self.assertEqual([e for _, e in st["equity_samples"]], [704.79])

    def test_the_day_cap_pass_still_samples(self):
        why, st = self.pre(704.79, {"opened_today": C.MAX_NEW_PER_DAY})
        self.assertIn("today", why)
        self.assertEqual([e for _, e in st["equity_samples"]], [704.79])

    def test_a_failed_query_samples_nothing_and_still_refuses(self):
        why, st = self.pre(None)
        self.assertIn("refusing to size blind", why)
        self.assertFalse(st.get("equity_samples"))

    def test_the_sample_is_persisted(self):
        st = {"owned": {}, "pending": {}, "realised_today": 0.0,
              "halted": True, "halt_kind": "manual", "halt_reason": "x"}
        with frozen(), mock.patch.object(T, "save_state") as saved:
            T.preflight(Backend(704.79), st)
        self.assertTrue(saved.called)

    def test_sampling_happens_in_exactly_one_place(self):
        """Two write sites drift apart; that is how the last bug started."""
        src = open("tools/live_levels_trader.py").read()
        self.assertEqual(src.count('state["equity_samples"] = samples'), 1)


class AutoRelease(unittest.TestCase):
    def test_a_drawdown_halt_releases_when_the_high_ages_out(self):
        st = {"owned": {}, "pending": {}, "realised_today": 0.0,
              "halted": True, "halt_kind": "drawdown",
              "halt_reason": "account drawdown 11.1%",
              "equity_samples": [ago(25, 793.05), ago(2, 710.0)]}
        with frozen(), mock.patch.object(T, "save_state"), \
                mock.patch.object(T.notify, "send") as sent:
            why = T.preflight(Backend(704.79), st)
        self.assertIsNone(why)
        self.assertFalse(st["halted"])
        self.assertTrue(sent.called)

    def test_a_drawdown_halt_does_not_release_while_it_still_applies(self):
        st = {"owned": {}, "pending": {}, "realised_today": 0.0,
              "halted": True, "halt_kind": "drawdown",
              "halt_reason": "account drawdown 11.1%",
              "equity_samples": [ago(2, 793.05)]}
        with frozen(), mock.patch.object(T, "save_state"):
            why = T.preflight(Backend(704.79), st)
        self.assertIn("halted", why)
        self.assertTrue(st["halted"])

    def test_a_daily_loss_halt_releases_once_the_day_has_reset(self):
        """The exact 10-13 case: realised_today reset twice, flag stayed."""
        st = {"owned": {}, "pending": {}, "realised_today": 0.0,
              "halted": True, "halt_kind": "daily_loss",
              "halt_reason": "daily loss stop: -39.42 against 709.61 equity"}
        with frozen(), mock.patch.object(T, "save_state"), \
                mock.patch.object(T.notify, "send"):
            why = T.preflight(Backend(704.79), st)
        self.assertIsNone(why)
        self.assertFalse(st["halted"])

    def test_a_daily_loss_halt_holds_while_the_day_is_still_down(self):
        st = {"owned": {}, "pending": {}, "realised_today": -40.0,
              "halted": True, "halt_kind": "daily_loss",
              "halt_reason": "daily loss stop"}
        with frozen(), mock.patch.object(T, "save_state"):
            why = T.preflight(Backend(704.79), st)
        self.assertIn("halted", why)

    def test_a_legacy_halt_is_read_from_its_reason(self):
        """State written before halts carried a kind -- the live 10-13 halt."""
        st = {"owned": {}, "pending": {}, "realised_today": 0.0,
              "halted": True,
              "halt_reason": "daily loss stop: -39.42 against 709.61 equity"}
        with frozen(), mock.patch.object(T, "save_state"), \
                mock.patch.object(T.notify, "send"):
            why = T.preflight(Backend(704.79), st)
        self.assertIsNone(why)
        self.assertFalse(st["halted"])

    def test_an_unrecognised_halt_is_never_auto_released(self):
        """Clearing a halt by guessing is the one thing this must not do."""
        for st in ({"halted": True, "halt_kind": "manual",
                    "halt_reason": "stopped by hand"},
                   {"halted": True, "halt_reason": "something unfamiliar"}):
            st.update({"owned": {}, "pending": {}, "realised_today": 0.0})
            with frozen(), mock.patch.object(T, "save_state"):
                why = T.preflight(Backend(704.79), st)
            self.assertIn("halted", why)
            self.assertTrue(st["halted"])

    def test_release_can_be_switched_off(self):
        saved = T.HALT_AUTO_RELEASE
        try:
            T.HALT_AUTO_RELEASE = False
            st = {"owned": {}, "pending": {}, "realised_today": 0.0,
                  "halted": True, "halt_kind": "daily_loss",
                  "halt_reason": "daily loss stop"}
            with frozen(), mock.patch.object(T, "save_state"):
                why = T.preflight(Backend(704.79), st)
            self.assertIn("halted", why)
        finally:
            T.HALT_AUTO_RELEASE = saved

    def test_release_clears_the_alert_key_so_the_next_halt_speaks(self):
        st = {"owned": {}, "pending": {}, "realised_today": 0.0,
              "halted": True, "halt_kind": "daily_loss",
              "halt_reason": "daily loss stop",
              "alert_keys": {"halt": 123.0, "other": 1.0}}
        with frozen(), mock.patch.object(T, "save_state"), \
                mock.patch.object(T.notify, "send"):
            T.preflight(Backend(704.79), st)
        self.assertNotIn("halt", st["alert_keys"])
        self.assertIn("other", st["alert_keys"])


if __name__ == "__main__":
    unittest.main()

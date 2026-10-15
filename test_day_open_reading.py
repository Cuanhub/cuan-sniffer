"""The drawdown window has to keep advancing while the trader is halted.

The reference is the highest of the last ACCOUNT_DD_LOOKBACK_DAYS opening
balances, which only works if an opening balance is actually recorded each
day. Two paths through preflight returned before the reading was taken: the
halted check, and the concurrency and day-count caps. A halted trader returns
early on EVERY pass, so from the 2026-10-13 halt to 10-15 it took no equity
readings at all and roll_day recorded the same frozen pre-halt figure as both
the 10-14 and the 10-15 open:

    day_open_equity = [793.05, 752.87, 720.729607, 720.729607]
                                      ^^^^^^^^^^  ^^^^^^^^^^

Left alone the window converges on copies of whatever was true at the moment
of the halt. If that figure was a local high the rail keeps measuring against
it and never releases, and the longer the halt lasts the less real history
survives in the window.
"""

import unittest
from unittest import mock

import tools.live_levels_trader as T
from research.levels import config as C


class Backend:
    """Counts reads, so a test can tell a skipped query from a cheap one."""

    def __init__(self, equity, used=0.0):
        self.equity, self.used, self.reads = equity, used, 0

    def get_margin_summary(self):
        self.reads += 1
        if self.equity is None:
            return None
        return (self.equity, self.used, self.equity - self.used)


def at(day):
    """Pin the UTC date the code sees."""
    class Clock:
        @staticmethod
        def now(tz=None):
            import datetime as dt
            return dt.datetime.fromisoformat(day + "T12:00:00+00:00")
    return mock.patch.object(T, "datetime", Clock)


class ReadingIsTakenOnEveryPass(unittest.TestCase):
    def pre(self, equity, state=None, day="2026-10-15"):
        st = {"owned": {}, "pending": {}, "realised_today": 0.0}
        st.update(state or {})
        b = Backend(equity)
        with at(day):
            why = T.preflight(b, st)
        return why, st, b

    def test_a_halted_pass_still_records_the_day_open(self):
        why, st, b = self.pre(704.79, {
            "halted": True, "halt_reason": "daily loss stop"})
        self.assertIn("halted", why)
        self.assertEqual(b.reads, 1)
        self.assertEqual(st["_equity_for_day_open"], 704.79)
        self.assertEqual(st["_equity_for_day_open_day"], "2026-10-15")

    def test_a_fully_loaded_pass_still_records_the_day_open(self):
        """Holding the maximum across midnight froze the reading too."""
        owned = {f"C{i}": {} for i in range(C.MAX_CONCURRENT)}
        why, st, b = self.pre(704.79, {"owned": owned})
        self.assertIn("max", why)
        self.assertEqual(st["_equity_for_day_open"], 704.79)

    def test_the_day_cap_pass_still_records_the_day_open(self):
        why, st, b = self.pre(704.79, {"opened_today": C.MAX_NEW_PER_DAY})
        self.assertIn("today", why)
        self.assertEqual(st["_equity_for_day_open"], 704.79)

    def test_a_failed_query_records_nothing_and_still_refuses(self):
        why, st, b = self.pre(None)
        self.assertIn("refusing to size blind", why)
        self.assertNotIn("_equity_for_day_open", st)

    def test_nonpositive_equity_is_not_recorded(self):
        why, st, b = self.pre(0.0)
        self.assertNotIn("_equity_for_day_open", st)

    def test_the_reading_is_taken_in_exactly_one_place(self):
        """Two write sites drift apart; that is how this bug started."""
        src = open("tools/live_levels_trader.py").read()
        self.assertEqual(src.count('state["_equity_for_day_open"] = '), 1)


class RollDayRefusesStaleReadings(unittest.TestCase):
    def roll(self, state, day):
        with at(day), mock.patch.object(T, "save_state"):
            return T.roll_day(state)

    def test_a_same_day_reading_is_recorded(self):
        st = {"day": "2026-10-15", "day_open_equity": [700.0],
              "_equity_for_day_open": 704.79,
              "_equity_for_day_open_day": "2026-10-15"}
        self.assertTrue(self.roll(st, "2026-10-16"))
        self.assertEqual(st["day_open_equity"], [700.0, 704.79])

    def test_a_frozen_reading_is_skipped_not_duplicated(self):
        """The exact 10-14 / 10-15 case: reading stamped 10-13, day is 10-14."""
        st = {"day": "2026-10-14", "day_open_equity": [793.05, 752.87],
              "_equity_for_day_open": 720.729607,
              "_equity_for_day_open_day": "2026-10-13"}
        self.assertTrue(self.roll(st, "2026-10-15"))
        self.assertEqual(st["day_open_equity"], [793.05, 752.87])

    def test_an_unstamped_reading_is_skipped(self):
        """State written by the previous build carries no stamp."""
        st = {"day": "2026-10-15", "day_open_equity": [700.0],
              "_equity_for_day_open": 720.729607}
        self.assertTrue(self.roll(st, "2026-10-16"))
        self.assertEqual(st["day_open_equity"], [700.0])

    def test_a_long_halt_cannot_fill_the_window_with_one_value(self):
        """Replays the bug: halted, so the reading never refreshes."""
        st = {"day": "2026-10-13", "day_open_equity": [793.05, 752.87],
              "_equity_for_day_open": 720.729607,
              "_equity_for_day_open_day": "2026-10-13"}
        self.roll(st, "2026-10-14")          # legitimate: stamped 10-13
        for d, nxt in (("2026-10-14", "2026-10-15"),
                       ("2026-10-15", "2026-10-16"),
                       ("2026-10-16", "2026-10-17")):
            st["day"] = d
            self.roll(st, nxt)
        self.assertEqual(st["day_open_equity"], [793.05, 752.87, 720.729607])
        self.assertEqual(st["day_open_equity"].count(720.729607), 1)

    def test_the_window_is_capped_at_the_lookback_plus_one(self):
        st = {"day": "2026-10-15", "day_open_equity": [1.0] * 9,
              "_equity_for_day_open": 2.0,
              "_equity_for_day_open_day": "2026-10-15"}
        self.roll(st, "2026-10-16")
        self.assertEqual(len(st["day_open_equity"]),
                         C.ACCOUNT_DD_LOOKBACK_DAYS + 1)
        self.assertEqual(st["day_open_equity"][-1], 2.0)

    def test_a_same_day_call_does_not_roll_or_record(self):
        st = {"day": "2026-10-15", "day_open_equity": [700.0],
              "_equity_for_day_open": 704.79,
              "_equity_for_day_open_day": "2026-10-15"}
        self.assertFalse(self.roll(st, "2026-10-15"))
        self.assertEqual(st["day_open_equity"], [700.0])


class ReferenceSurvivesAHalt(unittest.TestCase):
    """End to end: halt, sit for days, and the window still tells the truth."""

    def test_the_reference_ages_out_across_a_halt(self):
        st = {"owned": {}, "pending": {}, "realised_today": 0.0,
              "halted": True, "halt_reason": "daily loss stop",
              "day": "2026-10-15",
              "day_open_equity": [793.05, 752.87, 720.729607]}
        for day, nxt in (("2026-10-15", "2026-10-16"),
                         ("2026-10-16", "2026-10-17")):
            b = Backend(704.79)
            with at(day):
                T.preflight(b, st)
            with at(nxt), mock.patch.object(T, "save_state"):
                T.roll_day(st)
        # Two genuine 704.79 readings recorded, and the stale 793.05 gone.
        self.assertEqual(st["day_open_equity"], [752.87, 720.729607,
                                                 704.79, 704.79])
        ref = max(st["day_open_equity"])
        dd = (ref - 704.79) / ref * 100.0
        self.assertLess(dd, C.ACCOUNT_DD_HALT_PCT)


if __name__ == "__main__":
    unittest.main()

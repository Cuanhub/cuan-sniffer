"""Standing down after a run of losing trades.

The clearest pattern in the first 98 closed trades is that losses arrive in
day-sized clumps rather than spread out. Of 21 trading days, 9 were all-loss
and 6 all-win; 10-07 was 0W/13L and 10-13 was 0W/12L. The expectancy of the
NEXT trade collapses with the streak:

    streak 0   n=23   mean +$3.28   52% win
    streak 3   n= 5   mean -$3.50    0%
    streak 6+  n=42   mean -$2.28    7%

Forty-two of 98 trades were opened while already six or more losses deep.

3 losses / 24h is the only pair that survived the whole screen. Replaying the
book, skipping any trade whose entry fell inside a stand-down:

    baseline          -$62.03   PF 0.73   22% win   98 trades
    3 losses / 24h    +$58.95   PF 1.76   41% win   41 trades
    4 losses / 12h    +$23.15   PF 1.19   31% win   58 trades

Permutation test holding every timestamp fixed and reassigning which PnL
landed where: 3L/24h p=0.0002. Shuffled data still gains about +$23 on
average, because truncating any losing run helps -- which is why the test
matters. Out of sample 3L/24h is the only variant that improves in both
chronological halves.

Read honestly: n=98 over 21 days in one mostly-falling regime, and 3/24h was
the best of nine combinations tried, so the exact pair is partly fitted.
"""

import unittest
from unittest import mock
import datetime as dt

import tools.live_levels_trader as T
from research.levels import config as C

NOW = dt.datetime(2026, 10, 15, 12, 0, tzinfo=dt.timezone.utc)


class Clock:
    @staticmethod
    def now(tz=None):
        return NOW

    @staticmethod
    def fromisoformat(s):
        return dt.datetime.fromisoformat(s)


def frozen():
    return mock.patch.object(T, "datetime", Clock)


def at(hours):
    """An ISO stamp `hours` from NOW (negative for the past)."""
    return (NOW + dt.timedelta(hours=hours)).isoformat().replace("+00:00", "Z")


class Counting(unittest.TestCase):
    def note(self, state, pnl):
        with frozen(), mock.patch.object(T.notify, "send"):
            return T.note_closed_outcome(state, pnl)

    def test_a_win_resets_the_run(self):
        st = {"loss_streak": 2}
        self.assertFalse(self.note(st, 5.0))
        self.assertEqual(st["loss_streak"], 0)
        self.assertNotIn("streak_cooldown_until", st)

    def test_losses_accumulate_until_the_trigger(self):
        st = {}
        for i in range(1, C.LOSS_STREAK_TRIGGER):
            self.assertFalse(self.note(st, -2.0))
            self.assertEqual(st["loss_streak"], i)
            self.assertNotIn("streak_cooldown_until", st)

    def test_the_trigger_opens_a_stand_down(self):
        st = {}
        for _ in range(C.LOSS_STREAK_TRIGGER - 1):
            self.note(st, -2.0)
        self.assertTrue(self.note(st, -2.0))
        self.assertIn("streak_cooldown_until", st)
        with frozen():
            self.assertAlmostEqual(T.streak_cooldown_remaining(st),
                                   C.LOSS_STREAK_COOLDOWN_HOURS, places=3)

    def test_a_breakeven_close_counts_as_a_loss(self):
        """pnl of exactly 0 is a round trip paid in fees, not a win."""
        st = {}
        self.note(st, 0.0)
        self.assertEqual(st["loss_streak"], 1)

    def test_the_run_resets_on_triggering_so_one_run_buys_one_standdown(self):
        """Letting it keep counting would re-trigger on every later loss and
        extend the stand-down indefinitely. The replay measured a reset."""
        st = {}
        for _ in range(C.LOSS_STREAK_TRIGGER):
            self.note(st, -2.0)
        self.assertEqual(st["loss_streak"], 0)
        self.assertFalse(self.note(st, -2.0))

    def test_the_trigger_can_be_switched_off(self):
        saved = T.LOSS_STREAK_TRIGGER
        try:
            T.LOSS_STREAK_TRIGGER = 0
            st = {}
            for _ in range(20):
                self.assertFalse(self.note(st, -2.0))
            self.assertNotIn("streak_cooldown_until", st)
        finally:
            T.LOSS_STREAK_TRIGGER = saved


class Remaining(unittest.TestCase):
    def test_no_cooldown_reads_zero(self):
        with frozen():
            self.assertEqual(T.streak_cooldown_remaining({}), 0.0)

    def test_an_expired_cooldown_reads_zero(self):
        with frozen():
            self.assertEqual(
                T.streak_cooldown_remaining({"streak_cooldown_until": at(-1)}),
                0.0)

    def test_junk_reads_zero_rather_than_blocking_forever(self):
        with frozen():
            self.assertEqual(
                T.streak_cooldown_remaining({"streak_cooldown_until": "x"}),
                0.0)

    def test_a_live_cooldown_reports_hours_left(self):
        with frozen():
            self.assertAlmostEqual(
                T.streak_cooldown_remaining({"streak_cooldown_until": at(6)}),
                6.0, places=3)


class Preflight(unittest.TestCase):
    class Backend:
        def get_margin_summary(self):
            return (704.79, 0.0, 704.79)

    def pre(self, state):
        st = {"owned": {}, "pending": {}, "realised_today": 0.0}
        st.update(state)
        with frozen(), mock.patch.object(T, "save_state"):
            return T.preflight(self.Backend(), st), st

    def test_a_live_stand_down_blocks_opening(self):
        why, st = self.pre({"streak_cooldown_until": at(5)})
        self.assertIsNotNone(why)
        self.assertIn("losses in a row", why)
        self.assertIn("5.0h left", why)

    def test_an_expired_stand_down_does_not_block(self):
        why, st = self.pre({"streak_cooldown_until": at(-1)})
        self.assertIsNone(why)

    def test_a_halt_still_takes_precedence(self):
        why, st = self.pre({"streak_cooldown_until": at(5), "halted": True,
                            "halt_kind": "manual", "halt_reason": "by hand"})
        self.assertIn("halted", why)

    def test_the_stand_down_does_not_stop_the_equity_sample(self):
        """The drawdown window must keep advancing through a stand-down for
        the same reason it must through a halt."""
        why, st = self.pre({"streak_cooldown_until": at(5)})
        self.assertEqual([e for _, e in st["equity_samples"]], [704.79])


class EveryCloseIsCounted(unittest.TestCase):
    def test_accrue_close_is_the_only_caller(self):
        """A close that does not reach note_closed_outcome is invisible to
        this rail for the same reason it would be invisible to the daily
        loss stop."""
        src = open("tools/live_levels_trader.py").read()
        self.assertEqual(src.count("note_closed_outcome("), 2)  # def + 1 call


class ReplayMatchesTheMeasurement(unittest.TestCase):
    """The live rule must behave like the replay that justified it.

    Rebuilt here as a pure sequence so a future change to the thresholds is
    checked against the shape of the finding, not just against itself.
    """

    def simulate(self, closes, trigger, hours):
        """closes: list of (entry_hour, close_hour, pnl). Returns kept pnl."""
        state = {}
        kept = []
        for enter, close, pnl in closes:
            with mock.patch.object(T, "datetime", type("C", (), {
                    "now": staticmethod(lambda tz=None: NOW + dt.timedelta(hours=enter)),
                    "fromisoformat": staticmethod(dt.datetime.fromisoformat)})):
                if T.streak_cooldown_remaining(state) > 0:
                    continue
            kept.append(pnl)
            with mock.patch.object(T, "datetime", type("C", (), {
                    "now": staticmethod(lambda tz=None: NOW + dt.timedelta(hours=close)),
                    "fromisoformat": staticmethod(dt.datetime.fromisoformat)})), \
                    mock.patch.object(T.notify, "send"):
                saved_t, saved_h = T.LOSS_STREAK_TRIGGER, T.LOSS_STREAK_COOLDOWN_HOURS
                try:
                    T.LOSS_STREAK_TRIGGER, T.LOSS_STREAK_COOLDOWN_HOURS = trigger, hours
                    T.note_closed_outcome(state, pnl)
                finally:
                    T.LOSS_STREAK_TRIGGER, T.LOSS_STREAK_COOLDOWN_HOURS = saved_t, saved_h
        return kept

    def test_a_losing_run_is_truncated(self):
        """Thirteen losses an hour apart -- the 10-07 shape. A 3/24h rule
        must take three and stand down through the rest."""
        closes = [(i, i + 0.5, -3.0) for i in range(13)]
        kept = self.simulate(closes, 3, 24.0)
        self.assertEqual(len(kept), 3)
        self.assertEqual(sum(kept), -9.0)

    def test_wins_keep_it_trading(self):
        closes = [(i, i + 0.5, 5.0 if i % 2 else -3.0) for i in range(13)]
        kept = self.simulate(closes, 3, 24.0)
        self.assertEqual(len(kept), 13, "a win should reset the run")

    def test_it_re_arms_after_the_stand_down_expires(self):
        closes = [(0, 0.5, -3.0), (1, 1.5, -3.0), (2, 2.5, -3.0),
                  (3, 3.5, -3.0),      # inside the stand-down, skipped
                  (30, 30.5, -3.0)]    # after it, taken
        kept = self.simulate(closes, 3, 24.0)
        self.assertEqual(len(kept), 4)


if __name__ == "__main__":
    unittest.main()


class LiftingAStandDown(unittest.TestCase):
    """There was no way to lift one until 2026-10-16.

    /pause set a stand-down and nothing unset it. On 10-15 the rail fired at
    13:07, an hour before the market turned, and then locked the account out
    for the remaining eight hours of a recovery that was visible in the
    equity curve. The rail is still right on the measured sample; being
    unable to override it was not.
    """

    def test_clearing_removes_the_cooldown_and_reports_the_time_left(self):
        st = {"streak_cooldown_until": at(6), "loss_streak": 2}
        with frozen(), mock.patch.object(T, "save_state"):
            left = T.clear_standdown(st)
        self.assertAlmostEqual(left, 6.0, places=2)
        self.assertNotIn("streak_cooldown_until", st)

    def test_clearing_also_resets_the_losing_run(self):
        """Leaving it at 2 would re-trigger on the next single loss, which is
        not what lifting a stand-down means."""
        st = {"streak_cooldown_until": at(6), "loss_streak": 2}
        with frozen(), mock.patch.object(T, "save_state"):
            T.clear_standdown(st)
        self.assertEqual(st["loss_streak"], 0)

    def test_clearing_when_none_is_active_is_harmless(self):
        st = {}
        with frozen(), mock.patch.object(T, "save_state"):
            self.assertEqual(T.clear_standdown(st), 0.0)
        self.assertEqual(st["loss_streak"], 0)

    def test_a_cleared_standdown_no_longer_blocks_opening(self):
        class Backend:
            def get_margin_summary(self):
                return (704.79, 0.0, 704.79)
        st = {"owned": {}, "pending": {}, "realised_today": 0.0,
              "streak_cooldown_until": at(6)}
        with frozen(), mock.patch.object(T, "save_state"):
            self.assertIn("losses in a row", T.preflight(Backend(), st))
            T.clear_standdown(st)
            self.assertIsNone(T.preflight(Backend(), st))

    def test_a_queued_pause_zero_lifts_it(self):
        import json, os, pathlib as _p
        tmp = _p.Path(os.environ.get("TMPDIR", "/tmp")) / f"sd{os.getpid()}.json"
        saved = T.CONTROL_QUEUE
        T.CONTROL_QUEUE = tmp
        try:
            tmp.write_text(json.dumps([{"cmd": "pause", "arg": "0"}]))
            st = {"streak_cooldown_until": at(6), "loss_streak": 2}
            with frozen(), mock.patch.object(T, "save_state"):
                done = T.drain_control_queue(st)
            self.assertNotIn("streak_cooldown_until", st)
            self.assertTrue(any("cleared" in d for d in done), done)
        finally:
            T.CONTROL_QUEUE = saved
            tmp.unlink(missing_ok=True)

    def test_a_queued_positive_pause_still_sets_one(self):
        import json, os, pathlib as _p
        tmp = _p.Path(os.environ.get("TMPDIR", "/tmp")) / f"sp{os.getpid()}.json"
        saved = T.CONTROL_QUEUE
        T.CONTROL_QUEUE = tmp
        try:
            tmp.write_text(json.dumps([{"cmd": "pause", "arg": "4"}]))
            st = {}
            with frozen(), mock.patch.object(T, "save_state"):
                T.drain_control_queue(st)
            self.assertIn("streak_cooldown_until", st)
        finally:
            T.CONTROL_QUEUE = saved
            tmp.unlink(missing_ok=True)

    def test_the_clearing_logic_exists_in_exactly_one_place(self):
        src = open("tools/live_levels_trader.py").read()
        self.assertEqual(src.count('state.pop("streak_cooldown_until", None)'), 1)

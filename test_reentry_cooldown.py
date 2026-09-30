"""Leaving a coin alone after it stopped us out, and the daily cap holding
inside a pass as well as at the start of one.

Both are missing guards rather than tuning. They came out of the same three
days: 17 trades, 2 wins, -$19.99, in a universe whose median coin ROSE 1.5%.

LIT was $13.16 of that on its own -- bought three times inside four hours,
twice at the same level, 28 and 19 minutes after being stopped out of it,
while being the worst performer in the universe at -10%.

The cooldown keys off the REALISED RESULT, not merely off closing, and that
distinction is the whole design. Across the 26 live trades there were 12
re-entries: under 2h after a loss went 0W/3L for -$10.06, while the two after
a WIN went 2W/0L for +$26.63. A level that stopped us out is a level that
broke. A level that paid is a level that worked.
"""

import unittest
from datetime import datetime, timedelta, timezone

import tools.live_levels_trader as T


def iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def ago(hours):
    return iso(datetime.now(timezone.utc) - timedelta(hours=hours))


class Cooldown(unittest.TestCase):
    def test_a_fresh_state_cools_nothing(self):
        self.assertEqual(T.cooldown_remaining({}, "LIT"), 0.0)

    def test_a_losing_close_starts_it(self):
        st = {}
        T.start_cooldown(st, "LIT")
        self.assertGreater(T.cooldown_remaining(st, "LIT"), 0.0)

    def test_it_expires(self):
        st = {"cooled": {"LIT": ago(T.COIN_REENTRY_COOLDOWN_HOURS + 0.5)}}
        self.assertEqual(T.cooldown_remaining(st, "LIT"), 0.0)

    def test_it_still_bites_just_before_expiry(self):
        st = {"cooled": {"LIT": ago(T.COIN_REENTRY_COOLDOWN_HOURS - 0.25)}}
        self.assertGreater(T.cooldown_remaining(st, "LIT"), 0.0)

    def test_the_lit_re_entries_would_all_have_been_refused(self):
        # 0.5h and 0.3h after the previous stop-out, and the 0.8h HYPE one.
        for gap in (0.8, 0.5, 0.3):
            st = {"cooled": {"LIT": ago(gap)}}
            self.assertGreater(T.cooldown_remaining(st, "LIT"), 0.0,
                               f"a re-entry {gap}h after a stop-out must be refused")

    def test_the_profitable_re_entries_would_still_have_been_allowed(self):
        # PUMP came back 3.4h after a WIN (+20.35) and PONS 92h after one.
        # A win never stamps a cooldown, so the gap does not matter -- but the
        # losing re-entries that DID pay were 15.9h and later, so check those
        # are outside the window too.
        for gap in (15.9, 20.1, 21.6):
            st = {"cooled": {"LIT": ago(gap)}}
            self.assertEqual(T.cooldown_remaining(st, "LIT"), 0.0,
                             f"a re-entry {gap}h later is outside the evidence")

    def test_it_is_per_coin_not_per_level(self):
        """LIT's fourth entry was a DIFFERENT level (3.784 after 3.911), so a
        per-level cooldown would not have stopped it."""
        st = {"cooled": {"LIT": ago(0.3)}}
        self.assertGreater(T.cooldown_remaining(st, "LIT"), 0.0)
        self.assertEqual(T.cooldown_remaining(st, "HYPE"), 0.0)

    def test_the_coin_name_is_matched_case_insensitively(self):
        st = {}
        T.start_cooldown(st, "lit")
        self.assertGreater(T.cooldown_remaining(st, "LIT"), 0.0)

    def test_a_corrupt_stamp_does_not_cool_forever(self):
        for bad in ("", "not-a-date", None):
            self.assertEqual(T.cooldown_remaining({"cooled": {"LIT": bad}}, "LIT"), 0.0)

    def test_expired_stamps_are_pruned_so_state_cannot_grow(self):
        st = {"cooled": {"OLD": ago(T.COIN_REENTRY_COOLDOWN_HOURS + 1),
                         "NEW": ago(0.1)}}
        left = T.prune_cooldowns(st)
        self.assertEqual(left, 1)
        self.assertNotIn("OLD", st["cooled"])
        self.assertIn("NEW", st["cooled"])


class AccrueCloseStamps(unittest.TestCase):
    class Backend:
        def __init__(self, pnl):
            self.pnl = pnl

        def get_realised_pnl(self, coin, since):
            return self.pnl

    def accrue(self, pnl):
        st = {"realised_today": 0.0}
        note = T.accrue_close(self.Backend(pnl), st, "LIT",
                              {"opened_at": ago(3), "size_usd": 100.0,
                               "entry": 4.0, "stop": 3.9})
        return st, note

    def test_a_loss_puts_the_coin_on_cooldown(self):
        st, note = self.accrue(-2.28)
        self.assertGreater(T.cooldown_remaining(st, "LIT"), 0.0)
        self.assertIn("cooldown", note)

    def test_a_win_does_not(self):
        st, note = self.accrue(+6.28)
        self.assertEqual(T.cooldown_remaining(st, "LIT"), 0.0)
        self.assertNotIn("cooldown", note)

    def test_a_scratch_counts_as_a_loss(self):
        """Zero realised is a round trip paid in fees for nothing, which is the
        churn this is here to stop."""
        st, _ = self.accrue(0.0)
        self.assertGreater(T.cooldown_remaining(st, "LIT"), 0.0)

    def test_the_realised_total_is_still_accrued(self):
        st, _ = self.accrue(-2.28)
        self.assertAlmostEqual(st["realised_today"], -2.28)


class DailyCapInsideAPass(unittest.TestCase):
    """preflight checks the cap once per pass. On 2026-09-28 it saw
    opened_today=4, allowed the pass, and BTC then filled (5) with DOGE
    filling six seconds later (6) -- one over."""

    def test_the_open_loop_rechecks_the_daily_cap(self):
        import inspect
        src = inspect.getsource(T.main)
        head = src[src.index("concurrency cap reached"):]
        window = head[:900]
        self.assertIn("MAX_NEW_PER_DAY", window,
                      "the per-order guard does not re-check the daily cap")
        self.assertIn("daily cap reached", window)

    def test_preflight_still_checks_it_too(self):
        import inspect
        self.assertIn("MAX_NEW_PER_DAY", inspect.getsource(T.preflight))


if __name__ == "__main__":
    unittest.main()

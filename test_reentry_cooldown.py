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
import time
from core import streak
from tests.fake_venue import FakeHL, build_venue


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


class CooldownIsDerivedFromFills(unittest.TestCase):
    """Replaces AccrueCloseStamps.

    The cooldown used to be a timestamp stamped into the state file by
    accrue_close, whose PnL came from get_realised_pnl OR, when that query
    failed, from assumed_loss. So a rate limit could stamp a cooldown on a
    winning trade, and a restart between close and stamp lost it entirely.
    It is now computed from the venue's own fills, so it cannot be faked or
    lost -- same change as the losing run in core.streak.
    """

    def trades(self, pnl, hours_ago=1.0):
        """One completed round trip on LIT with this net result."""
        now = int(time.time() * 1000)
        closed = now - int(hours_ago * 3_600_000)
        fee = 0.05
        return [
            {"coin": "LIT", "time": closed - 1000, "side": "B", "sz": "25.0",
             "startPosition": "0.0", "closedPnl": "0.0", "fee": str(fee),
             "tid": 1},
            {"coin": "LIT", "time": closed, "side": "A", "sz": "25.0",
             "startPosition": "25.0", "closedPnl": str(pnl + 2 * fee),
             "fee": str(fee), "tid": 2},
        ]

    def cooling(self, pnl, hours_ago=1.0, window=None):
        fake = FakeHL()
        fake.fills_override = self.trades(pnl, hours_ago)
        return streak.coins_on_cooldown(
            build_venue(fake),
            window if window is not None else T.COIN_REENTRY_COOLDOWN_HOURS)

    def test_a_loss_puts_the_coin_on_cooldown(self):
        self.assertGreater(self.cooling(-2.28).get("LIT", 0.0), 0.0)

    def test_a_win_does_not(self):
        self.assertNotIn("LIT", self.cooling(+6.28))

    def test_a_scratch_counts_as_a_loss(self):
        """Zero realised is a round trip paid in fees for nothing, which is
        the churn this is here to stop. Net of fees it is a loss, so it falls
        out of the same comparison rather than needing a special case."""
        self.assertGreater(self.cooling(0.0).get("LIT", 0.0), 0.0)

    def test_the_cooldown_expires(self):
        self.assertNotIn("LIT", self.cooling(
            -2.28, hours_ago=T.COIN_REENTRY_COOLDOWN_HOURS + 1))

    def test_a_later_win_clears_an_earlier_loss(self):
        """Only the most recent trade on a coin counts -- "leave it alone
        until it behaves"."""
        fake = FakeHL()
        rows = self.trades(-2.28, hours_ago=2.0) + self.trades(+5.0, hours_ago=1.0)
        fake.fills_override = rows
        self.assertNotIn("LIT", streak.coins_on_cooldown(
            build_venue(fake), T.COIN_REENTRY_COOLDOWN_HOURS))

    def test_an_unreadable_venue_blocks_nothing(self):
        fake = FakeHL()
        fake.fills_override = self.trades(-2.28)
        v = build_venue(fake)
        fake.fail_info_always = True
        self.assertEqual(streak.coins_on_cooldown(v, 24.0), {})

    def test_the_entry_path_consults_it(self):
        import inspect
        self.assertIn("coins_on_cooldown", inspect.getsource(T.execute_one))



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

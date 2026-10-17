"""One entry at a time, an hour apart.

Not a tuned parameter but a correction to a design error. The open loop fills
every free slot in a single pass, so four triggers in one scan became four
positions in the SAME MINUTE -- APT, AVAX, HYPE and INJ at 00:06 on
2026-10-13. All six of that morning's entries closed inside the hour for
-$18.99 and the account drawdown halt fired.

In a universe whose median pairwise hourly correlation is 0.47, four
simultaneous longs are not four bets. They are one bet sized four times; four
positions behave like 2.4 independent ones.

It is the largest effect in the live book. Across 90 closed trades, split by
how many other entries landed within an hour:

    alone        33 trades   45% win   +$59.90
    +1 other     20 trades   15% win   -$32.83
    +2 others     9 trades    0% win   -$24.40
    +3 or more   28 trades   14% win   -$37.36

Isolated entries made money. Clustered ones lost $94.59.
"""

import unittest
from datetime import datetime, timedelta, timezone

import tools.live_levels_trader as T


def ago(minutes):
    return (datetime.now(timezone.utc) - timedelta(minutes=minutes)).strftime(
        "%Y-%m-%dT%H:%M:%SZ")


class Spacing(unittest.TestCase):
    def test_a_fresh_state_blocks_nothing(self):
        self.assertEqual(T.spacing_remaining({}), 0.0)

    def test_an_entry_starts_the_clock(self):
        st = {}
        T.mark_entry(st)
        self.assertGreater(T.spacing_remaining(st), 0.0)
        self.assertTrue(st["last_entry_at"].endswith("Z"))

    def test_it_blocks_inside_the_window(self):
        st = {"last_entry_at": ago(T.ENTRY_SPACING_MINUTES - 5)}
        left = T.spacing_remaining(st)
        self.assertGreater(left, 0)
        self.assertLessEqual(left, 6)

    def test_it_clears_after_the_window(self):
        st = {"last_entry_at": ago(T.ENTRY_SPACING_MINUTES + 1)}
        self.assertEqual(T.spacing_remaining(st), 0.0)

    def test_the_0006_cluster_would_have_been_refused(self):
        """APT, AVAX, HYPE and INJ all filled within the same minute."""
        st = {}
        T.mark_entry(st)                      # APT goes on
        for _ in ("AVAX", "HYPE", "INJ"):     # the rest arrive seconds later
            self.assertGreater(T.spacing_remaining(st), 0.0,
                               "a same-minute entry was not refused")

    def test_it_applies_across_coins_unlike_the_per_coin_cooldown(self):
        st = {}
        T.mark_entry(st)
        self.assertGreater(T.spacing_remaining(st), 0.0)
        self.assertEqual(T.cooldown_remaining(st, "ANYTHING"), 0.0)

    def test_a_corrupt_stamp_does_not_block_forever(self):
        for bad in ("", "not-a-date", None):
            self.assertEqual(T.spacing_remaining({"last_entry_at": bad}), 0.0)

    def test_it_can_be_switched_off(self):
        saved = T.ENTRY_SPACING_MINUTES
        try:
            T.ENTRY_SPACING_MINUTES = 0.0
            st = {}
            T.mark_entry(st)
            self.assertEqual(T.spacing_remaining(st), 0.0)
        finally:
            T.ENTRY_SPACING_MINUTES = saved


class WiredEverywhereItMatters(unittest.TestCase):
    def test_the_open_loop_breaks_rather_than_logging_a_skip_each_time(self):
        """Checking only inside execute_one would write a skip row for every
        remaining candidate instead of ending the pass."""
        import inspect
        src = inspect.getsource(T.main)
        head = src[src.index("concurrency cap reached"):]
        self.assertIn("spacing_remaining(state)", head[:1400])
        self.assertIn("stopping this pass", head[:1400])

    def test_the_market_path_refuses_and_stamps(self):
        import inspect
        src = inspect.getsource(T.execute_one)
        self.assertIn("spacing_remaining(state)", src)
        self.assertIn("entry_spacing", src)
        self.assertIn("mark_entry(state)", src)

    def test_a_resting_bid_placement_respects_it(self):
        """A bid commits future exposure even though it cannot be refused once
        it fills."""
        import inspect
        self.assertIn("spacing_remaining(state)",
                      inspect.getsource(T.rest_one_bid))

    def test_placing_a_bid_stamps_the_clock(self):
        """Stamped at PLACEMENT, not on fill.

        It used to be stamped on fill, by the function that turned a filled
        bid into a tracked position. That function is gone -- the venue is
        read directly now -- and placement is the better moment anyway: the
        exposure is committed there, and a fill can happen while nothing is
        watching.
        """
        import inspect
        self.assertIn("mark_entry(state)", inspect.getsource(T.rest_one_bid))

    def test_stamping_happens_once_per_path(self):
        import inspect
        for fn in (T.execute_one, T.rest_one_bid):
            self.assertEqual(inspect.getsource(fn).count("mark_entry(state)"), 1,
                             f"{fn.__name__} stamps more than once")


if __name__ == "__main__":
    unittest.main()

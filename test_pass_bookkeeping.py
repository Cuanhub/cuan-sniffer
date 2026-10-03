"""Bookkeeping that must happen on EVERY pass, not only the ones that trade.

preflight continues the loop whenever it refuses a pass -- cap full, daily stop,
halted, margin query failed. With the watchlist and cooldown pruning below that
early return, they stopped running exactly when the trader was busiest.

Measured 2026-10-03: four positions held for most of the previous day meant the
concurrency cap refused nearly every pass, and in that window 110 of 976 watched
levels aged past the 48h TTL without being dropped, the oldest at 60.4h. Six
expired cooldown stamps were still sitting in state.

The watchlist one is not cosmetic. build_intents gives a watched level
passes_quality(watched=True), which skips the depth and rest admission gates --
so a level the TTL had retired stayed eligible to trade.

Same shape as three bugs already fixed here: the day rollover, the asset-meta
load and the universe refresh were each placed where they ran on some cycles
rather than every cycle.
"""

import inspect
import unittest
from datetime import datetime, timedelta, timezone

import tools.live_levels_trader as T


def ago(hours):
    return (datetime.now(timezone.utc) - timedelta(hours=hours)).strftime(
        "%Y-%m-%dT%H:%M:%SZ")


class PruningRunsBeforeTheEarlyReturn(unittest.TestCase):
    """Asserted on source order, because the condition only shows up across
    passes and a unit test cannot run the daemon loop."""

    def setUp(self):
        self.src = inspect.getsource(T.main)

    def positions(self, *names):
        return [self.src.index(n) for n in names]

    def test_watchlist_pruning_precedes_preflight(self):
        prune, pre = self.positions("prune_watchlist(watching)",
                                    "why = preflight(backend, state)")
        self.assertLess(prune, pre,
                        "prune_watchlist sits after the preflight early return, "
                        "so the TTL stops enforcing whenever a pass is refused")

    def test_cooldown_pruning_precedes_preflight(self):
        prune, pre = self.positions("prune_cooldowns(state)",
                                    "why = preflight(backend, state)")
        self.assertLess(prune, pre)

    def test_the_early_return_is_still_there(self):
        """If this ever stops being true the ordering above is moot, and the
        reasoning in this file has gone stale rather than merely redundant."""
        after = self.src[self.src.index("why = preflight(backend, state)"):]
        self.assertIn("continue", after[:600])

    def test_pruning_happens_once_per_pass_not_twice(self):
        """Counted inside the daemon loop only. prune_watchlist is also called
        by the one-shot --seed-watchlist branch, which is a different path and
        legitimately its own call -- an earlier version of this test counted
        across the whole of main() and flagged that as a duplicate."""
        loop = self.src[self.src.index("reconcile_pending(backend, state)"):]
        self.assertEqual(loop.count("prune_watchlist(watching)"), 1)
        self.assertEqual(loop.count("prune_cooldowns(state)"), 1)


class PruningItself(unittest.TestCase):
    def test_expired_watch_entries_are_dropped(self):
        w = {"fresh": {"admitted_at": ago(1)},
             "stale": {"admitted_at": ago(T.WATCH_TTL_HOURS + 2)}}
        dropped = T.prune_watchlist(w)
        self.assertEqual(dropped, 1)
        self.assertEqual(set(w), {"fresh"})

    def test_an_entry_with_no_timestamp_is_dropped(self):
        w = {"broken": {}}
        self.assertEqual(T.prune_watchlist(w), 1)
        self.assertEqual(w, {})

    def test_expired_cooldowns_are_dropped(self):
        st = {"cooled": {"OLD": ago(T.COIN_REENTRY_COOLDOWN_HOURS + 1),
                         "NEW": ago(0.5)}}
        left = T.prune_cooldowns(st)
        self.assertEqual(left, 1)
        self.assertEqual(set(st["cooled"]), {"NEW"})

    def test_an_expired_stamp_never_blocked_a_trade_anyway(self):
        """The cooldown half of this was bounded: cooldown_remaining reads the
        timestamp, so a stale entry returns 0.0 and refuses nothing. Only the
        watchlist half could change a trading decision."""
        st = {"cooled": {"OLD": ago(T.COIN_REENTRY_COOLDOWN_HOURS + 10)}}
        self.assertEqual(T.cooldown_remaining(st, "OLD"), 0.0)


if __name__ == "__main__":
    unittest.main()

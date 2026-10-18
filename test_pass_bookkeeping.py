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


if __name__ == "__main__":
    unittest.main()

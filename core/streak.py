"""Consecutive losing trades, derived from venue fills.

The old stand-down kept a counter in the state file and incremented it from
`accrue_close`, whose PnL came from `get_realised_pnl` OR, when that query
failed, from `assumed_loss` -- which is negative by construction. So a rate
limit manufactured losing trades, and the counter then drifted from reality
with no way to re-derive it. That is the same root cause as everything else
this rebuild removes: a number the system believed instead of looked up.

Nothing is counted here. The run is computed from fills every time it is
asked for, so a wrong answer cannot persist, and a trade the venue does not
show cannot be in it.

Grouping fills into trades is the only real work. `closedPnl` is per FILL, not
per trade: one position closed in three fills is three rows. A trade is the
span from a fill whose `startPosition` is zero to the fill that brings the
position back to zero, and its result is the sum of `closedPnl` less fees over
that span.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from core.venue import Venue

TRIGGER = int(os.getenv("STREAK_TRIGGER", "3"))
COOLDOWN_HOURS = float(os.getenv("STREAK_COOLDOWN_HOURS", "24"))
# How far back to ask for fills. Long enough that a slow run of losses is
# still visible, short enough to stay one cheap request.
LOOKBACK_HOURS = float(os.getenv("STREAK_LOOKBACK_HOURS", "336"))
_FLAT = 1e-12


@dataclass(frozen=True)
class Trade:
    """One round trip, from flat back to flat."""
    coin: str
    opened_ms: int
    closed_ms: int
    pnl: float            # closedPnl summed over the trade, net of fees
    fills: int

    @property
    def lost(self) -> bool:
        return self.pnl < 0


def _f(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def trades_from_fills(fills: List[Dict[str, Any]]) -> List[Trade]:
    """Group fills into completed round trips, oldest close first.

    Two kinds of fill are deliberately excluded:

      - fills belonging to a trade whose opening is outside the window. Their
        entry fees are not in the data, so the result would be flattering by
        an unknown amount, and a loss limit must never be flattered.
      - fills of a position that is still open. It has no result yet.

    Position arithmetic uses each fill's own `startPosition` rather than an
    accumulated total, so float error cannot drift a position away from zero
    and merge two trades into one.
    """
    by_coin: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for f in fills:
        coin = str(f.get("coin", "")).upper()
        if coin:
            by_coin[coin].append(f)

    out: List[Trade] = []
    for coin, rows in by_coin.items():
        rows.sort(key=lambda r: (int(_f(r.get("time"))), int(_f(r.get("tid")))))
        opened_ms: Optional[int] = None
        pnl = 0.0
        n = 0
        for r in rows:
            start = _f(r.get("startPosition"))
            size = _f(r.get("sz"))
            signed = size if str(r.get("side", "")).upper() == "B" else -size
            after = start + signed
            if opened_ms is None:
                if abs(start) > _FLAT:
                    continue           # opened before the window: skip it
                opened_ms, pnl, n = int(_f(r.get("time"))), 0.0, 0
            pnl += _f(r.get("closedPnl")) - _f(r.get("fee"))
            n += 1
            if abs(after) <= _FLAT:
                out.append(Trade(coin, opened_ms, int(_f(r.get("time"))),
                                 pnl, n))
                opened_ms = None
    out.sort(key=lambda t: t.closed_ms)
    return out


def losing_run(trades: List[Trade]) -> List[Trade]:
    """The unbroken run of losses ending at the most recent trade.

    Empty if the last trade won. A trade that broke even is not a loss: the
    old counter treated anything not strictly positive as one, so a scratch
    trade extended a run it should have ended.
    """
    run: List[Trade] = []
    for t in reversed(trades):
        if not t.lost:
            break
        run.append(t)
    return list(reversed(run))


@dataclass(frozen=True)
class StandDown:
    """An active stand-down, and why."""
    until_ms: int
    run: List[Trade]

    @property
    def hours_left(self) -> float:
        return max(0.0, (self.until_ms - time.time() * 1000) / 3_600_000)

    @property
    def reason(self) -> str:
        coins = ", ".join(t.coin for t in self.run)
        total = sum(t.pnl for t in self.run)
        return (f"{len(self.run)} losing trades in a row ({coins}), "
                f"{total:.2f} total")


def current(venue: Venue, ignore_before_ms: int = 0,
            now_ms: Optional[int] = None) -> Optional[StandDown]:
    """The stand-down in force, or None. Derived, never stored.

    It expires on its own: once COOLDOWN_HOURS have passed since the run's
    last close, this simply stops returning it. A win also ends it, because
    the run is recomputed and no longer reaches TRIGGER. Neither outcome
    needs anyone to clear a flag -- which is what went wrong before, when a
    stuck counter had to be cleared by hand.

    `ignore_before_ms` is the operator's override: trades that closed at or
    before it are not counted, so clearing a stand-down cannot be undone by
    the same trades, while a fresh loss afterwards starts a new run.
    """
    now = now_ms if now_ms is not None else int(time.time() * 1000)
    if TRIGGER <= 0:
        return None
    since = now - int(LOOKBACK_HOURS * 3_600_000)
    fills = venue.fills(since)
    if fills is None:
        # Cannot ask, so cannot claim a stand-down. Unlike a loss limit, the
        # safe default here is to allow trading: refusing on an unreadable
        # venue would stop the bot every time the API hiccupped.
        return None
    trades = [t for t in trades_from_fills(fills)
              if t.closed_ms > ignore_before_ms]
    run = losing_run(trades)
    if len(run) < TRIGGER:
        return None
    until = run[-1].closed_ms + int(COOLDOWN_HOURS * 3_600_000)
    if until <= now:
        return None                    # the run is old enough to have expired
    return StandDown(until_ms=until, run=run)

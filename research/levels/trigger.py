"""
Entry trigger: has the level actually held?

The quality filters decide which levels are worth watching, while price is
still 5-10 sigma above them. This decides the moment of entry, and it is a
separate question -- the user's labels are explicit that arrival alone is not
enough. Every setup they marked "maybe" was one price was sitting on at that
moment, with the reason given as "good trade IF it holds" and "if we maintain
this level in a channel". A bot that fired on arrival would buy precisely the
setups they decline.

The rule implemented here is theirs, and is the first testable version of
"it held": price came down into the zone within the recent lookback, and has
stayed above the level for TRIGGER_HOLD_HOURS since.

Read-only. This module computes a boolean from candles and knows nothing
about orders.
"""

import time
from typing import Any, Dict, List, Tuple

from research.levels.config import (
    TRIGGER_HOLD_HOURS, TRIGGER_TOUCH_LOOKBACK_HOURS, TRIGGER_ZONE_SIGMA,
    TRIGGER_MAX_ABOVE_SIGMA,
)
from research.levels.levels import log_sigma, sigma_distance


# How much 1h history the trigger reads. A NAMED constant because the window
# is not a free parameter: level_has_held derives sigma with
# log_sigma(closed, len(closed) - 1), which slices backwards by VOL_LOOKBACK
# (120), so a series longer than that produces a DIFFERENT sigma -- and sigma
# gates both the touch test and the max-above ceiling. Anything that supplies
# these bars from elsewhere has to supply this same window, so the number lives
# in one place rather than being repeated at each source.
TRIGGER_1H_HOURS = 72


def _normalise_1h(raw, keep_from_ms: int = 0) -> List[Dict[str, Any]]:
    """Venue rows -> the bar dicts this module reads, oldest first.

    `keep_from_ms` drops anything older, so a caller holding a longer 1h series
    can hand over exactly the window fetch_1h would have returned.
    """
    out = []
    for b in raw:
        try:
            t = int(b["t"])
            if t < keep_from_ms:
                continue
            out.append({"t": t, "o": float(b["o"]), "h": float(b["h"]),
                        "l": float(b["l"]), "c": float(b["c"])})
        except (KeyError, TypeError, ValueError):
            continue
    return sorted(out, key=lambda x: x["t"])


def trigger_window(bars_1h: List[Dict[str, Any]],
                   hours: int = TRIGGER_1H_HOURS,
                   now_ms: int = 0) -> List[Dict[str, Any]]:
    """Trim a longer 1h series to what fetch_1h would have returned.

    call.find_candidates already pulls 1h bars for every coin and used to throw
    them away, while this module fetched the same series again seconds later --
    60 duplicate venue calls a pass. The trim is what makes reuse safe rather
    than merely cheap: see TRIGGER_1H_HOURS for why the length matters.
    """
    now_ms = now_ms or int(time.time() * 1000)
    return _normalise_1h(bars_1h, now_ms - hours * 3600_000)


def fetch_1h(client, coin: str, hours: int = TRIGGER_1H_HOURS) -> List[Dict[str, Any]]:
    now_ms = int(time.time() * 1000)
    raw = client.get_candles(coin, "1h", now_ms - hours * 3600_000, now_ms)
    return _normalise_1h(raw)


def level_has_held(bars_1h: List[Dict[str, Any]], level: float,
                   hold_hours: int = TRIGGER_HOLD_HOURS,
                   touch_lookback: int = TRIGGER_TOUCH_LOOKBACK_HOURS,
                   zone_sigma: float = TRIGGER_ZONE_SIGMA,
                   max_above_sigma: float = TRIGGER_MAX_ABOVE_SIGMA) -> Tuple[bool, str]:
    """(fired, reason). Fires when price reached the level recently and has
    since held above it for hold_hours.

    The last bar is excluded as still forming -- a hold is only a hold once
    the candle has closed, and counting a live bar would let a wick that is
    about to be filled count as two hours of support.
    """
    if level <= 0 or len(bars_1h) < hold_hours + 2:
        return False, "not enough 1h history"
    closed = bars_1h[:-1]
    sigma = log_sigma(closed, len(closed) - 1)
    if sigma <= 0:
        return False, "sigma undefined"

    hold = closed[-hold_hours:]
    window = closed[-touch_lookback:]

    touched = any(abs(sigma_distance(b["l"], level, sigma)) <= zone_sigma or b["l"] <= level
                  for b in window)
    if not touched:
        return False, f"price has not reached the level in {touch_lookback}h"

    broke = [b for b in hold if b["c"] < level]
    if broke:
        return False, f"closed below the level {len(broke)}x in the last {hold_hours}h"

    if closed[-1]["c"] <= level:
        return False, "latest close is not above the level"

    # Holding AT the level, not merely above it. "Above" alone is satisfied by
    # a level price has run away from -- UNI touched 7.5078, rallied to 8.64,
    # and that read as two hours of support 16% below the market.
    above = sigma_distance(closed[-1]["c"], level, sigma)
    if above > max_above_sigma:
        return False, (f"price is {above:.1f} sigma above the level "
                       f"(max {max_above_sigma}) -- it has left the zone")

    return True, f"reached the level and has held in the zone for {hold_hours}h"

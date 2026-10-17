"""Demand zones: the band an impulse launched from.

One idea, and the whole module is that idea:

    A demand zone is the base candle of an impulse that CLOSED ABOVE a prior
    swing high. The band runs from that candle's LOW to its CLOSE. It dies
    when price closes back below the low.

Everything else follows. The low is where the zone is invalidated, so it is
also where the stop belongs. The band has a top and a bottom, so an entry can
sit INSIDE it rather than above it.

Why this replaces levels.py's scoring
-------------------------------------
levels.py finds the PRICE a pivot sat at. That is a different object from the
band a move launched from, and the difference is what put real money in the
wrong place: on 2026-10-17 the trader bought ARB at 0.18482 while the operator's
band was 0.1705-0.1750 -- 5.6% above it. Across eight setups the operator
judged by hand, band position predicted every verdict: every skip was an entry
above the band, every take was inside it or 0.44% above. No feature levels.py
computes separated them at all (`level_hold_rate` separated takes from skips
by +0.019).

Fitted, not guessed
-------------------
The definition was chosen by reproducing seven bands the operator drew on his
own charts. Searching pivot window x zone-top definition, "close" beat "high",
"body", "open" and "mid":

    top=close   edge error 0.73%   mean width 2.97%   (operator's: 2.96%)
    top=body    edge error 1.65%   mean width 5.48%
    top=high    edge error 1.72%   mean width 5.77%

The width agreeing to 0.01pp is what says this found the same object rather
than something nearby. Pivot window is insensitive -- 2 and 3 are identical,
4 and 5 marginally worse -- so there is no fragile knob here.

tests/fixtures/zones/ holds those bands and the bars, and
test_zones.py keeps the definition pinned to them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from research.levels.levels import find_pivots

# A pivot needs this many bars either side. 2 and 3 give identical results
# against the operator's bands; the search is flat here on purpose.
ZONE_PIVOT_WINDOW = int(os.getenv("LEVELS_ZONE_PIVOT_WINDOW", "3"))
# How far under the band low the stop sits. The zone is invalidated by a close
# below the low, so the stop only needs to clear the noise of the touch
# itself -- not a volatility multiple.
ZONE_STOP_BUFFER_PCT = float(os.getenv("LEVELS_ZONE_STOP_BUFFER_PCT", "0.25"))
# Bands narrower than this are a single tight candle rather than a base worth
# trading; wider than this and the stop is too far for the position to size.
ZONE_MIN_WIDTH_PCT = float(os.getenv("LEVELS_ZONE_MIN_WIDTH_PCT", "0.3"))
ZONE_MAX_WIDTH_PCT = float(os.getenv("LEVELS_ZONE_MAX_WIDTH_PCT", "8.0"))
# A zone's base can be far back, and the detector only sees the bars it is
# given. Measured on the operator's seven: six bases sat 50-61 bars back, but
# HYPE's sat 740 back -- a band price had not closed under in 31 days, and the
# one he drew. Below the horizon the detector does not fail loudly, it finds a
# THINNER, more recent base instead: at 400 bars HYPE came out 77.44-77.675,
# 0.30% wide, which is a 0.40% stop that noise takes out before the idea is
# tested. So short history returns NOTHING rather than a band that happens to
# fit inside it.
#
# 800 bars of 1h is ~33 days. Raising this is cheap; a zone older than the
# window is simply invisible, and no guard can recover it.
ZONE_MIN_BARS = int(os.getenv("LEVELS_ZONE_MIN_BARS", "800"))


@dataclass(frozen=True)
class Zone:
    """One demand band. `lo` and `hi` are the whole of it."""
    coin: str
    lo: float
    hi: float
    base_i: int          # bar the band is built on
    bos_i: int           # bar whose close broke the prior swing high

    @property
    def mid(self) -> float:
        return (self.lo + self.hi) / 2.0

    @property
    def width_pct(self) -> float:
        return (self.hi - self.lo) / self.lo * 100.0 if self.lo > 0 else 0.0

    def contains(self, px: float) -> bool:
        return self.lo <= px <= self.hi

    def entry_px(self, px_now: Optional[float] = None) -> float:
        """Where to buy.

        Inside the band, middle-to-upper. The operator's rule, and his reason
        for it: entering at the band low risks missing the band entirely when
        price only penetrates part of it. The midpoint is the default, and if
        price is already inside the band above the midpoint there is nothing
        to wait for -- that IS the entry.
        """
        if px_now is not None and self.mid <= px_now <= self.hi:
            return px_now
        return self.mid

    def stop_px(self, buffer_pct: float = ZONE_STOP_BUFFER_PCT) -> float:
        """Just under the band low -- the price that invalidates the zone.

        The definition supplies the stop. A close below the low is what kills
        the zone, so there is nothing to tune: no sigma multiple, no next
        support, no structural guess.
        """
        return self.lo * (1.0 - buffer_pct / 100.0)


def demand_zones(bars: List[Dict[str, Any]], coin: str = "",
                 window: int = ZONE_PIVOT_WINDOW,
                 alive_only: bool = True,
                 min_bars: Optional[int] = None) -> List[Zone]:
    """Every demand zone in `bars`, oldest first.

    `alive_only` drops zones price has since closed below. A zone that has
    been broken is not a level that failed -- it is not a zone any more, and
    keeping it is how a dead band becomes a trade.
    """
    floor = ZONE_MIN_BARS if min_bars is None else min_bars
    if len(bars) < max(floor, window * 2 + 2):
        return []
    highs, _ = find_pivots(bars, window)
    out: List[Zone] = []
    for hi_i, hi_px in highs:
        bos = _first_close_above(bars, hi_i + 1, hi_px)
        if bos is None:
            continue
        base = min(range(hi_i, bos + 1), key=lambda k: bars[k]["l"])
        lo, hi = float(bars[base]["l"]), float(bars[base]["c"])
        if hi <= lo:
            continue                       # base closed at its low: no band
        z = Zone(coin=coin.upper(), lo=lo, hi=hi, base_i=base, bos_i=bos)
        if not ZONE_MIN_WIDTH_PCT <= z.width_pct <= ZONE_MAX_WIDTH_PCT:
            continue
        if alive_only and _closed_below(bars, bos, lo):
            continue
        out.append(z)
    out.sort(key=lambda z: z.base_i)
    return _merge_nested(_dedupe(out))


def nearest_below(zones: List[Zone], px: float) -> Optional[Zone]:
    """The highest live zone at or below `px` -- the one price is falling into."""
    below = [z for z in zones if z.lo <= px]
    return max(below, key=lambda z: z.hi) if below else None


def _first_close_above(bars: List[Dict[str, Any]], start: int,
                       level: float) -> Optional[int]:
    """Break of structure: the first bar to CLOSE above `level`.

    The close, not the wick. A wick through a swing high is the failure this
    is meant to exclude -- it is what a sweep looks like, not what a break
    looks like.
    """
    for j in range(start, len(bars)):
        if bars[j]["c"] > level:
            return j
    return None


def _closed_below(bars: List[Dict[str, Any]], frm: int, lo: float) -> bool:
    return any(b["c"] < lo for b in bars[frm:])


def _merge_nested(zones: List[Zone]) -> List[Zone]:
    """Drop a band that sits wholly INSIDE another.

    A tight base can sit inside the wider base of the same move. Both are
    genuine impulse bases, but they are one zone, and taking the inner one
    puts the stop inside the band instead of under it -- the one thing a stop
    must never be.

    Containment, not overlap. Merging anything that merely touches chains
    adjacent bands into one runaway zone: the first cut of this turned HYPE's
    77.03-78.63 into 75.15-78.63, a 4.6% band spanning two separate bases.
    Two bands that only overlap at the edges are two bands.
    """
    out: List[Zone] = []
    for z in sorted(zones, key=lambda z: (z.lo - z.hi)):      # widest first
        if any(o.lo <= z.lo and z.hi <= o.hi for o in out):
            continue
        out.append(z)
    return sorted(out, key=lambda z: z.base_i)


def _dedupe(zones: List[Zone]) -> List[Zone]:
    """Collapse zones built on the same base bar.

    Several swing highs can be broken by one impulse, which yields the same
    band several times over. This is the same duplication that gave ARB
    seventeen "levels" where there was one -- caught at the source here
    rather than patched downstream.
    """
    seen: Dict[int, Zone] = {}
    for z in zones:
        prev = seen.get(z.base_i)
        if prev is None or z.bos_i < prev.bos_i:
            seen[z.base_i] = z
    return [seen[k] for k in sorted(seen)]

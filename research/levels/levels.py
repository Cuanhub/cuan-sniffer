"""
Support/resistance level detection and quality scoring.

Levels are built from swing pivots clustered in units of the asset's own
volatility, never in raw percent. That is not a stylistic choice -- it is
the specific error that made an earlier hand-encoding of this strategy
unusable (2026-09-17, see DECISIONS.md): the same setup measured 124% on
CASHCAT and 9.8% on BTC, which no raw-percent threshold can catch together,
but 9.9 and 13.2 sigma respectively once scaled by each asset's own 4h
return volatility. Every width, distance and band in this module is
therefore expressed in sigma.

Quality is touch count plus hold rate, not touch count alone. The two
levels the user identified by eye scored very differently on holds --
BTC 74,900 held 10 of 10 tests, while a naive 4%-wide band around CASHCAT
0.14 held only 8 of 18 -- and it is the hold rate that distinguishes a
level price respects from a price it happens to cross often.

Detection only. This module places nothing and knows nothing about orders.
"""

import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# All tunables live in config.py, each tagged STRUCTURAL / EVIDENCED / GUESSED.
from research.levels.config import (               # noqa: E402
    PIVOT_WINDOW, CLUSTER_SIGMA, TOUCH_BAND_SIGMA, BREAK_SIGMA, BREAK_WINDOW,
    VOL_LOOKBACK,
)
from research.levels.config import MIN_TOUCHES_DETECT as MIN_TOUCHES  # noqa: E402


def log_sigma(bars: List[Dict[str, Any]], i: int, lookback: int = VOL_LOOKBACK) -> float:
    """Stdev of 4h log returns over the trailing window -- the unit everything
    else in this module is measured in. Returns 0.0 when undefined so callers
    can skip rather than divide by it."""
    w = bars[max(1, i - lookback):i + 1]
    rets = [math.log(w[j]["c"] / w[j - 1]["c"])
            for j in range(1, len(w)) if w[j - 1]["c"] > 0 and w[j]["c"] > 0]
    return statistics.pstdev(rets) if len(rets) > 5 else 0.0


def sigma_distance(a: float, b: float, sigma: float) -> float:
    """Distance from a to b in sigma units (signed: positive when a > b)."""
    if a <= 0 or b <= 0 or sigma <= 0:
        return 0.0
    return math.log(a / b) / sigma


@dataclass
class Level:
    price: float
    kind: str                      # "support" | "resistance"
    touches: int = 0
    holds: int = 0
    first_touch_idx: int = -1
    last_touch_idx: int = -1
    touch_indices: List[int] = field(default_factory=list)
    pivot_count: int = 0

    @property
    def hold_rate(self) -> float:
        return self.holds / self.touches if self.touches else 0.0


def find_pivots(bars: List[Dict[str, Any]], window: int = PIVOT_WINDOW):
    """Swing highs and lows: a bar strictly extreme over +/- window bars.
    Returns (highs, lows) as lists of (index, price)."""
    highs, lows = [], []
    for i in range(window, len(bars) - window):
        seg = bars[i - window:i + window + 1]
        if bars[i]["h"] >= max(x["h"] for x in seg):
            highs.append((i, bars[i]["h"]))
        if bars[i]["l"] <= min(x["l"] for x in seg):
            lows.append((i, bars[i]["l"]))
    return highs, lows


def _score_level(bars: List[Dict[str, Any]], price: float, kind: str,
                 upto: int, sigma: float) -> Level:
    """Count how often price tested this level and how often the level held.

    A support "holds" when, in the BREAK_WINDOW bars following the touch, no
    bar CLOSES more than BREAK_SIGMA below the level; resistance is the
    mirror. Two details matter here and both were wrong in the first cut:

    - the break is judged over the bars AFTER the touch, not within the
      touching bar. Judging it inside the bar required a single candle to dip
      one sigma and close three below, which essentially never happens, so
      every level scored a perfect 100% hold rate and the metric carried no
      information at all.
    - it keys off the close, not the wick. A wick through the level is what
      these setups look like when they work (CASHCAT printed 0.14211 against
      a 0.14 level and reversed), so counting wicks as breaks would discard
      exactly the tests worth counting.
    """
    lvl = Level(price=price, kind=kind)
    band = sigma * TOUCH_BAND_SIGMA
    in_touch = False
    for i in range(min(upto, len(bars))):
        b = bars[i]
        if price <= 0 or b["l"] <= 0:
            continue
        near = (abs(sigma_distance(b["l"], price, sigma)) <= TOUCH_BAND_SIGMA if kind == "support"
                else abs(sigma_distance(b["h"], price, sigma)) <= TOUCH_BAND_SIGMA)
        if near and not in_touch:
            in_touch = True
            lvl.touches += 1
            lvl.touch_indices.append(i)
            if lvl.first_touch_idx < 0:
                lvl.first_touch_idx = i
            lvl.last_touch_idx = i
            fwd = bars[i + 1:min(i + 1 + BREAK_WINDOW, upto)]
            if kind == "support":
                broke = any(sigma_distance(price, x["c"], sigma) > BREAK_SIGMA for x in fwd)
            else:
                broke = any(sigma_distance(x["c"], price, sigma) > BREAK_SIGMA for x in fwd)
            if not broke:
                lvl.holds += 1
        elif not near:
            in_touch = False
    return lvl


def detect_levels(bars: List[Dict[str, Any]], upto: Optional[int] = None,
                  kind: str = "support") -> List[Level]:
    """Cluster swing pivots into levels, scored by touches and hold rate.

    upto lets the caller evaluate the past as of a given bar, so a scan can be
    replayed historically without leaking future bars into the level set.
    """
    upto = len(bars) if upto is None else upto
    if upto < VOL_LOOKBACK // 2:
        return []
    sigma = log_sigma(bars, upto - 1)
    if sigma <= 0:
        return []
    highs, lows = find_pivots(bars[:upto])
    pts = lows if kind == "support" else highs
    if not pts:
        return []

    clusters: List[List[float]] = []
    for _, px in sorted(pts, key=lambda p: p[1]):
        if clusters and abs(sigma_distance(px, statistics.mean(clusters[-1]), sigma)) <= CLUSTER_SIGMA:
            clusters[-1].append(px)
        else:
            clusters.append([px])

    out = []
    for cl in clusters:
        price = statistics.median(cl)
        lvl = _score_level(bars, price, kind, upto, sigma)
        lvl.pivot_count = len(cl)
        if lvl.touches >= MIN_TOUCHES:
            out.append(lvl)
    out.sort(key=lambda L: (-L.hold_rate, -L.touches))
    return out


def flipped_supports(bars: List[Dict[str, Any]], upto: Optional[int] = None) -> List[Level]:
    """Resistance levels price has broken above, re-scored as supports.

    Classic S/R flip, and the detector had no concept of it: supports were
    only ever built from swing lows, so a level that spent months capping
    price and was then broken became invisible the moment it mattered most.
    On XMR 1h that hid 19 levels, including one with 35 tests sitting 1.4%
    under price -- exactly the kind of level the user reads as support after
    a break.

    The hold rate is recomputed in the SUPPORT sense rather than carried over.
    A resistance's hold rate measures how often it rejected price from below,
    which says nothing about whether it holds as a floor from above.
    """
    upto = len(bars) if upto is None else upto
    if upto < VOL_LOOKBACK // 2:
        return []
    sigma = log_sigma(bars, upto - 1)
    if sigma <= 0:
        return []
    price = bars[upto - 1]["c"]
    out = []
    for R in detect_levels(bars, upto=upto, kind="resistance"):
        if R.price >= price:
            continue
        lvl = _score_level(bars, R.price, "support", upto, sigma)
        lvl.pivot_count = R.pivot_count
        if lvl.touches >= MIN_TOUCHES:
            out.append(lvl)
    return out


def nearest_level(levels: List[Level], price: float, sigma: float,
                  below: bool = True) -> Optional[Level]:
    """Closest level under (or over) the current price, in sigma terms."""
    cands = [L for L in levels if (L.price <= price if below else L.price >= price)]
    if not cands:
        return None
    return min(cands, key=lambda L: abs(sigma_distance(price, L.price, sigma)))

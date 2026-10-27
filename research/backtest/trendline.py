"""Descending trendlines and their breaks, computed causally.

The most common way a trendline backtest lies is lookahead through the swing
points. A swing high at bar j is only a swing high once the k bars after it
have printed lower, so it is KNOWABLE at bar j + k, not at bar j. Drawing a
line through it any earlier uses the future, and a "breakout" detected that
way is partly a description of a reversal that had already happened.
Everything here is written so that the value at bar t depends on bars 0..t
only, and test_backtest_trendline proves it on real bars by recomputing on a
truncated series and requiring the identical answer.

The line (pre-registered in PREREG_trendline_breakout.md)
---------------------------------------------------------
At the close of bar t, using only swing highs confirmed by t (j + k <= t)
that sit within the last `lookback` bars:

  A  the highest such swing high -- where the decline starts;
  B  of the swing highs after A, the one giving the LEAST negative slope
     from A. That makes A-B the upper envelope: every confirmed swing high
     since A lies on or under the line, which is what a trader means by a
     trendline that "holds". It is the first segment of the upper convex
     hull, which makes it deterministic rather than a judgement call.

  Rejected when: the slope is not negative; B is fewer than `min_span` bars
  after A; any CLOSE strictly between A and B is above the line (it did not
  actually hold); or fewer than `min_touches` swing highs sit within
  `touch_tol_atr` ATRs under it (A and B always count).

The break
---------
Bar t breaks the line when its close is above the line by more than
`buffer_atr` ATRs AND no close since B has done so -- the FIRST close above
the line after its second anchor. Re-crossing a line that already broke is
not a new breakout.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from research.backtest.engine import atr_pct_series


@dataclass(frozen=True)
class TLParams:
    k: int = 3                  # = zones.ZONE_PIVOT_WINDOW: not a new knob
    lookback: int = 120
    min_span: int = 6
    min_touches: int = 2
    touch_tol_atr: float = 0.25
    buffer_atr: float = 0.0


@dataclass(frozen=True)
class Line:
    a_i: int
    a_px: float
    b_i: int
    b_px: float
    touches: int

    @property
    def slope(self) -> float:
        return (self.b_px - self.a_px) / (self.b_i - self.a_i)

    def at(self, t: int) -> float:
        return self.a_px + self.slope * (t - self.a_i)


def swing_highs(bars: List[Dict[str, Any]], k: int) -> List[int]:
    """Indices j with h[j] strictly above the k bars before it and at least
    the k bars after it. Strict on the left so a flat top yields one swing
    high, not several. Each is usable only from bar j + k onward."""
    out = []
    for j in range(k, len(bars) - k):
        h = bars[j]["h"]
        if h > max(b["h"] for b in bars[j - k:j]) and \
                h >= max(b["h"] for b in bars[j + 1:j + k + 1]):
            out.append(j)
    return out


def line_at(bars: List[Dict[str, Any]], highs: List[int], t: int,
            p: TLParams, atr_pct: List[float]) -> Optional[Line]:
    """The descending line as it could be drawn at the close of bar t."""
    lo = bisect_left(highs, t - p.lookback)
    hi = bisect_right(highs, t - p.k)          # confirmed: j + k <= t
    usable = highs[lo:hi]
    if len(usable) < 2:
        return None
    a = max(usable, key=lambda j: (bars[j]["h"], j))
    after = [j for j in usable if j > a]
    if not after:
        return None
    ha = bars[a]["h"]
    b = max(after, key=lambda j: ((bars[j]["h"] - ha) / (j - a), j))
    if b - a < p.min_span:
        return None
    hb = bars[b]["h"]
    if not hb < ha:
        return None
    line = Line(a, ha, b, hb, 0)
    for s in range(a + 1, b):
        if bars[s]["c"] > line.at(s):
            return None
    tol = p.touch_tol_atr * atr_pct[b] / 100.0 * bars[b]["c"]
    touches = 1 + sum(1 for j in after if line.at(j) - bars[j]["h"] <= tol)
    if touches < p.min_touches:
        return None
    return Line(a, ha, b, hb, touches)


def breaks(bars: List[Dict[str, Any]], p: TLParams = TLParams(),
           atr_pct: Optional[List[float]] = None
           ) -> Tuple[List[bool], List[Optional[Line]]]:
    """(is_break[t], line[t]) for every bar. Causal: see the module docstring.

    The line is redrawn every bar, but the checks that scan history run only
    when the line itself changes, which keeps a 5000-bar series to about a
    second.
    """
    n = len(bars)
    atr = atr_pct if atr_pct is not None else atr_pct_series(bars)
    highs = swing_highs(bars, p.k)
    is_break = [False] * n
    lines: List[Optional[Line]] = [None] * n
    key = None
    already = False
    for t in range(n):
        ln = line_at(bars, highs, t, p, atr)
        lines[t] = ln
        if ln is None:
            key = None
            continue
        buf = p.buffer_atr * atr[t] / 100.0 * bars[t]["c"]
        if (ln.a_i, ln.b_i) != key:
            key = (ln.a_i, ln.b_i)
            already = any(bars[s]["c"] > ln.at(s) + p.buffer_atr * atr[s]
                          / 100.0 * bars[s]["c"] for s in range(ln.b_i + 1, t))
        above = bars[t]["c"] > ln.at(t) + buf
        if above and not already:
            is_break[t] = True
        if above:
            already = True
    return is_break, lines


def last_break_before(is_break: List[bool], t: int, window: int) -> Optional[int]:
    """Bars since the most recent break in [t - window + 1, t], or None."""
    for s in range(t, max(-1, t - window), -1):
        if is_break[s]:
            return t - s
    return None

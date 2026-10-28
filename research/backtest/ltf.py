"""Lower-timeframe (15m) structure, computed causally, and the 1h -> 15m map.

Pre-registered in PREREG_ltf_15m.md. Every value at 15m bar t depends on
bars 0..t only; test_backtest_ltf proves it on recorded real 15m bars by
recomputing on a series cut at t.
"""

from __future__ import annotations

from bisect import bisect_right
from typing import Any, Dict, List, Optional

from research.backtest.engine import LIMIT, MARKET_CLOSE, Trade, atr_pct_series
from research.backtest.trendline import swing_highs

QUARTER_MS = 900_000
HOUR_MS = 3_600_000
K = 3


def swing_lows(bars: List[Dict[str, Any]], k: int = K) -> List[int]:
    """Mirror of trendline.swing_highs: strictly below the k bars before it,
    at most the k bars after. Usable from bar j + k."""
    out = []
    for j in range(k, len(bars) - k):
        low = bars[j]["l"]
        if low < min(b["l"] for b in bars[j - k:j]) and \
                low <= min(b["l"] for b in bars[j + 1:j + k + 1]):
            out.append(j)
    return out


def bos_breaks(bars: List[Dict[str, Any]], k: int = K) -> List[bool]:
    """is_break[t]: bar t's close is the FIRST close above the most recent
    swing high confirmed by t (index + k <= t).

    A just-confirmed swing high cannot already be broken: the k bars after it
    have highs at or under it, so their closes are too. The level then holds
    until a close goes through it or a newer swing high is confirmed.
    """
    highs = swing_highs(bars, k)
    out = [False] * len(bars)
    ptr = -1                # index into highs of the level in force
    broken = False
    for t in range(len(bars)):
        while ptr + 1 < len(highs) and highs[ptr + 1] + k <= t:
            ptr += 1
            broken = False
        if ptr < 0:
            continue
        if not broken and bars[t]["c"] > bars[highs[ptr]]["h"]:
            out[t] = True
            broken = True
    return out


def last_swing_low(bars: List[Dict[str, Any]], lows: List[int], t: int,
                   k: int = K) -> Optional[float]:
    """The low of the most recent swing low confirmed by bar t."""
    i = bisect_right(lows, t - k) - 1
    return bars[lows[i]]["l"] if i >= 0 else None


def ltf_stop(bars15: List[Dict[str, Any]], lows: List[int], atr15: List[float],
             t: int, band_stop: float) -> float:
    """B2's stop: the last confirmed 15m swing low minus max(0.25%, 1.5 x
    ATR(14) on 15m), or the band stop if that is tighter. Never wider than the
    band stop."""
    sl = last_swing_low(bars15, lows, t)
    if sl is None:
        return band_stop
    buf = max(0.25, 1.5 * atr15[t]) / 100.0
    return max(band_stop, sl * (1 - buf))


def arm_index(trade: Trade, bars_1h: List[Dict[str, Any]],
              t15: List[int], bars15: List[Dict[str, Any]]) -> Optional[int]:
    """Where a 1h setup reached its entry, on the 15m series, or None if
    that moment is outside the 15m data.

    - bid filled in 1h bar j: the first 15m bar of hour j whose low reaches
      the bid -- the fill, seen at 15m;
    - in-band market entry at the close of 1h bar i: the first 15m bar after
      that close.
    """
    if trade.kind == LIMIT:
        h0 = bars_1h[trade.entry_i]["t"]
        i = bisect_right(t15, h0 - 1)
        for q in range(i, min(i + 4, len(bars15))):
            if bars15[q]["t"] >= h0 + HOUR_MS:
                break
            if bars15[q]["l"] <= trade.entry_px:
                return q
        return None
    if trade.kind == MARKET_CLOSE:
        start = bars_1h[trade.entry_i]["t"] + HOUR_MS
        i = bisect_right(t15, start - 1)
        if i < len(bars15) and bars15[i]["t"] == start:
            return i
        return None
    return None


def atr15(bars15: List[Dict[str, Any]]) -> List[float]:
    return atr_pct_series(bars15)

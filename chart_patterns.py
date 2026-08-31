"""
Closed-candle chart pattern features.

The detector is intentionally conservative. It marks a pattern as trade-worthy
only when the geometric shape has already broken its confirmation level and the
latest closed candle has both volume and candlestick confirmation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


PATTERN_BOOL_COLUMNS = [
    "pattern_head_shoulders",
    "pattern_inverse_head_shoulders",
    "pattern_double_top",
    "pattern_double_bottom",
    "pattern_ascending_triangle",
    "pattern_descending_triangle",
    "pattern_bull_flag",
    "pattern_bear_flag",
    "pattern_bull_pennant",
    "pattern_bear_pennant",
    "chart_pattern_confirmed",
    "chart_pattern_volume_confirmed",
    "chart_pattern_candle_confirmed",
]

PATTERN_TEXT_COLUMNS = {
    "chart_pattern": "",
    "chart_pattern_family": "",
    "chart_pattern_side": "",
}

PATTERN_NUMERIC_COLUMNS = {
    "chart_pattern_score": 0.0,
}


@dataclass(frozen=True)
class PatternCandidate:
    name: str
    family: str
    side: str
    score: float
    volume_confirmed: bool
    candle_confirmed: bool

    @property
    def confirmed(self) -> bool:
        return self.volume_confirmed and self.candle_confirmed and self.score > 0


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _as_bool(value: Any) -> bool:
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass
    return bool(value)


def _is_monotonic(prices: List[float], direction: str) -> bool:
    """True only if every consecutive pair moves in `direction` ("up" or
    "down") -- a real converging trendline, not just endpoints that
    happen to land the right way with a reversal in between."""
    if len(prices) < 2:
        return False
    pairs = list(zip(prices, prices[1:]))
    if direction == "up":
        return all(b > a for a, b in pairs)
    return all(b < a for a, b in pairs)


def _linear_slope(values: pd.Series) -> float:
    if len(values) < 2:
        return 0.0
    y = values.astype(float).to_numpy()
    x = np.arange(len(y), dtype=float)
    try:
        return float(np.polyfit(x, y, 1)[0])
    except Exception:
        return 0.0


def _level_tolerance(row: pd.Series) -> float:
    price = max(_as_float(row.get("close")), 1e-9)
    atr = max(_as_float(row.get("atr_14")), 0.0)
    return max(0.006, min(0.025, 1.25 * atr / price))


def _break_buffer(row: pd.Series) -> float:
    price = max(_as_float(row.get("close")), 0.0)
    atr = max(_as_float(row.get("atr_14")), 0.0)
    return max(0.04 * atr, 0.0006 * price)


def _near(a: float, b: float, tolerance: float) -> bool:
    ref = max((abs(a) + abs(b)) / 2.0, 1e-9)
    return abs(a - b) / ref <= tolerance


def _volume_confirmed(hist: pd.DataFrame) -> bool:
    if len(hist) < 25 or "volume" not in hist:
        return False
    row = hist.iloc[-1]
    volume = _as_float(row.get("volume"))
    vol_mean = _as_float(hist["volume"].iloc[-21:-1].mean())
    return bool(_as_bool(row.get("vol_spike", False)) or (vol_mean > 0 and volume >= 1.15 * vol_mean))


def _bullish_candle(hist: pd.DataFrame) -> bool:
    if len(hist) < 2:
        return False
    row = hist.iloc[-1]
    prev = hist.iloc[-2]
    open_ = _as_float(row.get("open"))
    close = _as_float(row.get("close"))
    high = _as_float(row.get("high"))
    low = _as_float(row.get("low"))
    body_pct = _as_float(row.get("body_pct"))
    candle_range = max(high - low, 1e-9)
    lower_wick = min(open_, close) - low
    strong_body = close > open_ and body_pct >= 0.45
    bullish_engulfing = (
        close > open_
        and _as_float(prev.get("close")) < _as_float(prev.get("open"))
        and close >= _as_float(prev.get("open"))
        and open_ <= _as_float(prev.get("close"))
    )
    hammer_reclaim = close > open_ and lower_wick >= 0.45 * candle_range and body_pct >= 0.20
    return bool(strong_body or bullish_engulfing or hammer_reclaim)


def _bearish_candle(hist: pd.DataFrame) -> bool:
    if len(hist) < 2:
        return False
    row = hist.iloc[-1]
    prev = hist.iloc[-2]
    open_ = _as_float(row.get("open"))
    close = _as_float(row.get("close"))
    high = _as_float(row.get("high"))
    low = _as_float(row.get("low"))
    body_pct = _as_float(row.get("body_pct"))
    candle_range = max(high - low, 1e-9)
    upper_wick = high - max(open_, close)
    strong_body = close < open_ and body_pct >= 0.45
    bearish_engulfing = (
        close < open_
        and _as_float(prev.get("close")) > _as_float(prev.get("open"))
        and close <= _as_float(prev.get("open"))
        and open_ >= _as_float(prev.get("close"))
    )
    shooting_star = close < open_ and upper_wick >= 0.45 * candle_range and body_pct >= 0.20
    return bool(strong_body or bearish_engulfing or shooting_star)


def _prior_trend(hist: pd.DataFrame, direction: str) -> bool:
    if len(hist) < 35:
        return False
    row = hist.iloc[-1]
    if direction == "up" and _as_bool(row.get("trend_up", False)):
        return True
    if direction == "down" and _as_bool(row.get("trend_down", False)):
        return True

    probe = hist.iloc[-60:-8] if len(hist) >= 68 else hist.iloc[:-8]
    if len(probe) < 20:
        return False
    first = _as_float(probe["close"].iloc[0])
    last = _as_float(probe["close"].iloc[-1])
    if first <= 0:
        return False
    pct = (last - first) / first
    atr = max(_as_float(row.get("atr_14")), 0.0)
    price = max(_as_float(row.get("close")), 1e-9)
    threshold = max(0.018, 1.5 * atr / price)
    return pct >= threshold if direction == "up" else pct <= -threshold


def _fallback_pivots(
    hist: pd.DataFrame, kind: str, lookback: int, atr_amp_mult: float = 0.5
) -> List[Tuple[int, float]]:
    """
    Local-extreme pivots, used only when smc_structure.py's flag-based
    detect_swings() found fewer than 2 real pivots. Must apply the same
    ATR-based amplitude filter detect_swings() does (atr_amp_mult=0.5,
    matching its default) -- otherwise this fallback systematically
    re-admits noise exactly when the stricter primary method correctly
    judged there wasn't a real pivot (most likely in low-volatility
    regimes), which is the opposite of what a fallback should do.
    """
    pivots: List[Tuple[int, float]] = []
    start = max(3, len(hist) - lookback)
    end = len(hist) - 3
    column = "high" if kind == "high" else "low"
    has_atr = "atr_14" in hist.columns
    for i in range(start, end):
        window = hist[column].iloc[i - 3:i + 4].astype(float)
        value = _as_float(hist[column].iloc[i])

        if has_atr and not pd.isna(hist["atr_14"].iloc[i]):
            atr_val = float(hist["atr_14"].iloc[i])
            amplitude = float(
                hist["high"].iloc[i - 3:i + 4].astype(float).max()
                - hist["low"].iloc[i - 3:i + 4].astype(float).min()
            )
            if amplitude < atr_amp_mult * atr_val:
                continue

        if kind == "high" and value >= float(window.max()):
            pivots.append((i, value))
        elif kind == "low" and value <= float(window.min()):
            pivots.append((i, value))
    return pivots


def _pivots(hist: pd.DataFrame, kind: str, lookback: int = 90) -> List[Tuple[int, float]]:
    column = "high" if kind == "high" else "low"
    flag_col = "swing_high" if kind == "high" else "swing_low"
    start = max(0, len(hist) - lookback)
    pivots: List[Tuple[int, float]] = []
    if flag_col in hist.columns:
        for i in range(start, max(start, len(hist) - 1)):
            if _as_bool(hist[flag_col].iloc[i]):
                pivots.append((i, _as_float(hist[column].iloc[i])))
    if len(pivots) < 2:
        pivots = _fallback_pivots(hist, kind, lookback)
    return pivots


def _candidate(
    name: str,
    family: str,
    side: str,
    base_score: float,
    volume_ok: bool,
    candle_ok: bool,
) -> PatternCandidate:
    score = float(base_score) if volume_ok and candle_ok else 0.0
    return PatternCandidate(
        name=name,
        family=family,
        side=side,
        score=round(min(score, 0.12), 4),
        volume_confirmed=volume_ok,
        candle_confirmed=candle_ok,
    )


def _detect_double_top_bottom(hist: pd.DataFrame, row: pd.Series) -> List[PatternCandidate]:
    candidates: List[PatternCandidate] = []
    tol = _level_tolerance(row)
    buffer = _break_buffer(row)
    close = _as_float(row.get("close"))
    volume_ok = _volume_confirmed(hist)

    highs = _pivots(hist, "high", lookback=80)
    if len(highs) >= 2 and _prior_trend(hist, "up"):
        (h1_i, h1), (h2_i, h2) = highs[-2], highs[-1]
        if 5 <= h2_i - h1_i <= 45 and _near(h1, h2, tol):
            valley = float(hist["low"].iloc[h1_i:h2_i + 1].min())
            if close < valley - buffer:
                candidates.append(
                    _candidate("double_top", "reversal", "SHORT", 0.075, volume_ok, _bearish_candle(hist))
                )

    lows = _pivots(hist, "low", lookback=80)
    if len(lows) >= 2 and _prior_trend(hist, "down"):
        (l1_i, l1), (l2_i, l2) = lows[-2], lows[-1]
        if 5 <= l2_i - l1_i <= 45 and _near(l1, l2, tol):
            neckline = float(hist["high"].iloc[l1_i:l2_i + 1].max())
            if close > neckline + buffer:
                candidates.append(
                    _candidate("double_bottom", "reversal", "LONG", 0.075, volume_ok, _bullish_candle(hist))
                )

    return candidates


def _detect_head_shoulders(hist: pd.DataFrame, row: pd.Series) -> List[PatternCandidate]:
    candidates: List[PatternCandidate] = []
    tol = max(_level_tolerance(row) * 1.6, 0.012)
    buffer = _break_buffer(row)
    close = _as_float(row.get("close"))
    atr = max(_as_float(row.get("atr_14")), 0.0)
    price = max(_as_float(row.get("close")), 1e-9)
    prominence = max(0.35 * atr, 0.004 * price)
    volume_ok = _volume_confirmed(hist)

    highs = _pivots(hist, "high", lookback=105)
    if len(highs) >= 3 and _prior_trend(hist, "up"):
        (ls_i, ls), (head_i, head), (rs_i, rs) = highs[-3], highs[-2], highs[-1]
        if (
            4 <= head_i - ls_i <= 45
            and 4 <= rs_i - head_i <= 45
            and head > max(ls, rs) + prominence
            and _near(ls, rs, tol)
        ):
            left_neck = float(hist["low"].iloc[ls_i:head_i + 1].min())
            right_neck = float(hist["low"].iloc[head_i:rs_i + 1].min())
            neckline = (left_neck + right_neck) / 2.0
            if close < neckline - buffer:
                candidates.append(
                    _candidate("head_shoulders", "reversal", "SHORT", 0.09, volume_ok, _bearish_candle(hist))
                )

    lows = _pivots(hist, "low", lookback=105)
    if len(lows) >= 3 and _prior_trend(hist, "down"):
        (ls_i, ls), (head_i, head), (rs_i, rs) = lows[-3], lows[-2], lows[-1]
        if (
            4 <= head_i - ls_i <= 45
            and 4 <= rs_i - head_i <= 45
            and head < min(ls, rs) - prominence
            and _near(ls, rs, tol)
        ):
            left_neck = float(hist["high"].iloc[ls_i:head_i + 1].max())
            right_neck = float(hist["high"].iloc[head_i:rs_i + 1].max())
            neckline = (left_neck + right_neck) / 2.0
            if close > neckline + buffer:
                candidates.append(
                    _candidate("inverse_head_shoulders", "reversal", "LONG", 0.09, volume_ok, _bullish_candle(hist))
                )

    return candidates


def _detect_triangles(hist: pd.DataFrame, row: pd.Series) -> List[PatternCandidate]:
    candidates: List[PatternCandidate] = []
    highs = _pivots(hist, "high", lookback=60)
    lows = _pivots(hist, "low", lookback=60)
    if len(highs) < 2 or len(lows) < 2:
        return candidates

    tol = _level_tolerance(row)
    buffer = _break_buffer(row)
    close = _as_float(row.get("close"))
    volume_ok = _volume_confirmed(hist)

    recent_highs = highs[-3:]
    recent_lows = lows[-3:]
    high_prices = [p for _, p in recent_highs]
    low_prices = [p for _, p in recent_lows]

    flat_highs = (max(high_prices) - min(high_prices)) / max(np.mean(high_prices), 1e-9) <= tol
    flat_lows = (max(low_prices) - min(low_prices)) / max(np.mean(low_prices), 1e-9) <= tol
    # Convergence requires a genuinely monotonic trendline across every
    # sampled pivot, not just first-vs-last -- comparing only the
    # endpoints let a dip-then-recover (or spike-then-fade) sequence pass
    # as "ascending"/"descending" despite never forming a real converging
    # line (e.g. lows of 100 -> 80 -> 105 previously passed "ascending"
    # since 105 > 100 + threshold, even though the middle point breaks it).
    min_move = 0.15 * max(_as_float(row.get("atr_14")), 0.0)
    lows_ascending = (
        _is_monotonic(low_prices, "up")
        and low_prices[-1] > low_prices[0] + min_move
    )
    highs_descending = (
        _is_monotonic(high_prices, "down")
        and high_prices[-1] < high_prices[0] - min_move
    )

    resistance = max(high_prices)
    support = min(low_prices)
    if flat_highs and lows_ascending and close > resistance + buffer and _prior_trend(hist, "up"):
        candidates.append(
            _candidate("ascending_triangle", "continuation", "LONG", 0.075, volume_ok, _bullish_candle(hist))
        )
    if flat_lows and highs_descending and close < support - buffer and _prior_trend(hist, "down"):
        candidates.append(
            _candidate("descending_triangle", "continuation", "SHORT", 0.075, volume_ok, _bearish_candle(hist))
        )
    return candidates


def _detect_flags_pennants(hist: pd.DataFrame, row: pd.Series) -> List[PatternCandidate]:
    candidates: List[PatternCandidate] = []
    if len(hist) < 45:
        return candidates

    atr = max(_as_float(row.get("atr_14")), 0.0)
    price = max(_as_float(row.get("close")), 1e-9)
    impulse = hist.iloc[-38:-14]
    coil = hist.iloc[-14:-1]
    if len(impulse) < 15 or len(coil) < 8:
        return candidates

    impulse_start = _as_float(impulse["close"].iloc[0])
    impulse_end = _as_float(impulse["close"].iloc[-1])
    if impulse_start <= 0:
        return candidates
    impulse_pct = (impulse_end - impulse_start) / impulse_start
    impulse_threshold = max(0.025, 2.2 * atr / price)
    if abs(impulse_pct) < impulse_threshold:
        return candidates

    buffer = _break_buffer(row)
    close = _as_float(row.get("close"))
    volume_ok = _volume_confirmed(hist)
    coil_high = float(coil["high"].max())
    coil_low = float(coil["low"].min())
    high_slope = _linear_slope(coil["high"])
    low_slope = _linear_slope(coil["low"])
    close_slope = _linear_slope(coil["close"])
    prior_vol = _as_float(impulse["volume"].mean())
    coil_vol = _as_float(coil["volume"].mean())
    vol_contracting = prior_vol > 0 and coil_vol <= 0.90 * prior_vol

    first_range = float((coil["high"].iloc[:5].max() - coil["low"].iloc[:5].min()))
    last_range = float((coil["high"].iloc[-5:].max() - coil["low"].iloc[-5:].min()))
    range_contracting = first_range > 0 and last_range <= 0.72 * first_range

    if impulse_pct > 0:
        if vol_contracting and close_slope < 0 and close > coil_high + buffer:
            candidates.append(
                _candidate("bull_flag", "continuation", "LONG", 0.07, volume_ok, _bullish_candle(hist))
            )
        if range_contracting and high_slope < 0 and low_slope > 0 and close > coil_high + buffer:
            candidates.append(
                _candidate("bull_pennant", "continuation", "LONG", 0.08, volume_ok, _bullish_candle(hist))
            )
    else:
        if vol_contracting and close_slope > 0 and close < coil_low - buffer:
            candidates.append(
                _candidate("bear_flag", "continuation", "SHORT", 0.07, volume_ok, _bearish_candle(hist))
            )
        if range_contracting and high_slope < 0 and low_slope > 0 and close < coil_low - buffer:
            candidates.append(
                _candidate("bear_pennant", "continuation", "SHORT", 0.08, volume_ok, _bearish_candle(hist))
            )

    return candidates


def _detect_at(hist: pd.DataFrame) -> Optional[PatternCandidate]:
    if len(hist) < 35:
        return None
    row = hist.iloc[-1]
    candidates: List[PatternCandidate] = []
    candidates.extend(_detect_head_shoulders(hist, row))
    candidates.extend(_detect_double_top_bottom(hist, row))
    candidates.extend(_detect_triangles(hist, row))
    candidates.extend(_detect_flags_pennants(hist, row))
    confirmed = [c for c in candidates if c.confirmed]
    if not confirmed:
        return None
    return max(confirmed, key=lambda c: c.score)


def _empty_pattern_row() -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    row.update({col: False for col in PATTERN_BOOL_COLUMNS})
    row.update(PATTERN_TEXT_COLUMNS)
    row.update(PATTERN_NUMERIC_COLUMNS)
    return row


def add_chart_pattern_features(
    df: pd.DataFrame,
    *,
    min_history: int = 60,
    lookback_limit: int = 130,
) -> pd.DataFrame:
    """
    Add chart pattern features to a closed-candle feature frame.

    The latest row can then be scored by the signal engine as confluence:
    reversal patterns produce LONG/SHORT reversal confirmation; continuation
    patterns produce LONG/SHORT continuation confirmation.
    """
    out = df.copy()
    for col in PATTERN_BOOL_COLUMNS:
        if col not in out.columns:
            out[col] = False
    for col, default in PATTERN_TEXT_COLUMNS.items():
        if col not in out.columns:
            out[col] = default
    for col, default in PATTERN_NUMERIC_COLUMNS.items():
        if col not in out.columns:
            out[col] = default

    if len(out) < min_history:
        return out

    for end_idx in range(min_history - 1, len(out)):
        start_idx = max(0, end_idx - lookback_limit + 1)
        hist = out.iloc[start_idx:end_idx + 1].copy()
        detected = _detect_at(hist)
        if detected is None:
            continue

        row_updates = _empty_pattern_row()
        row_updates["chart_pattern"] = detected.name
        row_updates["chart_pattern_family"] = detected.family
        row_updates["chart_pattern_side"] = detected.side
        row_updates["chart_pattern_score"] = detected.score
        row_updates["chart_pattern_confirmed"] = True
        row_updates["chart_pattern_volume_confirmed"] = detected.volume_confirmed
        row_updates["chart_pattern_candle_confirmed"] = detected.candle_confirmed
        bool_col = "pattern_" + detected.name
        if bool_col in row_updates:
            row_updates[bool_col] = True

        for col, value in row_updates.items():
            out.at[out.index[end_idx], col] = value

    return out

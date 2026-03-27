# smc_sweeps.py
"""
Liquidity sweep and equal-high/low features.

Upgrades included:
  - multi-bar sweep window
  - rolling equal-high / equal-low cluster detection
  - optional ATR-based minimum breach filter
  - optional minimum reclaim / rejection close filter

Definitions:
  Bullish sweep:
    - price trades below a prior swing low
    - closes back above that level
    - lower wick is meaningfully larger than body
    - breach is large enough to matter

  Bearish sweep:
    - price trades above a prior swing high
    - closes back below that level
    - upper wick is meaningfully larger than body
    - breach is large enough to matter
"""

import pandas as pd
import numpy as np


SWEEP_WINDOW = 15          # prior bars used to define the liquidity level
EQ_CLUSTER_WINDOW = 4      # bars used to detect equal highs/lows clusters


def _ensure_atr(df: pd.DataFrame, atr_col: str = "atr_14", period: int = 14) -> pd.DataFrame:
    """
    Ensure ATR exists. Uses EMA smoothing for consistency with features.py.
    """
    if atr_col in df.columns:
        return df

    df = df.copy()
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df[atr_col] = tr.ewm(span=period, adjust=False).mean()
    return df


def detect_equal_highs_lows(
    df: pd.DataFrame,
    threshold_pct: float = 0.10,
    cluster_window: int = EQ_CLUSTER_WINDOW,
) -> pd.DataFrame:
    """
    Detect equal highs / lows across a small rolling cluster rather than only
    adjacent bars.

    Two levels are considered "equal" if their distance is within:
      threshold_pct% of price

    Marks all bars in the matching cluster as eq_high / eq_low.
    """
    df = df.copy()
    n = len(df)

    eq_high = [False] * n
    eq_low = [False] * n

    highs = df["high"].values
    lows = df["low"].values

    for i in range(cluster_window - 1, n):
        start = i - cluster_window + 1
        end = i + 1

        high_window = highs[start:end]
        low_window = lows[start:end]

        max_h = float(np.max(high_window))
        min_h = float(np.min(high_window))
        max_l = float(np.max(low_window))
        min_l = float(np.min(low_window))

        high_ref = float(np.mean(high_window))
        low_ref = float(np.mean(low_window))

        max_dev_h = abs(high_ref) * threshold_pct / 100.0
        max_dev_l = abs(low_ref) * threshold_pct / 100.0

        if (max_h - min_h) <= max_dev_h:
            for j in range(start, end):
                eq_high[j] = True

        if (max_l - min_l) <= max_dev_l:
            for j in range(start, end):
                eq_low[j] = True

    df["eq_high"] = pd.Series(eq_high, index=df.index, dtype=bool)
    df["eq_low"] = pd.Series(eq_low, index=df.index, dtype=bool)
    return df


def detect_liquidity_sweeps(
    df: pd.DataFrame,
    wick_factor: float = 2.0,
    window: int = SWEEP_WINDOW,
    atr_col: str = "atr_14",
    min_breach_atr_frac: float = 0.05,
    min_reclaim_body_frac: float = 0.10,
) -> pd.DataFrame:
    """
    Detect liquidity sweeps against multi-bar swing levels.

    Bullish sweep requires:
      - current low < prior window swing low
      - current close > swing low
      - lower wick > body * wick_factor
      - breach magnitude >= min_breach_atr_frac * ATR
      - reclaim close above level by at least min_reclaim_body_frac * candle range

    Bearish sweep requires:
      - current high > prior window swing high
      - current close < swing high
      - upper wick > body * wick_factor
      - breach magnitude >= min_breach_atr_frac * ATR
      - rejection close below level by at least min_reclaim_body_frac * candle range
    """
    df = _ensure_atr(df, atr_col=atr_col).copy()
    n = len(df)

    bullish_sweep = [False] * n
    bearish_sweep = [False] * n

    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    opens = df["open"].values
    atrs = df[atr_col].values

    for i in range(window, n):
        prior_window_lows = lows[i - window:i]
        prior_window_highs = highs[i - window:i]

        swing_low = float(np.min(prior_window_lows))
        swing_high = float(np.max(prior_window_highs))

        o = float(opens[i])
        c = float(closes[i])
        h = float(highs[i])
        l = float(lows[i])
        atr_v = float(atrs[i]) if not pd.isna(atrs[i]) else 0.0

        candle_range = h - l
        body = abs(c - o)

        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l

        min_breach = min_breach_atr_frac * atr_v if atr_v > 0 else 0.0
        reclaim_buffer = min_reclaim_body_frac * candle_range if candle_range > 0 else 0.0

        # Bullish sweep
        breach_dn = swing_low - l
        reclaims_level = c > (swing_low + reclaim_buffer)

        if (
            l < swing_low and
            breach_dn >= min_breach and
            reclaims_level and
            lower_wick > body * wick_factor
        ):
            bullish_sweep[i] = True

        # Bearish sweep
        breach_up = h - swing_high
        rejects_level = c < (swing_high - reclaim_buffer)

        if (
            h > swing_high and
            breach_up >= min_breach and
            rejects_level and
            upper_wick > body * wick_factor
        ):
            bearish_sweep[i] = True

    df["sweep_bull"] = pd.Series(bullish_sweep, index=df.index, dtype=bool)
    df["sweep_bear"] = pd.Series(bearish_sweep, index=df.index, dtype=bool)
    return df


def add_sweep_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add equal-high/low and liquidity-sweep features.
    """
    df = detect_equal_highs_lows(df)
    df = detect_liquidity_sweeps(df)
    return df
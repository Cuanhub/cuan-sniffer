# smc_zones.py
"""
SMC zones:
  - Order Blocks (OB)
  - Fair Value Gaps (FVG)

Order Block definition:
  Bullish OB = the last bearish candle before a bullish displacement.
  Bearish OB = the last bullish candle before a bearish displacement.

Fair Value Gap definition:
  Uses a standard 3-candle imbalance:
    - Bullish FVG when low[i] > high[i-2]
    - Bearish FVG when high[i] < low[i-2]

This version includes:
  - ATR EMA fallback if atr_14 is missing
  - minimum displacement / body filters
  - minimum FVG size filter
  - zone aging
  - distance-to-zone features
"""

from typing import Tuple, Optional
import numpy as np
import pandas as pd


def _ensure_atr(df: pd.DataFrame, atr_col: str = "atr_14", period: int = 14) -> pd.DataFrame:
    """
    Ensure ATR column exists.
    Uses EMA-smoothed ATR for consistency with features.py.
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


def add_smc_zones(
    df: pd.DataFrame,
    atr_col: str = "atr_14",
    atr_mult_displacement: float = 1.2,
    min_body_pct: float = 0.5,
    max_zone_age_bars: int = 80,
    ob_lookback: int = 5,
    min_fvg_atr_frac: float = 0.10,
) -> pd.DataFrame:
    """
    Adds order block and fair value gap features.

    OB detection:
      1. Detect a displacement candle:
           range >= atr_mult_displacement * ATR
           body  >= min_body_pct * candle range
      2. For bullish displacement:
           find the last bearish candle in prior ob_lookback bars
      3. For bearish displacement:
           find the last bullish candle in prior ob_lookback bars

    FVG detection:
      Standard 3-candle imbalance:
        bullish: low[i] > high[i-2]
        bearish: high[i] < low[i-2]

      Only keep gaps whose size is at least:
        min_fvg_atr_frac * ATR
    """
    if df.empty:
        return df

    df = df.copy()
    df = _ensure_atr(df, atr_col=atr_col)

    n = len(df)

    bull_ob_low = [np.nan] * n
    bull_ob_high = [np.nan] * n
    bear_ob_low = [np.nan] * n
    bear_ob_high = [np.nan] * n
    in_bull_ob = [0] * n
    in_bear_ob = [0] * n
    ob_bull = [0] * n
    ob_bear = [0] * n

    bull_fvg_low = [np.nan] * n
    bull_fvg_high = [np.nan] * n
    bear_fvg_low = [np.nan] * n
    bear_fvg_high = [np.nan] * n
    in_bull_fvg = [0] * n
    in_bear_fvg = [0] * n
    fvg_bull = [0] * n
    fvg_bear = [0] * n

    dist_to_bull_ob = [np.nan] * n
    dist_to_bear_ob = [np.nan] * n
    dist_to_bull_fvg = [np.nan] * n
    dist_to_bear_fvg = [np.nan] * n

    last_bull_ob: Optional[Tuple[float, float, int]] = None
    last_bear_ob: Optional[Tuple[float, float, int]] = None
    last_bull_fvg: Optional[Tuple[float, float, int]] = None
    last_bear_fvg: Optional[Tuple[float, float, int]] = None

    for i in range(n):
        row = df.iloc[i]
        high = float(row["high"])
        low = float(row["low"])
        open_ = float(row["open"])
        close = float(row["close"])
        atr_v = float(row.get(atr_col, 0.0) or 0.0)

        rng = high - low
        body = abs(close - open_)

        # ── Order Blocks ──────────────────────────────────────────────────────
        if atr_v > 0 and rng > 0:
            is_displacement = (
                rng >= atr_mult_displacement * atr_v and
                body >= min_body_pct * rng
            )

            if is_displacement:
                if close > open_:
                    # Bullish displacement → last bearish candle before move
                    found = None
                    for j in range(i - 1, max(-1, i - ob_lookback - 1), -1):
                        c_j = float(df["close"].iloc[j])
                        o_j = float(df["open"].iloc[j])
                        if c_j < o_j:
                            found = j
                            break

                    if found is not None:
                        ob_l = float(df["low"].iloc[found])
                        ob_h = float(df["high"].iloc[found])
                        last_bull_ob = (ob_l, ob_h, i)

                elif close < open_:
                    # Bearish displacement → last bullish candle before move
                    found = None
                    for j in range(i - 1, max(-1, i - ob_lookback - 1), -1):
                        c_j = float(df["close"].iloc[j])
                        o_j = float(df["open"].iloc[j])
                        if c_j > o_j:
                            found = j
                            break

                    if found is not None:
                        ob_l = float(df["low"].iloc[found])
                        ob_h = float(df["high"].iloc[found])
                        last_bear_ob = (ob_l, ob_h, i)

        # Age out old OB zones
        if last_bull_ob is not None and i - last_bull_ob[2] > max_zone_age_bars:
            last_bull_ob = None
        if last_bear_ob is not None and i - last_bear_ob[2] > max_zone_age_bars:
            last_bear_ob = None

        price = close

        if last_bull_ob is not None:
            ob_l, ob_h, _ = last_bull_ob
            bull_ob_low[i] = ob_l
            bull_ob_high[i] = ob_h
            inside = (low <= ob_h) and (high >= ob_l)
            if inside:
                in_bull_ob[i] = 1
                ob_bull[i] = 1
            dist_to_bull_ob[i] = 0.0 if inside else min(abs(price - ob_l), abs(price - ob_h))

        if last_bear_ob is not None:
            ob_l, ob_h, _ = last_bear_ob
            bear_ob_low[i] = ob_l
            bear_ob_high[i] = ob_h
            inside = (low <= ob_h) and (high >= ob_l)
            if inside:
                in_bear_ob[i] = 1
                ob_bear[i] = 1
            dist_to_bear_ob[i] = 0.0 if inside else min(abs(price - ob_l), abs(price - ob_h))

        # ── Fair Value Gaps ───────────────────────────────────────────────────
        # Standard 3-candle imbalance:
        # bullish FVG if low[i] > high[i-2]
        # bearish FVG if high[i] < low[i-2]
        if i >= 2 and atr_v > 0:
            high_2 = float(df["high"].iloc[i - 2])
            low_2 = float(df["low"].iloc[i - 2])

            bull_gap_size = low - high_2
            bear_gap_size = low_2 - high

            min_gap = min_fvg_atr_frac * atr_v

            if bull_gap_size > min_gap:
                # zone between prior high and current low
                last_bull_fvg = (high_2, low, i)

            if bear_gap_size > min_gap:
                # zone between current high and prior low
                last_bear_fvg = (high, low_2, i)

        # Age out old FVG zones
        if last_bull_fvg is not None and i - last_bull_fvg[2] > max_zone_age_bars:
            last_bull_fvg = None
        if last_bear_fvg is not None and i - last_bear_fvg[2] > max_zone_age_bars:
            last_bear_fvg = None

        if last_bull_fvg is not None:
            f_l, f_h, _ = last_bull_fvg
            bull_fvg_low[i] = f_l
            bull_fvg_high[i] = f_h
            inside = (low <= f_h) and (high >= f_l)
            if inside:
                in_bull_fvg[i] = 1
                fvg_bull[i] = 1
            dist_to_bull_fvg[i] = 0.0 if inside else min(abs(price - f_l), abs(price - f_h))

        if last_bear_fvg is not None:
            f_l, f_h, _ = last_bear_fvg
            bear_fvg_low[i] = f_l
            bear_fvg_high[i] = f_h
            inside = (low <= f_h) and (high >= f_l)
            if inside:
                in_bear_fvg[i] = 1
                fvg_bear[i] = 1
            dist_to_bear_fvg[i] = 0.0 if inside else min(abs(price - f_l), abs(price - f_h))

    df["bull_ob_low"] = bull_ob_low
    df["bull_ob_high"] = bull_ob_high
    df["bear_ob_low"] = bear_ob_low
    df["bear_ob_high"] = bear_ob_high
    df["in_bull_ob"] = pd.Series(in_bull_ob, index=df.index, dtype=int)
    df["in_bear_ob"] = pd.Series(in_bear_ob, index=df.index, dtype=int)
    df["ob_bull"] = pd.Series(ob_bull, index=df.index, dtype=int)
    df["ob_bear"] = pd.Series(ob_bear, index=df.index, dtype=int)

    df["bull_fvg_low"] = bull_fvg_low
    df["bull_fvg_high"] = bull_fvg_high
    df["bear_fvg_low"] = bear_fvg_low
    df["bear_fvg_high"] = bear_fvg_high
    df["in_bull_fvg"] = pd.Series(in_bull_fvg, index=df.index, dtype=int)
    df["in_bear_fvg"] = pd.Series(in_bear_fvg, index=df.index, dtype=int)
    df["fvg_bull"] = pd.Series(fvg_bull, index=df.index, dtype=int)
    df["fvg_bear"] = pd.Series(fvg_bear, index=df.index, dtype=int)

    df["dist_to_bull_ob"] = dist_to_bull_ob
    df["dist_to_bear_ob"] = dist_to_bear_ob
    df["dist_to_bull_fvg"] = dist_to_bull_fvg
    df["dist_to_bear_fvg"] = dist_to_bear_fvg

    return df
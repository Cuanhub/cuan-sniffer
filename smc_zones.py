# smc_zones.py

from typing import Tuple, Optional

import numpy as np
import pandas as pd


def _ensure_atr(df: pd.DataFrame, atr_col: str = "atr_14", period: int = 14) -> pd.DataFrame:
    """
    Ensure we have an ATR column. If it's not present (e.g. calling zones before features),
    we compute a basic ATR here. If it exists, we leave it as is.
    """
    if atr_col in df.columns:
        return df

    high = df["high"]
    low = df["low"]
    close = df["close"]

    prev_close = close.shift(1)
    tr = np.maximum(high - low, np.maximum((high - prev_close).abs(), (low - prev_close).abs()))
    atr = tr.rolling(period, min_periods=period).mean()
    df[atr_col] = atr
    return df


def add_smc_zones(
    df: pd.DataFrame,
    atr_col: str = "atr_14",
    atr_mult_displacement: float = 1.2,
    min_body_pct: float = 0.5,
    max_zone_age_bars: int = 80,
) -> pd.DataFrame:
    """
    Institutional-style SMC zones:

    1) Order blocks (displacement candles):
       - Bullish OB:
         * close > open
         * body >= min_body_pct * (high - low)
         * (high - low) >= atr_mult_displacement * ATR
       - Bearish OB:
         * close < open
         * body >= min_body_pct * (high - low)
         * (high - low) >= atr_mult_displacement * ATR
       - We keep *only the latest* valid OB of each side, up to max_zone_age_bars.

    2) Fair Value Gaps (FVG) – clean 1-bar gaps:
       - Bullish FVG between prior high and current low:
         * low > prev_high  (up-gap)
       - Bearish FVG between prior low and current high:
         * high < prev_low  (down-gap)
       - Again: only the latest gap of each side, up to max_zone_age_bars.

    Outputs (per row):
        bull_ob_low, bull_ob_high, bear_ob_low, bear_ob_high
        in_bull_ob, in_bear_ob, ob_bull, ob_bear
        bull_fvg_low, bull_fvg_high, bear_fvg_low, bear_fvg_high
        in_bull_fvg, in_bear_fvg, fvg_bull, fvg_bear
        dist_to_bull_ob, dist_to_bear_ob, dist_to_bull_fvg, dist_to_bear_fvg

    Signal engine primarily uses:
        ob_bull, ob_bear, fvg_bull, fvg_bear
    The distance metrics are for future refinement/ML.
    """
    if df.empty:
        return df

    df = df.copy()
    df = _ensure_atr(df, atr_col=atr_col)

    # Prepare columns
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

    # Rolling latest zones
    last_bull_ob: Optional[Tuple[float, float, int]] = None  # (low, high, idx)
    last_bear_ob: Optional[Tuple[float, float, int]] = None

    last_bull_fvg: Optional[Tuple[float, float, int]] = None  # (low, high, idx)
    last_bear_fvg: Optional[Tuple[float, float, int]] = None

    for i in range(n):
        row = df.iloc[i]
        high = float(row["high"])
        low = float(row["low"])
        open_ = float(row["open"])
        close = float(row["close"])
        atr = float(row.get(atr_col, 0.0) or 0.0)

        rng = high - low
        body = abs(close - open_)

        # ------------------------------------------------------------------
        # 1) Update Order Blocks (displacement candles)
        # ------------------------------------------------------------------
        if atr > 0 and rng > 0:
            # Displacement condition
            is_displacement = rng >= atr_mult_displacement * atr and body >= min_body_pct * rng

            # Bullish displacement OB → we use the last down candle range or just this bar
            if is_displacement and close > open_:
                # For simplicity: OB range = [low, high] of this candle
                last_bull_ob = (low, high, i)

            # Bearish displacement OB
            if is_displacement and close < open_:
                last_bear_ob = (low, high, i)

        # Age out old OBs
        if last_bull_ob is not None:
            ob_low, ob_high, idx_ob = last_bull_ob
            if i - idx_ob > max_zone_age_bars:
                last_bull_ob = None

        if last_bear_ob is not None:
            ob_low, ob_high, idx_ob = last_bear_ob
            if i - idx_ob > max_zone_age_bars:
                last_bear_ob = None

        # Mark OB values + membership
        price = close

        if last_bull_ob is not None:
            ob_low, ob_high, idx_ob = last_bull_ob
            bull_ob_low[i] = ob_low
            bull_ob_high[i] = ob_high

            inside = (low <= ob_high) and (high >= ob_low)
            if inside:
                in_bull_ob[i] = 1
                ob_bull[i] = 1

            dist_to_bull_ob[i] = 0.0 if inside else min(abs(price - ob_low), abs(price - ob_high))

        if last_bear_ob is not None:
            ob_low, ob_high, idx_ob = last_bear_ob
            bear_ob_low[i] = ob_low
            bear_ob_high[i] = ob_high

            inside = (low <= ob_high) and (high >= ob_low)
            if inside:
                in_bear_ob[i] = 1
                ob_bear[i] = 1

            dist_to_bear_ob[i] = 0.0 if inside else min(abs(price - ob_low), abs(price - ob_high))

        # ------------------------------------------------------------------
        # 2) Fair Value Gaps (1-bar gaps)
        # ------------------------------------------------------------------
        if i > 0:
            prev_row = df.iloc[i - 1]
            prev_high = float(prev_row["high"])
            prev_low = float(prev_row["low"])

            # Bullish gap: current low > previous high
            if low > prev_high:
                last_bull_fvg = (prev_high, low, i)

            # Bearish gap: current high < previous low
            if high < prev_low:
                last_bear_fvg = (high, prev_low, i)

        # Age out old FVGs
        if last_bull_fvg is not None:
            f_low, f_high, idx_f = last_bull_fvg
            if i - idx_f > max_zone_age_bars:
                last_bull_fvg = None

        if last_bear_fvg is not None:
            f_low, f_high, idx_f = last_bear_fvg
            if i - idx_f > max_zone_age_bars:
                last_bear_fvg = None

        # Mark FVG membership
        if last_bull_fvg is not None:
            f_low, f_high, idx_f = last_bull_fvg
            bull_fvg_low[i] = f_low
            bull_fvg_high[i] = f_high

            inside = (low <= f_high) and (high >= f_low)
            if inside:
                in_bull_fvg[i] = 1
                fvg_bull[i] = 1

            dist_to_bull_fvg[i] = 0.0 if inside else min(abs(price - f_low), abs(price - f_high))

        if last_bear_fvg is not None:
            f_low, f_high, idx_f = last_bear_fvg
            bear_fvg_low[i] = f_low
            bear_fvg_high[i] = f_high

            inside = (low <= f_high) and (high >= f_low)
            if inside:
                in_bear_fvg[i] = 1
                fvg_bear[i] = 1

            dist_to_bear_fvg[i] = 0.0 if inside else min(abs(price - f_low), abs(price - f_high))

    # Attach columns
    df["bull_ob_low"] = bull_ob_low
    df["bull_ob_high"] = bull_ob_high
    df["bear_ob_low"] = bear_ob_low
    df["bear_ob_high"] = bear_ob_high

    df["in_bull_ob"] = in_bull_ob
    df["in_bear_ob"] = in_bear_ob
    df["ob_bull"] = ob_bull
    df["ob_bear"] = ob_bear

    df["bull_fvg_low"] = bull_fvg_low
    df["bull_fvg_high"] = bull_fvg_high
    df["bear_fvg_low"] = bear_fvg_low
    df["bear_fvg_high"] = bear_fvg_high

    df["in_bull_fvg"] = in_bull_fvg
    df["in_bear_fvg"] = in_bear_fvg
    df["fvg_bull"] = fvg_bull
    df["fvg_bear"] = fvg_bear

    df["dist_to_bull_ob"] = dist_to_bull_ob
    df["dist_to_bear_ob"] = dist_to_bear_ob
    df["dist_to_bull_fvg"] = dist_to_bull_fvg
    df["dist_to_bear_fvg"] = dist_to_bear_fvg

    return df

import pandas as pd
import numpy as np


def detect_equal_highs_lows(df: pd.DataFrame, threshold_pct: float = 0.1) -> pd.DataFrame:
    """
    Identifies equal highs and equal lows (liquidity pools).

    threshold_pct = % difference allowed between candle highs/lows 
    to consider them "equal".
    """
    df = df.copy()
    highs = df["high"]
    lows = df["low"]

    eq_high = [False] * len(df)
    eq_low = [False] * len(df)

    for i in range(1, len(df)):
        # percent deviation allowed
        max_dev_high = highs.iloc[i] * threshold_pct / 100
        max_dev_low = lows.iloc[i] * threshold_pct / 100

        if abs(highs.iloc[i] - highs.iloc[i - 1]) <= max_dev_high:
            eq_high[i] = True
            eq_high[i - 1] = True

        if abs(lows.iloc[i] - lows.iloc[i - 1]) <= max_dev_low:
            eq_low[i] = True
            eq_low[i - 1] = True

    df["eq_high"] = eq_high
    df["eq_low"] = eq_low

    return df


def detect_liquidity_sweeps(df: pd.DataFrame, wick_factor: float = 2.0) -> pd.DataFrame:
    """
    Detects bullish and bearish liquidity sweeps:
    
    Bullish Sweep (grab liquidity below):
        - Current low < previous swing low
        - Candle closes back above previous low
        - Wick size significantly larger than candle body

    Bearish Sweep (grab liquidity above):
        - Current high > previous swing high
        - Candle closes back below previous high
        - Wick size significantly larger than body
    """

    df = df.copy()
    bullish_sweep = [False] * len(df)
    bearish_sweep = [False] * len(df)

    highs = df["high"]
    lows = df["low"]
    closes = df["close"]
    opens = df["open"]

    for i in range(1, len(df)):
        prev_high = highs.iloc[i - 1]
        prev_low = lows.iloc[i - 1]

        body = abs(closes.iloc[i] - opens.iloc[i])
        wick_down = opens.iloc[i] - lows.iloc[i] if opens.iloc[i] > closes.iloc[i] else closes.iloc[i] - lows.iloc[i]
        wick_up = highs.iloc[i] - closes.iloc[i] if closes.iloc[i] > opens.iloc[i] else highs.iloc[i] - opens.iloc[i]

        # ------------------------
        # Bullish Liquidity Sweep
        # ------------------------
        if lows.iloc[i] < prev_low and closes.iloc[i] > prev_low and wick_down > body * wick_factor:
            bullish_sweep[i] = True

        # ------------------------
        # Bearish Liquidity Sweep
        # ------------------------
        if highs.iloc[i] > prev_high and closes.iloc[i] < prev_high and wick_up > body * wick_factor:
            bearish_sweep[i] = True

    df["sweep_bull"] = bullish_sweep
    df["sweep_bear"] = bearish_sweep

    return df


def add_sweep_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full sweep detection engine:
    - equal highs/lows (liquidity pools)
    - sweep candles
    """

    df = detect_equal_highs_lows(df)
    df = detect_liquidity_sweeps(df)
    return df

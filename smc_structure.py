# smc_structure.py
import pandas as pd
from typing import Literal, Optional


Trend = Literal["bull", "bear", "neutral"]


def detect_swings(
    df: pd.DataFrame,
    lookback: int = 5,
    min_separation: int = 2,
    atr_amp_mult: float = 0.5,
) -> pd.DataFrame:
    """
    Detect swing highs and lows using a symmetric lookback window.

    A swing high at index i requires:
      - high[i] is the maximum in [i-lookback, i+lookback]
      - high[i] is greater than the left and right edge highs
      - window amplitude >= atr_amp_mult * ATR (if ATR exists)
      - not too close to the prior accepted swing high

    A swing low at index i requires:
      - low[i] is the minimum in [i-lookback, i+lookback]
      - low[i] is lower than the left and right edge lows
      - window amplitude >= atr_amp_mult * ATR (if ATR exists)
      - not too close to the prior accepted swing low

    If a bar qualifies as both swing high and swing low, we suppress both.
    That usually indicates a noisy or extreme bar better handled downstream.
    """
    df = df.copy()
    n = len(df)

    swing_high = [False] * n
    swing_low = [False] * n

    has_atr = "atr_14" in df.columns
    last_swing_high_idx: Optional[int] = None
    last_swing_low_idx: Optional[int] = None

    for i in range(lookback, n - lookback):
        window = df.iloc[i - lookback: i + lookback + 1]

        h = float(df["high"].iloc[i])
        l = float(df["low"].iloc[i])

        if has_atr and not pd.isna(df["atr_14"].iloc[i]):
            atr_val = float(df["atr_14"].iloc[i])
            amplitude = float(window["high"].max() - window["low"].min())
            min_amp = atr_amp_mult * atr_val
        else:
            amplitude = 1.0
            min_amp = 0.0

        passes_amp = amplitude >= min_amp

        is_candidate_high = (
            h >= float(window["high"].max()) and
            h > float(window["high"].iloc[0]) and
            h > float(window["high"].iloc[-1]) and
            passes_amp
        )

        is_candidate_low = (
            l <= float(window["low"].min()) and
            l < float(window["low"].iloc[0]) and
            l < float(window["low"].iloc[-1]) and
            passes_amp
        )

        # Suppress ambiguous outside-bar style pivots
        if is_candidate_high and is_candidate_low:
            continue

        if is_candidate_high:
            if last_swing_high_idx is None or (i - last_swing_high_idx) >= min_separation:
                swing_high[i] = True
                last_swing_high_idx = i

        if is_candidate_low:
            if last_swing_low_idx is None or (i - last_swing_low_idx) >= min_separation:
                swing_low[i] = True
                last_swing_low_idx = i

    df["swing_high"] = pd.Series(swing_high, index=df.index, dtype=bool)
    df["swing_low"] = pd.Series(swing_low, index=df.index, dtype=bool)
    return df


def build_structure(
    df: pd.DataFrame,
    lookback: int = 5,
    min_separation: int = 2,
    atr_amp_mult: float = 0.5,
) -> pd.DataFrame:
    """
    Build SMC-style market structure on OHLC data.

    Outputs:
      - swing_high / swing_low
      - structure_label: HH, HL, LH, LL
      - bos_bull / bos_bear
      - choch_bull / choch_bear
      - trend: bull / bear / neutral

    Notes:
      - HH => bullish break of previous accepted swing high
      - LL => bearish break of previous accepted swing low
      - CHoCH flips trend only when the prior trend was opposite
      - HL / LH do not by themselves flip trend
    """
    df = detect_swings(
        df,
        lookback=lookback,
        min_separation=min_separation,
        atr_amp_mult=atr_amp_mult,
    )

    df["structure_label"] = ""
    df["bos_bull"] = False
    df["bos_bear"] = False
    df["choch_bull"] = False
    df["choch_bear"] = False
    df["trend"] = "neutral"

    pivot_indices = df.index[(df["swing_high"] | df["swing_low"])].tolist()

    last_high_idx: Optional[int] = None
    last_low_idx: Optional[int] = None
    last_trend: Trend = "neutral"

    for idx in pivot_indices:
        row = df.loc[idx]
        is_high = bool(row["swing_high"])
        is_low = bool(row["swing_low"])
        price_h = float(row["high"])
        price_l = float(row["low"])

        label = ""
        bos_bull = False
        bos_bear = False
        choch_bull = False
        choch_bear = False
        trend: Trend = last_trend

        if is_high:
            if last_high_idx is None:
                label = "HH"
                if last_trend == "neutral":
                    trend = "bull"
            else:
                prev_high = float(df.loc[last_high_idx, "high"])
                if price_h > prev_high:
                    label = "HH"
                    bos_bull = True
                    if last_trend == "bear":
                        choch_bull = True
                    trend = "bull"
                else:
                    label = "LH"

            last_high_idx = idx

        elif is_low:
            if last_low_idx is None:
                label = "LL"
                if last_trend == "neutral":
                    trend = "bear"
            else:
                prev_low = float(df.loc[last_low_idx, "low"])
                if price_l < prev_low:
                    label = "LL"
                    bos_bear = True
                    if last_trend == "bull":
                        choch_bear = True
                    trend = "bear"
                else:
                    label = "HL"

            last_low_idx = idx

        if trend != last_trend:
            last_trend = trend

        df.at[idx, "structure_label"] = label
        df.at[idx, "bos_bull"] = bos_bull
        df.at[idx, "bos_bear"] = bos_bear
        df.at[idx, "choch_bull"] = choch_bull
        df.at[idx, "choch_bear"] = choch_bear
        df.at[idx, "trend"] = trend

    # Forward-fill trend so non-pivot bars inherit the last known structure bias
    df["trend"] = df["trend"].replace("neutral", pd.NA).ffill().fillna("neutral")

    # Ensure boolean columns stay boolean
    for col in ["bos_bull", "bos_bear", "choch_bull", "choch_bear", "swing_high", "swing_low"]:
        df[col] = df[col].fillna(False).astype(bool)

    return df
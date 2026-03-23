import pandas as pd
from typing import Literal, Optional


Trend = Literal["bull", "bear", "neutral"]


def detect_swings(df: pd.DataFrame, lookback: int = 3) -> pd.DataFrame:
    """
    Detects swing highs and swing lows using a simple window-based method.

    A swing high at index i:
        high[i] is the maximum high in [i - lookback, i + lookback]
    A swing low at index i:
        low[i] is the minimum low in [i - lookback, i + lookback]

    This is a basic yet effective SMC-style swing definition.
    """
    df = df.copy()
    n = len(df)
    swing_high = [False] * n
    swing_low = [False] * n

    for i in range(lookback, n - lookback):
        window = df.iloc[i - lookback : i + lookback + 1]

        h = df["high"].iloc[i]
        l = df["low"].iloc[i]

        if h >= window["high"].max() and h > window["high"].iloc[0] and h > window["high"].iloc[-1]:
            swing_high[i] = True

        if l <= window["low"].min() and l < window["low"].iloc[0] and l < window["low"].iloc[-1]:
            swing_low[i] = True

    df["swing_high"] = swing_high
    df["swing_low"] = swing_low
    return df


def build_structure(
    df: pd.DataFrame,
    lookback: int = 3,
) -> pd.DataFrame:
    """
    Builds basic Smart Money Concepts structure on top of OHLC data:

    - swing_high / swing_low (via detect_swings)
    - structure_label: "HH", "HL", "LH", "LL", or "" for non-pivot bars
    - bos_bull / bos_bear (Break of Structure)
    - choch_bull / choch_bear (Change of Character)
    - trend: "bull", "bear", "neutral"

    Assumes df has columns: time, open, high, low, close, volume
    """

    df = detect_swings(df, lookback=lookback)

    df["structure_label"] = ""
    df["bos_bull"] = False
    df["bos_bear"] = False
    df["choch_bull"] = False
    df["choch_bear"] = False
    df["trend"] = "neutral"

    # Collect pivot indices in chronological order
    pivot_indices = df.index[(df["swing_high"] | df["swing_low"])].tolist()

    last_high_idx: Optional[int] = None
    last_low_idx: Optional[int] = None

    last_trend: Trend = "neutral"

    for idx in pivot_indices:
        row = df.loc[idx]

        is_high = bool(row["swing_high"])
        is_low = bool(row["swing_low"])

        price_high = float(row["high"])
        price_low = float(row["low"])

        # Determine current label and BOS/CHoCH
        label = ""
        bos_bull = False
        bos_bear = False
        choch_bull = False
        choch_bear = False
        trend: Trend = last_trend

        if is_high:
            # Compare against previous swing high
            if last_high_idx is not None:
                prev_high = float(df.loc[last_high_idx, "high"])
                if price_high > prev_high:
                    label = "HH"
                    bos_bull = True  # broke previous high
                    if last_trend == "bear":
                        choch_bull = True  # change of character to bull
                        trend = "bull"
                    elif last_trend == "neutral":
                        trend = "bull"
                else:
                    label = "LH"
            else:
                # first high
                label = "HH"  # arbitrary; will refine over time
                trend = "neutral"

            last_high_idx = idx

        elif is_low:
            # Compare against previous swing low
            if last_low_idx is not None:
                prev_low = float(df.loc[last_low_idx, "low"])
                if price_low < prev_low:
                    label = "LL"
                    bos_bear = True  # broke previous low
                    if last_trend == "bull":
                        choch_bear = True  # change of character to bear
                        trend = "bear"
                    elif last_trend == "neutral":
                        trend = "bear"
                else:
                    label = "HL"
            else:
                # first low
                label = "LL"
                trend = "neutral"

            last_low_idx = idx

        # Update trend if we set it above
        if trend != last_trend:
            last_trend = trend

        # Write results into df
        df.at[idx, "structure_label"] = label
        df.at[idx, "bos_bull"] = bos_bull
        df.at[idx, "bos_bear"] = bos_bear
        df.at[idx, "choch_bull"] = choch_bull
        df.at[idx, "choch_bear"] = choch_bear
        df.at[idx, "trend"] = trend

    # Forward-fill trend into non-pivot bars
    df["trend"] = df["trend"].replace("neutral", pd.NA).ffill().fillna("neutral")

    return df

"""
HTF and macro regime classification — pure functions.

Extracted from signal_engine.py. Candle integrity policy preserved exactly:
  - Layer 1: forming 1H candle (last row) excluded before resampling
  - Layer 2: last resampled bar excluded (incomplete 4H or incomplete day)

These functions take a DataFrame and return (regime_str, notes_list).
They do NOT import signal_engine.py.
"""

from typing import List, Tuple

import pandas as pd


def compute_htf_regime(
    df_ohlcv: pd.DataFrame,
    debug: bool = False,
) -> Tuple[str, List[str]]:
    """
    4H regime from 1H bars — HTF context for signal generation.

    Candle policy:
      1. Drop last 1H row (forming candle)
      2. Drop last 4H bar (incomplete period)
    """
    notes: List[str] = []

    if df_ohlcv is None or df_ohlcv.empty:
        return "unknown", ["htf_no_data"]

    df_raw = df_ohlcv.copy()

    if len(df_raw) > 1:
        df_raw = df_raw.iloc[:-1]

    if "time" in df_raw.columns:
        df_raw["time"] = pd.to_datetime(df_raw["time"])
        df_raw = df_raw.set_index("time")
    elif not isinstance(df_raw.index, pd.DatetimeIndex):
        return "unknown", ["htf_no_time_index"]

    try:
        df_4h = (
            df_raw.resample("4h")
            .agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            })
            .dropna()
        )
    except Exception as e:
        if debug:
            print("[HTF_REGIME_DEBUG] Resample 4h failed: " + str(e))
        return "unknown", ["htf_resample_error"]

    if len(df_4h) > 1:
        df_4h = df_4h.iloc[:-1]

    if len(df_4h) < 10:
        return "unknown", ["htf_insufficient_bars"]

    df_4h["ema_fast"] = df_4h["close"].ewm(span=10, adjust=False).mean()
    df_4h["ema_slow"] = df_4h["close"].ewm(span=30, adjust=False).mean()

    last = df_4h.iloc[-1]
    prev = df_4h.iloc[-3] if len(df_4h) >= 3 else df_4h.iloc[0]

    ema_fast = float(last["ema_fast"])
    ema_slow = float(last["ema_slow"])
    ema_fast_prev = float(prev["ema_fast"])
    ema_slow_prev = float(prev["ema_slow"])

    fast_slope = ema_fast - ema_fast_prev
    slow_slope = ema_slow - ema_slow_prev

    if ema_fast > ema_slow and fast_slope > 0 and slow_slope >= 0:
        regime = "up"
        notes.append("htf_ema_trend_up")
    elif ema_fast < ema_slow and fast_slope < 0 and slow_slope <= 0:
        regime = "down"
        notes.append("htf_ema_trend_down")
    else:
        regime = "chop"
        notes.append("htf_ema_chop")

    if debug:
        print(
            "[HTF_REGIME_DEBUG] 4h regime=" + regime +
            " ema_fast=" + str(round(ema_fast, 2)) +
            " ema_slow=" + str(round(ema_slow, 2))
        )

    return regime, notes


def compute_macro_regime(
    df_ohlcv: pd.DataFrame,
    debug: bool = False,
) -> Tuple[str, List[str]]:
    """
    Daily macro regime from 1H bars.

    Candle policy:
      1. Drop last 1H row (forming candle)
      2. Drop last 1D bar (incomplete day)
    """
    notes: List[str] = []

    if df_ohlcv is None or df_ohlcv.empty:
        return "unknown", ["macro_no_data"]

    df_raw = df_ohlcv.copy()

    if len(df_raw) > 1:
        df_raw = df_raw.iloc[:-1]

    if "time" in df_raw.columns:
        df_raw["time"] = pd.to_datetime(df_raw["time"])
        df_raw = df_raw.set_index("time")
    elif not isinstance(df_raw.index, pd.DatetimeIndex):
        return "unknown", ["macro_no_time_index"]

    try:
        df_1d = (
            df_raw.resample("1D")
            .agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            })
            .dropna()
        )
    except Exception as e:
        if debug:
            print("[MACRO_REGIME_DEBUG] Resample 1D failed: " + str(e))
        return "unknown", ["macro_resample_error"]

    if len(df_1d) > 1:
        df_1d = df_1d.iloc[:-1]

    if len(df_1d) < 8:
        return "unknown", ["macro_insufficient_bars"]

    df_1d["ema_fast"] = df_1d["close"].ewm(span=5, adjust=False).mean()
    df_1d["ema_slow"] = df_1d["close"].ewm(span=10, adjust=False).mean()

    last = df_1d.iloc[-1]
    prev = df_1d.iloc[-2] if len(df_1d) >= 2 else df_1d.iloc[0]

    ema_fast = float(last["ema_fast"])
    ema_slow = float(last["ema_slow"])
    ema_fast_prev = float(prev["ema_fast"])
    ema_slow_prev = float(prev["ema_slow"])

    fast_slope = ema_fast - ema_fast_prev
    slow_slope = ema_slow - ema_slow_prev

    if ema_fast > ema_slow and fast_slope > 0 and slow_slope >= 0:
        regime = "up"
        notes.append("macro_ema_trend_up")
    elif ema_fast < ema_slow and fast_slope < 0 and slow_slope <= 0:
        regime = "down"
        notes.append("macro_ema_trend_down")
    else:
        regime = "chop"
        notes.append("macro_ema_chop")

    if debug:
        print(
            "[MACRO_REGIME_DEBUG] 1D regime=" + regime +
            " ema_fast=" + str(round(ema_fast, 2)) +
            " ema_slow=" + str(round(ema_slow, 2))
        )

    return regime, notes

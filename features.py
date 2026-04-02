# features.py
import pandas as pd
import numpy as np


# ── Basic indicators ───────────────────────────────────────────────────────────

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    RSI with EMA/Wilder-style smoothing.
    More responsive and more standard for live trading than simple rolling mean.
    """
    delta = series.diff()

    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)

    roll_up = up.ewm(span=period, adjust=False).mean()
    roll_down = down.ewm(span=period, adjust=False).mean()

    rs = roll_up / (roll_down + 1e-9)
    return 100.0 - (100.0 / (1.0 + rs))


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    ATR with EMA smoothing.
    Better suited for live systems than simple rolling mean ATR.
    """
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def vwap_session(df: pd.DataFrame) -> pd.Series:
    """
    Session-anchored VWAP — resets at 00:00 UTC each day.

    Works with either:
      - a 'time' column
      - a DatetimeIndex

    Falls back to cumulative VWAP if time cannot be resolved.
    """
    df = df.copy()

    # Resolve timestamps safely into UTC
    if "time" in df.columns:
        times = pd.to_datetime(df["time"], utc=True)
    elif isinstance(df.index, pd.DatetimeIndex):
        if df.index.tz is None:
            idx_utc = df.index.tz_localize("UTC")
        else:
            idx_utc = df.index.tz_convert("UTC")
        times = pd.Series(idx_utc, index=df.index)
    else:
        # Graceful fallback: cumulative VWAP
        pv = (df["close"] * df["volume"]).cumsum()
        vv = df["volume"].cumsum() + 1e-9
        return pv / vv

    # Use vectorized daily grouping, aligned to index
    dates = times.dt.date
    pv = df["close"] * df["volume"]

    cum_pv = pv.groupby(dates).cumsum()
    cum_v = df["volume"].groupby(dates).cumsum() + 1e-9

    return cum_pv / cum_v


# ── Feature builder ────────────────────────────────────────────────────────────

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds TA + context features used by the signal engine.

    Includes:
      - trend EMAs
      - RSI momentum
      - ATR volatility state
      - volume spike / collapse
      - session-anchored VWAP
      - candle body quality (body_pct)
    """
    df = df.copy()

    # Trend EMAs
    df["ema_50"] = ema(df["close"], 50)
    df["ema_200"] = ema(df["close"], 200)
    df["trend_up"] = (df["ema_50"] > df["ema_200"]).astype(int)
    df["trend_down"] = (df["ema_50"] < df["ema_200"]).astype(int)

    # Momentum
    df["rsi_14"] = rsi(df["close"], 14)

    # Volatility
    df["atr_14"] = atr(df, 14)
    atr_mean_20 = df["atr_14"].rolling(20).mean()
    df["vol_expansion"] = (df["atr_14"] > atr_mean_20).astype(int)
    df["vol_compression"] = (df["atr_14"] < atr_mean_20).astype(int)

    # Volume context
    vol_mean_30 = df["volume"].rolling(30).mean()
    df["vol_spike"] = (df["volume"] > 1.5 * vol_mean_30).astype(int)
    df["vol_collapse"] = (df["volume"] < 0.5 * vol_mean_30).astype(int)

    # Session-anchored VWAP
    df["vwap"] = vwap_session(df)
    df["above_vwap"] = (df["close"] > df["vwap"]).astype(int)
    df["below_vwap"] = (df["close"] < df["vwap"]).astype(int)
    df["vwap_dev"] = df["close"] - df["vwap"]

    # ── NEW: Candle body quality ───────────────────────────────────────────
    # body_pct = fraction of candle range that is body (0.0 = doji, 1.0 = marubozu)
    # Used by signal_engine for trigger quality scoring
    candle_range = df["high"] - df["low"]
    df["body_pct"] = (df["close"] - df["open"]).abs() / (candle_range + 1e-9)
    df["body_pct"] = df["body_pct"].clip(0.0, 1.0)

    return df
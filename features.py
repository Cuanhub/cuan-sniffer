import pandas as pd
import numpy as np


# -------------------------------------------------------------
# BASIC INDICATORS
# -------------------------------------------------------------

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = np.where(delta > 0, delta, 0)
    down = np.where(delta < 0, -delta, 0)

    roll_up = pd.Series(up).rolling(period).mean()
    roll_down = pd.Series(down).rolling(period).mean()

    rs = roll_up / (roll_down + 1e-9)
    return 100.0 - (100.0 / (1.0 + rs))


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def vwap(df: pd.DataFrame) -> pd.Series:
    pv = (df["close"] * df["volume"]).cumsum()
    vv = df["volume"].cumsum() + 1e-9
    return pv / vv


# -------------------------------------------------------------
# FEATURE BUILDER
# -------------------------------------------------------------

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds all key TA indicators the Hybrid Strategy needs.
    Returns a new DataFrame with feature columns.
    """
    df = df.copy()

    # Trend
    df["ema_50"] = ema(df["close"], 50)
    df["ema_200"] = ema(df["close"], 200)
    df["trend_up"] = (df["ema_50"] > df["ema_200"]).astype(int)
    df["trend_down"] = (df["ema_50"] < df["ema_200"]).astype(int)

    # Momentum
    df["rsi_14"] = rsi(df["close"], 14)

    # Volatility / Regime
    df["atr_14"] = atr(df, 14)
    df["vol_expansion"] = (df["atr_14"] > df["atr_14"].rolling(20).mean()).astype(int)
    df["vol_compression"] = (df["atr_14"] < df["atr_14"].rolling(20).mean()).astype(int)

    # Volume context
    vol_mean = df["volume"].rolling(50).mean()
    df["vol_spike"] = (df["volume"] > 1.5 * vol_mean).astype(int)
    df["vol_collapse"] = (df["volume"] < 0.5 * vol_mean).astype(int)

    # VWAP (mean reversion reference)
    df["vwap"] = vwap(df)
    df["above_vwap"] = (df["close"] > df["vwap"]).astype(int)
    df["below_vwap"] = (df["close"] < df["vwap"]).astype(int)

    # Structure: deviations & reversion signals
    df["vwap_dev"] = df["close"] - df["vwap"]

    # Return enhanced dataframe
    return df

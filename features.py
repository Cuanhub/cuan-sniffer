# features.py
import os
import pandas as pd
import numpy as np


# ── Basic indicators ───────────────────────────────────────────────────────────

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    RSI with true Wilder smoothing (alpha=1/period), matching the
    original formula and every standard charting tool. Previously used
    ewm(span=period) instead, which is alpha=2/(period+1) -- roughly
    1.9x more reactive than real Wilder's RSI for period=14 (alpha
    0.133 vs 0.071), giving materially different values than anything
    computed the standard way (fixed 2026-08-25).
    """
    delta = series.diff()

    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)

    roll_up = up.ewm(alpha=1.0 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1.0 / period, adjust=False).mean()

    rs = roll_up / (roll_down + 1e-9)
    return 100.0 - (100.0 / (1.0 + rs))


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    ATR with true Wilder smoothing (alpha=1/period) -- same fix as rsi()
    above, for the same reason: ewm(span=period) is not Wilder's method
    despite being commonly mislabeled as such.
    """
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


# Session open hours in UTC — configurable via VWAP_SESSION_OPENS env var.
# Defaults: 00 (Asia), 07 (London), 13 (NY).  Sessions partition each 24H day;
# VWAP resets at the start of each session rather than at midnight only.
_VWAP_SESSION_OPENS: list[int] = sorted(
    int(h.strip())
    for h in os.getenv("VWAP_SESSION_OPENS", "0,7,13").split(",")
    if h.strip().isdigit()
) or [0, 7, 13]


def _resolve_times_utc(df: pd.DataFrame) -> pd.Series | None:
    """Return a Series of tz-aware UTC timestamps aligned to df.index, or None."""
    if "time" in df.columns:
        return pd.to_datetime(df["time"], utc=True)
    if isinstance(df.index, pd.DatetimeIndex):
        idx = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
        return pd.Series(idx, index=df.index)
    return None


def _typical_price(df: pd.DataFrame) -> pd.Series:
    """(H+L+C)/3 — the textbook VWAP price input. All four vwap_* functions
    below previously used close alone, which is not standard VWAP (fixed
    2026-08-25)."""
    return (df["high"] + df["low"] + df["close"]) / 3.0


def _cumulative_vwap(df: pd.DataFrame) -> pd.Series:
    """Full-history VWAP — fallback when timestamps cannot be resolved."""
    pv = (_typical_price(df) * df["volume"]).cumsum()
    return pv / (df["volume"].cumsum() + 1e-9)


def vwap_session(df: pd.DataFrame) -> pd.Series:
    """
    Intraday session-anchored VWAP.

    Resets at the start of each major trading session (Asia/London/NY) defined
    by VWAP_SESSION_OPENS (UTC hours, default 0/7/13). Within each session,
    the VWAP accumulates cleanly from the session open — not from midnight.

    This eliminates the stale-anchor problem: a 13:00 UTC (NY open) signal is
    no longer compared against 13 hours of Asia/London drift.
    """
    df = df.copy()
    times = _resolve_times_utc(df)
    if times is None:
        return _cumulative_vwap(df)

    # Assign each candle to its session anchor.
    # Session key: "YYYY-MM-DD_HH" where HH is the session-open hour.
    hours = times.dt.hour
    dates = times.dt.date.astype(str)

    def _session_open_hour(h: int) -> int:
        # Walk backwards through session opens to find which session we're in.
        open_h = _VWAP_SESSION_OPENS[0]
        for so in _VWAP_SESSION_OPENS:
            if h >= so:
                open_h = so
        return open_h

    session_open_hour = hours.map(_session_open_hour)
    session_key = dates + "_" + session_open_hour.astype(str).str.zfill(2)

    pv = _typical_price(df) * df["volume"]
    cum_pv = pv.groupby(session_key).cumsum()
    cum_v = df["volume"].groupby(session_key).cumsum() + 1e-9
    return cum_pv / cum_v


def vwap_daily(df: pd.DataFrame) -> pd.Series:
    """
    Classic daily VWAP — resets at 00:00 UTC.  Retained as a secondary
    reference anchor; primary intraday signal uses vwap_session().
    """
    df = df.copy()
    times = _resolve_times_utc(df)
    if times is None:
        return _cumulative_vwap(df)

    dates = times.dt.date
    pv = _typical_price(df) * df["volume"]
    cum_pv = pv.groupby(dates).cumsum()
    cum_v = df["volume"].groupby(dates).cumsum() + 1e-9
    return cum_pv / cum_v


def vwap_weekly(df: pd.DataFrame) -> pd.Series:
    """
    Weekly VWAP — resets Monday 00:00 UTC.

    Provides medium-term institutional reference.  A price >1 % below the
    weekly VWAP at a key SMC trigger point signals deep discount; above signals
    premium.  Used by swing scoring in signal_engine.
    """
    df = df.copy()
    times = _resolve_times_utc(df)
    if times is None:
        return _cumulative_vwap(df)

    # ISO week period — resets Monday.
    # Strip timezone before to_period() to avoid pandas UserWarning about tz drop.
    week_key = times.dt.tz_localize(None).dt.to_period("W").astype(str)
    pv = _typical_price(df) * df["volume"]
    cum_pv = pv.groupby(week_key).cumsum()
    cum_v = df["volume"].groupby(week_key).cumsum() + 1e-9
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

    # ── Multi-anchor VWAP ─────────────────────────────────────────────────
    # vwap_session  = intraday anchor (Asia/London/NY — primary signal)
    # vwap_d1       = daily anchor (midnight UTC — secondary reference)
    # vwap_w1       = weekly anchor (Monday reset — swing context)
    #
    # vwap_dev      = deviation from session VWAP (used for all intraday scoring)
    # vwap_dev_d1   = deviation from daily VWAP (diagnostic / legacy compat)
    # vwap_dev_w1   = deviation from weekly VWAP (swing premium/discount context)
    df["vwap_session"] = vwap_session(df)
    df["vwap_d1"] = vwap_daily(df)
    df["vwap_w1"] = vwap_weekly(df)

    # Primary VWAP used by signal engine — session anchor, not midnight reset
    df["vwap"] = df["vwap_session"]
    df["above_vwap"] = (df["close"] > df["vwap"]).astype(int)
    df["below_vwap"] = (df["close"] < df["vwap"]).astype(int)
    df["vwap_dev"] = df["close"] - df["vwap_session"]
    df["vwap_dev_d1"] = df["close"] - df["vwap_d1"]
    df["vwap_dev_w1"] = df["close"] - df["vwap_w1"]

    # ── NEW: Candle body quality ───────────────────────────────────────────
    # body_pct = fraction of candle range that is body (0.0 = doji, 1.0 = marubozu)
    # Used by signal_engine for trigger quality scoring
    candle_range = df["high"] - df["low"]
    df["body_pct"] = (df["close"] - df["open"]).abs() / (candle_range + 1e-9)
    df["body_pct"] = df["body_pct"].clip(0.0, 1.0)

    return df
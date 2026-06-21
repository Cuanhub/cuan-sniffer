"""
Daily premium/discount zone classification — pure function.

Extracted from signal_engine.py. Candle integrity policy preserved:
  - Layer 1: forming 1H candle (last row) excluded before resampling
  - Layer 2: last 1D bar excluded (incomplete current day)

Does NOT import signal_engine.py.
"""

import os
from typing import Any, Dict

import pandas as pd

DAILY_ZONE_PREMIUM_THRESHOLD = float(os.getenv("DAILY_ZONE_PREMIUM_THRESHOLD", "0.65"))
DAILY_ZONE_DISCOUNT_THRESHOLD = float(os.getenv("DAILY_ZONE_DISCOUNT_THRESHOLD", "0.35"))


def compute_daily_zone(
    df_ohlcv: pd.DataFrame,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Classify current price within the daily premium/discount range.

    Returns dict with keys: zone, zone_pct, trend, swing_high, swing_low.
    Returns {"zone": "unknown"} on insufficient data.

    Candle policy:
      1. Drop last 1H row (forming candle)
      2. Drop last 1D bar (incomplete current day)
    """
    empty: Dict[str, Any] = {"zone": "unknown", "zone_pct": 0.5, "trend": "unknown"}

    if df_ohlcv is None or df_ohlcv.empty:
        return empty

    df_raw = df_ohlcv.copy()

    if len(df_raw) > 1:
        df_raw = df_raw.iloc[:-1]

    if "time" in df_raw.columns:
        df_raw["time"] = pd.to_datetime(df_raw["time"])
        df_raw = df_raw.set_index("time")
    elif not isinstance(df_raw.index, pd.DatetimeIndex):
        return empty

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
    except Exception:
        return empty

    if len(df_1d) > 1:
        df_1d = df_1d.iloc[:-1]

    if len(df_1d) < 5:
        return empty

    lookback = min(20, len(df_1d))
    recent = df_1d.tail(lookback)
    swing_high = float(recent["high"].max())
    swing_low = float(recent["low"].min())
    range_width = swing_high - swing_low
    if range_width <= 0:
        return empty

    current_close = float(df_1d["close"].iloc[-1])
    zone_pct = (current_close - swing_low) / range_width

    if zone_pct > DAILY_ZONE_PREMIUM_THRESHOLD:
        zone = "premium"
    elif zone_pct < DAILY_ZONE_DISCOUNT_THRESHOLD:
        zone = "discount"
    else:
        zone = "equilibrium"

    df_1d["ema_fast"] = df_1d["close"].ewm(span=5, adjust=False).mean()
    df_1d["ema_slow"] = df_1d["close"].ewm(span=10, adjust=False).mean()
    d_last = df_1d.iloc[-1]
    d_prev = df_1d.iloc[-2] if len(df_1d) >= 2 else df_1d.iloc[0]
    d_fast = float(d_last["ema_fast"])
    d_slow = float(d_last["ema_slow"])
    d_fast_slope = d_fast - float(d_prev["ema_fast"])
    if d_fast > d_slow and d_fast_slope > 0:
        daily_trend = "up"
    elif d_fast < d_slow and d_fast_slope < 0:
        daily_trend = "down"
    else:
        daily_trend = "chop"

    if debug:
        print(
            f"[DAILY_ZONE] zone={zone} pct={zone_pct:.2f}"
            f" swing_hi={swing_high:.4f} swing_lo={swing_low:.4f}"
            f" daily_trend={daily_trend}"
        )

    return {
        "zone": zone,
        "zone_pct": round(zone_pct, 3),
        "trend": daily_trend,
        "swing_high": swing_high,
        "swing_low": swing_low,
    }

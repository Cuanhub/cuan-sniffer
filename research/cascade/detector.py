"""
Cascade-fade detection and resolution logic -- ported directly from the
manually-supplied backtest_cascade.py, same thresholds, unchanged. This
is the forward/live counterpart: given a fresh window of recent 1m
candles, detect a cascade on the newest closed bar and construct the
same fade trade the backtest would have. Resolution (stop/TP1/TP2/time-
stop) is checked against real subsequent candles once enough time has
passed -- same walk-forward logic as backtest_cascade.py's
_resolve_trade, not reimplemented differently.

Fixed rule, no re-tuning for the forward collector -- same discipline as
every other candidate this session. If the thresholds ever need
revisiting, that's a deliberate, logged decision (DECISIONS.md), not a
silent parameter change.
"""

import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional

MIN_MOVE_PCT = 0.003
MIN_VOL_MULTIPLE = 3.0
VOL_LOOKBACK_BARS = 60
ENTRY_OFFSET_PCT = 0.003
FILL_WINDOW_BARS = 3
STOP_PCT = 0.005
TP1_R = 1.0
TP2_R = 2.0
TP1_FRACTION = 0.60
TIME_STOP_BARS = 10
FEE_PER_SIDE = 0.00045


@dataclass
class CascadeEvent:
    symbol: str
    detected_ts_ms: int
    cascade_move_pct: float
    cascade_vol_ratio: float
    fade_side: str  # "buy" or "sell"
    entry_price: float
    stop_price: float
    tp1_price: float
    tp2_price: float


def detect_cascade(symbol: str, candles: List[Dict]) -> Optional[CascadeEvent]:
    """candles: chronological list of {ts_ms, open, high, low, close, volume},
    at least VOL_LOOKBACK_BARS + 1 long. Checks only the LAST candle for a
    cascade signature against the rolling median of the bars before it."""
    if len(candles) < VOL_LOOKBACK_BARS + 1:
        return None
    window = candles[-(VOL_LOOKBACK_BARS + 1):]
    c = window[-1]
    history = window[:-1]

    o, cl = c["open"], c["close"]
    if o <= 0:
        return None
    move_pct = (cl - o) / o

    vol_med = statistics.median(v["volume"] for v in history)
    if vol_med <= 0:
        return None
    vol_ratio = c["volume"] / vol_med

    if abs(move_pct) < MIN_MOVE_PCT or vol_ratio < MIN_VOL_MULTIPLE:
        return None

    if move_pct > 0:
        fade_side = "sell"
        entry_price = cl * (1 + ENTRY_OFFSET_PCT)
        stop_price = entry_price * (1 + STOP_PCT)
        tp1_price = entry_price * (1 - STOP_PCT * TP1_R)
        tp2_price = entry_price * (1 - STOP_PCT * TP2_R)
    else:
        fade_side = "buy"
        entry_price = cl * (1 - ENTRY_OFFSET_PCT)
        stop_price = entry_price * (1 - STOP_PCT)
        tp1_price = entry_price * (1 + STOP_PCT * TP1_R)
        tp2_price = entry_price * (1 + STOP_PCT * TP2_R)

    return CascadeEvent(
        symbol=symbol, detected_ts_ms=int(c["ts_ms"]),
        cascade_move_pct=round(move_pct * 100, 4), cascade_vol_ratio=round(vol_ratio, 3),
        fade_side=fade_side, entry_price=entry_price,
        stop_price=stop_price, tp1_price=tp1_price, tp2_price=tp2_price,
    )


def try_fill_and_resolve(event: CascadeEvent, candles_after: List[Dict]) -> Optional[Dict]:
    """candles_after: chronological 1m candles strictly after the detection
    bar. Returns a resolution dict once the trade either fills-then-
    resolves, never fills within FILL_WINDOW_BARS, or there isn't yet
    enough data to know (returns None in that last case -- caller should
    retry once more time has passed, never guess)."""
    needed = FILL_WINDOW_BARS + TIME_STOP_BARS
    if len(candles_after) < FILL_WINDOW_BARS:
        return None  # not enough elapsed time yet to know even the fill outcome

    fill_bar = None
    for j in range(min(FILL_WINDOW_BARS, len(candles_after))):
        bar = candles_after[j]
        if event.fade_side == "sell" and bar["high"] >= event.entry_price:
            fill_bar = j
            break
        if event.fade_side == "buy" and bar["low"] <= event.entry_price:
            fill_bar = j
            break

    if fill_bar is None:
        return {"filled": False, "exit_reason": "not_filled", "net_pnl_pct": 0.0, "pnl_r": 0.0, "bars_held": 0}

    if len(candles_after) < needed:
        return None  # filled, but not enough time has passed to know the exit yet

    remaining = 1.0
    tp1_hit = False
    weighted_exit = 0.0
    exit_reason = ""
    exit_idx = min(fill_bar + TIME_STOP_BARS, len(candles_after) - 1)

    for j in range(fill_bar + 1, exit_idx + 1):
        bar = candles_after[j]
        h, l = bar["high"], bar["low"]
        if event.fade_side == "sell":
            if h >= event.stop_price:
                weighted_exit += remaining * event.stop_price
                exit_reason = "tp1+stop" if tp1_hit else "stop"
                exit_idx, remaining = j, 0.0
                break
            if not tp1_hit and l <= event.tp1_price:
                tp1_hit = True
                amt = min(TP1_FRACTION, remaining)
                weighted_exit += amt * event.tp1_price
                remaining -= amt
            if tp1_hit and l <= event.tp2_price:
                weighted_exit += remaining * event.tp2_price
                exit_reason, exit_idx, remaining = "tp2", j, 0.0
                break
        else:
            if l <= event.stop_price:
                weighted_exit += remaining * event.stop_price
                exit_reason = "tp1+stop" if tp1_hit else "stop"
                exit_idx, remaining = j, 0.0
                break
            if not tp1_hit and h >= event.tp1_price:
                tp1_hit = True
                amt = min(TP1_FRACTION, remaining)
                weighted_exit += amt * event.tp1_price
                remaining -= amt
            if tp1_hit and h >= event.tp2_price:
                weighted_exit += remaining * event.tp2_price
                exit_reason, exit_idx, remaining = "tp2", j, 0.0
                break

    if remaining > 0:
        weighted_exit += remaining * candles_after[exit_idx]["close"]
        exit_reason = "time_stop" if not exit_reason else "tp1+time"

    if event.fade_side == "sell":
        gross = (event.entry_price - weighted_exit) / event.entry_price
    else:
        gross = (weighted_exit - event.entry_price) / event.entry_price
    fees = FEE_PER_SIDE * 2
    net = gross - fees
    pnl_r = gross / STOP_PCT if STOP_PCT > 0 else 0.0

    return {
        "filled": True, "fill_ts_ms": int(candles_after[fill_bar]["ts_ms"]),
        "exit_reason": exit_reason, "avg_exit_price": round(weighted_exit, 8),
        "gross_pnl_pct": round(gross * 100, 4), "fees_pct": round(fees * 100, 4),
        "net_pnl_pct": round(net * 100, 4), "pnl_r": round(pnl_r, 4),
        "bars_held": exit_idx - fill_bar,
    }

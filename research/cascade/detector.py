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

import os
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

# Multi-arm resolution (2026-09-16, see DECISIONS.md). Fees are a fixed
# fraction of notional while R is defined by the stop, so fee drag in R terms
# is 2*fee/stop -- 0.18R at the 0.5% baseline under the old flat-taker model.
# Stop width is therefore the one lever that attacks that arithmetic rather
# than nibbling at it, and it cannot be tested after the fact: resolutions
# store outcomes, not the post-entry path, and this venue drops 1m candles
# after ~3-4 days. So every detected event is resolved at each arm in
# parallel, from the same candles, at no extra API cost. Arm 0 is and stays
# STOP_PCT -- the pre-registered rule -- so the baseline remains directly
# comparable to everything collected before this change.
def _parse_stop_arms() -> tuple:
    """Arms from CASCADE_STOP_ARMS, with STOP_PCT guaranteed first.

    The baseline is an invariant, not a convention: every comparison back to
    data collected before 2026-09-16 assumes arm 0 is the pre-registered
    rule, and the collector prints that claim at startup. An override that
    omitted or reordered STOP_PCT would quietly falsify both, so it is
    re-inserted rather than trusted to be there. Duplicates are dropped and
    the remaining order is preserved."""
    raw = [float(x) for x in os.getenv("CASCADE_STOP_ARMS", "0.005,0.0075,0.01").split(",") if x.strip()]
    arms = [STOP_PCT] + [a for a in raw if a != STOP_PCT]
    seen, out = set(), []
    for a in arms:
        if a > 0 and a not in seen:
            seen.add(a)
            out.append(a)
    return tuple(out)


STOP_ARMS = _parse_stop_arms()

# Venue fee schedule, read from userFees for this account on 2026-09-16.
# The entry is a resting limit 0.3% beyond the cascade close that only fills
# if price comes back to it, and TP1/TP2 are limits -- all maker. Only stops
# and time-stops cross the spread. The previous model charged taker on both
# sides unconditionally (FEE_PER_SIDE * 2 = 0.09%), overstating cost by about
# 0.04% per trade, which was enough on its own to invert the sign of the
# measured edge. Corrected below; FEE_PER_SIDE is kept only as the legacy
# reference value so old rows stay interpretable.
MAKER_FEE_PER_SIDE = 0.00015
TAKER_FEE_PER_SIDE = 0.00045
FEE_PER_SIDE = TAKER_FEE_PER_SIDE


def arm_prices(fade_side: str, entry_price: float, stop_pct: float):
    """Stop/TP1/TP2 for one arm. TPs are defined in R, so they scale with the
    arm's stop width -- same construction as build-time, not a second rule."""
    if fade_side == "sell":
        return (entry_price * (1 + stop_pct),
                entry_price * (1 - stop_pct * TP1_R),
                entry_price * (1 - stop_pct * TP2_R))
    return (entry_price * (1 - stop_pct),
            entry_price * (1 + stop_pct * TP1_R),
            entry_price * (1 + stop_pct * TP2_R))


def exit_fee_per_side(exit_reason: str) -> float:
    """Exit-side fee as a fraction of notional, by how the position actually
    left the book. A partial TP1 followed by a stop or a time-stop pays maker
    on the TP1_FRACTION that rested and taker on the remainder."""
    if exit_reason == "tp2":
        return MAKER_FEE_PER_SIDE
    if exit_reason in ("tp1+stop", "tp1+time"):
        return TP1_FRACTION * MAKER_FEE_PER_SIDE + (1.0 - TP1_FRACTION) * TAKER_FEE_PER_SIDE
    return TAKER_FEE_PER_SIDE  # stop, time_stop -- crossed the spread


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


def try_fill_and_resolve(event: CascadeEvent, candles_after: List[Dict],
                         stop_pct: Optional[float] = None) -> Optional[Dict]:
    """candles_after: chronological 1m candles strictly after the detection
    bar. Returns a resolution dict once the trade either fills-then-
    resolves, never fills within FILL_WINDOW_BARS, or there isn't yet
    enough data to know (returns None in that last case -- caller should
    retry once more time has passed, never guess).

    stop_pct selects the arm; None means the pre-registered STOP_PCT and
    reproduces this function's original behaviour exactly. The fill test is
    arm-independent (entry and FILL_WINDOW_BARS do not vary), so every arm
    becomes resolvable at the same moment -- the caller can resolve them
    together and drop the event from pending in one step."""
    stop_pct = STOP_PCT if stop_pct is None else stop_pct
    stop_price, tp1_price, tp2_price = arm_prices(event.fade_side, event.entry_price, stop_pct)
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
        return {"filled": False, "exit_reason": "not_filled", "net_pnl_pct": 0.0,
                "pnl_r": 0.0, "bars_held": 0, "arm_stop_pct": stop_pct}

    # How far past the entry the filling bar actually traded. The entry rests
    # in the book, so a bar that merely tags the level does not establish a
    # fill -- real queue position needs price to trade through. This records
    # the evidence rather than assuming either way: near-zero penetration
    # marks the fills that are most likely optimistic, so the forward sample
    # can be re-read with a minimum-penetration filter instead of guessing at
    # a haircut now. See DECISIONS.md 2026-09-16.
    _fb = candles_after[fill_bar]
    if event.fade_side == "sell":
        penetration = (_fb["high"] - event.entry_price) / event.entry_price
    else:
        penetration = (event.entry_price - _fb["low"]) / event.entry_price

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
            if h >= stop_price:
                weighted_exit += remaining * stop_price
                exit_reason = "tp1+stop" if tp1_hit else "stop"
                exit_idx, remaining = j, 0.0
                break
            if not tp1_hit and l <= tp1_price:
                tp1_hit = True
                amt = min(TP1_FRACTION, remaining)
                weighted_exit += amt * tp1_price
                remaining -= amt
            if tp1_hit and l <= tp2_price:
                weighted_exit += remaining * tp2_price
                exit_reason, exit_idx, remaining = "tp2", j, 0.0
                break
        else:
            if l <= stop_price:
                weighted_exit += remaining * stop_price
                exit_reason = "tp1+stop" if tp1_hit else "stop"
                exit_idx, remaining = j, 0.0
                break
            if not tp1_hit and h >= tp1_price:
                tp1_hit = True
                amt = min(TP1_FRACTION, remaining)
                weighted_exit += amt * tp1_price
                remaining -= amt
            if tp1_hit and h >= tp2_price:
                weighted_exit += remaining * tp2_price
                exit_reason, exit_idx, remaining = "tp2", j, 0.0
                break

    if remaining > 0:
        # Reaching here always means no break fired, so exit_reason is still
        # "" and the old `if not exit_reason` test could never choose
        # "tp1+time" -- a TP1 that then timed out was silently labelled
        # "time_stop". Keyed off tp1_hit instead, which is what the label
        # meant. This also fixes the exit fee: that residual leaves as a
        # market order but TP1_FRACTION already left as a resting limit.
        weighted_exit += remaining * candles_after[exit_idx]["close"]
        exit_reason = "tp1+time" if tp1_hit else "time_stop"

    if event.fade_side == "sell":
        gross = (event.entry_price - weighted_exit) / event.entry_price
    else:
        gross = (weighted_exit - event.entry_price) / event.entry_price
    entry_fee = MAKER_FEE_PER_SIDE          # resting limit, always maker
    exit_fee = exit_fee_per_side(exit_reason)
    fees = entry_fee + exit_fee
    net = gross - fees
    pnl_r = gross / stop_pct if stop_pct > 0 else 0.0
    net_pnl_r = net / stop_pct if stop_pct > 0 else 0.0

    return {
        "filled": True, "fill_ts_ms": int(candles_after[fill_bar]["ts_ms"]),
        "exit_reason": exit_reason, "avg_exit_price": round(weighted_exit, 8),
        "gross_pnl_pct": round(gross * 100, 4), "fees_pct": round(fees * 100, 4),
        "net_pnl_pct": round(net * 100, 4), "pnl_r": round(pnl_r, 4),
        "bars_held": exit_idx - fill_bar,
        "arm_stop_pct": stop_pct,
        "net_pnl_r": round(net_pnl_r, 4),
        "entry_fee_pct": round(entry_fee * 100, 4),
        "exit_fee_pct": round(exit_fee * 100, 4),
        "fill_penetration_pct": round(penetration * 100, 5),
    }

"""
Stop redesign logic — pure computation, no class coupling.

Extracted from executor.py._apply_entry_stop_redesign.
Computes widened stop, validates RR, returns rejection reason or None.
"""

import os
from typing import Any, Dict, Optional, Tuple

from executor_modules.telemetry import log_executor_reject

# ── Constants (read from env at import time, same as executor.py) ─────
STOP_ATR_FLOOR_MULT_INTRADAY = float(os.getenv("STOP_ATR_FLOOR_MULT_INTRADAY", "1.10"))
STOP_ATR_FLOOR_MULT_SWING = float(os.getenv("STOP_ATR_FLOOR_MULT_SWING", "1.40"))
STOP_BUFFER_ATR_MULT = float(os.getenv("STOP_BUFFER_ATR_MULT", "0.10"))
MIN_STOP_ATR_REJECT = float(os.getenv("MIN_STOP_ATR_REJECT", "0.80"))
MIN_STOP_REDESIGN_RR = float(os.getenv("MIN_STOP_REDESIGN_RR", "1.60"))
STOP_REDESIGN_RR_TOLERANCE = float(os.getenv("STOP_REDESIGN_RR_TOLERANCE", "0.05"))
HIGH_CONF_STOP_REDESIGN_MIN_CONFIDENCE = float(
    os.getenv("HIGH_CONF_STOP_REDESIGN_MIN_CONFIDENCE", "0.93")
)
HIGH_CONF_STOP_REDESIGN_RR_TOLERANCE = float(
    os.getenv("HIGH_CONF_STOP_REDESIGN_RR_TOLERANCE", "0.08")
)
HIGH_CONF_STOP_REDESIGN_FAMILIES = {
    s.strip().lower()
    for s in os.getenv("HIGH_CONF_STOP_REDESIGN_FAMILIES", "reversal,swing").split(",")
    if s.strip()
}
STOP_REDESIGN_MAX_WIDEN_MULT = float(os.getenv("STOP_REDESIGN_MAX_WIDEN_MULT", "1.50"))


def apply_entry_stop_redesign(
    *,
    coin: str,
    side: str,
    entry: float,
    structural_stop: float,
    tp: float,
    atr: float,
    track: str,
    confidence: float,
    setup_family: str,
    market_regime: str,
    timeframe: str = "1h",
    meta: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[str], float, Dict[str, Any]]:
    """
    Compute the widened stop and validate RR.

    Returns:
        (reject_reason or None, final_stop, updated meta dict)

    If reject_reason is not None, the signal should be rejected.
    If None, final_stop and meta are ready for use.
    """
    meta = dict(meta) if meta else {}

    if entry <= 0 or structural_stop <= 0 or tp <= 0:
        return "invalid_signal_levels", 0.0, meta
    if atr <= 0:
        return "invalid_atr_for_stop_redesign", 0.0, meta

    if side == "LONG":
        if structural_stop >= entry:
            return (
                f"invalid_structural_stop_long "
                f"(stop={structural_stop:.6f} >= entry={entry:.6f})"
            ), 0.0, meta
        if tp <= entry:
            return (
                f"invalid_tp_long "
                f"(tp={tp:.6f} <= entry={entry:.6f})"
            ), 0.0, meta
    elif side == "SHORT":
        if structural_stop <= entry:
            return (
                f"invalid_structural_stop_short "
                f"(stop={structural_stop:.6f} <= entry={entry:.6f})"
            ), 0.0, meta
        if tp >= entry:
            return (
                f"invalid_tp_short "
                f"(tp={tp:.6f} >= entry={entry:.6f})"
            ), 0.0, meta
    else:
        return f"invalid_side:{side}", 0.0, meta

    floor_mult = (
        STOP_ATR_FLOOR_MULT_SWING if track == "swing"
        else STOP_ATR_FLOOR_MULT_INTRADAY
    )
    floor_mult = max(0.0, float(floor_mult))
    buffer_mult = max(0.0, float(STOP_BUFFER_ATR_MULT))
    min_stop_atr = max(0.0, float(MIN_STOP_ATR_REJECT))

    atr_floor_dist = floor_mult * atr
    min_stop_dist = min_stop_atr * atr
    buffer_dist = buffer_mult * atr

    if side == "LONG":
        atr_floor_stop = entry - atr_floor_dist
        floor_applied_stop = min(structural_stop, atr_floor_stop)
        buffered_stop = floor_applied_stop - buffer_dist
        final_stop = buffered_stop
        if (entry - final_stop) < min_stop_dist:
            final_stop = entry - min_stop_dist
    else:
        atr_floor_stop = entry + atr_floor_dist
        floor_applied_stop = max(structural_stop, atr_floor_stop)
        buffered_stop = floor_applied_stop + buffer_dist
        final_stop = buffered_stop
        if (final_stop - entry) < min_stop_dist:
            final_stop = entry + min_stop_dist

    original_stop_dist = abs(entry - structural_stop)
    final_stop_dist = abs(entry - final_stop)
    if final_stop_dist <= 0:
        return "invalid_final_stop_distance", 0.0, meta

    widen_mult = (
        final_stop_dist / original_stop_dist
        if original_stop_dist > 0 else 0.0
    )

    if STOP_REDESIGN_MAX_WIDEN_MULT > 0 and widen_mult > STOP_REDESIGN_MAX_WIDEN_MULT:
        reason = (
            f"stop_redesign_too_wide"
            f" (widen={widen_mult:.2f}x > max={STOP_REDESIGN_MAX_WIDEN_MULT:.2f}x)"
        )
        print(
            f"[STOP_REDESIGN] {coin} {side} stop_redesign_too_wide"
            f" | widen_mult={widen_mult:.2f}"
            f" | max_widen_mult={STOP_REDESIGN_MAX_WIDEN_MULT:.2f}"
            f" | original_sd={original_stop_dist:.8f}"
            f" | final_sd={final_stop_dist:.8f}"
        )
        log_executor_reject(
            symbol=coin, side=side,
            confidence=confidence,
            rr=0.0,
            reject_reason=reason,
            setup_family=setup_family,
            market_regime=market_regime,
            timeframe=timeframe,
        )
        return reason, 0.0, meta

    tp_dist = abs(tp - entry)
    original_rr = tp_dist / original_stop_dist if original_stop_dist > 0 else 0.0
    final_rr = tp_dist / final_stop_dist if final_stop_dist > 0 else 0.0
    min_rr = float(MIN_STOP_REDESIGN_RR)

    high_conf_rr_tolerance = (
        confidence >= HIGH_CONF_STOP_REDESIGN_MIN_CONFIDENCE
        and market_regime != "chop"
        and setup_family in HIGH_CONF_STOP_REDESIGN_FAMILIES
    )
    min_rr_tolerance = (
        HIGH_CONF_STOP_REDESIGN_RR_TOLERANCE
        if high_conf_rr_tolerance
        else STOP_REDESIGN_RR_TOLERANCE
    )
    min_rr_effective = max(0.0, min_rr - min_rr_tolerance)

    if final_rr < min_rr_effective:
        reason = (
            f"stop_redesign_rr_destroyed"
            f" (original_rr={original_rr:.2f}"
            f" final_rr={final_rr:.2f} < min_rr={min_rr_effective:.2f}"
            f" widen={widen_mult:.2f}x)"
        )
        print(
            f"[STOP_REDESIGN] {coin} {side} stop_redesign_rr_destroyed"
            f" | original_rr={original_rr:.3f}"
            f" | final_rr={final_rr:.3f}"
            f" | min_rr={min_rr_effective:.3f}"
            f" | widen_mult={widen_mult:.2f}"
            f" | original_sd={original_stop_dist:.8f}"
            f" | final_sd={final_stop_dist:.8f}"
        )
        log_executor_reject(
            symbol=coin, side=side,
            confidence=confidence,
            rr=final_rr, required_rr=min_rr_effective,
            reject_reason=(
                f"stop_redesign_rr_destroyed"
                f" (original_rr={original_rr:.2f}"
                f" final_rr={final_rr:.2f}"
                f" widen={widen_mult:.2f}x)"
            ),
            setup_family=setup_family,
            market_regime=market_regime,
            timeframe=timeframe,
        )
        return reason, 0.0, meta

    # ── Success: build meta ──────────────────────────────────────
    engine_stop_method = str(meta.get("stop_method", "atr") or "atr")
    stop_was_redesigned = abs(float(final_stop) - float(structural_stop)) > 1e-12

    meta["original_stop"] = round(structural_stop, 8)
    meta["original_rr"] = round(original_rr, 4)
    meta["final_entry"] = round(entry, 8)
    meta["final_stop"] = round(final_stop, 8)
    meta["final_tp"] = round(tp, 8)
    meta["final_rr"] = round(final_rr, 4)
    meta["final_stop_method"] = (
        f"{engine_stop_method}+executor_redesign"
        if stop_was_redesigned else engine_stop_method
    )
    meta["stop_was_redesigned"] = stop_was_redesigned
    meta["stop_widen_mult"] = round(widen_mult, 4)
    meta["stop_structural"] = round(structural_stop, 8)
    meta["stop_atr_floor"] = round(atr_floor_stop, 8)
    meta["stop_buffered"] = round(buffered_stop, 8)
    meta["stop_final"] = round(final_stop, 8)
    meta["rr_original"] = round(original_rr, 4)
    meta["rr_final"] = round(final_rr, 4)
    meta["stop_floor_mult"] = round(floor_mult, 4)
    meta["stop_buffer_mult"] = round(buffer_mult, 4)
    meta["stop_min_atr_reject"] = round(min_stop_atr, 4)
    meta["stop_rr_min"] = round(min_rr, 4)
    meta["stop_rr_tolerance"] = round(min_rr_tolerance, 4)
    meta["stop_rr_min_effective"] = round(min_rr_effective, 4)
    meta["stop_rr_tolerance_reason"] = (
        "high_conf_non_chop"
        if high_conf_rr_tolerance else "base"
    )
    meta["stop_track"] = track

    print(
        f"[STOP_REDESIGN] {coin} {side} track={track} "
        f"structural_stop={structural_stop:.6f} "
        f"atr_floor_stop={atr_floor_stop:.6f} "
        f"buffered_stop={buffered_stop:.6f} "
        f"final_stop={final_stop:.6f} "
        f"original_rr={original_rr:.2f} "
        f"final_rr={final_rr:.2f} "
        f"widen={widen_mult:.2f}x "
        f"min_rr={min_rr_effective:.2f}"
    )

    return None, final_stop, meta

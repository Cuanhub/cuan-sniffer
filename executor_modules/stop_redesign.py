"""
Stop redesign logic — pure computation, no class coupling.

Extracted from executor.py._apply_entry_stop_redesign.
Computes widened stop, validates RR, returns rejection reason or None.
"""

import os
from typing import Any, Dict, Optional, Tuple

from executor_modules.execution_policy import (
    ExecutionPolicyConfig,
    evaluate_execution_policy,
)

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
) -> Tuple[Optional[str], float, float, Dict[str, Any]]:
    """
    Compute the widened stop and validate RR.

    Returns:
        (reject_reason or None, final_stop, final_tp, updated meta dict)

    If reject_reason is not None, the signal should be rejected.
    If None, final_stop and meta are ready for use.
    """
    meta = dict(meta) if meta else {}

    config = ExecutionPolicyConfig(
        stop_atr_floor_mult_intraday=STOP_ATR_FLOOR_MULT_INTRADAY,
        stop_atr_floor_mult_swing=STOP_ATR_FLOOR_MULT_SWING,
        stop_buffer_atr_mult=STOP_BUFFER_ATR_MULT,
        min_stop_atr_reject=MIN_STOP_ATR_REJECT,
        min_stop_redesign_rr=MIN_STOP_REDESIGN_RR,
        stop_redesign_rr_tolerance=STOP_REDESIGN_RR_TOLERANCE,
        high_conf_stop_redesign_min_confidence=HIGH_CONF_STOP_REDESIGN_MIN_CONFIDENCE,
        high_conf_stop_redesign_rr_tolerance=HIGH_CONF_STOP_REDESIGN_RR_TOLERANCE,
        high_conf_stop_redesign_families=HIGH_CONF_STOP_REDESIGN_FAMILIES,
        stop_redesign_max_widen_mult=STOP_REDESIGN_MAX_WIDEN_MULT,
        apply_chop_block=False,
        apply_dual_chop_block=False,
        apply_stop_redesign=True,
        apply_tp_cap=False,
        apply_effective_rr=False,
        block_continuation_in_weak_trend=False,
        block_reversal_in_weak_trend=False,
    )
    result = evaluate_execution_policy(
        entry=entry,
        stop=structural_stop,
        tp=tp,
        side=side,
        atr=atr,
        timeframe=timeframe,
        session=str(meta.get("session", "")),
        market_regime=market_regime,
        htf_regime=str(meta.get("regime_htf_1h", "")),
        macro_regime=str(meta.get("regime_macro_4h", "")),
        setup_family=setup_family,
        confidence=confidence,
        config=config,
        track=track,
        metadata=meta,
    )

    if not result.approved:
        reason = result.reject_reason or "execution_policy_reject"
        original_stop_dist = abs(entry - structural_stop)
        final_stop_dist = abs(entry - result.redesigned_stop)
        widen_mult = result.widen_mult
        final_rr = result.final_rr
        original_rr = result.original_rr

        if reason.startswith("stop_redesign_too_wide"):
            print(
                f"[STOP_REDESIGN] {coin} {side} stop_redesign_too_wide"
                f" | widen_mult={widen_mult:.2f}"
                f" | max_widen_mult={STOP_REDESIGN_MAX_WIDEN_MULT:.2f}"
                f" | original_sd={original_stop_dist:.8f}"
                f" | final_sd={final_stop_dist:.8f}"
            )
            return reason, 0.0, tp, meta

        if reason.startswith("stop_redesign_rr_destroyed"):
            min_rr_effective = float(result.metadata.get("stop_rr_min_effective", 0.0) or 0.0)
            print(
                f"[STOP_REDESIGN] {coin} {side} stop_redesign_rr_destroyed"
                f" | original_rr={original_rr:.3f}"
                f" | final_rr={final_rr:.3f}"
                f" | min_rr={min_rr_effective:.3f}"
                f" | widen_mult={widen_mult:.2f}"
                f" | original_sd={original_stop_dist:.8f}"
                f" | final_sd={final_stop_dist:.8f}"
            )
            return reason, 0.0, tp, meta

        return reason, 0.0, tp, meta

    final_stop = result.redesigned_stop
    final_tp = result.final_tp
    updated_meta = dict(result.metadata)
    original_rr = result.original_rr
    final_rr = result.final_rr
    widen_mult = result.widen_mult
    original_stop_dist = abs(entry - structural_stop)
    final_stop_dist = abs(entry - final_stop)
    atr_floor_stop = float(updated_meta.get("stop_atr_floor", final_stop))
    buffered_stop = float(updated_meta.get("stop_buffered", final_stop))
    floor_mult = float(updated_meta.get("stop_floor_mult", 0.0))
    buffer_mult = float(updated_meta.get("stop_buffer_mult", 0.0))
    min_stop_atr = float(updated_meta.get("stop_min_atr_reject", 0.0))
    min_rr = float(updated_meta.get("stop_rr_min", MIN_STOP_REDESIGN_RR))
    min_rr_tolerance = float(updated_meta.get("stop_rr_tolerance", STOP_REDESIGN_RR_TOLERANCE))
    min_rr_effective = float(updated_meta.get("stop_rr_min_effective", max(0.0, min_rr - min_rr_tolerance)))

    engine_stop_method = str(meta.get("stop_method", "atr") or "atr")
    stop_was_redesigned = abs(float(final_stop) - float(structural_stop)) > 1e-12
    updated_meta["final_stop_method"] = (
        f"{engine_stop_method}+executor_redesign"
        if stop_was_redesigned else engine_stop_method
    )
    updated_meta["stop_was_redesigned"] = stop_was_redesigned
    updated_meta["stop_structural"] = round(structural_stop, 8)
    updated_meta["stop_atr_floor"] = round(atr_floor_stop, 8)
    updated_meta["stop_buffered"] = round(buffered_stop, 8)
    updated_meta["stop_final"] = round(final_stop, 8)
    updated_meta["tp_original"] = round(tp, 8)
    updated_meta["tp_final"] = round(final_tp, 8)
    updated_meta["rr_original"] = round(original_rr, 4)
    updated_meta["rr_final"] = round(final_rr, 4)
    updated_meta["stop_floor_mult"] = round(floor_mult, 4)
    updated_meta["stop_buffer_mult"] = round(buffer_mult, 4)
    updated_meta["stop_min_atr_reject"] = round(min_stop_atr, 4)
    updated_meta["stop_rr_min"] = round(min_rr, 4)
    updated_meta["stop_rr_tolerance"] = round(min_rr_tolerance, 4)
    updated_meta["stop_rr_min_effective"] = round(min_rr_effective, 4)

    print(
        f"[STOP_REDESIGN] {coin} {side} track={track} "
        f"structural_stop={structural_stop:.6f} "
        f"atr_floor_stop={atr_floor_stop:.6f} "
        f"buffered_stop={buffered_stop:.6f} "
        f"final_stop={final_stop:.6f} "
        f"final_tp={final_tp:.6f} "
        f"original_rr={original_rr:.2f} "
        f"final_rr={final_rr:.2f} "
        f"widen={widen_mult:.2f}x "
        f"min_rr={min_rr_effective:.2f}"
    )

    return None, final_stop, final_tp, updated_meta

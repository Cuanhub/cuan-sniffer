"""
Pure execution geometry policy shared by production and replay.

This module has no venue calls, order placement, CSV writes, or logging side
effects. It centralizes the deterministic stop/RR/TP-cap decisions so replay
can model production executor geometry exactly.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set


def _env_bool(name: str, default: str) -> bool:
    return os.getenv(name, default).lower() in {"true", "1", "yes", "on"}


def _parse_family_set(raw: str) -> Set[str]:
    return {s.strip().lower() for s in str(raw or "").split(",") if s.strip()}


@dataclass(frozen=True)
class ExecutionPolicyConfig:
    stop_atr_floor_mult_intraday: float = field(
        default_factory=lambda: float(os.getenv("STOP_ATR_FLOOR_MULT_INTRADAY", "1.10"))
    )
    stop_atr_floor_mult_swing: float = field(
        default_factory=lambda: float(os.getenv("STOP_ATR_FLOOR_MULT_SWING", "1.40"))
    )
    stop_buffer_atr_mult: float = field(
        default_factory=lambda: float(os.getenv("STOP_BUFFER_ATR_MULT", "0.10"))
    )
    min_stop_atr_reject: float = field(
        default_factory=lambda: float(os.getenv("MIN_STOP_ATR_REJECT", "0.80"))
    )
    min_stop_redesign_rr: float = field(
        default_factory=lambda: float(os.getenv("MIN_STOP_REDESIGN_RR", "1.60"))
    )
    stop_redesign_rr_tolerance: float = field(
        default_factory=lambda: float(os.getenv("STOP_REDESIGN_RR_TOLERANCE", "0.05"))
    )
    high_conf_stop_redesign_min_confidence: float = field(
        default_factory=lambda: float(os.getenv("HIGH_CONF_STOP_REDESIGN_MIN_CONFIDENCE", "0.93"))
    )
    high_conf_stop_redesign_rr_tolerance: float = field(
        default_factory=lambda: float(os.getenv("HIGH_CONF_STOP_REDESIGN_RR_TOLERANCE", "0.08"))
    )
    high_conf_stop_redesign_families: Set[str] = field(
        default_factory=lambda: _parse_family_set(os.getenv("HIGH_CONF_STOP_REDESIGN_FAMILIES", "reversal,swing"))
    )
    stop_redesign_max_widen_mult: float = field(
        default_factory=lambda: float(os.getenv("STOP_REDESIGN_MAX_WIDEN_MULT", "1.50"))
    )
    min_execution_effective_rr: float = field(
        default_factory=lambda: float(os.getenv("MIN_EXECUTION_EFFECTIVE_RR", os.getenv("MIN_EFFECTIVE_RR", "1.55")))
    )
    regime_tp_cap_r: float = field(
        default_factory=lambda: float(os.getenv("REGIME_TP_CAP_R", "1.75"))
    )
    hard_block_chop: bool = field(default_factory=lambda: _env_bool("HARD_BLOCK_CHOP", "true"))
    chop_reversal_exception: bool = field(default_factory=lambda: _env_bool("CHOP_REVERSAL_EXCEPTION", "false"))
    chop_reversal_min_confidence: float = field(
        default_factory=lambda: float(os.getenv("CHOP_REVERSAL_MIN_CONFIDENCE", os.getenv("UNIVERSAL_MIN_CONFIDENCE", "0.90")))
    )
    block_continuation_in_chop: bool = field(
        default_factory=lambda: _env_bool("BLOCK_CONTINUATION_IN_CHOP", "true")
    )
    apply_chop_block: bool = True
    apply_dual_chop_block: bool = True
    apply_stop_redesign: bool = True
    apply_tp_cap: bool = True
    apply_effective_rr: bool = True


@dataclass
class ExecutionPolicyResult:
    approved: bool
    reject_reason: Optional[str]
    entry: float
    original_stop: float
    redesigned_stop: float
    original_tp: float
    final_tp: float
    original_rr: float
    final_rr: float
    widen_mult: float
    tp_capped: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


def infer_execution_track(
    *,
    timeframe: str,
    setup_family: str,
    execution_track: str = "",
) -> str:
    explicit = str(execution_track or "").strip().lower()
    if explicit in {"intraday", "swing"}:
        return explicit
    tf = str(timeframe or "").strip().lower()
    family = str(setup_family or "").strip().lower()
    if tf in {"1h", "4h"} or family == "swing":
        return "swing"
    return "intraday"


def _initial_result(
    *,
    entry: float,
    stop: float,
    tp: float,
    metadata: Optional[Dict[str, Any]],
) -> ExecutionPolicyResult:
    return ExecutionPolicyResult(
        approved=True,
        reject_reason=None,
        entry=entry,
        original_stop=stop,
        redesigned_stop=stop,
        original_tp=tp,
        final_tp=tp,
        original_rr=0.0,
        final_rr=0.0,
        widen_mult=1.0,
        tp_capped=False,
        metadata=dict(metadata) if metadata else {},
    )


def _rr_from_entry(entry: float, stop: float, tp: float) -> float:
    risk = abs(entry - stop)
    if risk <= 0:
        return 0.0
    return abs(tp - entry) / risk


def _reject(result: ExecutionPolicyResult, reason: str) -> ExecutionPolicyResult:
    result.approved = False
    result.reject_reason = reason
    return result


def _is_chop_blocked(
    *,
    market_regime: str,
    htf_regime: str,
    setup_family: str,
    confidence: float,
    config: ExecutionPolicyConfig,
) -> bool:
    if not (config.apply_chop_block and config.hard_block_chop):
        return False
    if market_regime != "chop":
        return False
    if (
        config.chop_reversal_exception
        and setup_family == "reversal"
        and htf_regime == "up"
        and confidence >= config.chop_reversal_min_confidence
    ):
        return False
    return True


def _cap_tp_if_needed(
    *,
    result: ExecutionPolicyResult,
    side: str,
    market_regime: str,
    macro_regime: str,
    regime: str,
    config: ExecutionPolicyConfig,
) -> None:
    regime_text = str(regime or "").strip().lower()
    cap_applies = (
        market_regime == "chop"
        or market_regime == "weak_trend"
        or macro_regime == "chop"
        or "mkt_chop" in regime_text
        or "mkt_weak_trend" in regime_text
        or "macro_chop" in regime_text
    )
    if not cap_applies:
        return

    stop_dist = abs(result.entry - result.redesigned_stop)
    if stop_dist <= 0:
        return

    cap_dist = config.regime_tp_cap_r * stop_dist
    if side == "LONG":
        capped_tp = result.entry + cap_dist
        should_cap = result.final_tp > capped_tp
    else:
        capped_tp = result.entry - cap_dist
        should_cap = result.final_tp < capped_tp

    if should_cap:
        result.final_tp = capped_tp
        result.tp_capped = True
        result.metadata["tp_capped"] = True
        result.metadata["original_tp"] = round(result.original_tp, 8)
        result.metadata["final_tp"] = round(capped_tp, 8)
        result.metadata["regime_tp_cap_r"] = round(config.regime_tp_cap_r, 4)


def evaluate_execution_policy(
    *,
    entry: float,
    stop: float,
    tp: float,
    side: str,
    atr: float,
    timeframe: str,
    session: str,
    market_regime: str,
    htf_regime: str,
    macro_regime: str,
    setup_family: str,
    confidence: float,
    config: Optional[ExecutionPolicyConfig] = None,
    current_price: Optional[float] = None,
    regime: str = "",
    track: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> ExecutionPolicyResult:
    """
    Evaluate deterministic executor geometry.

    current_price defaults to entry for replay parity when no venue midpoint is
    available. Production passes the live midpoint for effective-RR checks.
    """
    config = config or ExecutionPolicyConfig()
    side = str(side or "").upper()
    timeframe = str(timeframe or "").strip().lower()
    session = str(session or "").strip().lower()
    market_regime = str(market_regime or "").strip().lower()
    htf_regime = str(htf_regime or "").strip().lower()
    macro_regime = str(macro_regime or "").strip().lower()
    setup_family = str(setup_family or "").strip().lower()
    track = infer_execution_track(
        timeframe=timeframe,
        setup_family=setup_family,
        execution_track=track,
    )
    result = _initial_result(entry=entry, stop=stop, tp=tp, metadata=metadata)
    result.metadata["session"] = session
    result.metadata["stop_track"] = track

    if _is_chop_blocked(
        market_regime=market_regime,
        htf_regime=htf_regime,
        setup_family=setup_family,
        confidence=confidence,
        config=config,
    ):
        return _reject(result, "market_regime_block:chop")

    if (
        config.apply_chop_block
        and config.block_continuation_in_chop
        and setup_family == "continuation"
        and market_regime == "chop"
    ):
        return _reject(result, "market_regime_block:continuation_in_chop")

    regime_text = str(regime or "").strip().lower()
    if (
        config.apply_dual_chop_block
        and config.apply_chop_block
        and "macro_chop" in regime_text
        and "mkt_chop" in regime_text
    ):
        return _reject(result, "blocked_chop_dual_regime")

    if entry <= 0 or stop <= 0 or tp <= 0:
        return _reject(result, "invalid_signal_levels")

    if side == "LONG":
        if stop >= entry:
            return _reject(result, f"invalid_structural_stop_long (stop={stop:.6f} >= entry={entry:.6f})")
        if tp <= entry:
            return _reject(result, f"invalid_tp_long (tp={tp:.6f} <= entry={entry:.6f})")
    elif side == "SHORT":
        if stop <= entry:
            return _reject(result, f"invalid_structural_stop_short (stop={stop:.6f} <= entry={entry:.6f})")
        if tp >= entry:
            return _reject(result, f"invalid_tp_short (tp={tp:.6f} >= entry={entry:.6f})")
    else:
        return _reject(result, f"invalid_side:{side}")

    original_stop_dist = abs(entry - stop)
    result.original_rr = _rr_from_entry(entry, stop, tp)
    result.final_rr = result.original_rr

    if config.apply_stop_redesign:
        if atr <= 0:
            return _reject(result, "invalid_atr_for_stop_redesign")

        floor_mult = (
            config.stop_atr_floor_mult_swing
            if track == "swing"
            else config.stop_atr_floor_mult_intraday
        )
        floor_mult = max(0.0, float(floor_mult))
        buffer_mult = max(0.0, float(config.stop_buffer_atr_mult))
        min_stop_atr = max(0.0, float(config.min_stop_atr_reject))

        atr_floor_dist = floor_mult * atr
        min_stop_dist = min_stop_atr * atr
        buffer_dist = buffer_mult * atr

        if side == "LONG":
            atr_floor_stop = entry - atr_floor_dist
            floor_applied_stop = min(stop, atr_floor_stop)
            buffered_stop = floor_applied_stop - buffer_dist
            final_stop = buffered_stop
            if (entry - final_stop) < min_stop_dist:
                final_stop = entry - min_stop_dist
        else:
            atr_floor_stop = entry + atr_floor_dist
            floor_applied_stop = max(stop, atr_floor_stop)
            buffered_stop = floor_applied_stop + buffer_dist
            final_stop = buffered_stop
            if (final_stop - entry) < min_stop_dist:
                final_stop = entry + min_stop_dist

        final_stop_dist = abs(entry - final_stop)
        if final_stop_dist <= 0:
            return _reject(result, "invalid_final_stop_distance")

        result.redesigned_stop = final_stop
        result.widen_mult = final_stop_dist / original_stop_dist if original_stop_dist > 0 else 0.0
        result.final_rr = abs(tp - entry) / final_stop_dist

        result.metadata.update({
            "stop_structural": round(stop, 8),
            "stop_atr_floor": round(atr_floor_stop, 8),
            "stop_buffered": round(buffered_stop, 8),
            "stop_final": round(final_stop, 8),
            "stop_floor_mult": round(floor_mult, 4),
            "stop_buffer_mult": round(buffer_mult, 4),
            "stop_min_atr_reject": round(min_stop_atr, 4),
        })

        if (
            config.stop_redesign_max_widen_mult > 0
            and result.widen_mult > config.stop_redesign_max_widen_mult
        ):
            return _reject(
                result,
                f"stop_redesign_too_wide"
                f" (widen={result.widen_mult:.2f}x > max={config.stop_redesign_max_widen_mult:.2f}x)",
            )

        high_conf_rr_tolerance = (
            confidence >= config.high_conf_stop_redesign_min_confidence
            and market_regime != "chop"
            and setup_family in config.high_conf_stop_redesign_families
        )
        min_rr_tolerance = (
            config.high_conf_stop_redesign_rr_tolerance
            if high_conf_rr_tolerance
            else config.stop_redesign_rr_tolerance
        )
        min_rr_effective = max(0.0, config.min_stop_redesign_rr - min_rr_tolerance)
        result.metadata.update({
            "stop_rr_min": round(config.min_stop_redesign_rr, 4),
            "stop_rr_tolerance": round(min_rr_tolerance, 4),
            "stop_rr_min_effective": round(min_rr_effective, 4),
            "stop_rr_tolerance_reason": "high_conf_non_chop" if high_conf_rr_tolerance else "base",
        })

        if result.final_rr < min_rr_effective:
            return _reject(
                result,
                f"stop_redesign_rr_destroyed"
                f" (original_rr={result.original_rr:.2f}"
                f" final_rr={result.final_rr:.2f} < min_rr={min_rr_effective:.2f}"
                f" widen={result.widen_mult:.2f}x)",
            )

    if config.apply_tp_cap:
        _cap_tp_if_needed(
            result=result,
            side=side,
            market_regime=market_regime,
            macro_regime=macro_regime,
            regime=regime,
            config=config,
        )
        result.final_rr = _rr_from_entry(entry, result.redesigned_stop, result.final_tp)
        if result.tp_capped and result.final_rr <= config.min_execution_effective_rr:
            return _reject(
                result,
                f"regime_tp_cap_rr_impossible"
                f" (rr={result.final_rr:.2f} <= {config.min_execution_effective_rr:.2f})",
            )

    if config.apply_effective_rr:
        fill_price = float(current_price) if current_price is not None else entry
        if side == "LONG":
            rr_num = result.final_tp - fill_price
            rr_den = fill_price - result.redesigned_stop
        else:
            rr_num = fill_price - result.final_tp
            rr_den = result.redesigned_stop - fill_price

        result.metadata["execution_rr_num"] = round(rr_num, 8)
        result.metadata["execution_rr_den"] = round(rr_den, 8)
        if rr_den <= 0:
            return _reject(result, "fill_rr_invalid_geometry")

        effective_rr = rr_num / rr_den
        result.final_rr = effective_rr
        result.metadata["execution_effective_rr"] = round(effective_rr, 4)
        result.metadata["execution_current_price"] = round(fill_price, 8)
        if effective_rr < config.min_execution_effective_rr:
            return _reject(
                result,
                f"fill_rr_below_threshold"
                f" (rr={effective_rr:.2f} < {config.min_execution_effective_rr:.2f})",
            )

    result.metadata.update({
        "original_stop": round(stop, 8),
        "original_rr": round(result.original_rr, 4),
        "final_entry": round(entry, 8),
        "final_stop": round(result.redesigned_stop, 8),
        "final_tp": round(result.final_tp, 8),
        "final_rr": round(result.final_rr, 4),
        "stop_was_redesigned": abs(float(result.redesigned_stop) - float(stop)) > 1e-12,
        "stop_widen_mult": round(result.widen_mult, 4),
        "rr_original": round(result.original_rr, 4),
        "rr_final": round(result.final_rr, 4),
    })
    return result

"""
Position replacement decision logic — scoring and eligibility.

Extracted from executor.py. Pure decision functions that determine:
- Whether a position is eligible for replacement
- Quality scoring for replacement comparisons
- Whether an incoming signal is strong enough to replace

These functions do NOT execute trades — they return decisions that
the Executor acts on.
"""

import os
from datetime import datetime, timezone
from typing import Any, Optional, Tuple

# ── Constants ────────────────────────────────────────────────────────
ENABLE_POSITION_REPLACEMENT = (
    os.getenv("ENABLE_POSITION_REPLACEMENT", "true").lower() == "true"
)
POSITION_REPLACEMENT_MIN_SCORE_DELTA = float(
    os.getenv("POSITION_REPLACEMENT_MIN_SCORE_DELTA", "0.16")
)
MIN_HOLD_TIME_BEFORE_REPLACEMENT_SEC = int(
    os.getenv("MIN_HOLD_TIME_BEFORE_REPLACEMENT_SEC", "120")
)
POSITION_REPLACEMENT_PROTECT_PARTIALED = (
    os.getenv("POSITION_REPLACEMENT_PROTECT_PARTIALED", "true").lower() == "true"
)
POSITION_REPLACEMENT_PROTECT_NEAR_TP_R = float(
    os.getenv("POSITION_REPLACEMENT_PROTECT_NEAR_TP_R", "1.2")
)
POSITION_REPLACEMENT_PROTECT_IN_PROFIT_R = float(
    os.getenv("POSITION_REPLACEMENT_PROTECT_IN_PROFIT_R", "0.5")
)
POSITION_REPLACEMENT_PREFERRED_BONUS = float(
    os.getenv("POSITION_REPLACEMENT_PREFERRED_BONUS", "0.05")
)


def is_capacity_reject_reason(reason: str) -> bool:
    text = str(reason or "").strip().lower()
    return (
        text.startswith("max positions (")
        or text.startswith("intraday max positions (")
        or text.startswith("swing max positions (")
    )


def signal_replacement_quality(
    *,
    total_score: float,
    setup_family: str,
    market_regime: str,
    trend_aligned: bool,
) -> float:
    score = float(total_score)
    if (
        setup_family == "continuation"
        and market_regime == "strong_trend"
        and trend_aligned
    ):
        score += POSITION_REPLACEMENT_PREFERRED_BONUS
    elif setup_family == "continuation" and market_regime == "weak_trend":
        score -= 0.03
    elif setup_family == "reversal":
        score -= 0.03

    if market_regime == "chop":
        score -= 0.10
    return score


def position_replacement_quality(
    *,
    total_score: float,
    setup_family: str,
    market_regime: str,
    trend_aligned: bool,
) -> float:
    score = float(total_score)
    if (
        setup_family == "continuation"
        and market_regime == "strong_trend"
        and trend_aligned
    ):
        score += POSITION_REPLACEMENT_PREFERRED_BONUS
    elif setup_family == "continuation" and market_regime == "weak_trend":
        score -= 0.05
    elif setup_family == "reversal":
        score -= 0.03
    elif setup_family not in {"continuation", "reversal", "swing"}:
        score -= 0.04

    if market_regime == "chop":
        score -= 0.12
    return score


def is_position_protected(
    *,
    pos_coin: str,
    pos_side: str,
    pos_track: str,
    incoming_coin: str,
    restrict_track: str,
    exit_requested: bool,
    opened_at: Optional[datetime],
    partial_closed: bool,
    current_r: Optional[float],
    tp_remaining_r: Optional[float],
) -> Tuple[bool, str]:
    """
    Returns (protected: bool, reason: str).
    If protected=True, this position cannot be replaced.
    """
    if pos_coin == incoming_coin:
        return True, "same_coin"

    if restrict_track and pos_track != restrict_track:
        return True, f"track_mismatch"

    if exit_requested:
        return True, "exit_already_pending"

    if MIN_HOLD_TIME_BEFORE_REPLACEMENT_SEC > 0:
        if opened_at is None:
            return True, "missing_opened_at_for_min_hold"
        age_sec = (datetime.now(timezone.utc) - opened_at).total_seconds()
        if age_sec < float(MIN_HOLD_TIME_BEFORE_REPLACEMENT_SEC):
            return True, f"min_hold_not_met"

    if POSITION_REPLACEMENT_PROTECT_PARTIALED and partial_closed:
        return True, "partial_closed"

    if (
        current_r is not None
        and POSITION_REPLACEMENT_PROTECT_IN_PROFIT_R > 0
        and current_r >= POSITION_REPLACEMENT_PROTECT_IN_PROFIT_R
    ):
        return True, "in_profit"

    if (
        tp_remaining_r is not None
        and POSITION_REPLACEMENT_PROTECT_NEAR_TP_R > 0
        and tp_remaining_r <= POSITION_REPLACEMENT_PROTECT_NEAR_TP_R
    ):
        return True, "near_tp"

    return False, ""


def incoming_beats_weakest(
    incoming_quality: float,
    weakest_quality: float,
) -> bool:
    required = weakest_quality + POSITION_REPLACEMENT_MIN_SCORE_DELTA
    return incoming_quality >= required

"""
Session and market regime filtering — pure decision functions.

Extracted from executor.py. These functions take primitive inputs and return
pass/block decisions. They do not import executor.py or any trading classes.
"""

import os
from typing import Set, Tuple


# ── Session parsing ──────────────────────────────────────────────────
def _parse_session_set(env_name: str, default: str) -> set:
    raw = os.getenv(env_name, default)
    if raw is None:
        return set()
    raw = raw.strip()
    if not raw:
        return set()
    lowered = raw.lower()
    if lowered in {"none", "null", "false", "off", "no"}:
        return set()
    return {s.strip().lower() for s in raw.split(",") if s.strip()}


# ── Session block constants ──────────────────────────────────────────
MAJORS_BLOCKED_SESSIONS = _parse_session_set("MAJORS_BLOCKED_SESSIONS", "dead_zone")
SOL_BETA_BLOCKED_SESSIONS = _parse_session_set("SOL_BETA_BLOCKED_SESSIONS", "dead_zone")
ALT_BETA_BLOCKED_SESSIONS = _parse_session_set("ALT_BETA_BLOCKED_SESSIONS", "dead_zone")
OTHER_BLOCKED_SESSIONS = _parse_session_set("OTHER_BLOCKED_SESSIONS", "dead_zone")
GLOBAL_BLOCKED_SESSIONS = _parse_session_set("BLOCKED_SESSIONS", "")

SOFT_BLOCKED_SESSIONS = _parse_session_set("SOFT_BLOCKED_SESSIONS", "ny_pm")
SESSION_OVERRIDE_MIN_SCORE = float(os.getenv("SESSION_OVERRIDE_MIN_SCORE", "0.80"))
SESSION_OVERRIDE_FAMILIES = {
    s.strip().lower()
    for s in os.getenv("SESSION_OVERRIDE_FAMILIES", "continuation").split(",")
    if s.strip()
}
SWING_SESSION_OVERRIDE = os.getenv("SWING_SESSION_OVERRIDE", "false").lower() == "true"

DEAD_ZONE_SOFT_OVERRIDE_ENABLED = (
    os.getenv("DEAD_ZONE_SOFT_OVERRIDE_ENABLED", "true").lower() == "true"
)
DEAD_ZONE_OVERRIDE_MIN_SCORE = float(os.getenv("DEAD_ZONE_OVERRIDE_MIN_SCORE", "0.84"))
DEAD_ZONE_OVERRIDE_FAMILIES = {
    s.strip().lower()
    for s in os.getenv("DEAD_ZONE_OVERRIDE_FAMILIES", "continuation").split(",")
    if s.strip()
}
DEAD_ZONE_OVERRIDE_REQUIRE_TREND_ALIGN = (
    os.getenv("DEAD_ZONE_OVERRIDE_REQUIRE_TREND_ALIGN", "true").lower() == "true"
)

# ── Market regime / profitability gate constants ─────────────────────
HARD_BLOCKED_COINS: Set[str] = {
    c.strip().upper()
    for c in os.getenv("HARD_BLOCKED_COINS", "").split(",")
    if c.strip()
}
HARD_BLOCK_CONTINUATION = (
    os.getenv("HARD_BLOCK_CONTINUATION", "false").lower() == "true"
)
HARD_BLOCK_CHOP = os.getenv("HARD_BLOCK_CHOP", "false").lower() == "true"
CHOP_REVERSAL_EXCEPTION = os.getenv("CHOP_REVERSAL_EXCEPTION", "false").lower() == "true"
CHOP_REVERSAL_MIN_CONFIDENCE = float(
    os.getenv("CHOP_REVERSAL_MIN_CONFIDENCE", os.getenv("UNIVERSAL_MIN_CONFIDENCE", "0.90"))
)
HARD_BLOCK_UNKNOWN_SESSION = (
    os.getenv("HARD_BLOCK_UNKNOWN_SESSION", "true").lower() == "true"
)
HARD_BLOCKED_TIMEFRAMES: Set[str] = {
    tf.strip().lower()
    for tf in os.getenv("HARD_BLOCKED_TIMEFRAMES", "").split(",")
    if tf.strip()
}
BLOCK_CONTINUATION_IN_CHOP = (
    os.getenv("BLOCK_CONTINUATION_IN_CHOP", "true").lower() == "true"
)
BLOCK_CONTINUATION_IN_WEAK_TREND = (
    os.getenv("BLOCK_CONTINUATION_IN_WEAK_TREND", "false").lower() == "true"
)
BLOCK_REVERSAL_AGAINST_DUAL_TREND = (
    os.getenv("BLOCK_REVERSAL_AGAINST_DUAL_TREND", "true").lower() == "true"
)


# ── Pure decision functions ──────────────────────────────────────────

def get_hard_blocked_sessions(bucket: str) -> set:
    if GLOBAL_BLOCKED_SESSIONS:
        return GLOBAL_BLOCKED_SESSIONS
    if bucket == "majors":
        return MAJORS_BLOCKED_SESSIONS
    if bucket == "sol_beta":
        return SOL_BETA_BLOCKED_SESSIONS
    if bucket == "alt_beta":
        return ALT_BETA_BLOCKED_SESSIONS
    return OTHER_BLOCKED_SESSIONS


def is_trend_aligned(side: str, regime: str) -> bool:
    side = str(side).upper()
    regime = str(regime or "").lower()
    if side == "LONG":
        return "htf_up" in regime and "macro_up" in regime
    return "htf_down" in regime and "macro_down" in regime


def can_override_soft_block(
    *,
    session: str,
    score: float,
    setup_family: str,
    side: str,
    regime: str,
    is_swing_timeframe: bool,
) -> bool:
    if SWING_SESSION_OVERRIDE and is_swing_timeframe:
        return True

    if session == "dead_zone" and DEAD_ZONE_SOFT_OVERRIDE_ENABLED:
        min_score = DEAD_ZONE_OVERRIDE_MIN_SCORE
        families = DEAD_ZONE_OVERRIDE_FAMILIES
        require_trend = DEAD_ZONE_OVERRIDE_REQUIRE_TREND_ALIGN
    elif session in SOFT_BLOCKED_SESSIONS:
        min_score = SESSION_OVERRIDE_MIN_SCORE
        families = SESSION_OVERRIDE_FAMILIES
        require_trend = True
    else:
        return False

    if score < min_score:
        return False

    if setup_family not in families:
        return False

    if require_trend and not is_trend_aligned(side, regime):
        return False

    return True


def evaluate_session_block(
    *,
    session: str,
    bucket: str,
    score: float,
    setup_family: str,
    side: str,
    regime: str,
    is_swing_timeframe: bool,
) -> Tuple[bool, str]:
    """
    Returns (blocked: bool, reason: str).
    If blocked=True, reason explains why.
    """
    if HARD_BLOCK_UNKNOWN_SESSION and session in {"", "unknown", "none", "null"}:
        return True, "session_blocked:unknown"

    hard_blocked = get_hard_blocked_sessions(bucket)
    if session in hard_blocked:
        if not (session == "dead_zone" and DEAD_ZONE_SOFT_OVERRIDE_ENABLED):
            return True, f"session_blocked:{session}"

    session_soft_blocked = (
        (session in SOFT_BLOCKED_SESSIONS)
        or (session == "dead_zone" and DEAD_ZONE_SOFT_OVERRIDE_ENABLED)
    )
    if session_soft_blocked:
        if not can_override_soft_block(
            session=session,
            score=score,
            setup_family=setup_family,
            side=side,
            regime=regime,
            is_swing_timeframe=is_swing_timeframe,
        ):
            return True, f"session_soft_blocked:{session}"

    return False, ""


def evaluate_regime_block(
    *,
    coin: str,
    side: str,
    setup_family: str,
    market_regime: str,
    confidence: float,
    htf_regime: str,
    macro_regime: str,
    timeframe: str,
) -> Tuple[bool, str]:
    """
    Returns (blocked: bool, reason: str) for regime/profitability gates.
    """
    if coin in HARD_BLOCKED_COINS:
        return True, f"hard_blocked_coin:{coin}"

    if timeframe in HARD_BLOCKED_TIMEFRAMES:
        return True, f"hard_blocked_timeframe:{timeframe}"

    if HARD_BLOCK_CONTINUATION and setup_family == "continuation":
        return True, "hard_blocked_setup_family:continuation"

    if HARD_BLOCK_CHOP and market_regime == "chop":
        if (
            CHOP_REVERSAL_EXCEPTION
            and setup_family == "reversal"
            and htf_regime == "up"
            and confidence >= CHOP_REVERSAL_MIN_CONFIDENCE
        ):
            pass  # exception
        else:
            return True, "market_regime_block:chop"

    if (
        BLOCK_CONTINUATION_IN_WEAK_TREND
        and setup_family == "continuation"
        and market_regime == "weak_trend"
    ):
        return True, "market_regime_block:continuation_in_weak_trend"

    if (
        BLOCK_CONTINUATION_IN_CHOP
        and setup_family == "continuation"
        and market_regime == "chop"
    ):
        return True, "market_regime_block:continuation_in_chop"

    if BLOCK_REVERSAL_AGAINST_DUAL_TREND and setup_family == "reversal":
        if (
            (side == "LONG" and htf_regime == "down" and macro_regime == "down")
            or (side == "SHORT" and htf_regime == "up" and macro_regime == "up")
        ):
            return True, f"market_regime_block:reversal_against_dual_trend(htf={htf_regime},macro={macro_regime})"

    return False, ""

"""
Extracted executor sub-modules.

Behavioral contracts:
- telemetry.py: CSV logging for executor rejects (no trade logic)
- execution_policy.py: pure stop/RR/TP-cap geometry shared by replay/live
- stop_redesign.py: stop widening + RR validation telemetry wrapper
- session_filters.py: session/regime blocking decisions (pure functions)
- position_replacement.py: replacement eligibility + quality scoring
"""

from executor_modules.telemetry import (
    EXECUTOR_REJECTS_PATH,
    _EXECUTOR_REJECT_FIELDS,
    _exec_ensure_csv,
    log_executor_reject,
    _stage_missed_context,
    _take_missed_context,
)
from executor_modules.stop_redesign import (
    STOP_ATR_FLOOR_MULT_INTRADAY,
    STOP_ATR_FLOOR_MULT_SWING,
    STOP_BUFFER_ATR_MULT,
    MIN_STOP_ATR_REJECT,
    MIN_STOP_REDESIGN_RR,
    STOP_REDESIGN_RR_TOLERANCE,
    HIGH_CONF_STOP_REDESIGN_MIN_CONFIDENCE,
    HIGH_CONF_STOP_REDESIGN_RR_TOLERANCE,
    HIGH_CONF_STOP_REDESIGN_FAMILIES,
    STOP_REDESIGN_MAX_WIDEN_MULT,
    apply_entry_stop_redesign,
)
from executor_modules.execution_policy import (
    ExecutionPolicyConfig,
    ExecutionPolicyResult,
    evaluate_execution_policy,
    infer_execution_track,
)
from executor_modules.session_filters import (
    GLOBAL_BLOCKED_SESSIONS,
    MAJORS_BLOCKED_SESSIONS,
    SOL_BETA_BLOCKED_SESSIONS,
    ALT_BETA_BLOCKED_SESSIONS,
    OTHER_BLOCKED_SESSIONS,
    SOFT_BLOCKED_SESSIONS,
    SESSION_OVERRIDE_MIN_SCORE,
    SESSION_OVERRIDE_FAMILIES,
    SWING_SESSION_OVERRIDE,
    DEAD_ZONE_SOFT_OVERRIDE_ENABLED,
    DEAD_ZONE_OVERRIDE_MIN_SCORE,
    DEAD_ZONE_OVERRIDE_FAMILIES,
    DEAD_ZONE_OVERRIDE_REQUIRE_TREND_ALIGN,
    HARD_BLOCKED_COINS,
    HARD_BLOCK_CONTINUATION,
    HARD_BLOCK_CHOP,
    CHOP_REVERSAL_EXCEPTION,
    CHOP_REVERSAL_MIN_CONFIDENCE,
    HARD_BLOCK_UNKNOWN_SESSION,
    HARD_BLOCKED_TIMEFRAMES,
    BLOCK_CONTINUATION_IN_CHOP,
    BLOCK_CONTINUATION_IN_WEAK_TREND,
    BLOCK_REVERSAL_IN_WEAK_TREND,
    BLOCK_REVERSAL_AGAINST_DUAL_TREND,
    get_hard_blocked_sessions,
    is_trend_aligned,
    can_override_soft_block,
    evaluate_session_block,
    evaluate_regime_block,
)
from executor_modules.position_replacement import (
    ENABLE_POSITION_REPLACEMENT,
    POSITION_REPLACEMENT_MIN_SCORE_DELTA,
    MIN_HOLD_TIME_BEFORE_REPLACEMENT_SEC,
    POSITION_REPLACEMENT_PROTECT_PARTIALED,
    POSITION_REPLACEMENT_PROTECT_NEAR_TP_R,
    POSITION_REPLACEMENT_PROTECT_IN_PROFIT_R,
    POSITION_REPLACEMENT_PREFERRED_BONUS,
    is_capacity_reject_reason,
    signal_replacement_quality,
    position_replacement_quality,
    is_position_protected,
    incoming_beats_weakest,
)

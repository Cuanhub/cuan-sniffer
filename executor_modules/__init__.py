"""
Extracted executor sub-modules.

Behavioral contracts:
- telemetry.py: CSV logging for executor rejects (no trade logic)
- stop_redesign.py: stop widening + RR validation (pure computation)

These modules are imported by executor.py. They do not import executor.py
(no circular dependencies).
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

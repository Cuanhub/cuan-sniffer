"""
Shared scoring helpers used by score_v3 (archived), score_v3b (shadow-only,
never live), and score_v3c (live).

Extracted 2026-09-01 from score_v3.py so score_v3.py could be archived to
archive/v3_2026_06/ without breaking its importers -- both score_v3c.py
and score_v3b.py depended on it. Functions are pure and side-effect-free.

PREFERRED_SYMBOLS/NEGATIVE_SYMBOLS/NEGATIVE_SESSIONS are score_v3's
ORIGINAL, unzeroed classifications, kept here only so score_v3 (archived)
and score_v3b (shadow-only, not part of this pivot's scope) keep their
exact prior behavior. score_v3c.py does NOT use these -- it zeroed its own
copies of the symbol sets 2026-09-01 (see DECISIONS.md) rather than import
these; that decision does not apply to score_v3b, which is untouched.
"""

from typing import Any, Dict

PREFERRED_SYMBOLS = {"FARTCOIN", "JTO", "SOL", "WIF", "SUI"}
NEGATIVE_SYMBOLS = {"ETH", "ZEC", "BNB"}
NEGATIVE_SESSIONS = {"ny_pm", "london_late", "asia_late"}


def _safe_str(ctx: Dict[str, Any], *keys: str, default: str = "") -> str:
    for key in keys:
        val = ctx.get(key)
        if val is not None:
            try:
                return str(val).strip()
            except Exception:
                pass
    return default


def _safe_bool(ctx: Dict[str, Any], key: str) -> bool:
    val = ctx.get(key)
    if val is None:
        return False
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() == "true"


def _has_trigger(ctx: Dict[str, Any], trigger_name: str) -> bool:
    for key in ("triggers", "trigger", "trigger_type", "setup_trigger", "reason", "reasons"):
        val = ctx.get(key)
        if val and trigger_name.lower() in str(val).lower():
            return True
    if trigger_name == "fvg":
        if _safe_bool(ctx, "fvg_bull") or _safe_bool(ctx, "fvg_bear"):
            return True
        if _safe_bool(ctx, "in_bull_fvg") or _safe_bool(ctx, "in_bear_fvg"):
            return True
    if trigger_name == "ob":
        if _safe_bool(ctx, "ob_bull") or _safe_bool(ctx, "ob_bear"):
            return True
        if _safe_bool(ctx, "in_bull_ob") or _safe_bool(ctx, "in_bear_ob"):
            return True
    if trigger_name == "choch":
        if _safe_bool(ctx, "choch_bull") or _safe_bool(ctx, "choch_bear"):
            return True
    if trigger_name == "sweep":
        if _safe_bool(ctx, "sweep_bull") or _safe_bool(ctx, "sweep_bear"):
            return True
    return False

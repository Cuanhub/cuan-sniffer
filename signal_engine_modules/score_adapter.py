"""
Shadow score adapter — glue between signal_engine and score_v2/v3.

Handles safe import, context construction, failure handling, and result
packaging for all shadow scorers. Never affects live signal gating.
"""

from typing import Any, Dict


def compute_score_v2_for_signal(
    meta: Dict[str, Any],
    coin: str,
    side: str,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Compute shadow score_v2 and return fields to merge into signal meta.
    On failure, returns empty dict (never raises).
    """
    try:
        from score_v2 import compute_shadow_score_v2
        v2_ctx = dict(meta)
        v2_ctx.update({"symbol": coin, "side": side, "coin": coin})
        v2 = compute_shadow_score_v2(v2_ctx)
        return {
            "score_v2": v2["score_v2"],
            "score_v2_version": v2["score_v2_version"],
            "score_v2_tags": ",".join(v2["score_v2_tags"]),
            "score_v2_reason": v2["score_v2_reason"],
        }
    except Exception as e:
        if debug:
            print(f"[SCORE_V2] compute failed: {e}")
        return {}


def compute_score_v3_for_signal(
    meta: Dict[str, Any],
    coin: str,
    side: str,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Compute shadow score_v3 and return fields to merge into signal meta.
    On failure, returns empty dict (never raises).
    """
    try:
        from score_v3 import compute_shadow_score_v3
        v3_ctx = dict(meta)
        v3_ctx.update({"symbol": coin, "side": side, "coin": coin})
        v3 = compute_shadow_score_v3(v3_ctx)
        return {
            "score_v3": v3["score_v3"],
            "score_v3_version": v3["score_v3_version"],
            "score_v3_tags": ",".join(v3["score_v3_tags"]),
            "score_v3_reason": v3["score_v3_reason"],
        }
    except Exception as e:
        if debug:
            print(f"[SCORE_V3] compute failed: {e}")
        return {}


def compute_all_shadow_scores(
    meta: Dict[str, Any],
    coin: str,
    side: str,
    debug: bool = False,
) -> Dict[str, Any]:
    """Compute v2 + v3 and return merged fields for signal meta."""
    fields = {}
    fields.update(compute_score_v2_for_signal(meta, coin, side, debug))
    fields.update(compute_score_v3_for_signal(meta, coin, side, debug))
    return fields

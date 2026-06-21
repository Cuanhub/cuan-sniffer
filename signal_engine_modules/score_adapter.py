"""
Score v2 adapter — glue between signal_engine and score_v2.

This module handles:
  - Safe import of score_v2
  - Context construction for the shadow scorer
  - Failure handling (never crashes the engine)
  - Result packaging into signal meta

Does NOT modify score_v2.py or its weights.
Does NOT affect live signal gating.
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

    Returns a dict with keys: score_v2, score_v2_version, score_v2_tags, score_v2_reason.
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

"""
Shadow score v2 — factor-based candidate quality model.

Research-only. Does NOT affect live execution. Logged alongside score_v1
for comparative evaluation. Based on factor decomposition of 249 replayed
signals (2026-06-16 to 2026-06-20, post-candle-fix).

Factor weights are derived from standalone expectancy deltas, not from
curve-fitting or ML. Each factor's contribution is interpretable and
auditable from the ranked factor table.

Usage:
    from score_v2 import compute_shadow_score_v2
    result = compute_shadow_score_v2(signal_context)
    # result["score_v2"] → float in [0.0, 1.0]
    # result["score_v2_tags"] → list of factor tags that fired
"""

from typing import Any, Dict, List

VERSION = "v2_2026_06_factor_shadow"

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


def _safe_float(ctx: Dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        val = ctx.get(key)
        if val is not None:
            try:
                import pandas as pd
                if pd.isna(val):
                    continue
            except Exception:
                pass
            try:
                return float(val)
            except (TypeError, ValueError):
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
    if trigger_name == "bos":
        if _safe_bool(ctx, "bos_bull") or _safe_bool(ctx, "bos_bear"):
            return True
    return False


def compute_shadow_score_v2(signal_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute a shadow quality score from signal context.

    Returns a dict with score_v2, tags, reason, and version.
    Never mutates the input. Never affects live execution.
    """
    ctx = signal_context or {}
    tags: List[str] = []
    score = 0.50

    # ── Extract features ─────────────────────────────────────────
    symbol = _safe_str(ctx, "symbol", "coin").upper()
    side = _safe_str(ctx, "side").upper()
    setup_family = _safe_str(ctx, "setup_family", "regime_local").lower()
    session = _safe_str(ctx, "session").lower()
    market_regime = _safe_str(ctx, "market_regime").lower()
    htf_regime = _safe_str(ctx, "htf_regime", "regime_htf_1h").lower()
    macro_regime = _safe_str(ctx, "macro_regime", "regime_macro_4h").lower()
    stop_method = _safe_str(ctx, "stop_method", "stop_basis", "stop_source",
                            "structural_stop_type").lower()
    score_v1 = _safe_float(ctx, "score", "total_score", "confidence")
    timeframe = _safe_str(ctx, "timeframe").lower()

    has_fvg = _has_trigger(ctx, "fvg")
    has_ob = _has_trigger(ctx, "ob")
    has_choch = _has_trigger(ctx, "choch")
    has_sweep = _has_trigger(ctx, "sweep")
    has_bos = _has_trigger(ctx, "bos")
    ob_stop = "ob" in stop_method

    missing = []
    if not symbol:
        missing.append("missing_symbol")
    if not setup_family:
        missing.append("missing_setup_family")
    if not session:
        missing.append("missing_session")
    if not market_regime:
        missing.append("missing_market_regime")
    if not macro_regime:
        missing.append("missing_macro_regime")
    tags.extend(missing)

    # ── Positive factors ─────────────────────────────────────────
    if setup_family == "continuation":
        score += 0.12
        tags.append("+continuation")

    if has_fvg:
        score += 0.15
        tags.append("+fvg")

    if ob_stop:
        score += 0.10
        tags.append("+ob_stop")

    if session == "ny_open":
        score += 0.12
        tags.append("+ny_open")

    if macro_regime == "chop":
        score += 0.08
        tags.append("+macro_chop")

    if (market_regime == "chop"
            and setup_family == "continuation"
            and has_fvg):
        score += 0.06
        tags.append("+cont_fvg_mkt_chop")

    if symbol in PREFERRED_SYMBOLS:
        score += 0.05
        tags.append("+preferred_symbol")

    # ── Negative factors ─────────────────────────────────────────
    if setup_family == "reversal":
        score -= 0.20
        tags.append("-reversal")

    if has_choch:
        score -= 0.15
        tags.append("-choch")

    if has_sweep and not has_fvg:
        score -= 0.12
        tags.append("-sweep_no_fvg")

    if session in NEGATIVE_SESSIONS:
        score -= 0.12
        tags.append(f"-session_{session}")

    if symbol in NEGATIVE_SYMBOLS:
        score -= 0.10
        tags.append("-negative_symbol")

    if macro_regime == "up" and setup_family == "continuation":
        score -= 0.08
        tags.append("-macro_up_cont")

    if market_regime == "strong_trend":
        score -= 0.08
        tags.append("-strong_trend")

    if score_v1 >= 0.90 and not has_fvg:
        score -= 0.10
        tags.append("-high_v1_no_fvg")

    # ── Synergy bonuses ──────────────────────────────────────────
    if setup_family == "continuation" and has_fvg and session == "ny_open":
        score += 0.10
        tags.append("+synergy_cont_fvg_ny")

    if setup_family == "continuation" and has_fvg and ob_stop:
        score += 0.08
        tags.append("+synergy_cont_fvg_ob")

    if setup_family == "continuation" and has_fvg and symbol in PREFERRED_SYMBOLS:
        score += 0.06
        tags.append("+synergy_cont_fvg_pref_sym")

    # ── Caps / floors ────────────────────────────────────────────
    if setup_family == "reversal":
        reversal_exception = has_fvg and session == "ny_open"
        if not reversal_exception and score > 0.70:
            score = 0.70
            tags.append("cap_reversal_070")

    if has_choch and not has_fvg:
        if score > 0.60:
            score = 0.60
            tags.append("cap_choch_no_fvg_060")

    if symbol in NEGATIVE_SYMBOLS:
        if score > 0.75:
            score = 0.75
            tags.append("cap_negative_symbol_075")

    # Final clamp
    score = max(0.0, min(1.0, round(score, 4)))

    reason_parts = [t for t in tags if t.startswith("+") or t.startswith("-")]
    reason = ", ".join(reason_parts) if reason_parts else "base_only"

    return {
        "score_v2": score,
        "score_v2_reason": reason,
        "score_v2_tags": tags,
        "score_v2_version": VERSION,
    }

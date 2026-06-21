"""
Shadow score v3 — expectancy-grounded factor model.

Research-only. Does NOT affect live execution. Derived from post-candle-fix
factor decomposition (249 replayed signals, 2026-06-16 to 2026-06-20).

Key differences from v2:
  - Weights normalized to realized avg_r per factor
  - htf=up and macro=up are NEGATIVE (inverted from v1)
  - OB structural stop is the highest-weighted single factor
  - Symbol quality is a first-class factor (not afterthought)
  - No continuation base bonus (v2 gave +0.12 for continuation;
    v3 treats it as neutral since all signals in sample are continuation)

Usage:
    from score_v3 import compute_shadow_score_v3
    result = compute_shadow_score_v3(signal_context)
"""

from typing import Any, Dict, List

VERSION = "v3_2026_06_factor_shadow"

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
    return False


def compute_shadow_score_v3(signal_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute shadow score v3 from signal context.

    Returns dict with score_v3, tags, reason, version.
    Never mutates input. Never affects live execution.
    """
    ctx = signal_context if signal_context is not None else {}
    tags: List[str] = []
    score = 0.50

    # ── Extract features ─────────────────────────────────────────
    symbol = _safe_str(ctx, "symbol", "coin").upper()
    setup_family = _safe_str(ctx, "setup_family", "regime_local").lower()
    session = _safe_str(ctx, "session").lower()
    market_regime = _safe_str(ctx, "market_regime").lower()
    htf_regime = _safe_str(ctx, "htf_regime", "regime_htf_1h").lower()
    macro_regime = _safe_str(ctx, "macro_regime", "regime_macro_4h").lower()
    stop_method = _safe_str(ctx, "stop_method", "stop_basis", "stop_source").lower()

    has_fvg = _has_trigger(ctx, "fvg")
    has_ob = _has_trigger(ctx, "ob")
    has_choch = _has_trigger(ctx, "choch")
    has_sweep = _has_trigger(ctx, "sweep")
    ob_stop = "ob" in stop_method

    missing = []
    if not symbol:
        missing.append("missing_symbol")
    if not session:
        missing.append("missing_session")
    if not market_regime:
        missing.append("missing_market_regime")
    if not macro_regime:
        missing.append("missing_macro_regime")
    tags.extend(missing)

    # ── STRUCTURE ────────────────────────────────────────────────
    if ob_stop:
        score += 0.14
        tags.append("+ob_stop")

    if has_fvg:
        score += 0.08
        tags.append("+fvg")

    if has_ob:
        score += 0.04
        tags.append("+ob")

    if has_choch:
        score -= 0.12
        tags.append("-choch")

    if has_sweep and not has_fvg:
        score -= 0.10
        tags.append("-sweep_no_fvg")

    # ── CONTEXT ──────────────────────────────────────────────────
    if macro_regime == "chop":
        score += 0.10
        tags.append("+macro_chop")

    if htf_regime == "down":
        score += 0.06
        tags.append("+htf_down")

    if market_regime == "chop":
        score += 0.05
        tags.append("+mkt_chop")

    if macro_regime == "up":
        score -= 0.06
        tags.append("-macro_up")

    if htf_regime == "up":
        score -= 0.04
        tags.append("-htf_up")

    if market_regime == "strong_trend":
        score -= 0.05
        tags.append("-strong_trend")

    # ── SESSION ──────────────────────────────────────────────────
    if session == "ny_open":
        score += 0.15
        tags.append("+ny_open")

    if session == "asia_open":
        score += 0.05
        tags.append("+asia_open")

    if session == "ny_pm":
        score -= 0.10
        tags.append("-ny_pm")

    if session == "london_late":
        score -= 0.10
        tags.append("-london_late")

    if session == "asia_late":
        score -= 0.08
        tags.append("-asia_late")

    # ── SYMBOL ───────────────────────────────────────────────────
    if symbol in PREFERRED_SYMBOLS:
        score += 0.06
        tags.append("+preferred_symbol")

    if symbol in NEGATIVE_SYMBOLS:
        score -= 0.15
        tags.append("-negative_symbol")

    # ── CAPS ─────────────────────────────────────────────────────
    if symbol in NEGATIVE_SYMBOLS and score > 0.65:
        score = 0.65
        tags.append("cap_negative_sym_065")

    if has_choch and not has_fvg and score > 0.55:
        score = 0.55
        tags.append("cap_choch_no_fvg_055")

    if has_sweep and not has_fvg and score > 0.60:
        score = 0.60
        tags.append("cap_sweep_no_fvg_060")

    # ── CLAMP ────────────────────────────────────────────────────
    score = max(0.0, min(1.0, round(score, 4)))

    # ── RECIPE TAG ───────────────────────────────────────────────
    is_full_recipe = (
        setup_family == "continuation"
        and has_fvg
        and score >= 0.70
        and symbol in PREFERRED_SYMBOLS
        and session not in NEGATIVE_SESSIONS
        and symbol not in NEGATIVE_SYMBOLS
    )
    if is_full_recipe:
        tags.append("v3_full_recipe")

    # ── BUILD RESULT ─────────────────────────────────────────────
    reason_parts = [t for t in tags if t.startswith("+") or t.startswith("-")]
    reason = ", ".join(reason_parts) if reason_parts else "base_only"

    return {
        "score_v3": score,
        "score_v3_reason": reason,
        "score_v3_tags": tags,
        "score_v3_version": VERSION,
    }

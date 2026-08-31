"""
Shadow score v3b — regime-scoring redesign of score_v3's CONTEXT block.

Research-only. Does NOT affect live execution — logged alongside score_v3
for out-of-sample validation before any promotion decision. See
score_v3.py's own docstring for why that phrase needs to be taken
seriously here: this model MUST clear a genuine forward-testing bar before
it ever gates a live trade, not just a good backtest on the data it was
designed from.

Why this exists (2026-08-25): score_v3's CONTEXT block penalizes
htf_regime/macro_regime=="up" and rewards "down" unconditionally, with no
side-awareness and no symmetric "macro_regime==down" bonus at all. That
means the score depresses uniformly as a market trends up — the exact
opposite of what a trader anticipating a bull market would want, and not
even a coherent trend-following or mean-reversion thesis, just an
asymmetric artifact of the small sample (249 signals, 5 days, all
continuation) score_v3 was originally derived from.

Replaced with two data-grounded changes (see project memory for the full
replay analysis this was derived from — 352 resolved historical V3-eligible
candidates, replayed against real Hyperliquid candle data):

  1. market_regime becomes the primary context signal instead of
     htf/macro up/down. Empirically: strong_trend is the best regime for
     BOTH setup families (continuation PF=1.87 n=23, reversal PF=2.50 n=9,
     consistent across sides where sample allows); chop is the worst
     regime for continuation specifically (PF=0.06 n=15); weak_trend is
     negative for both (already hard-blocked at the executor level via
     BLOCK_CONTINUATION_IN_WEAK_TREND/BLOCK_REVERSAL_IN_WEAK_TREND — this
     term just keeps the score internally consistent with that reality
     instead of contradicting it).

  2. A reversal-alignment penalty, side-aware (score_v3 never reads
     `side` despite having it available): a reversal trigger already
     confirmed by both htf and macro regime in the trade's own direction
     is empirically a late/lagging signal, not a genuine reversal —
     LONG aligned-reversal PF=0.53 (n=28), SHORT aligned-reversal PF=0.66
     (n=113), consistent direction both sides.

  Continuation-vs-regime alignment is deliberately left UNSCORED: LONG
  (n=20, PF=1.33) vs SHORT (n=94, PF=0.66) was inconsistent enough that
  asserting a directional bonus there would fit noise, not signal. That's
  the "no bias" call — absence of a confident signal means don't invent
  one.

In-sample replay comparison (same 1,928 candidates, same replay mechanism
as score_v3): overall PF 0.79->1.05, up-regime PF 0.62->0.98, LONG PF
0.90->1.12, SHORT PF 0.76->0.99. Encouraging, but this was tested on the
data it was designed from — that is exactly why it's shadow-only for now,
not a live cutover.

Usage:
    from score_v3b import compute_shadow_score_v3b
    result = compute_shadow_score_v3b(signal_context)
"""

from typing import Any, Dict, List

from score_v3 import (
    _safe_str,
    _has_trigger,
    PREFERRED_SYMBOLS,
    NEGATIVE_SYMBOLS,
    NEGATIVE_SESSIONS,
)

VERSION = "v3b_2026_08_25_regime_redesign"


def compute_shadow_score_v3b(signal_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute shadow score v3b from signal context.

    Returns dict with score_v3b, tags, reason, version.
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
    side = _safe_str(ctx, "side").upper()

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

    # ── STRUCTURE (unchanged from score_v3) ────────────────────────
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

    # ── CONTEXT (redesigned — see module docstring) ────────────────
    if market_regime == "strong_trend":
        score += 0.08
        tags.append("+mkt_strong_trend")
    elif market_regime == "chop":
        if setup_family == "continuation":
            score -= 0.12
            tags.append("-mkt_chop_continuation")
        else:
            score -= 0.04
            tags.append("-mkt_chop")
    elif market_regime == "weak_trend":
        score -= 0.03
        tags.append("-mkt_weak_trend")

    if setup_family == "reversal" and side in ("LONG", "SHORT"):
        up_votes = sum(1 for r in (htf_regime, macro_regime) if r == "up")
        down_votes = sum(1 for r in (htf_regime, macro_regime) if r == "down")
        aligned = (
            (side == "LONG" and up_votes > down_votes)
            or (side == "SHORT" and down_votes > up_votes)
        )
        if aligned:
            score -= 0.07
            tags.append("-reversal_late_aligned")

    # ── SESSION (unchanged from score_v3) ──────────────────────────
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

    # ── SYMBOL (unchanged from score_v3) ───────────────────────────
    if symbol in PREFERRED_SYMBOLS:
        score += 0.06
        tags.append("+preferred_symbol")

    if symbol in NEGATIVE_SYMBOLS:
        score -= 0.15
        tags.append("-negative_symbol")

    # ── CAPS (unchanged from score_v3) ─────────────────────────────
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

    # ── RECIPE TAG (unchanged definition from score_v3) ────────────
    is_full_recipe = (
        setup_family == "continuation"
        and has_fvg
        and score >= 0.70
        and symbol in PREFERRED_SYMBOLS
        and session not in NEGATIVE_SESSIONS
        and symbol not in NEGATIVE_SYMBOLS
    )
    if is_full_recipe:
        tags.append("v3b_full_recipe")

    # ── BUILD RESULT ─────────────────────────────────────────────
    reason_parts = [t for t in tags if t.startswith("+") or t.startswith("-")]
    reason = ", ".join(reason_parts) if reason_parts else "base_only"

    return {
        "score_v3b": score,
        "score_v3b_reason": reason,
        "score_v3b_tags": tags,
        "score_v3b_version": VERSION,
    }

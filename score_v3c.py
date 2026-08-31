"""
Score v3c — side-conditioned rescore of score_v3's STRUCTURE/CONTEXT
factors. LIVE model when LIVE_ELIGIBILITY_MODEL=v3c.

Why this exists (2026-08-27): score_v3 showed no real predictive edge
across its full score range (503 candidates, 45 days, resolved against
real Hyperliquid price action -- Pearson r = -0.008, PF bounced 0.63-1.10
with no monotonic relationship, and the live-eligible slice score_v3>=0.80
was net-negative: AvgR=-0.080, PF=0.875). Per-factor decomposition (parsing
score_v3_tags against the same resolved outcomes) found the root cause:
score_v3 applies ONE flat weight per regime tag regardless of trade side,
but the real edge is highly side-conditional -- a flat weight cancels out
or inverts opposite-signed real effects instead of capturing them.

Concrete, side-conditioned findings this model encodes (n given per
factor; all n>=59 unless noted):
  - macro_regime=="chop": SHORT PF=2.14 (n=40) vs LONG PF=0.82 (n=23).
    score_v3 rewards both equally (+0.10); should reward SHORT much more,
    LONG barely at all.
  - htf_regime=="down" (the "trend-aligned" case): SHORT PF=0.427 (n=92)
    vs SHORT-not-aligned PF=1.209 (n=95) -- a late-entry/chasing trap.
    score_v3 rewards this for both sides (+0.06); must be flipped negative
    for SHORT. LONG+htf_down is too rare in the data (n=2) to touch.
  - macro_regime=="up": correctly penalized for SHORT (PF 0.734 vs 0.801,
    n=73/114) but backwards for LONG (PF 0.961 vs 0.895 for macro!=up,
    n=293/25) -- the LONG penalty is removed, not flipped to a bonus
    (thin "without" sample for LONG, so no overreach).
  - STRUCTURE: the OB trigger (+ob, not the stop-basis one) is inverted
    outright -- PF=0.79 with vs PF=1.37 without (n=414/91) -- flipped to a
    small penalty. ob_stop, score_v3's single largest weight (+0.14), had
    a near-zero/slightly-negative real edge (PF 0.86 vs 0.90, n=215/290)
    and is cut to +0.02. fvg presence showed no edge either way (0.883 vs
    0.871, n=392/113) and is cut to +0.02.
  - SESSION/SYMBOL/CAPS: left unchanged. ny_open and preferred_symbol
    confirmed correctly signed; choch, strong_trend, london_late confirmed
    correctly signed. ny_pm and asia_late showed inverted signs too
    (PF 1.37/1.16 while currently penalized) but both sessions are also
    hard-blocked at the executor session gate -- fixing the score alone
    has no live effect, so that's left for a dedicated follow-up rather
    than bundled in here.

In-sample validation (same 505 candidates the diagnosis was run on,
recomputing this formula from the same already-resolved outcomes):
Pearson r improved from -0.008 to +0.130, and threshold-sweep PF is
monotonically increasing from >=0.50 (PF=1.24) through >=0.80 (n=7, too
thin to trust) -- unlike score_v3's flat/noisy curve. This is still
in-sample (same window used to derive the weights); LIVE_V3C_ELIGIBILITY_
THRESHOLD should be revisited once genuine forward data accumulates.

Update 2026-08-28 -- trend-confirmed structure (BOS/EQH/EQL): tested
against the manual-trading thesis that LuxAlgo-style structure breaks are
far more reliable with higher-timeframe trend confirmation. Unlike
everything above, this was validated on TWO non-overlapping 75-80 day
windows using raw forward price return (close-to-close, 8h/24h horizons),
independent of the scoring/gating pipeline entirely -- a genuine
out-of-sample test, not an in-sample fit:
  - BOS bull + htf_regime=="up": 24h forward return +1.25% (n=106,
    window 1) and +0.95% (n=127, window 2) -- both 1.5-2.8x the
    unconditional BOS-bull average in their own window. Reproduces
    cleanly on raw forward return. Originally added as a LONG-side
    scoring bonus -- REMOVED 2026-08-29: htf=="up" is unconditionally
    hard-blocked at the live eligibility gate (_v3_eligibility_reject_
    reason) regardless of score, so this bonus could never affect a real
    trade. A full-pipeline replay testing a narrow carve-out (LONG +
    bos_bull + htf=up, real generate_signal(), real outcomes) found the
    combination too rare to validate (n=2 in a 120-day window, n=0 in a
    second non-overlapping 120-day window). The broader question -- would
    reopening the htf=up gate for all LONG signals be profitable under
    v3c -- was also tested directly and does not hold up: window 1
    (n=186) showed PF=1.165, window 2/OOS (n=134) showed PF=0.875
    (net-negative), blended AvgR across both = +0.026R (essentially
    breakeven, driven entirely by one window canceling the other). The
    htf=up gate stays as-is; this factor is documented here rather than
    silently dropped from the record.
  - BOS bear: negative in every regime split, in BOTH windows, even when
    "trend-aligned" (htf=="down"). No positive edge found anywhere --
    deliberately NOT rewarded (absence of a confirmed signal means don't
    invent one, same principle as score_v3b's design).
  - EQH cluster (rolling equal-highs compression, smc_sweeps.py's
    detect_equal_highs_lows): the textbook "equal highs = bearish
    liquidity grab" reading is WRONG in this data -- -0.90% (n=81) and
    -0.91% (n=50) at 24h in windows 1 and 2 respectively, meaning the
    bearish thesis lost both times, consistently and by nearly the same
    magnitude. Added as a small side-conditioned modifier: bearish lean
    penalized, bullish lean rewarded (the opposite of naive SMC reading).
  - EQL cluster: directionally consistent bullish lean both windows but
    unstable magnitude (+0.25% n=52 vs +1.28% n=39) -- added at roughly
    half EQH's weight to reflect the lower confidence.
  - CHoCH aligned-with-existing-trend penalty: an early read of window 1
    alone appeared to independently corroborate score_v3b's
    -reversal_late_aligned logic (WR 27.3% when CHoCH agreed with an
    already-established htf trend). This did NOT reproduce in window 2
    (WR 44.1%, unremarkable) -- retracted. score_v3b's own penalty is
    left untouched since it was derived from a different methodology
    (R-multiple trade replay, not raw forward return) and isn't
    invalidated by this being inconclusive, but this diagnostic does not
    independently strengthen it either.
  - Sweep bull/bear: no reproducible edge in either window -- no action.

Usage:
    from score_v3c import compute_shadow_score_v3c
    result = compute_shadow_score_v3c(signal_context)
"""

from typing import Any, Dict, List

from score_v3 import (
    _safe_str,
    _safe_bool,
    _has_trigger,
    PREFERRED_SYMBOLS,
    NEGATIVE_SYMBOLS,
    NEGATIVE_SESSIONS,
)

VERSION = "v3c_2026_08_29_eqh_eql_only"


def compute_shadow_score_v3c(signal_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute score v3c from signal context.

    Returns dict with score_v3c, tags, reason, version.
    Never mutates input.
    """
    ctx = signal_context if signal_context is not None else {}
    tags: List[str] = []
    score = 0.50

    # ── Extract features ─────────────────────────────────────────
    symbol = _safe_str(ctx, "symbol", "coin").upper()
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
    eq_high = _safe_bool(ctx, "eq_high")
    eq_low = _safe_bool(ctx, "eq_low")

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

    # ── STRUCTURE (reweighted -- see module docstring) ─────────────
    if ob_stop:
        score += 0.02
        tags.append("+ob_stop")

    if has_fvg:
        score += 0.02
        tags.append("+fvg")

    if has_ob:
        score -= 0.05
        tags.append("-ob")

    if has_choch:
        score -= 0.12
        tags.append("-choch")

    if has_sweep and not has_fvg:
        score -= 0.10
        tags.append("-sweep_no_fvg")

    # ── CONTEXT (side-conditioned -- the core redesign) ─────────────
    if side == "SHORT":
        if macro_regime == "chop":
            score += 0.15
            tags.append("+macro_chop_short")
        if htf_regime == "down":
            score -= 0.08
            tags.append("-htf_down_short_aligned")
        if macro_regime == "up":
            score -= 0.08
            tags.append("-macro_up_short")
    else:  # LONG (or unknown side, defaults to the LONG-side treatment)
        if macro_regime == "chop":
            score += 0.02
            tags.append("+macro_chop_long")
        if htf_regime == "down":
            score += 0.06
            tags.append("+htf_down_long")
        # macro_up penalty removed for LONG -- empirically backwards
        # (n=293/25); no penalty and no bonus applied.

    if market_regime == "chop":
        score += 0.05
        tags.append("+mkt_chop")

    if market_regime == "strong_trend":
        score -= 0.05
        tags.append("-strong_trend")

    if htf_regime == "up":
        score -= 0.04
        tags.append("-htf_up")

    # ── TREND-CONFIRMED STRUCTURE (EQH/EQL only -- see module docstring
    # for why a BOS-bull+htf-up bonus was tried and removed 2026-08-29) ──
    if side == "SHORT":
        if eq_high:
            score -= 0.05
            tags.append("-eq_high_short")
        if eq_low:
            score -= 0.03
            tags.append("-eq_low_short")
    else:  # LONG
        if eq_high:
            score += 0.05
            tags.append("+eq_high_long")
        if eq_low:
            score += 0.03
            tags.append("+eq_low_long")

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

    # ── BUILD RESULT ─────────────────────────────────────────────
    reason_parts = [t for t in tags if t.startswith("+") or t.startswith("-")]
    reason = ", ".join(reason_parts) if reason_parts else "base_only"

    return {
        "score_v3c": score,
        "score_v3c_reason": reason,
        "score_v3c_tags": tags,
        "score_v3c_version": VERSION,
    }

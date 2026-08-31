#!/usr/bin/env python3
"""
param_suggester.py — Level 4 self-learning: nightly parameter suggestion engine.

Reads missed_signals.csv to identify gates that are blocking signals with
positive directional edge. Writes suggested_params.json for human review.

IMPORTANT: price_move_r is an at-logging-time directional proxy. It measures
how far price moved toward TP (positive) or SL (negative) at the moment the
rejection was logged — not a full TP/SL simulation. Use tools/research/backtest_missed.py
for full simulation before applying any suggestion to .env.

Usage:
    python3 tools/research/param_suggester.py                   # last 7 days
    python3 tools/research/param_suggester.py --days 14         # last 14 days
    python3 tools/research/param_suggester.py --days 3 --verbose
"""

import os
import json
import argparse
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd

# Load .env so standalone runs use the same thresholds as the live system.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed — os.getenv defaults remain


# ── Paths ─────────────────────────────────────────────────────────────────────
CSV_PATH        = os.getenv("MISSED_SIGNALS_CSV",       "missed_signals.csv")
FILTER_STATE    = os.getenv("STRATEGY_FILTER_STATE_FILE", "strategy_filter_state.json")
OUTPUT_PATH     = os.getenv("PARAM_SUGGESTION_OUTPUT",   "suggested_params.json")

# ── Gate thresholds (read from env so they match the live system) ──────────────
SWING_MIN_CONFIDENCE        = float(os.getenv("SWING_MIN_CONFIDENCE",          "0.90"))
REVERSAL_CHOP_MIN_SCORE     = float(os.getenv("REVERSAL_CHOP_MIN_SCORE",       "0.78"))
WEAK_TREND_MIN_CONFIDENCE   = float(os.getenv("WEAK_TREND_MIN_CONFIDENCE",     "0.80"))

# ── Suggester calibration ─────────────────────────────────────────────────────
ANALYSIS_DAYS       = int(os.getenv("SUGGESTER_ANALYSIS_DAYS",     "7"))
MIN_EVIDENCE_N      = int(os.getenv("SUGGESTER_MIN_EVIDENCE_N",    "8"))
MIN_POSITIVE_RATE   = float(os.getenv("SUGGESTER_MIN_POSITIVE_RATE", "0.50"))
MIN_MEAN_R          = float(os.getenv("SUGGESTER_MIN_MEAN_R",       "0.20"))
SCAN_BAND           = float(os.getenv("SUGGESTER_SCAN_BAND",        "0.08"))  # how far below threshold to scan
MAX_STEP            = float(os.getenv("SUGGESTER_MAX_STEP",         "0.05"))  # largest single threshold move


# ── Data loading ───────────────────────────────────────────────────────────────

def _load_missed(path: str, days: int) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    df = df[df["timestamp"] >= cutoff].copy()
    for col in ["confidence", "total_score", "price_move_r"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["price_move_r", "confidence"])
    return df.reset_index(drop=True)


def _load_filter_state(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


# ── Stats helpers ──────────────────────────────────────────────────────────────

def _stats(df: pd.DataFrame) -> Dict[str, Any]:
    n = len(df)
    if n == 0:
        return {"n": 0, "mean_r": 0.0, "pct_positive": 0.0, "pct_win_proxy": 0.0}
    r = df["price_move_r"]
    return {
        "n": n,
        "mean_r": round(float(r.mean()), 3),
        "pct_positive": round(float((r > 0).mean()), 3),
        "pct_win_proxy": round(float((r >= 0.8).mean()), 3),  # reached ≥80% of TP when logged
    }


def _has_edge(stats: Dict) -> bool:
    return (
        stats["n"] >= MIN_EVIDENCE_N
        and stats["pct_positive"] >= MIN_POSITIVE_RATE
        and stats["mean_r"] >= MIN_MEAN_R
    )


def _find_optimal_threshold(
    df: pd.DataFrame,
    current: float,
    value_col: str = "confidence",
    direction: str = "lower",   # "lower" = scan downward, "raise" scans upward
) -> Tuple[float, Dict]:
    """
    Walk candidate thresholds away from current and return the most aggressive
    threshold where the cumulative subset still shows edge.

    For direction="lower": subset grows as candidate decreases (more signals included).
    We do NOT break on the first failing step — a narrow sub-band at 0.88 may fail
    while the cumulative set at 0.85 passes once enough signals are included.
    We walk all steps and take the furthest candidate that maintained edge.
    """
    best_threshold = current
    best_stats: Dict = {}

    steps = [round(current - s, 3) for s in [0.02, 0.03, 0.04, 0.05, MAX_STEP]] if direction == "lower" \
        else [round(current + s, 3) for s in [0.02, 0.03, 0.04, 0.05, MAX_STEP]]

    for candidate in steps:
        if direction == "lower":
            subset = df[df[value_col] >= candidate]
        else:
            subset = df[df[value_col] <= candidate]
        s = _stats(subset)
        if _has_edge(s):
            best_threshold = candidate
            best_stats = s
        # No break — walk all steps; subset grows so later steps may regain edge

    return best_threshold, best_stats


# ── Gate-specific analysis ─────────────────────────────────────────────────────

def analyze_swing_conf_gate(df: pd.DataFrame) -> Dict[str, Any]:
    """Signals blocked by SWING_MIN_CONFIDENCE."""
    blocked = df[df["reject_reason"].str.startswith("swing_conf_gate", na=False)].copy()
    if blocked.empty:
        return {"checked": 0, "suggestion": None, "note": "no swing_conf_gate rejections in window"}

    # Band immediately below the current floor
    band = blocked[
        (blocked["confidence"] >= SWING_MIN_CONFIDENCE - SCAN_BAND) &
        (blocked["confidence"] < SWING_MIN_CONFIDENCE)
    ]
    band_stats = _stats(band)

    suggestion = None
    if _has_edge(band_stats):
        new_floor, evidence = _find_optimal_threshold(blocked, SWING_MIN_CONFIDENCE, "confidence", "lower")
        if new_floor < SWING_MIN_CONFIDENCE:
            suggestion = {
                "param": "SWING_MIN_CONFIDENCE",
                "current": SWING_MIN_CONFIDENCE,
                "suggested": new_floor,
                "evidence": evidence or band_stats,
                "warning": "All 1H/4H signals are swing-classified. Lower this only with strong evidence.",
            }

    return {
        "checked": len(blocked),
        "scan_band": band_stats,
        "suggestion": suggestion,
    }


def analyze_reversal_score_gate(df: pd.DataFrame) -> Dict[str, Any]:
    """Signals blocked by blocked_low_quality_reversal_in_chop."""
    blocked = df[df["reject_reason"] == "blocked_low_quality_reversal_in_chop"].copy()
    if blocked.empty:
        return {"checked": 0, "suggestion": None, "note": "no reversal score-gate rejections in window"}

    band = blocked[
        (blocked["total_score"] >= REVERSAL_CHOP_MIN_SCORE - SCAN_BAND) &
        (blocked["total_score"] < REVERSAL_CHOP_MIN_SCORE)
    ]
    band_stats = _stats(band)

    suggestion = None
    if _has_edge(band_stats):
        new_floor, evidence = _find_optimal_threshold(blocked, REVERSAL_CHOP_MIN_SCORE, "total_score", "lower")
        if new_floor < REVERSAL_CHOP_MIN_SCORE:
            suggestion = {
                "param": "REVERSAL_CHOP_MIN_SCORE",
                "current": REVERSAL_CHOP_MIN_SCORE,
                "suggested": new_floor,
                "evidence": evidence or band_stats,
                "warning": None,
            }

    return {
        "checked": len(blocked),
        "scan_band": band_stats,
        "suggestion": suggestion,
    }


def analyze_continuation_gate(df: pd.DataFrame) -> Dict[str, Any]:
    """Check if continuation signals show enough live edge to reconsider the hard block."""
    blocked = df[df["reject_reason"].str.contains("continuation", case=False, na=False)].copy()
    if blocked.empty:
        return {"checked": 0, "suggestion": None, "note": "no continuation rejections in window"}

    all_stats = _stats(blocked)

    suggestion = None
    if _has_edge(all_stats):
        suggestion = {
            "param": "HARD_BLOCK_CONTINUATION",
            "current": "true",
            "suggested": "false (with score gate — use WEAK_CONTINUATION_MIN_SCORE)",
            "evidence": all_stats,
            "warning": "Continuation was −19.91R lifetime. Run tools/research/backtest_missed.py before changing.",
        }

    return {
        "checked": len(blocked),
        "all_stats": all_stats,
        "suggestion": suggestion,
    }


# ── Per-coin analysis ──────────────────────────────────────────────────────────

def analyze_per_coin(df: pd.DataFrame, verbose: bool = False) -> Dict[str, Any]:
    """Per-coin missed-signal stats + adaptive floor suggestion."""
    results = {}
    for coin in sorted(df["coin"].unique()):
        coin_df = df[df["coin"] == coin]
        s = _stats(coin_df)
        by_reason = (
            coin_df.groupby("reject_reason")["price_move_r"]
            .agg(n="count", mean_r="mean")
            .round(3)
            .to_dict(orient="index")
        )
        results[coin] = {
            "total_missed": s,
            "by_reason": by_reason if verbose else {},
        }
    return results


# ── Regime heatmap ─────────────────────────────────────────────────────────────

def regime_heatmap(df: pd.DataFrame) -> Dict[str, Any]:
    """Cross-tab of mean price_move_r by (regime, reject_reason)."""
    if "regime" not in df.columns:
        return {}
    pivot = (
        df.groupby(["regime", "reject_reason"])["price_move_r"]
        .agg(n="count", mean_r="mean")
        .reset_index()
        .query("n >= 5")
        .round({"mean_r": 3})
        .to_dict(orient="records")
    )
    return {"rows": pivot}


# ── Report ─────────────────────────────────────────────────────────────────────

def _print_report(data: Dict, verbose: bool):
    suggestions = data["suggestions"]
    print(f"\n{'='*64}")
    print(f"PARAM SUGGESTER — {data['generated_at'][:19]} UTC")
    print(f"Window: last {data['analysis_period_days']}d  |  "
          f"Missed signals: {data['n_total_missed']}")
    print(f"{'='*64}")

    print(f"\n── GATE ANALYSIS ──────────────────────────────────────────")
    for gate, result in data["gate_analysis"].items():
        n = result.get("checked", 0)
        s = result.get("suggestion")
        note = result.get("note", "")
        print(f"\n  {gate}  (n={n})")
        if note:
            print(f"    {note}")
        if s:
            print(f"    ✦ SUGGESTION: {s['param']} {s['current']} → {s['suggested']}")
            ev = s["evidence"]
            print(f"      evidence: n={ev['n']} mean_r={ev['mean_r']:+.3f} pct_pos={ev['pct_positive']:.0%}")
            if s.get("warning"):
                print(f"      ⚠  {s['warning']}")
        else:
            band = result.get("scan_band") or result.get("all_stats") or result.get("stats")
            if band and band.get("n", 0) >= 3:
                # Distinguish "edge exists but too thin for a specific threshold" from "no edge"
                min_n = int(os.getenv("SUGGESTER_MIN_EVIDENCE_N", "8"))
                has_directional = (
                    band["pct_positive"] >= MIN_POSITIVE_RATE
                    and band["mean_r"] >= MIN_MEAN_R
                )
                if has_directional and band["n"] < min_n:
                    label = f"directional signal (n={band['n']} < min={min_n} for threshold rec)"
                elif has_directional:
                    label = "edge in band — threshold walk found no stable cut"
                else:
                    label = "no edge"
                print(f"    scan_band: n={band['n']} mean_r={band['mean_r']:+.3f} "
                      f"pct_pos={band['pct_positive']:.0%} — {label}")

    print(f"\n── PER-COIN MISSED SIGNAL PROXY ────────────────────────────")
    header = f"  {'Coin':<10} {'n':>4}  {'mean_r':>7}  {'pct_pos':>7}  {'win_proxy':>9}"
    print(header)
    for coin, cd in sorted(data["per_coin"].items(),
                           key=lambda x: -x[1]["total_missed"]["mean_r"]):
        s = cd["total_missed"]
        if s["n"] < 3:
            continue
        flag = "  ← edge" if s["pct_positive"] >= MIN_POSITIVE_RATE and s["mean_r"] >= MIN_MEAN_R else ""
        print(f"  {coin:<10} {s['n']:>4}  {s['mean_r']:>+7.3f}  "
              f"{s['pct_positive']:>6.0%}  {s['pct_win_proxy']:>8.0%}{flag}")

    if suggestions:
        print(f"\n── SUMMARY: {len(suggestions)} SUGGESTION(S) ─────────────────────────")
        for s in suggestions:
            print(f"  {s['param']:36s}  {s['current']} → {s['suggested']}")
    else:
        print(f"\n── SUMMARY: no parameter changes suggested ──────────────")

    print(f"\nFull output: {data.get('output_path', OUTPUT_PATH)}")
    print(
        "\nNOTE: price_move_r is a directional proxy at logging time — not a full "
        "TP/SL simulation.\nRun tools/research/backtest_missed.py before applying any suggestion."
    )


# ── Entry point ────────────────────────────────────────────────────────────────

def run(days: int = ANALYSIS_DAYS, output: str = OUTPUT_PATH, verbose: bool = False):
    print(f"[SUGGESTER] Loading {CSV_PATH} (last {days} days)…")
    try:
        df = _load_missed(CSV_PATH, days)
    except FileNotFoundError:
        print(f"[SUGGESTER] ERROR: {CSV_PATH} not found")
        return
    except Exception as e:
        print(f"[SUGGESTER] ERROR loading CSV: {e}")
        return

    if df.empty:
        print(f"[SUGGESTER] No missed signals in last {days} days — nothing to analyse")
        return

    print(f"[SUGGESTER] {len(df)} signals | {df['coin'].nunique()} coins | "
          f"{df['reject_reason'].nunique()} unique reject reasons")

    swing_conf   = analyze_swing_conf_gate(df)
    rev_score    = analyze_reversal_score_gate(df)
    continuation = analyze_continuation_gate(df)
    per_coin     = analyze_per_coin(df, verbose=verbose)
    heatmap      = regime_heatmap(df) if verbose else {}

    reject_summary = (
        df.groupby("reject_reason")["price_move_r"]
        .agg(n="count", mean_r="mean", pct_positive=lambda x: (x > 0).mean())
        .round(3)
        .sort_values("n", ascending=False)
        .head(20)
        .to_dict(orient="index")
    )

    all_suggestions: List[Dict] = [
        s for s in [
            swing_conf.get("suggestion"),
            rev_score.get("suggestion"),
            continuation.get("suggestion"),
        ]
        if s is not None
    ]

    output_data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "analysis_period_days": days,
        "n_total_missed": len(df),
        "output_path": output,
        "disclaimer": (
            "price_move_r is an at-logging-time directional proxy — positive means price "
            "moved toward TP when the rejection was recorded. Run tools/research/backtest_missed.py for "
            "full TP/SL simulation before applying any suggestion to .env."
        ),
        "gate_thresholds_used": {
            "SWING_MIN_CONFIDENCE": SWING_MIN_CONFIDENCE,
            "REVERSAL_CHOP_MIN_SCORE": REVERSAL_CHOP_MIN_SCORE,
            "WEAK_TREND_MIN_CONFIDENCE": WEAK_TREND_MIN_CONFIDENCE,
        },
        "suggester_calibration": {
            "min_evidence_n": MIN_EVIDENCE_N,
            "min_positive_rate": MIN_POSITIVE_RATE,
            "min_mean_r": MIN_MEAN_R,
            "scan_band": SCAN_BAND,
            "max_step": MAX_STEP,
        },
        "suggestions": all_suggestions,
        "gate_analysis": {
            "swing_conf_gate":      swing_conf,
            "reversal_score_gate":  rev_score,
            "continuation_gate":    continuation,
        },
        "per_coin": per_coin,
        "reject_reason_summary": reject_summary,
        "regime_heatmap": heatmap,
    }

    with open(output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, sort_keys=True)

    _print_report(output_data, verbose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self-learning parameter suggestion engine")
    parser.add_argument("--days",    type=int,  default=ANALYSIS_DAYS, help="Analysis window in days (default: 7)")
    parser.add_argument("--output",             default=OUTPUT_PATH,   help="Output JSON path")
    parser.add_argument("--verbose", action="store_true",              help="Include per-reason breakdown and regime heatmap")
    args = parser.parse_args()
    run(days=args.days, output=args.output, verbose=args.verbose)

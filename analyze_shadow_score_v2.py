#!/usr/bin/env python3
"""
Daily comparison: score_v1 vs score_v2.

Reads shadow_scores.csv and optionally shadow_trades.csv / trades.csv.
Prints bucket distributions, recipe monitoring, and recommendations.

Usage:
    python3 analyze_shadow_score_v2.py
"""

import csv
import os
import sys
from collections import Counter, defaultdict

SHADOW_SCORES_PATH = os.getenv("SHADOW_SCORES_PATH", "shadow_scores.csv")
SHADOW_TRADES_PATH = os.getenv("SHADOW_LOG_PATH", "shadow_trades.csv")
TRADES_PATH = "trades.csv"

PREFERRED_SYMBOLS = {"FARTCOIN", "JTO", "SOL", "WIF", "SUI"}
NEGATIVE_SYMBOLS = {"ETH", "ZEC", "BNB"}
NEGATIVE_SESSIONS = {"ny_pm", "london_late", "asia_late"}


def _load_csv(path):
    if not os.path.exists(path):
        return []
    try:
        with open(path, newline="") as f:
            return list(csv.DictReader(f))
    except Exception as e:
        print(f"  Warning: could not load {path}: {e}")
        return []


def _safe_float(row, *keys, default=0.0):
    for k in keys:
        v = row.get(k, "")
        if v and str(v).strip():
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return default


def _stats(rows, score_key="score_v2"):
    if not rows:
        return None
    n = len(rows)
    scores = [_safe_float(r, score_key) for r in rows]
    return {"n": n, "mean": sum(scores) / n, "min": min(scores), "max": max(scores)}


def main():
    print("=" * 76)
    print("  SHADOW SCORE V2 — DAILY ANALYSIS")
    print("=" * 76)

    # ── Load data ────────────────────────────────────────────────
    shadow = _load_csv(SHADOW_SCORES_PATH)
    trades_shadow = _load_csv(SHADOW_TRADES_PATH)
    trades = _load_csv(TRADES_PATH)

    print(f"\n  Data sources:")
    print(f"    shadow_scores.csv:  {len(shadow):>6} rows")
    print(f"    shadow_trades.csv:  {len(trades_shadow):>6} rows")
    print(f"    trades.csv:         {len(trades):>6} rows")

    if not shadow:
        print("\n  No shadow_scores.csv data yet. Run the system to collect shadow data.")
        print("  Minimum 30 days recommended before drawing conclusions.")
        return

    # ── Score v2 bucket distribution ─────────────────────────────
    print(f"\n{'─'*76}")
    print(f"  SCORE V2 BUCKET DISTRIBUTION")
    print(f"{'─'*76}")

    buckets = [
        (0.00, 0.60, "[0.00-0.60)"),
        (0.60, 0.72, "[0.60-0.72)"),
        (0.72, 0.80, "[0.72-0.80)"),
        (0.80, 0.88, "[0.80-0.88)"),
        (0.88, 0.95, "[0.88-0.95)"),
        (0.95, 1.01, "[0.95-1.00]"),
    ]

    print(f"  {'Bucket':<14s} {'Count':>7s} {'%':>7s} {'Live Accept':>12s}")
    print(f"  {'─'*14} {'─'*7} {'─'*7} {'─'*12}")
    for lo, hi, label in buckets:
        rows = [r for r in shadow if lo <= _safe_float(r, "score_v2") < hi]
        accepted = sum(1 for r in rows if r.get("would_live_execute", "").lower() == "true")
        pct = len(rows) / len(shadow) * 100 if shadow else 0
        print(f"  {label:<14s} {len(rows):>7} {pct:>6.1f}% {accepted:>12}")

    # ── Score v1 vs v2 comparison ────────────────────────────────
    print(f"\n{'─'*76}")
    print(f"  SCORE V1 vs V2 COMPARISON")
    print(f"{'─'*76}")

    v1_scores = [_safe_float(r, "score_v1") for r in shadow if _safe_float(r, "score_v1") > 0]
    v2_scores = [_safe_float(r, "score_v2") for r in shadow if _safe_float(r, "score_v2") > 0]

    if v1_scores:
        print(f"  V1: n={len(v1_scores)}  mean={sum(v1_scores)/len(v1_scores):.3f}  "
              f"min={min(v1_scores):.3f}  max={max(v1_scores):.3f}")
    if v2_scores:
        print(f"  V2: n={len(v2_scores)}  mean={sum(v2_scores)/len(v2_scores):.3f}  "
              f"min={min(v2_scores):.3f}  max={max(v2_scores):.3f}")

    # Correlation between v1 and v2
    paired = [(r, _safe_float(r, "score_v1"), _safe_float(r, "score_v2"))
              for r in shadow if _safe_float(r, "score_v1") > 0 and _safe_float(r, "score_v2") > 0]
    if len(paired) >= 10:
        xs = [p[1] for p in paired]
        ys = [p[2] for p in paired]
        n = len(paired)
        mx, my = sum(xs)/n, sum(ys)/n
        cov = sum((x-mx)*(y-my) for x, y in zip(xs, ys)) / n
        sx = (sum((x-mx)**2 for x in xs)/n)**0.5
        sy = (sum((y-my)**2 for y in ys)/n)**0.5
        corr = cov/(sx*sy) if sx > 0 and sy > 0 else 0
        print(f"  V1↔V2 correlation: {corr:+.3f}  (n={n})")

    # ── Recipe monitor ───────────────────────────────────────────
    print(f"\n{'─'*76}")
    print(f"  RECIPE MONITOR: cont + FVG + preferred + good sessions")
    print(f"{'─'*76}")

    recipe_rows = [
        r for r in shadow
        if r.get("setup_family", "").strip().lower() == "continuation"
        and _safe_float(r, "score_v2") >= 0.72
        and r.get("symbol", "").upper() not in NEGATIVE_SYMBOLS
        and r.get("session", "").lower() not in NEGATIVE_SESSIONS
    ]

    v2_tags_with_fvg = [
        r for r in recipe_rows
        if "fvg" in str(r.get("score_v2_tags", "")).lower()
    ]

    print(f"  Recipe matches (cont + v2>=0.72 + good sym/session): {len(recipe_rows)}")
    print(f"  With FVG tag: {len(v2_tags_with_fvg)}")

    if v2_tags_with_fvg:
        accepted = sum(1 for r in v2_tags_with_fvg
                       if r.get("would_live_execute", "").lower() == "true")
        print(f"  Live-accepted: {accepted}  Live-rejected: {len(v2_tags_with_fvg) - accepted}")

    dates = set(r.get("timestamp", "")[:10] for r in shadow if r.get("timestamp"))
    num_days = max(1, len(dates))

    # ── Recommendation ───────────────────────────────────────────
    print(f"\n{'─'*76}")
    print(f"  RECOMMENDATION")
    print(f"{'─'*76}")

    enough = len(shadow) >= 500 and num_days >= 10
    print(f"  Data days: {num_days}")
    print(f"  Total shadow rows: {len(shadow)}")
    print(f"  Enough samples: {'YES' if enough else 'NO — need 10+ days and 500+ rows'}")
    print(f"  Score v2 better than v1: INSUFFICIENT DATA (need realized outcomes)")
    print(f"  Estimated candidates/day: {len(shadow) / num_days:.1f}")

    if v2_tags_with_fvg:
        print(f"  Recipe candidates/day: {len(v2_tags_with_fvg) / num_days:.1f}")

    print(f"\n  Action: continue collecting shadow data. Re-run after 30 days.")


if __name__ == "__main__":
    main()

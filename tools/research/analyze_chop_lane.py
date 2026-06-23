#!/usr/bin/env python3
"""
Chop lane research analysis + live vs rejected comparison.

Reads shadow_chop_lane.csv (broad) and shadow_chop_exception.csv (narrow),
optionally replays against candle data, and prints cohort breakdowns.

Also compares live execution results against chop-blocked replay outcomes.

Usage:
    python3 tools/research/analyze_chop_lane.py
    python3 tools/research/analyze_chop_lane.py --replay --days 30
"""

import argparse
import csv
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CHOP_LANE_PATH = os.getenv("SHADOW_CHOP_LANE_PATH", str(PROJECT_ROOT / "shadow_chop_lane.csv"))
CHOP_EXCEPTION_PATH = os.getenv("SHADOW_CHOP_EXCEPTION_PATH", str(PROJECT_ROOT / "shadow_chop_exception.csv"))
TRADES_PATH = str(PROJECT_ROOT / "trades.csv")


def _load(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _sf(val, default=0.0):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _stats(rs):
    if not rs:
        return {"n": 0, "wr": 0, "avg_r": 0, "total_r": 0, "pf": 0}
    n = len(rs)
    wins = sum(1 for r in rs if r > 0)
    gw = sum(r for r in rs if r > 0)
    gl = abs(sum(r for r in rs if r < 0))
    return {"n": n, "wr": wins / n * 100, "avg_r": sum(rs) / n, "total_r": sum(rs),
            "pf": gw / gl if gl > 0 else (999 if gw > 0 else 0)}


def _print_table(title, groups, key_fn):
    print(f"\n  {title}:")
    print(f"  {'Key':<15s} │ {'n':>4s} │ {'WR%':>6s} │ {'AvgR':>7s} │ {'TotalR':>8s} │ {'PF':>6s}")
    print(f"  {'─'*15} ┼ {'─'*4} ┼ {'─'*6} ┼ {'─'*7} ┼ {'─'*8} ┼ {'─'*6}")
    buckets = defaultdict(list)
    for row in groups:
        buckets[key_fn(row)].append(_sf(row.get("realized_R", row.get("exit_r", 0))))
    for key in sorted(buckets.keys(), key=lambda k: -sum(buckets[k])):
        s = _stats(buckets[key])
        print(f"  {str(key):<15s} │ {s['n']:>4} │ {s['wr']:>5.1f}% │ {s['avg_r']:>+6.3f} │ {s['total_r']:>+7.2f} │ {s['pf']:>6.3f}")


def main():
    parser = argparse.ArgumentParser(description="Chop lane analysis")
    parser.add_argument("--replay", action="store_true", help="Replay against candle data")
    parser.add_argument("--days", type=int, default=30, help="Lookback days for replay")
    args = parser.parse_args()

    lane = _load(CHOP_LANE_PATH)
    exception = _load(CHOP_EXCEPTION_PATH)
    trades = _load(TRADES_PATH)

    print("=" * 70)
    print("  CHOP LANE RESEARCH REPORT")
    print("=" * 70)
    print(f"\n  Data sources:")
    print(f"    shadow_chop_lane.csv:      {len(lane):>6} rows")
    print(f"    shadow_chop_exception.csv: {len(exception):>6} rows")
    print(f"    trades.csv:                {len(trades):>6} rows")

    if not lane and not exception:
        print("\n  No chop lane data yet. Run the bot to collect candidates.")
        return

    candidates = lane if lane else exception
    dates = set(r.get("timestamp", "")[:10] for r in candidates if r.get("timestamp"))
    num_days = max(1, len(dates))

    print(f"    Days: {num_days}  ({min(dates) if dates else '?'} → {max(dates) if dates else '?'})")

    # ── Overall ──────────────────────────────────────────────────
    # We don't have realized R without replay — show raw counts
    print(f"\n{'─'*70}")
    print(f"  CANDIDATE DISTRIBUTION (shadow_chop_lane)")
    print(f"{'─'*70}")
    print(f"  Total candidates: {len(candidates)}")
    print(f"  Candidates/day:   {len(candidates) / num_days:.1f}")

    _print_table("By symbol", candidates, lambda r: r.get("symbol", "?"))
    _print_table("By session", candidates, lambda r: r.get("session", "?"))
    _print_table("By setup_family", candidates, lambda r: r.get("setup_family", "?"))
    _print_table("By side", candidates, lambda r: r.get("side", "?"))

    def v3_bucket(row):
        v = _sf(row.get("score_v3", 0))
        if v >= 0.80: return ">=0.80"
        if v >= 0.70: return "0.70-0.80"
        if v >= 0.60: return "0.60-0.70"
        return "<0.60"
    _print_table("By score_v3 bucket", candidates, v3_bucket)
    _print_table("By htf_regime", candidates, lambda r: r.get("htf_regime", "?"))
    _print_table("By macro_regime", candidates, lambda r: r.get("macro_regime", "?"))
    _print_table("By date", candidates, lambda r: r.get("timestamp", "")[:10])

    # ── LIVE VS REJECTED ─────────────────────────────────────────
    closed = [t for t in trades if t.get("state") == "closed"]
    if closed:
        print(f"\n{'─'*70}")
        print(f"  LIVE VS CHOP-BLOCKED COMPARISON")
        print(f"{'─'*70}")
        live_r = [_sf(t.get("realized_r", 0)) for t in closed]
        ls = _stats(live_r)
        print(f"\n  Live executed:  n={ls['n']}  WR={ls['wr']:.1f}%  AvgR={ls['avg_r']:+.3f}  TotalR={ls['total_r']:+.2f}  PF={ls['pf']:.3f}")
        print(f"  Chop blocked:   n={len(candidates)} candidates (need replay for realized R)")
        print(f"\n  Live trades:")
        for t in closed:
            print(f"    {t.get('coin','?')} {t.get('side','?')} → {_sf(t.get('realized_r',0)):+.4f}R")

    # ── PROMOTION CRITERIA ────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  PROMOTION CRITERIA")
    print(f"{'─'*70}")

    criteria = {
        "chop lane trades >= 100": len(candidates) >= 100,
        f"data days >= 10 (have {num_days})": num_days >= 10,
        "need replay for PF/avgR/stability": False,
    }
    all_pass = all(criteria.values())
    for c, passed in criteria.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {c}")
    print(f"\n  PROMOTE_CHOP_LANE = {'YES' if all_pass else 'NO'}")
    print(f"\n  Run with --replay to resolve outcomes against candle data.")


if __name__ == "__main__":
    main()

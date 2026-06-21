"""
audit_gate_rejections.py — Gate rejection leaderboard.

Reads gate_rejects.csv and executor_rejects.csv and produces:
  - Top rejection reasons by count and percentage
  - Score distribution analysis (mean, median, gap from threshold)
  - Per-symbol and per-regime breakdowns
  - Architecture duplicate filter detection

Usage:
    python tools/research/audit_gate_rejections.py
    python tools/research/audit_gate_rejections.py --hours 24
    python tools/research/audit_gate_rejections.py --gate-only
    python tools/research/audit_gate_rejections.py --executor-only
"""

import argparse
import csv
import os
import sys
from collections import Counter
from datetime import datetime, timezone, timedelta

GATE_REJECTS_PATH = os.getenv("GATE_REJECTS_PATH", "gate_rejects.csv")
EXECUTOR_REJECTS_PATH = os.getenv("EXECUTOR_REJECTS_PATH", "executor_rejects.csv")
SCORE_DIST_PATH = os.getenv("SCORE_DIST_PATH", "score_distribution.csv")

BAR_WIDTH = 40
SEP = "─" * 70


def _load_csv(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _parse_ts(ts_str: str) -> datetime | None:
    try:
        return datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _filter_recent(rows: list[dict], hours: float | None) -> list[dict]:
    if not hours:
        return rows
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    return [r for r in rows if (_parse_ts(r.get("timestamp", "")) or datetime.min.replace(tzinfo=timezone.utc)) >= cutoff]


def _safe_float(val: str, default: float = 0.0) -> float:
    try:
        return float(val) if val else default
    except (ValueError, TypeError):
        return default


def _bar(count: int, total: int, width: int = BAR_WIDTH) -> str:
    filled = int(round(count / total * width)) if total > 0 else 0
    return "█" * filled + "░" * (width - filled)


def _pct(count: int, total: int) -> str:
    return f"{count / total * 100:.1f}%" if total > 0 else "0.0%"


def _bucket_reason(reason: str) -> str:
    """Normalise score_below_threshold:0.612<0.640 → score_below_threshold."""
    for prefix in (
        "score_below_threshold",
        "rr_too_low",
        "4h_rr_below_min",
        "insufficient_swing_confluence",
        "session_blocked",
        "session_soft_blocked",
        "cooldown",
        "coin_reentry_cooldown",
        "bucket_reentry_cooldown",
        "bucket_dir_limit",
        "stop_redesign_reject",
        "market_regime_block",
        "weak_trend_conf_gate",
        "swing_conf_gate",
        "fill_rejected",
        "risk_check",
        "entry_validate",
        "margin_reject",
    ):
        if reason.startswith(prefix):
            return prefix
    return reason


# ──────────────────────────────────────────────────────────────────────────────

def print_gate_leaderboard(rows: list[dict]) -> None:
    if not rows:
        print(f"\n[gate_rejects.csv] No data found at {GATE_REJECTS_PATH}")
        return

    total = len(rows)
    reason_counter: Counter = Counter()
    for r in rows:
        reason_counter[_bucket_reason(r.get("reject_reason", "unknown"))] += 1

    print(f"\n{'ENGINE GATE REJECTIONS':^70}")
    print(SEP)
    print(f"  Total logged rejects: {total}")
    print(SEP)
    print(f"  {'#':<3} {'Reason':<45} {'Count':>6}  {'%':>6}  Bar")
    print(f"  {'─'*3} {'─'*45} {'─'*6}  {'─'*6}  {'─'*BAR_WIDTH}")
    for rank, (reason, count) in enumerate(reason_counter.most_common(20), 1):
        bar = _bar(count, total)
        print(f"  {rank:<3} {reason:<45} {count:>6}  {_pct(count, total):>6}  {bar}")
    print()

    # Per-regime breakdown
    regime_counter: Counter = Counter()
    for r in rows:
        regime_counter[r.get("market_regime", "unknown")] += 1
    print(f"  {'By market_regime':<45}")
    for regime, count in regime_counter.most_common():
        print(f"    {regime:<30} {count:>5}  {_pct(count, total):>6}")
    print()

    # Per-symbol breakdown (top 10)
    sym_counter: Counter = Counter()
    for r in rows:
        sym_counter[r.get("symbol", "?")] += 1
    print(f"  {'By symbol (top 10)':<45}")
    for sym, count in sym_counter.most_common(10):
        print(f"    {sym:<20} {count:>5}  {_pct(count, total):>6}")
    print()

    # Session breakdown
    sess_counter: Counter = Counter()
    for r in rows:
        sess_counter[r.get("session", "unknown")] += 1
    print(f"  {'By session':<45}")
    for sess, count in sess_counter.most_common():
        print(f"    {sess:<25} {count:>5}  {_pct(count, total):>6}")
    print(SEP)


def print_score_analysis(rows: list[dict]) -> None:
    if not rows:
        print(f"\n[score_distribution.csv] No data found at {SCORE_DIST_PATH}")
        return

    scores = [_safe_float(r.get("score")) for r in rows]
    thresholds = [_safe_float(r.get("threshold")) for r in rows]
    gaps = [s - t for s, t in zip(scores, thresholds)]

    rejected = [r for r in rows if _safe_float(r.get("score")) < _safe_float(r.get("threshold"))]
    passed = [r for r in rows if _safe_float(r.get("score")) >= _safe_float(r.get("threshold"))]

    def _median(vals: list[float]) -> float:
        if not vals:
            return 0.0
        s = sorted(vals)
        n = len(s)
        return (s[n // 2 - 1] + s[n // 2]) / 2 if n % 2 == 0 else s[n // 2]

    def _mean(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    print(f"\n{'SCORE DISTRIBUTION ANALYSIS':^70}")
    print(SEP)
    print(f"  Total candidates logged:  {len(rows)}")
    print(f"  Passed score gate:        {len(passed)}  ({_pct(len(passed), len(rows))})")
    print(f"  Rejected by score gate:   {len(rejected)}  ({_pct(len(rejected), len(rows))})")
    print(SEP)
    print(f"  ALL candidates:")
    print(f"    Mean score:   {_mean(scores):.4f}")
    print(f"    Median score: {_median(scores):.4f}")
    print(f"    Mean gap from threshold: {_mean(gaps):+.4f}")
    print()

    if rejected:
        rej_scores = [_safe_float(r.get("score")) for r in rejected]
        rej_thresholds = [_safe_float(r.get("threshold")) for r in rejected]
        rej_gaps = [s - t for s, t in zip(rej_scores, rej_thresholds)]
        print(f"  REJECTED candidates:")
        print(f"    Mean rejected score:   {_mean(rej_scores):.4f}")
        print(f"    Median rejected score: {_median(rej_scores):.4f}")
        print(f"    Mean gap from threshold: {_mean(rej_gaps):+.4f}  ← KEY METRIC")
        print(f"    Tightest gap (closest miss): {max(rej_gaps):.4f}")
        print(f"    Widest gap (most missing):   {min(rej_gaps):.4f}")
        print()
        print("    Gap distribution (rejected):")
        buckets = [
            ("within 0.02 of threshold (near-miss)", lambda g: g >= -0.02),
            ("0.02–0.05 below threshold",             lambda g: -0.05 <= g < -0.02),
            ("0.05–0.10 below threshold",             lambda g: -0.10 <= g < -0.05),
            ("more than 0.10 below threshold",        lambda g: g < -0.10),
        ]
        for label, fn in buckets:
            count = sum(1 for g in rej_gaps if fn(g))
            print(f"      {label:<40} {count:>5}  {_pct(count, len(rej_gaps)):>6}")
        print()
        print("    ► If >50% are near-miss (within 0.02), the threshold is too tight.")
        print("    ► If <20% are near-miss, regime/trigger quality is the primary issue.")
    print(SEP)


def print_executor_leaderboard(rows: list[dict]) -> None:
    if not rows:
        print(f"\n[executor_rejects.csv] No data found at {EXECUTOR_REJECTS_PATH}")
        return

    total = len(rows)
    reason_counter: Counter = Counter()
    for r in rows:
        reason_counter[_bucket_reason(r.get("reject_reason", "unknown"))] += 1

    print(f"\n{'EXECUTOR GATE REJECTIONS':^70}")
    print(SEP)
    print(f"  Total logged rejects: {total}")
    print(SEP)
    print(f"  {'#':<3} {'Reason':<45} {'Count':>6}  {'%':>6}  Bar")
    print(f"  {'─'*3} {'─'*45} {'─'*6}  {'─'*6}  {'─'*BAR_WIDTH}")
    for rank, (reason, count) in enumerate(reason_counter.most_common(20), 1):
        bar = _bar(count, total)
        print(f"  {rank:<3} {reason:<45} {count:>6}  {_pct(count, total):>6}  {bar}")
    print()

    # Confidence distribution of executor rejects
    confs = [_safe_float(r.get("confidence")) for r in rows if r.get("confidence")]
    if confs:
        conf_buckets = [
            ("<0.80",      lambda c: c < 0.80),
            ("0.80–0.84",  lambda c: 0.80 <= c < 0.85),
            ("0.85–0.89",  lambda c: 0.85 <= c < 0.90),
            ("0.90–0.91",  lambda c: 0.90 <= c < 0.92),
            ("0.92–0.94",  lambda c: 0.92 <= c < 0.95),
            ("0.95",       lambda c: c >= 0.95),
        ]
        print(f"  Confidence of executor-rejected signals:")
        for label, fn in conf_buckets:
            count = sum(1 for c in confs if fn(c))
            print(f"    {label:<15} {count:>5}  {_pct(count, len(confs)):>6}")
        print()
        print("  ► Signals in 0.90+ bucket rejected by executor reveal the double-filter.")
    print(SEP)


def print_duplicate_filter_analysis(gate_rows: list[dict], exec_rows: list[dict]) -> None:
    """Identify signals that cleared the engine score gate but were rejected by executor."""
    print(f"\n{'DUPLICATE FILTER ANALYSIS':^70}")
    print(SEP)

    # Engine stats
    engine_total = len(gate_rows)
    engine_score_rejects = sum(
        1 for r in gate_rows if r.get("reject_reason", "").startswith("score_below_threshold")
    )
    engine_other_rejects = engine_total - engine_score_rejects

    # Executor stats — signals that made it through engine but rejected at executor
    exec_total = len(exec_rows)
    exec_conf_rejects = sum(
        1 for r in exec_rows if "conf" in r.get("reject_reason", "").lower()
    )

    print(f"  Engine rejects (all gates):        {engine_total}")
    print(f"    → score_below_threshold:          {engine_score_rejects}")
    print(f"    → other engine gates:             {engine_other_rejects}")
    print()
    print(f"  Executor rejects:                  {exec_total}")
    print(f"    → confidence-related:             {exec_conf_rejects}")
    print()
    print("  ARCHITECTURE:")
    print("  ┌──────────────────────────────────────────────────────┐")
    print("  │  Engine score gate (threshold=0.64)                  │")
    print("  │    confidence = clamp(score, 0.50, 0.95)             │")
    print("  │    Signal emitted if score >= 0.64                   │")
    print("  ├──────────────────────────────────────────────────────┤")
    print("  │  Executor confidence gate (UNIVERSAL_MIN_CONF=0.90)  │")
    print("  │    Signal accepted if confidence >= 0.90             │")
    print("  │    i.e. raw_score >= 0.90 (since conf == score here) │")
    print("  └──────────────────────────────────────────────────────┘")
    print()
    print("  POTENTIAL DOUBLE-FILTER:")
    print("  Engine passes signals with score in [0.64, 0.90).")
    print("  Executor REJECTS any of those with confidence < 0.90.")
    print("  These 'passing' engine signals are dead on arrival.")
    print()
    print("  KEY QUESTION: Are most engine score-passing signals in [0.64, 0.90)?")
    print("  If yes → the engine score gate is generating useless work.")
    print("  Fix: raise ENGINE score gate to match UNIVERSAL_MIN_CONFIDENCE,")
    print("       or lower UNIVERSAL_MIN_CONFIDENCE to match the engine gate.")
    print()
    if exec_total > 0 and exec_conf_rejects > 0:
        pct = exec_conf_rejects / exec_total * 100
        print(f"  → {exec_conf_rejects}/{exec_total} executor rejects ({pct:.1f}%) were confidence-related.")
        if pct > 40:
            print("  ► [WARNING] >40% of executor rejects are confidence gates.")
            print("    The engine is emitting signals that can never execute.")
            print("    Consider: signal_engine.py score gate → 0.90 to match executor.")
    print(SEP)


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate rejection leaderboard")
    parser.add_argument("--hours", type=float, default=None,
                        help="Only include data from the last N hours")
    parser.add_argument("--gate-only", action="store_true",
                        help="Only show engine gate rejections")
    parser.add_argument("--executor-only", action="store_true",
                        help="Only show executor rejections")
    args = parser.parse_args()

    gate_rows = _filter_recent(_load_csv(GATE_REJECTS_PATH), args.hours)
    exec_rows = _filter_recent(_load_csv(EXECUTOR_REJECTS_PATH), args.hours)
    score_rows = _filter_recent(_load_csv(SCORE_DIST_PATH), args.hours)

    time_label = f"last {args.hours:.0f}h" if args.hours else "all time"
    print(f"\n{'CUAN SNIFFER — GATE REJECTION AUDIT':^70}")
    print(f"{'(' + time_label + ')':^70}")
    print(SEP)

    if not args.executor_only:
        print_gate_leaderboard(gate_rows)
        print_score_analysis(score_rows)

    if not args.gate_only:
        print_executor_leaderboard(exec_rows)

    if not args.gate_only and not args.executor_only:
        print_duplicate_filter_analysis(gate_rows, exec_rows)

    print(f"\n  Files read:")
    print(f"    {GATE_REJECTS_PATH}    ({len(gate_rows)} rows in window)")
    print(f"    {EXECUTOR_REJECTS_PATH} ({len(exec_rows)} rows in window)")
    print(f"    {SCORE_DIST_PATH}  ({len(score_rows)} rows in window)")
    print()


if __name__ == "__main__":
    main()

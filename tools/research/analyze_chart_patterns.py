#!/usr/bin/env python3
"""
Chart-pattern edge measurement: do double-top / double-bottom entries beat a
coin flip?

The only number that matters here is the win rate relative to the random-walk
baseline for the RR actually used. For a driftless walk,
P(target before stop) = 1/(1+RR), which is exactly break-even at every RR --
so a strategy's RR is a free parameter and cannot create edge. What can is the
entry. This report therefore always prints observed WR next to 1/(1+RR) for
the same trades, and the gap between them IS the measurement.

Join: signals.csv (which carries chart_pattern_* as of 2026-09-17) to
trades.csv on signal_id. Rows logged before that date have empty pattern
columns and are excluded from the pattern arms -- they are not evidence of
"no pattern", they are evidence of nothing, and counting them as no-pattern
would silently bias the baseline.

Usage:
    python -m tools.research.analyze_chart_patterns
    python -m tools.research.analyze_chart_patterns --patterns double_top,double_bottom
"""

import argparse
import csv
import math
import os
from collections import defaultdict

SIGNALS = os.getenv("SIGNAL_LOG_PATH", "signals.csv")
TRADES = os.getenv("TRADES_FILE", "trades.csv")

# Pattern provenance was wired into signal_log on this date. A signal logged
# before it has empty chart_pattern columns because nothing was ever written
# there -- indistinguishable, by value, from a signal that was evaluated and
# found no pattern. Treating the former as "no pattern" would pad that arm
# with trades that carry no information and bias the comparison, so they are
# excluded by timestamp instead.
PROVENANCE_START = os.getenv("PATTERN_PROVENANCE_START", "2026-09-17")


def _f(row, key, default=0.0):
    try:
        return float(row.get(key) or default)
    except (TypeError, ValueError):
        return default


def _wilson(k, n, z=1.96):
    """Wilson interval -- honest at the small n this will run at for weeks."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z*z/n
    c = p + z*z/(2*n)
    m = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return ((c-m)/d, (c+m)/d)


def load():
    if not os.path.exists(SIGNALS) or not os.path.exists(TRADES):
        return [], []
    with open(SIGNALS, newline="") as f:
        sig = list(csv.DictReader(f))
    with open(TRADES, newline="") as f:
        tr = list(csv.DictReader(f))
    return sig, tr


def join(signals, trades):
    """One row per closed trade, carrying its signal's pattern provenance."""
    by_sig = {}
    for s in signals:
        sid = (s.get("signal_id") or "").strip()
        if sid:
            by_sig[sid] = s
    out = []
    for t in trades:
        if (t.get("state") or "") != "closed":
            continue
        s = by_sig.get((t.get("signal_id") or "").strip())
        if s is None:
            continue
        out.append({
            "position_id": t.get("position_id", ""),
            "coin": t.get("coin", ""),
            "side": t.get("side", ""),
            "realized_r": _f(t, "realized_r"),
            "pnl_usd": _f(t, "pnl_usd"),
            "rr_planned": _f(t, "rr_planned"),
            "timeframe": s.get("timeframe", ""),
            "pattern": (s.get("chart_pattern") or "").strip(),
            "pattern_side": (s.get("chart_pattern_side") or "").strip(),
            "confirmed": str(s.get("chart_pattern_confirmed", "")).lower() == "true",
            "vol_confirmed": str(s.get("chart_pattern_volume_confirmed", "")).lower() == "true",
            "has_pattern_cols": "chart_pattern" in s,
            "ts": (s.get("timestamp_utc") or "").strip(),
        })
    return out


def has_provenance(row):
    """True only if this trade's signal was logged after pattern provenance
    existed, so an empty chart_pattern genuinely means 'no pattern found'."""
    return bool(row["ts"]) and row["ts"][:10] >= PROVENANCE_START


def report(rows, label):
    n = len(rows)
    if n == 0:
        print(f"  {label:28} n=0")
        return
    wins = sum(1 for r in rows if r["realized_r"] > 0)
    wr = wins / n
    rrs = [r["rr_planned"] for r in rows if r["rr_planned"] > 0]
    rr = sorted(rrs)[len(rrs)//2] if rrs else 0.0
    base = 1/(1+rr) if rr > 0 else float("nan")
    mean_r = sum(r["realized_r"] for r in rows) / n
    lo, hi = _wilson(wins, n)
    edge = (wr - base) * 100 if rr > 0 else float("nan")
    sig = ""
    if rr > 0:
        sig = "  <-- CI excludes baseline" if (lo > base or hi < base) else "  (CI contains baseline)"
    print(f"  {label:28} n={n:<4} WR={wr*100:5.1f}% [{lo*100:4.1f},{hi*100:5.1f}]  "
          f"RR={rr:4.2f} base={base*100:5.1f}%  edge={edge:+5.1f}pp  meanR={mean_r:+.3f}{sig}")


def main():
    ap = argparse.ArgumentParser(description="Measure chart-pattern entry edge")
    ap.add_argument("--patterns", default="double_top,double_bottom")
    args = ap.parse_args()
    wanted = [p.strip() for p in args.patterns.split(",") if p.strip()]

    signals, trades = load()
    if not signals:
        print(f"[!] {SIGNALS} or {TRADES} not found."); return 1
    rows = join(signals, trades)

    have_cols = "chart_pattern" in (signals[0] if signals else {})
    tagged = [r for r in rows if r["pattern"]]
    print("="*104)
    print("CHART-PATTERN ENTRY EDGE")
    print("="*104)
    print(f"signals rows {len(signals)}   closed trades joined {len(rows)}   "
          f"carrying a pattern tag {len(tagged)}")
    if not have_cols:
        print("\n[!] signals.csv has no chart_pattern column yet -- run the agent once to migrate it.")
        return 0
    if not tagged:
        print("\n[!] No trade yet carries a pattern tag. Pattern provenance starts 2026-09-17;")
        print("    trades logged before that cannot be attributed and are excluded, not counted")
        print("    as 'no pattern'. This report becomes meaningful as forward trades accumulate.")
    print()
    print("edge = observed WR minus the random-walk baseline 1/(1+RR) at the SAME RR.")
    print("positive = the entry beats a coin flip. RR changes move both columns together.")
    print()
    withprov = [r for r in rows if has_provenance(r)]
    noprov = len(rows) - len(withprov)
    print("ALL JOINED TRADES")
    report(rows, "all (incl. pre-provenance)")
    if noprov:
        print(f"  {'excluded: pre-' + PROVENANCE_START:28} n={noprov:<4} "
              f"(no pattern provenance -- cannot be attributed either way)")
    report(withprov, "with provenance")
    report([r for r in withprov if not r["pattern"]], "  evaluated, no pattern")
    print()
    print("BY PATTERN")
    for p in wanted:
        sub = [r for r in rows if r["pattern"] == p]
        report(sub, p)
        report([r for r in sub if r["confirmed"]], f"  {p} (confirmed)")
        for tf in sorted({r["timeframe"] for r in sub if r["timeframe"]}):
            report([r for r in sub if r["timeframe"] == tf], f"  {p} @{tf}")
    print()
    print("DIRECTION AGREEMENT  (did the trade take the side the pattern implied?)")
    agree = [r for r in tagged if r["pattern_side"] and r["side"].upper() == r["pattern_side"].upper()]
    disagree = [r for r in tagged if r["pattern_side"] and r["side"].upper() != r["pattern_side"].upper()]
    report(agree, "traded WITH pattern")
    report(disagree, "traded AGAINST pattern")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

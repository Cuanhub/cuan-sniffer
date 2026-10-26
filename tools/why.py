#!/usr/bin/env python3
"""Why are we losing? The band strategy's own record, read with the guards on.

    python -m tools.why                 # the report
    python -m tools.why --resolve       # resolve outstanding outcomes first
    python -m tools.why --min-n 40      # require more per bucket

Every guard here exists because the 2026-10-26 analysis hit the trap first,
and a tool that produces a confident false finding is worse than no tool:

  ORDERED WINS      hit_tp and hit_stop can both be true. The resolver stores
                    `won` once, ordered; nothing here reads the bare flags.
  BREAK-EVEN        a 17% win rate is good at rr 6 and terrible at rr 1, so
                    every win rate is reported against 1/(1+rr), never raw.
  CONSTANT FEATURES a median split on a zero-variance column divides by LIST
                    POSITION, which is chronological order -- it made a
                    constant score identically to a timestamp. Skipped.
  TIME CONFOUND     a feature that tracks time reproduces whatever changed
                    over time. Each one is scored for time correlation and
                    flagged.
  SIGNIFICANCE      a mean that cannot clear its own standard error is not a
                    finding. t is printed next to every split.
  PERSISTENCE       pooled significance that does not replicate across halves
                    is history, not edge. Per-coin edge failed exactly this.

The refused rows are the point. Nothing else in this project can say whether
a REFUSAL was right, because the level scanner never resolved what it turned
down.
"""

import argparse
import math
import statistics
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence

from dotenv import load_dotenv

load_dotenv()

from research.levels import storage                            # noqa: E402

SKIP = {"zone_episode_id", "coin", "detected_at_utc", "refused_reason",
        "taken", "immediate"}


def num(row: Dict[str, Any], key: str) -> Optional[float]:
    try:
        v = float(row.get(key, ""))
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def flag(v: Any) -> Optional[bool]:
    s = str(v).strip().lower()
    if s in ("1", "true", "yes"):
        return True
    if s in ("0", "false", "no"):
        return False
    return None


def joined() -> List[Dict[str, Any]]:
    """Episodes with their resolved outcome, as one row each."""
    outs: Dict[str, Dict[str, Any]] = {}
    import csv
    path = storage.DATA_DIR / "zone_outcomes.csv"
    if path.exists():
        with path.open(newline="") as fh:
            for raw in csv.DictReader(fh):
                outs[str(raw.get("zone_episode_id") or "")] = dict(raw)
    rows: List[Dict[str, Any]] = []
    for e in storage.load_zone_episodes():
        o = outs.get(str(e.get("zone_episode_id") or ""))
        if not o:
            continue
        r: Dict[str, Any] = dict(e)
        r["_won"] = flag(o.get("won"))
        r["_filled"] = flag(o.get("filled"))
        r["_r"] = num(o, "outcome_r")
        r["_mae"] = num(o, "mae_pct")
        r["_taken"] = flag(e.get("taken"))
        rows.append(r)
    return rows


def _corr(a: List[float], b: List[float]) -> float:
    """Pearson correlation. Used only to FLAG a feature that tracks the
    calendar, because such a feature reproduces whatever changed over time
    rather than telling you anything about the setup."""
    if len(a) < 3 or len(a) != len(b):
        return 0.0
    ma, mb = statistics.mean(a), statistics.mean(b)
    den = math.sqrt(sum((x - ma) ** 2 for x in a)
                    * sum((y - mb) ** 2 for y in b))
    if den == 0:
        return 0.0
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / den


def stat(g: Sequence[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    rs = [x["_r"] for x in g if x["_r"] is not None]
    wins = [x for x in g if x["_won"] is True]
    if not rs:
        return None
    m = statistics.mean(rs)
    sd = statistics.stdev(rs) if len(rs) > 1 else 0.0
    se = sd / math.sqrt(len(rs)) if rs else 0.0
    rrs = [v for v in (num(x, "rr") for x in g) if v is not None]
    be = 1.0 / (1.0 + statistics.median(rrs)) if rrs else float("nan")
    w = len(wins) / len(g)
    return {"n": float(len(g)), "win": w, "mean_r": m, "se": se,
            "t": m / se if se else 0.0, "break_even": be,
            "edge_pp": (w - be) * 100.0}


def line(label: str, s: Optional[Dict[str, float]]) -> str:
    if not s:
        return f"  {label:30} (no resolved outcomes)"
    return (f"  {label:30} n={int(s['n']):5} win {s['win']*100:5.1f}% "
            f"(break-even {s['break_even']*100:5.1f}%, "
            f"edge {s['edge_pp']:+6.1f}pp)  mean R {s['mean_r']:+.3f} "
            f"t={s['t']:+5.2f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resolve", action="store_true",
                    help="resolve outstanding outcomes before reporting")
    ap.add_argument("--min-n", type=int, default=25,
                    help="minimum rows per bucket (default 25)")
    args = ap.parse_args()

    if args.resolve:
        from research.levels.venue_client import HyperliquidReadOnlyClient
        from research.levels import zone_log
        n = zone_log.resolve(HyperliquidReadOnlyClient())
        print(f"  resolved {n} new outcome(s)\n")

    rows = joined()
    if not rows:
        print("  No resolved band episodes yet.\n"
              "  The pass records one row per band it considers, and\n"
              "  `--resolve` turns those into outcomes once enough bars have\n"
              "  passed. Come back when the trader has run for a day.")
        return 0

    print(f"=== {len(rows)} resolved band episodes ===\n")
    filled = [x for x in rows if x["_filled"] is True]
    print(line("ALL (filled)", stat(filled)))
    print(line("  of those, TAKEN", stat([x for x in filled if x["_taken"]])))
    print(line("  of those, REFUSED", stat([x for x in filled
                                            if x["_taken"] is False])))
    print(f"\n  never filled: {sum(1 for x in rows if x['_filled'] is False)}"
          f" of {len(rows)} -- bands price never reached\n")

    print("=== WERE THE REFUSALS RIGHT? ===")
    print("  (nothing else in this project can answer this: the level")
    print("   scanner never resolved what it turned down)")
    byr: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for x in filled:
        if x["_taken"] is False:
            byr[str(x.get("refused_reason") or "?").split(" <")[0]].append(x)
    for reason, g in sorted(byr.items(), key=lambda kv: -len(kv[1])):
        if len(g) >= args.min_n:
            print(line(reason[:28], stat(g)))
    print()

    print("=== FEATURE SCAN (filled rows only) ===")
    results = []
    for key in storage.ZONE_EPISODE_FIELDNAMES:
        if key in SKIP:
            continue
        pairs: List[tuple] = []
        for x in filled:
            v = num(x, key)
            if v is not None:
                pairs.append((v, x))
        if len(pairs) < args.min_n * 2:
            continue
        if len({v for v, _ in pairs}) < 10:
            continue                       # the constant-split trap
        pairs.sort(key=lambda t: float(t[0]))
        vals = pairs
        h = len(vals) // 2
        lo = stat([x for _, x in vals[:h]])
        hi = stat([x for _, x in vals[h:]])
        if not (lo and hi):
            continue
        # time correlation, so a feature that merely tracks the calendar is
        # flagged rather than celebrated
        fv: List[float] = [float(v) for v, _ in vals]
        tv: List[float] = [float(num(x, "bar_ts_ms") or 0.0) for _, x in vals]
        rho = _corr(fv, tv)
        results.append((abs(hi["mean_r"] - lo["mean_r"]), key, vals[h][0],
                        lo, hi, rho))
    results.sort(reverse=True)
    if not results:
        print("  not enough resolved rows per bucket yet\n")
    for spread, key, cut, lo, hi, rho in results[:10]:
        warn = "  << tracks time, treat as confounded" if abs(rho) > 0.5 else ""
        print(f"  {key}  (split at {cut:.4g}, R spread {spread:.3f}){warn}")
        print(line("    lower half", lo))
        print(line("    upper half", hi))
    print()

    print("=== PERSISTENCE: does any of it replicate? ===")
    fs = sorted(filled, key=lambda x: num(x, "bar_ts_ms") or 0)
    half = len(fs) // 2
    if half < args.min_n:
        print(f"  too few resolved rows to split ({len(fs)}); "
              f"need about {args.min_n*2}\n")
    else:
        for nm, part in (("first half", fs[:half]), ("second half", fs[half:])):
            print(line(nm, stat(part)))
        print("  a finding that flips sign between these is history, not edge")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

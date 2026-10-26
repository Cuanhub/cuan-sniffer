#!/usr/bin/env python3
"""The bands worth your verdict, and a one-liner to record it.

    python -m tools.boundary                      # what to look at
    python -m tools.boundary --label zone:TIA:... take -n "4h is clean"
    python -m tools.boundary --score              # were you right?

Why the BOUNDARY and not the best setups
----------------------------------------
Your verdict is the scarce input here -- 69 labels in a month -- so it should
be spent where it carries information. A band at rr 0.02 teaches nothing: the
gate refuses it and so would you. A band at rr 1.94 against a 2.0 floor is
the only place a label can say the line is in the wrong place.

So this ranks by DISTANCE TO THE DECISION LINE, nearest first, and shows
which side the gate came down on. It is not a list of good trades.

What it is for
--------------
MIN_RR has been argued from a different strategy's episodes all week. Twenty
verdicts on bands sitting either side of 2.0, joined to what those bands
actually did, answers it with this strategy's own data.

`--score` is the other half, and nobody had ever run it: your existing 69
labels already have 34 resolved outcomes.
"""

import argparse
import csv
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

from research.levels import storage, zone_log                  # noqa: E402
from research.levels.config import MIN_RR                      # noqa: E402

VERDICTS = ("take", "skip", "maybe")


def _labels() -> Dict[str, Dict[str, Any]]:
    path = storage.DATA_DIR / "labels.csv"
    if not path.exists():
        return {}
    with path.open(newline="") as fh:
        return {r["episode_id"]: dict(r) for r in csv.DictReader(fh)}


def _num(row: Dict[str, Any], key: str) -> Optional[float]:
    try:
        v = float(row.get(key, ""))
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def show(min_rr: float, limit: int) -> int:
    rows = storage.load_zone_episodes()
    if not rows:
        print("  No band episodes recorded yet -- the trader writes them each"
              " pass.")
        return 0
    cases = zone_log.boundary_cases(rows, min_rr, labelled=list(_labels()),
                                    limit=limit)
    if not cases:
        print(f"  {len(rows)} episodes recorded, none within "
              f"{zone_log.BOUNDARY_FRAC:.0%} of the rr {min_rr} line that is"
              f" not already labelled.")
        return 0
    print(f"  {len(rows)} band episodes recorded. The {len(cases)} sitting "
          f"closest to the rr {min_rr} line:\n")
    print(f"  {'coin':9} {'rr':>6} {'gate':>13} {'band':>22} "
          f"{'risk%':>7} {'stop/ATR':>9} {'target%':>8}")
    for r in cases:
        lo, hi = _num(r, "zone_lo"), _num(r, "zone_hi")
        print(f"  {r['coin']:9} {_num(r,'rr') or 0:6.2f} {r['_side']:>13} "
              f"{f'{lo:.6g}-{hi:.6g}' if lo and hi else '?':>22} "
              f"{_num(r,'risk_pct') or 0:6.2f}% "
              f"{_num(r,'stop_atr_mult') or 0:8.2f}x "
              f"{_num(r,'target_dist_pct') or 0:7.2f}%")
    print(f"\n  record one with:\n"
          f"    python -m tools.boundary --label {cases[0]['zone_episode_id']}"
          f" take -n \"why\"")
    return 0


def label(eid: str, verdict: str, confidence: str, note: str) -> int:
    verdict = verdict.lower().strip()
    if verdict not in VERDICTS:
        print(f"  verdict must be one of {VERDICTS}")
        return 1
    known = {str(r.get("zone_episode_id")) for r in storage.load_zone_episodes()}
    if eid not in known:
        print(f"  {eid} is not a recorded band episode. "
              f"`python -m tools.boundary` lists the ones waiting.")
        return 1
    ok = storage.append_label({
        "episode_id": eid, "labelled_at_utc": storage.now_iso(),
        "verdict": verdict, "confidence": confidence, "note": note or ""})
    print(f"  {'labelled' if ok else 'already labelled'}: {eid} -> {verdict}")
    return 0


def score() -> int:
    """Were the verdicts right? Joins labels to resolved outcomes.

    Works on BOTH label sets -- the level episodes you have been labelling
    since September, and band episodes as they resolve -- because the join is
    on episode id and the id tells you which kind it is.
    """
    labs = _labels()
    if not labs:
        print("  no labels recorded yet")
        return 0
    outs: Dict[str, Dict[str, Any]] = {}
    for name, key in (("outcomes.csv", "episode_id"),
                      ("zone_outcomes.csv", "zone_episode_id")):
        path = storage.DATA_DIR / name
        if not path.exists():
            continue
        with path.open(newline="") as fh:
            for r in csv.DictReader(fh):
                outs[str(r.get(key) or "")] = dict(r)

    def won_of(o: Dict[str, Any]) -> Optional[bool]:
        # zone outcomes store `won` already resolved. Level outcomes do not,
        # and their two flags can BOTH be true -- which is the trap that
        # contaminated the first pass of the 2026-10-26 analysis.
        w = str(o.get("won", "")).strip().lower()
        if w in ("true", "1", "yes"):
            return True
        if w in ("false", "0", "no"):
            return False
        tp = str(o.get("hit_suggested_tp", "")).strip().lower() in ("true", "1")
        st = str(o.get("hit_suggested_stop", "")).strip().lower() in ("true", "1")
        if not (tp or st):
            return None
        if tp and st:
            btp, bst = _num(o, "bars_to_tp"), _num(o, "bars_to_stop")
            if btp is None or bst is None or btp == bst:
                return None
            return btp < bst
        return tp

    rows = []
    for eid, lab in labs.items():
        o = outs.get(eid)
        if not o:
            continue
        w = won_of(o)
        if w is None:
            continue
        rows.append((lab["verdict"].strip().lower(), w, _num(o, "outcome_r"),
                     "band" if eid.startswith("zone:") else "level"))
    if not rows:
        print(f"  {len(labs)} labels, none with a resolved outcome yet")
        return 0

    print(f"  {len(rows)} of {len(labs)} labels have a resolved outcome\n")
    print(f"  {'kind':6} {'verdict':8} {'n':>4} {'won':>7} {'mean R':>9} {'t':>7}")
    for kind in sorted({r[3] for r in rows}):
        for v in sorted({r[0] for r in rows if r[3] == kind}):
            g = [r for r in rows if r[3] == kind and r[0] == v]
            rs = [r[2] for r in g if r[2] is not None]
            if not rs:
                continue
            m = statistics.mean(rs)
            se = (statistics.stdev(rs) / math.sqrt(len(rs))
                  if len(rs) > 1 else 0.0)
            print(f"  {kind:6} {v:8} {len(g):4} "
                  f"{sum(1 for r in g if r[1])/len(g)*100:6.1f}% {m:+9.3f} "
                  f"{m/se if se else 0:+7.2f}")
    print("\n  A verdict column that cannot clear its own standard error is")
    print("  not evidence about your judgement. On the level labels the")
    print("  take-vs-skip gap was t=1.33 -- suggestive, not established, and")
    print("  largely the October regime.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", nargs=2, metavar=("EPISODE_ID", "VERDICT"))
    ap.add_argument("-n", "--note", default="")
    ap.add_argument("--confidence", default="")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--min-rr", type=float, default=MIN_RR)
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()
    if args.label:
        return label(args.label[0], args.label[1], args.confidence, args.note)
    if args.score:
        return score()
    return show(args.min_rr, args.top)


if __name__ == "__main__":
    raise SystemExit(main())

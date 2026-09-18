"""
Review queue and labelling CLI.

The scanner deliberately logs every gated episode -- its thresholds are
placeholders fitted to two levels, and a gate that only records what it is
already confident about can never learn where its own boundary belongs. A
recent pass produced 39 episodes against roughly 9 trades a month by hand, so
the gate currently fires about 25x too often. That gap is the thing being
measured, not a bug to tune away before there is data to tune against.

This module does the separating: everything is logged, a ranked shortlist is
surfaced. Ranking is a transparent product of level quality, test count and
reward/risk -- deliberately simple, because a learned ranker fitted to n=2
would just launder the same guess.

    python -m research.levels.review                    # top candidates
    python -m research.levels.review --all              # everything unlabelled
    python -m research.levels.review --label <id> take  --note "..."
    python -m research.levels.review --label <id> skip  --note "too far"
    python -m research.levels.review --stats            # gate vs your labels
"""

import argparse
import csv
from collections import Counter
from typing import Any, Dict

from research.levels import storage

VERDICTS = ("take", "skip", "maybe")


def _f(row: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key) or default)
    except (TypeError, ValueError):
        return default


def sort_key(ep: Dict[str, Any]):
    """Shortlist ordering: hold rate first, then reward/risk.

    This used to be a weighted sum of four features with coefficients
    0.45/0.30/0.15/0.10 that I invented. Those weights encoded a claim about
    relative importance that nothing supports, and dressing a guess up as a
    score makes it harder to see that it is one. Hold rate leads because it
    is the one level property with a measurement behind it -- BTC 74,900 held
    10 of 10 tests where a naive band around CASHCAT 0.14 held 8 of 18. The
    ordering is a convenience for reviewing, not a model.
    """
    return (_f(ep, "level_hold_rate"), _f(ep, "suggested_rr"))


def labelled_ids():
    path = storage.DATA_DIR / "labels.csv"
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        return {r["episode_id"]: r for r in csv.DictReader(f)}


def cmd_list(args):
    eps = storage.load_episodes()
    if not eps:
        print("no episodes yet -- run the scanner first")
        return 0
    done = labelled_ids()
    pending = [e for e in eps if e["episode_id"] not in done]
    pending.sort(key=sort_key, reverse=True)
    show = pending if args.all else pending[:args.top]
    print(f"{len(eps)} episodes logged, {len(done)} labelled, {len(pending)} pending"
          f" -- showing {len(show)}\n")
    print(f"{'coin':10} {'level':>12} {'hold':>5} {'tch':>4} {'run':>6} {'pull':>6} "
          f"{'rr':>5} {'entry':>12}  episode_id")
    for e in show:
        print(f"{e['coin']:10} {_f(e,'level_price'):>12.6g} "
              f"{_f(e,'level_hold_rate')*100:>4.0f}% {int(_f(e,'level_touches')):>4} "
              f"{_f(e,'run_sigma'):>5.1f}s {_f(e,'pull_sigma'):>5.1f}s "
              f"{_f(e,'suggested_rr'):>5.2f} {_f(e,'suggested_entry'):>12.6g}  {e['episode_id']}")
    print("\nlabel with:  python -m research.levels.review --label <episode_id> take|skip|maybe")
    return 0


def cmd_label(args):
    eid, verdict = args.label
    verdict = verdict.lower().strip()
    if verdict not in VERDICTS:
        print(f"verdict must be one of {VERDICTS}")
        return 1
    if not any(e["episode_id"] == eid for e in storage.load_episodes()):
        print(f"no episode with id {eid}")
        return 1
    ok = storage.append_label({
        "episode_id": eid, "labelled_at_utc": storage.now_iso(),
        "verdict": verdict, "confidence": args.confidence, "note": args.note or "",
    })
    print(f"{'labelled' if ok else 'already labelled'}: {eid} -> {verdict}")
    return 0


def cmd_stats(args):
    eps = {e["episode_id"]: e for e in storage.load_episodes()}
    labs = labelled_ids()
    if not labs:
        print("no labels yet -- the gate cannot be calibrated until you label some episodes")
        return 0
    c = Counter(v["verdict"] for v in labs.values())
    print(f"labels: {dict(c)}   of {len(eps)} episodes\n")
    takes = [eps[i] for i, v in labs.items() if v["verdict"] == "take" and i in eps]
    skips = [eps[i] for i, v in labs.items() if v["verdict"] == "skip" and i in eps]
    if not takes or not skips:
        print("need both takes and skips before features can be compared")
        return 0
    print("feature separation between what you take and what you skip:")
    print(f"  {'feature':28} {'take avg':>10} {'skip avg':>10} {'gap':>9}")
    for k in ("level_hold_rate", "level_touches", "run_sigma", "pull_sigma", "retests",
              "near_sigma", "suggested_rr", "vol_ratio_recent_ref", "vol_trend_slope",
              "vol_down_up_ratio", "bars_since_last_retest"):
        t = sum(_f(e, k) for e in takes) / len(takes)
        s = sum(_f(e, k) for e in skips) / len(skips)
        print(f"  {k:28} {t:>10.3f} {s:>10.3f} {t-s:>+9.3f}")
    print("\nthe largest gaps are the features worth gating on. A near-zero gap on all")
    print("three volume readings would settle that question in the negative.")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Review and label level episodes")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--label", nargs=2, metavar=("EPISODE_ID", "VERDICT"))
    ap.add_argument("--note", default="")
    ap.add_argument("--confidence", default="")
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()
    if args.label:
        return cmd_label(args)
    if args.stats:
        return cmd_stats(args)
    return cmd_list(args)


if __name__ == "__main__":
    raise SystemExit(main())

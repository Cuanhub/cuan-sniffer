"""Screen the collected venue-context features against trade expectancy.

Collection started 2026-10-07. The rule this exists to serve: a feature is
not allowed to influence sizing or admission until it survives the whole
screen, because the first pass through ~30 features found exactly one that
did, and several that looked strong on the full sample reversed out of
sample.

What the screen demands, in order:

  1. Score on expectancy (outcome_r), never on win rate. Win rate is a
     function of how far the target sits, so a feature that quietly shortens
     targets buys a better win rate and a worse account.
  2. Stratify by reward/risk tercile. rr correlates with nearly everything
     here, so an unstratified split mostly rediscovers rr.
  3. Permutation test. With this many features at this sample size, a
     p < 0.05 on a single split is close to free.
  4. Chronological out-of-sample split. The halves have to agree in SIGN.
     This is the test that killed the 24h-trend and breadth features.

Usage: python -m tools.screen_context [--min-n 40] [--perms 20000]
"""

import argparse
import csv
import pathlib
import random
import statistics
from typing import Any, Dict, List, Optional, Sequence, Tuple

DATA = pathlib.Path(__file__).resolve().parent.parent / "research/levels/data"

# The twelve columns the scanner records from the venue, plus the ratios that
# are the actual hypotheses. Raw oi_notional_usd is a proxy for coin size, so
# it is screened only as a ratio against volume.
RATIOS = {
    # Mark above oracle means longs are paying to hold. This is the one
    # feature that survived the first screen (p=0.0035).
    "basis_bps": lambda r: (r["mark_px"] - r["oracle_px"]) / r["oracle_px"] * 1e4
    if r.get("oracle_px") else None,
    "funding_bps": lambda r: r["funding_rate"] * 1e4,
    # Open interest relative to the day's volume: crowded versus liquid.
    "oi_to_volume": lambda r: r["oi_notional_usd"] / r["day_ntl_vlm"]
    if r.get("day_ntl_vlm") else None,
    "spread_bps": lambda r: r["impact_spread_bps"],
    # How far the entry sits from the previous day's close.
    "vs_prev_day_pct": lambda r: (r["price"] - r["prev_day_px"]) / r["prev_day_px"] * 100.0
    if r.get("prev_day_px") else None,
    "oi_notional_usd": lambda r: r["oi_notional_usd"],
    "day_ntl_vlm": lambda r: r["day_ntl_vlm"],
}


def _f(row: Dict[str, str], key: str) -> Optional[float]:
    v = (row.get(key) or "").strip()
    if not v:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def load() -> List[Dict[str, Any]]:
    """Episodes that have both a venue context and a resolved outcome."""
    outcomes = {}
    with (DATA / "outcomes.csv").open() as fh:
        for row in csv.DictReader(fh):
            r = _f(row, "outcome_r")
            if r is not None:
                outcomes[row["episode_id"]] = row

    joined = []
    with (DATA / "episodes.csv").open() as fh:
        for row in csv.DictReader(fh):
            out = outcomes.get(row["episode_id"])
            if out is None or not (row.get("ctx_captured_at_utc") or "").strip():
                continue
            rec: Dict[str, Any] = {
                "episode_id": row["episode_id"],
                "coin": row["coin"],
                "at": row.get("ctx_captured_at_utc") or row.get("detected_at_utc"),
                "outcome_r": _f(out, "outcome_r"),
                "rr": _f(row, "suggested_rr"),
                "price": _f(row, "price"),
            }
            for col in ("oi_base", "oi_notional_usd", "funding_rate", "premium",
                        "mark_px", "oracle_px", "impact_bid", "impact_ask",
                        "impact_spread_bps", "day_ntl_vlm", "prev_day_px"):
                rec[col] = _f(row, col)
            for name, fn in RATIOS.items():
                try:
                    rec[name] = fn(rec)
                except (TypeError, ZeroDivisionError, KeyError):
                    rec[name] = None
            if rec["outcome_r"] is not None:
                joined.append(rec)
    joined.sort(key=lambda r: r["at"] or "")
    return joined


def terciles(vals: Sequence[float]) -> Tuple[float, float]:
    s = sorted(vals)
    return s[len(s) // 3], s[2 * len(s) // 3]


def split_mean_gap(rows: Sequence[Dict[str, Any]], feat: str
                   ) -> Optional[Tuple[float, float, float, int, int]]:
    """Mean outcome_r below vs above the feature median, rr-stratified.

    Stratifying means taking the median WITHIN each rr tercile and pooling the
    halves, so the comparison cannot be a restatement of rr.
    """
    usable = [r for r in rows if r.get(feat) is not None and r["rr"] is not None]
    if len(usable) < 12:
        return None
    lo_rr, hi_rr = terciles([r["rr"] for r in usable])
    low: List[float] = []
    high: List[float] = []
    for bound in ((None, lo_rr), (lo_rr, hi_rr), (hi_rr, None)):
        a, b = bound
        band = [r for r in usable
                if (a is None or r["rr"] >= a) and (b is None or r["rr"] < b)]
        if len(band) < 4:
            continue
        med = statistics.median(r[feat] for r in band)
        for r in band:
            (low if r[feat] <= med else high).append(r["outcome_r"])
    if len(low) < 4 or len(high) < 4:
        return None
    return (statistics.mean(low), statistics.mean(high),
            statistics.mean(high) - statistics.mean(low), len(low), len(high))


def permutation_p(rows: Sequence[Dict[str, Any]], feat: str, perms: int,
                  rng: random.Random) -> Optional[float]:
    """Two-sided p for the stratified gap, shuffling outcomes against it."""
    base = split_mean_gap(rows, feat)
    if base is None:
        return None
    observed = abs(base[2])
    usable = [dict(r) for r in rows
              if r.get(feat) is not None and r["rr"] is not None]
    outs = [r["outcome_r"] for r in usable]
    hits = 0
    for _ in range(perms):
        rng.shuffle(outs)
        for r, o in zip(usable, outs):
            r["outcome_r"] = o
        trial = split_mean_gap(usable, feat)
        if trial is not None and abs(trial[2]) >= observed:
            hits += 1
    return (hits + 1) / (perms + 1)


def quartile_table(rows: Sequence[Dict[str, Any]], feat: str
                   ) -> List[Tuple[int, float, float, float]]:
    """Mean outcome by quartile of `feat`, for reading the SHAPE of an effect.

    Ties are broken at random, not left to the stable sort. load() returns
    rows in time order, so a stable sort puts tied feature values in
    chronological order -- and the regime drifts hard enough across this
    sample that the buckets then separate by DATE rather than by the feature.
    funding_bps is 75% tied at 0.1250, because Hyperliquid pins funding at its
    base rate, so three quarters of that table was reading the calendar.
    Caught on 2026-10-16 by test_ties_are_not_ordered_by_outcome.

    The p-values are not affected: split_mean_gap compares against a median
    rather than ranking, so tied values all land on the same side of it.
    """
    usable = [r for r in rows if r.get(feat) is not None]
    random.Random(0xC0FFEE).shuffle(usable)
    usable.sort(key=lambda r: r[feat])
    if len(usable) < 8:
        return []
    n = len(usable)
    out = []
    for q in range(4):
        band = usable[q * n // 4:(q + 1) * n // 4]
        if not band:
            continue
        out.append((len(band),
                    statistics.mean(r[feat] for r in band),
                    statistics.mean(r["outcome_r"] for r in band),
                    sum(1 for r in band if r["outcome_r"] > 0) / len(band) * 100))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-n", type=int, default=40)
    ap.add_argument("--perms", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20261015)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    rows = load()
    if len(rows) < args.min_n:
        print(f"only {len(rows)} joined rows, need {args.min_n}")
        return 1

    span = (rows[0]["at"] or "")[:10], (rows[-1]["at"] or "")[:10]
    rs = [r["outcome_r"] for r in rows]
    print(f"\n{'='*74}\nCONTEXT SCREEN -- {len(rows)} resolved episodes with venue "
          f"context\n  span {span[0]} .. {span[1]}   "
          f"coins {len({r['coin'] for r in rows})}\n"
          f"  outcome_r: mean {statistics.mean(rs):+.3f}  "
          f"median {statistics.median(rs):+.3f}  "
          f"positive {sum(1 for r in rs if r>0)/len(rs)*100:.0f}%\n{'='*74}")

    half = len(rows) // 2
    first, second = rows[:half], rows[half:]

    results = []
    for feat in RATIOS:
        gap = split_mean_gap(rows, feat)
        if gap is None:
            print(f"\n  {feat:<16} too few rows with this field")
            continue
        lo, hi, delta, nlo, nhi = gap
        p = permutation_p(rows, feat, args.perms, rng)
        g1 = split_mean_gap(first, feat)
        g2 = split_mean_gap(second, feat)
        agree = (g1 is not None and g2 is not None
                 and g1[2] != 0 and g2[2] != 0
                 and (g1[2] > 0) == (g2[2] > 0))
        results.append((p if p is not None else 1.0, feat, lo, hi, delta,
                        nlo, nhi, g1, g2, agree))

    results.sort()
    for p, feat, lo, hi, delta, nlo, nhi, g1, g2, agree in results:
        verdict = ("SURVIVES" if p < 0.05 and agree else
                   "reverses out of sample" if p < 0.05 else "no signal")
        print(f"\n  {feat}")
        print(f"    low  half  mean r {lo:+.3f}  (n={nlo})")
        print(f"    high half  mean r {hi:+.3f}  (n={nhi})")
        print(f"    gap {delta:+.3f}   permutation p = {p:.4f}   -> {verdict}")
        if g1 and g2:
            print(f"    out of sample: first half gap {g1[2]:+.3f}, "
                  f"second {g2[2]:+.3f}  "
                  f"({'same sign' if agree else 'SIGN FLIPS'})")
        for i, (n, mean_f, mean_r, wr) in enumerate(quartile_table(rows, feat)):
            print(f"      Q{i+1} n={n:<4} {feat}~{mean_f:>12.4f}   "
                  f"mean r {mean_r:+.3f}   {wr:.0f}% positive")

    print(f"\n{'='*74}")
    kept = [f for p, f, *_ , agree in results if p < 0.05 and agree]
    print(f"  survived: {', '.join(kept) if kept else 'nothing'}")
    print(f"{'='*74}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

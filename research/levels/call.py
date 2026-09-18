"""
Live call sheet -- surface levels for a human judgement, record the verdict.

This inverts the rest of the package. The scanner filters with twelve
thresholds I invented and asks the user to check the result; that filter has
never produced a validated finding and measures -0.0413R against random
entry. Here the user judges and their judgements become the definition.

Only the nine knobs that earned their place are used -- level detection and
sigma normalisation. Every gate threshold is OFF. Context is still computed
and SHOWN (run, pullback, retests) because the user needs it to judge, but
nothing is filtered on it, so their take/skip boundary can land wherever it
actually is rather than inside a box I drew first.

Each call writes two rows: an episode (the full feature snapshot, so the
label has something to join to) and a label. Skips matter as much as takes --
a boundary needs both sides, and the current filter's failure is that it
rejects nothing the user would reject.

    python -m research.levels.call                     # today's call sheet
    python -m research.levels.call --take CALL_ID --note "clean retest"
    python -m research.levels.call --skip CALL_ID --note "too far from level"
    python -m research.levels.call --progress          # how far through 30-50
"""

import argparse
import math
from typing import Any, Dict, List

from research.levels import config as C
from research.levels import storage
from research.levels.gate import evaluate
from research.levels.levels import detect_levels, log_sigma, sigma_distance, nearest_level
from research.levels.scanner import fetch, _atr_pct
from research.levels.venue_client import HyperliquidReadOnlyClient, VenueFetchError
from research.funding.universe import select_universe

# How near price must be to a level to be worth looking at. Deliberately loose:
# this is a viewing window, not a filter, and a tight one would pre-decide the
# very boundary these calls exist to locate.
CANDIDATE_NEAR_SIGMA = 4.0
TARGET_LABELS = 50


def _f(d, k, dv=0.0):
    try:
        return float(d.get(k) or dv)
    except (TypeError, ValueError):
        return dv


def find_candidates(client, universe) -> List[Dict[str, Any]]:
    out = []
    for u in universe:
        coin = u["symbol"]
        try:
            bars = fetch(client, coin)
        except VenueFetchError:
            continue
        if len(bars) < 120:
            continue
        i = len(bars) - 1
        sigma = log_sigma(bars, i)
        if sigma <= 0:
            continue
        px = bars[i]["c"]
        supports = detect_levels(bars, upto=i + 1, kind="support")
        resistances = detect_levels(bars, upto=i + 1, kind="resistance")
        for L in supports:
            if L.price >= px:
                continue
            near = sigma_distance(px, L.price, sigma)
            if near > CANDIDATE_NEAR_SIGMA:
                continue
            feats = evaluate(bars, i, L.price) or {}
            entry = L.price * math.exp(C.ENTRY_BUFFER_SIGMA * sigma)
            stop = L.price * math.exp(-C.STOP_SIGMA_BELOW * sigma)
            # Show the resistance LADDER rather than pre-picking a target.
            # Taking the nearest resistance produced RRs of 0.4-1.1 on every
            # candidate, which would have had the user skipping setups for a
            # reason the tool invented: their own CASHCAT target was 0.25,
            # far above the nearest level. Which rung they aim at is exactly
            # the judgement these calls exist to capture, so it is displayed,
            # not decided. suggested_tp keeps the nearest rung for continuity
            # with episodes.csv, and rr_ladder carries the alternatives.
            above = sorted([L for L in resistances if L.price > px], key=lambda L: L.price)[:3]
            opp = above[0] if above else None
            tp = opp.price if opp else px * math.exp(8 * sigma)
            rr = ((tp - entry) / (entry - stop)) if entry > stop else 0.0
            ladder = []
            for Lr in above:
                r = ((Lr.price - entry) / (entry - stop)) if entry > stop else 0.0
                ladder.append((Lr.price, r, Lr.touches, Lr.hold_rate))
            # 7d trajectory, so the user can see how price arrived here
            w = bars[max(0, i - 42):i + 1]
            hi7 = max(x["h"] for x in w); lo7 = min(x["l"] for x in w)
            out.append({
                "episode_id": storage.make_episode_id(coin, bars[i]["t"], L.price),
                "coin": coin, "detected_at_utc": storage.now_iso(),
                "bar_ts_ms": bars[i]["t"], "level_price": round(L.price, 10),
                "level_kind": "support", "level_touches": L.touches,
                "level_holds": L.holds, "level_hold_rate": round(L.hold_rate, 4),
                "level_pivot_count": L.pivot_count,
                "level_bars_since_last_touch": (i - L.last_touch_idx) if L.last_touch_idx >= 0 else -1,
                "atr_pct": round(_atr_pct(bars, i), 4),
                "entry_buffer_sigma": C.ENTRY_BUFFER_SIGMA,
                "suggested_entry": round(entry, 10), "suggested_stop": round(stop, 10),
                "suggested_tp": round(tp, 10), "suggested_rr": round(rr, 3),
                "opposing_level_price": round(opp.price, 10) if opp else "",
                "_px": px, "_near": near, "_hi7": hi7, "_lo7": lo7, "_ladder": ladder,
                "_from_hi7": (px - hi7) / hi7 * 100, "_stop_pct": (entry - stop) / entry * 100,
                **feats,
            })
    out.sort(key=lambda r: r["_near"])
    return out


def cmd_list(args):
    client = HyperliquidReadOnlyClient()
    universe = select_universe(client)
    cands = find_candidates(client, universe)
    labelled = _labelled()
    cands = [c for c in cands if c["episode_id"] not in labelled]
    if not cands:
        print("no candidates near a level right now")
        return 0
    show = cands[: args.top]
    print(f"{len(cands)} level(s) within {CANDIDATE_NEAR_SIGMA}s of price -- showing {len(show)}")
    print(f"labels so far: {len(labelled)} / {TARGET_LABELS}\n")
    for n, c in enumerate(show, 1):
        print(f"[{n}] {c['coin']}   price {c['_px']:.6g}")
        print(f"     level {c['level_price']:.6g}  ({-c['_near']:.2f}s below price, "
              f"{(c['level_price']-c['_px'])/c['_px']*100:+.2f}%)")
        print(f"     history: {c['level_touches']} tests, {c['level_holds']} held "
              f"({c['level_hold_rate']*100:.0f}%), last tested {c['level_bars_since_last_touch']} bars ago")
        print(f"     context: ran {_f(c,'run_sigma'):.1f}s, pulled back {_f(c,'pull_sigma'):.1f}s, "
              f"{int(_f(c,'retests'))} retests, {c['_from_hi7']:+.1f}% from 7d high")
        print(f"     trade  : entry {c['suggested_entry']:.6g}  stop {c['suggested_stop']:.6g} "
              f"({c['_stop_pct']:.1f}%)")
        if c["_ladder"]:
            rungs = "   ".join(f"{px_:.6g} (rr {r:.2f}, {t} tests {h*100:.0f}%)"
                               for px_, r, t, h in c["_ladder"])
            print(f"     targets: {rungs}")
        print(f"     id     : {c['episode_id']}")
        print()
    print("take:  python -m research.levels.call --take <id> --note \"...\"")
    print("skip:  python -m research.levels.call --skip <id> --note \"...\"")
    return 0


def _labelled() -> Dict[str, Any]:
    import csv
    p = storage.DATA_DIR / "labels.csv"
    if not p.exists():
        return {}
    with open(p, newline="") as f:
        return {r["episode_id"]: r for r in csv.DictReader(f)}


def cmd_label(eid: str, verdict: str, note: str):
    """Record the verdict, writing the episode snapshot first so the label has
    something to join to."""
    client = HyperliquidReadOnlyClient()
    universe = select_universe(client)
    match = next((c for c in find_candidates(client, universe) if c["episode_id"] == eid), None)
    if match is None:
        existing = {e["episode_id"] for e in storage.load_episodes()}
        if eid not in existing:
            print(f"no candidate or episode with id {eid}")
            return 1
    else:
        storage.append_episode({k: v for k, v in match.items() if not k.startswith("_")})
    ok = storage.append_label({"episode_id": eid, "labelled_at_utc": storage.now_iso(),
                               "verdict": verdict, "confidence": "", "note": note or ""})
    n = len(_labelled())
    print(f"{'recorded' if ok else 'already labelled'}: {eid} -> {verdict}")
    print(f"labels: {n} / {TARGET_LABELS}")
    return 0


def cmd_progress():
    import csv
    from collections import Counter
    labs = _labelled()
    if not labs:
        print(f"0 / {TARGET_LABELS} labels. Run the call sheet to start.")
        return 0
    c = Counter(v["verdict"] for v in labs.values())
    print(f"labels: {len(labs)} / {TARGET_LABELS}   {dict(c)}")
    takes = sum(1 for v in labs.values() if v["verdict"] == "take")
    skips = sum(1 for v in labs.values() if v["verdict"] == "skip")
    if takes < 5 or skips < 5:
        print("need at least ~5 of each before any feature comparison means anything")
    else:
        print("run:  python -m research.levels.review --stats")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Live call sheet for level setups")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--take", metavar="ID")
    ap.add_argument("--skip", metavar="ID")
    ap.add_argument("--note", default="")
    ap.add_argument("--progress", action="store_true")
    a = ap.parse_args()
    if a.progress:
        return cmd_progress()
    if a.take:
        return cmd_label(a.take, "take", a.note)
    if a.skip:
        return cmd_label(a.skip, "skip", a.note)
    return cmd_list(a)


if __name__ == "__main__":
    raise SystemExit(main())

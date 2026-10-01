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
from pathlib import Path
from typing import Any, Dict, List, Optional

from research.levels import config as C
from research.levels import storage
from research.levels.gate import evaluate
from research.levels.levels import (
    choose_target, detect_levels, flipped_supports, log_sigma, sigma_distance,
    structural_stop,
)
from research.levels.scanner import _atr_pct
from research.levels.venue_client import HyperliquidReadOnlyClient, VenueFetchError
from research.levels.universe import select_universe

# How far below price to look. Set from the user's own calls, not guessed: the
# levels they said they would actually trade sat 5.45s (CASHCAT 0.149), 5.56s
# (ETH 2355) and 8.03s (PUMP 0.0034) below price. The original 4.0s window
# excluded ALL THREE -- the tool was showing shallow dips while they were
# waiting for real structure. 12.0 covers their range with room either side.
CANDIDATE_NEAR_SIGMA = 12.0

# The user reads 1h and 4h and gets different levels from each -- CASHCAT 0.18
# on the 1h against 0.149 on the 4h, both of which they named as entries. A
# 4h-only tool cannot show half of what they trade.
TIMEFRAMES = ("4h", "1h")

# A level price left behind long ago cannot be broken, so it scores a perfect
# hold rate while meaning "abandoned" rather than "respected". CASHCAT's Aug
# 21-22 liquidation wicks ranked 2nd and 3rd on hold rate at 81% and 84% below
# price, untested for 162 and 304 bars, while the level the user actually
# named ranked 12th of 12. Levels are flagged, never dropped -- which of these
# matters is the user's call to make, not mine to pre-empt.
STALE_TOUCH_BARS = 120

# Two levels within this distance are the same level
CLUSTER_MERGE_SIGMA = 0.5

TARGET_LABELS = 50


def _f(d, k, dv=0.0):
    try:
        return float(d.get(k) or dv)
    except (TypeError, ValueError):
        return dv


# Seconds in each timeframe's bar, for boundary-aligned cache expiry.
_TF_SECONDS = {"4h": 4 * 3600, "1h": 3600}
# Only this timeframe is cached across passes. 1h is NOT: the trigger reads the
# last couple of hours of it to decide whether a level has held, so a 1h series
# carried between passes would be exactly the data that must not be stale.
_CACHEABLE_TF = ("4h",)


def _bar_open_ms(now_ms: int, tf: str) -> int:
    """Start of the bar `now_ms` falls in. Venue bars are UTC-aligned, so this
    is plain truncation -- and it is what makes the cache correct rather than
    merely time-boxed: the CLOSED history of a timeframe cannot change until
    the next bar opens."""
    step = _TF_SECONDS.get(tf, 0) * 1000
    return (now_ms // step) * step if step else now_ms


def _apply_live_price(bars, mid: float):
    """Refresh the forming bar from a live mid.

    The closed history behind a cached series is immutable, but the last bar is
    still being built, and its close is the price everything downstream treats
    as "now" -- including the depth the admission gate measures. High and low
    only ever widen within a bar, so ratcheting them against each observed mid
    moves toward the true extremes and can never overstate them. The whole
    series is refetched at the next boundary, so nothing accumulates past one
    bar.
    """
    if not bars or mid <= 0:
        return bars
    last = bars[-1]
    last["c"] = mid
    if mid > last["h"]:
        last["h"] = mid
    if mid < last["l"]:
        last["l"] = mid
    return bars


def _fetch_tf(client, coin: str, tf: str, cache=None, mids=None):
    """Bars for one coin and timeframe, reusing a cached series where it is
    still valid for the current bar.

    A 4h series was being refetched every 450s pass -- about 32 times per bar,
    30 coins deep, for data that by definition cannot change until the bar
    closes. That was 30 of the pass's ~63 remaining venue calls.
    """
    import time
    now = int(time.time() * 1000)
    if cache is not None and tf in _CACHEABLE_TF:
        key = (coin, tf)
        hit = cache.get(key)
        if hit and hit[0] == _bar_open_ms(now, tf):
            return _apply_live_price(hit[1], float((mids or {}).get(coin, 0.0) or 0.0))
    days = C.HISTORY_DAYS if tf == "4h" else 60
    raw = client.get_candles(coin, tf, now - days * 86400000, now)
    out = []
    for b in raw:
        try:
            out.append({"t": int(b["t"]), "o": float(b["o"]), "h": float(b["h"]),
                        "l": float(b["l"]), "c": float(b["c"]), "v": float(b.get("v", 0.0))})
        except (KeyError, TypeError, ValueError):
            continue
    out = sorted(out, key=lambda x: x["t"])
    if cache is not None and tf in _CACHEABLE_TF and out:
        cache[(coin, tf)] = (_bar_open_ms(now, tf), out)
    return out


def find_candidates(client, universe, timeframes=TIMEFRAMES,
                    bars_out: Optional[Dict[str, List[Dict[str, Any]]]] = None,
                    tf_cache: Optional[Dict[Any, Any]] = None,
                    mids: Optional[Dict[str, float]] = None
                    ) -> List[Dict[str, Any]]:
    """Candidates across the universe. `bars_out`, when given, receives the 1h
    series per coin so the caller need not fetch it a second time.

    These bars were being discarded. The live trader then asked the venue for
    the same 1h series again, per coin, to run the trigger -- 30 duplicate
    calls a pass out of about 93, against an endpoint that had already been
    returning rate limits. Handing them over costs nothing and removes the
    duplicate outright.

    Deliberately the RAW series, not a trimmed one: the trigger's window is a
    trading concern and this module stays ignorant of it. The caller trims with
    trigger.trigger_window, which exists because that window is load-bearing
    rather than cosmetic.

    `tf_cache` is a caller-owned dict that survives between calls, holding the
    4h series per coin until its bar closes; `mids` supplies the live price
    that keeps the forming bar honest while the rest of the series is reused.
    Both optional, and the module stays stateless without them -- lifetime
    belongs to whoever is polling, not to this function.
    """
    out = []
    for u in universe:
        coin = u["symbol"]
        for tf in timeframes:
            try:
                bars = _fetch_tf(client, coin, tf, cache=tf_cache, mids=mids)
            except VenueFetchError:
                continue
            if bars_out is not None and tf == "1h":
                bars_out[coin] = bars
            out.extend(_candidates_for(coin, bars, tf))
    # Round-robin across coins rather than sorting by any quality measure.
    # Sorting by hold rate is what buried the user's own levels behind
    # liquidation wicks, and sorting by coin put six AAVE rows at the top of a
    # 199-row sheet. Interleaving gives breadth across the universe without
    # ranking anything, so a labelling pass sees a spread of coins, timeframes
    # and distances instead of one asset's ladder.
    by_coin: Dict[str, List[Dict[str, Any]]] = {}
    for r in out:
        by_coin.setdefault(r["coin"], []).append(r)
    for rows in by_coin.values():
        rows.sort(key=lambda r: r["_near"])
    interleaved = []
    depth = max((len(v) for v in by_coin.values()), default=0)
    for k in range(depth):
        for coin in sorted(by_coin):
            if k < len(by_coin[coin]):
                interleaved.append(by_coin[coin][k])
    return interleaved


def _candidates_for(coin: str, bars, tf: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if len(bars) < 120:
            return out
        i = len(bars) - 1
        sigma = log_sigma(bars, i)
        if sigma <= 0:
            return out
        px = bars[i]["c"]
        supports = detect_levels(bars, upto=i + 1, kind="support")
        resistances = detect_levels(bars, upto=i + 1, kind="resistance")
        # A broken resistance is a support candidate. Merge them in, keeping
        # the swing-low version where both find the same price.
        origin = {round(L.price, 10): "swing_low" for L in supports}
        for F in flipped_supports(bars, upto=i + 1):
            if not any(abs(sigma_distance(F.price, L.price, sigma)) <= CLUSTER_MERGE_SIGMA
                       for L in supports):
                supports.append(F)
                origin[round(F.price, 10)] = "flipped_resistance"
        for L in supports:
            if L.price >= px:
                continue
            near = sigma_distance(px, L.price, sigma)
            if near > CANDIDATE_NEAR_SIGMA:
                continue
            feats = evaluate(bars, i, L.price) or {}
            entry = L.price * math.exp(C.ENTRY_BUFFER_SIGMA * sigma)
            # Below the lows that formed the level, not a fixed multiple of
            # sigma. The fixed rule stopped UNI 8.5254 -- a double bottom with
            # lows at 8.4516 -- down at 8.073, which made a 3.3 reward/risk
            # read as 0.91 and had the floor reject it. Falls back when the
            # level has no touch in the lookback to measure against.
            stop = structural_stop(bars, i, L.price, sigma) \
                or L.price * math.exp(-C.STOP_SIGMA_BELOW * sigma)
            # Show the resistance LADDER rather than pre-picking a target.
            # Taking the nearest resistance produced RRs of 0.4-1.1 on every
            # candidate, which would have had the user skipping setups for a
            # reason the tool invented: their own CASHCAT target was 0.25,
            # far above the nearest level. Which rung they aim at is exactly
            # the judgement these calls exist to capture, so it is displayed,
            # not decided. suggested_tp keeps the nearest rung for continuity
            # with episodes.csv, and rr_ladder carries the alternatives.
            # Which rung, not just the first one. choose_target is shared with
            # the scanner so the research set and the traded path cannot drift
            # apart the way the stop rule did.
            opp, above, _qualified = choose_target(resistances, entry, stop, sigma)
            # Fallback measured from the entry too, for the same reason the
            # ladder is: px is where price happens to be, not where the trade
            # starts.
            tp = opp.price if opp else entry * math.exp(8 * sigma)
            rr = ((tp - entry) / (entry - stop)) if entry > stop else 0.0
            ladder = []
            for Lr in above:
                r = ((Lr.price - entry) / (entry - stop)) if entry > stop else 0.0
                ladder.append((Lr.price, r, Lr.touches, Lr.hold_rate))

            # Headroom: how far the first resistance sits above the ENTRY, in
            # sigma. Added because the user skipped AAVE for "sitting right
            # below overhead resistance" and rejected an ETH entry beneath
            # 2,480 for the same reason -- buying support with resistance
            # directly overhead is a rule they apply and nothing here measured.
            # Captured and displayed, never gated on.
            # No detected resistance above is NOT infinite headroom -- it means
            # price is at or near its highs with no mapped target, so there is
            # no trade thesis to act on. Reported as None rather than a
            # sentinel 99.0, which was being read as "maximum headroom" and
            # floated exactly the untradeable setups to the top of a scan.
            # To the rung actually being aimed at. Measuring it against the
            # nearest rung while targeting a further one would report headroom
            # the trade is not taking, and the gate reads this number.
            head_sigma = sigma_distance(opp.price, entry, sigma) if opp else None
            head_pct = ((opp.price - entry) / entry * 100) if opp else None
            # How long this level spent CAPPING price, and how recently. The
            # user reads "many are trying to flip resistance levels" off the
            # chart and treats those as needing observation, while calling a
            # level price has never traded below "more aligned with a buy
            # zone". level_origin does not capture that: it only marks levels
            # the resistance detector happened to find, so SOL 101.72 scored
            # swing_low while having capped price for 95 of the last 120 bars,
            # last only 12 bars ago. Time spent under the level is the measure
            # that actually separates the two.
            look = bars[max(0, i - 120):i + 1]
            capped = sum(1 for x in look if x["c"] < L.price)
            last_capped = max((k for k, x in enumerate(look) if x["c"] < L.price), default=None)
            bars_since_capped = (len(look) - 1 - last_capped) if last_capped is not None else -1

            # 7d trajectory, so the user can see how price arrived here
            w = bars[max(0, i - 42):i + 1]
            hi7 = max(x["h"] for x in w); lo7 = min(x["l"] for x in w)
            out.append({
                "episode_id": storage.make_episode_id(coin + "-" + tf, bars[i]["t"], L.price),
                "coin": coin, "timeframe": tf, "detected_at_utc": storage.now_iso(),
                "bar_ts_ms": bars[i]["t"], "level_price": round(L.price, 10),
                "level_kind": "support",
                "level_origin": origin.get(round(L.price, 10), "swing_low"),
                "level_touches": L.touches,
                "level_holds": L.holds, "level_hold_rate": round(L.hold_rate, 4),
                "level_pivot_count": L.pivot_count,
                "level_bars_since_last_touch": (i - L.last_touch_idx) if L.last_touch_idx >= 0 else -1,
                "atr_pct": round(_atr_pct(bars, i), 4),
                "entry_buffer_sigma": C.ENTRY_BUFFER_SIGMA,
                "suggested_entry": round(entry, 10), "suggested_stop": round(stop, 10),
                "suggested_tp": round(tp, 10), "suggested_rr": round(rr, 3),
                "opposing_level_price": round(opp.price, 10) if opp else "",
                "_px": px, "_near": near, "_hi7": hi7, "_lo7": lo7, "_ladder": ladder,
                "_tf": tf,
                "capped_bars": capped,
                "capped_frac": round(capped / max(len(look), 1), 3),
                "bars_since_capped": bars_since_capped,
                "_head_sigma": round(head_sigma, 2) if head_sigma is not None else None,
                "_head_pct": round(head_pct, 2) if head_pct is not None else None,
                "_no_target": head_sigma is None,
                "overhead_sigma": round(head_sigma, 3) if head_sigma is not None else "",
                "overhead_pct": round(head_pct, 3) if head_pct is not None else "",
                "_stale": (i - L.last_touch_idx) > STALE_TOUCH_BARS if L.last_touch_idx >= 0 else True,
                "_from_hi7": (px - hi7) / hi7 * 100, "_stop_pct": (entry - stop) / entry * 100,
                **feats,
            })
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
        stale = "  [STALE -- price left this behind]" if c.get("_stale") else ""
        flip = "  [FLIPPED resistance -> support]" if c.get("level_origin") == "flipped_resistance" else ""
        print(f"[{n}] {c['coin']} {c['_tf']}   price {c['_px']:.6g}{flip}")
        print(f"     level {c['level_price']:.6g}  ({-c['_near']:.2f}s below price, "
              f"{(c['level_price']-c['_px'])/c['_px']*100:+.2f}%)")
        print(f"     history: {c['level_touches']} tests, {c['level_holds']} held "
              f"({c['level_hold_rate']*100:.0f}%), last tested "
              f"{c['level_bars_since_last_touch']} bars ago{stale}")
        print(f"     context: ran {_f(c,'run_sigma'):.1f}s, pulled back {_f(c,'pull_sigma'):.1f}s, "
              f"{int(_f(c,'retests'))} retests, {c['_from_hi7']:+.1f}% from 7d high")
        print(f"     trade  : entry {c['suggested_entry']:.6g}  stop {c['suggested_stop']:.6g} "
              f"({c['_stop_pct']:.1f}%)")
        if c.get("_no_target"):
            print("     headroom: NO resistance mapped above -- no target, not a trade")
        else:
            cap = c.get("capped_bars", 0)
        if cap:
            since = c.get("bars_since_capped", -1)
            print(f"     was resistance: capped price {cap}/120 bars"
                  + (f", last {since} bars ago" if since >= 0 else ""))
        print(f"     headroom: first resistance {c['_head_sigma']:.2f}s above entry "
                  f"({c['_head_pct']:+.2f}%)")
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


def _same_level_labels(eid: str) -> List[Dict[str, Any]]:
    """Existing labels for the SAME level, under a different episode id.

    episode_id is COIN-TF_BARTIMESTAMP_LEVELPRICE, so one level gets a fresh
    id on every bar it gates, and storage dedupes on that id -- which cannot
    see that two rows describe one level. On 2026-09-19 USELESS 1H 0.25817
    was held as both a take and a skip a day apart. The filter is validated
    on "keeps 8 of 11 takes and admits 0 of 5 skips", and a level sitting in
    both halves makes that number unanswerable.

    Relabelling is legitimate -- context moves -- so this warns rather than
    blocks. It just must not happen silently.
    """
    import re
    m = re.match(r"^(.+?)-(1H|4H)_(\d+)_(.+)$", eid)
    if not m:
        return []
    coin, tf, level = m.group(1), m.group(2), m.group(4)
    out = []
    for prior_id, row in _labelled().items():
        pm = re.match(r"^(.+?)-(1H|4H)_(\d+)_(.+)$", prior_id)
        if not pm or prior_id == eid:
            continue
        if (pm.group(1), pm.group(2), pm.group(4)) == (coin, tf, level):
            out.append(row)
    return out


def _record(match, eid: str, verdict: str, note: str) -> bool:
    """Write the episode snapshot, then the label. Shared by both paths so a
    batch cannot drift from a single label."""
    for prior in _same_level_labels(eid):
        flag = "CONTRADICTS" if prior["verdict"] != verdict else "repeats"
        print(f"  !! {flag} an existing label on this same level: "
              f"{prior['verdict']} at {prior['labelled_at_utc']}"
              + (f" -- {prior['note'][:60]}" if prior.get("note") else ""))
    if match is not None:
        storage.append_episode({k: v for k, v in match.items()
                                if not k.startswith("_")})
    return storage.append_label({"episode_id": eid,
                                 "labelled_at_utc": storage.now_iso(),
                                 "verdict": verdict, "confidence": "",
                                 "note": note or ""})


def cmd_batch(path: str):
    """Label many calls from ONE scan.

    cmd_label re-scans the universe per label -- 45 coins across two
    timeframes, so about 90 venue requests EACH. Sixteen verdicts in one
    sitting cost ~1,440 requests against an endpoint that has been returning
    rate limits. This does one scan and applies every line against it.

    The file is one verdict per line:

        COIN LEVEL VERDICT  optional note to end of line
        AAVE 142.84 take    operator's blue zone from the chart
        HYPE 86.747 skip    entry would be at 85

    Matched on (coin, level) rather than episode_id deliberately: an id
    carries its bar timestamp, so the ones printed by a board go stale as soon
    as the next bar opens, which is exactly when somebody is sitting down to
    label them. A level price does not move.
    """
    lines = []
    for n, raw in enumerate(Path(path).read_text().splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 3)
        if len(parts) < 3:
            print(f"  line {n}: expected 'COIN LEVEL VERDICT [note]', got {line!r}")
            return 1
        coin, level_s, verdict = parts[0].upper(), parts[1], parts[2].lower()
        note = parts[3] if len(parts) > 3 else ""
        if verdict not in ("take", "skip", "maybe"):
            print(f"  line {n}: verdict must be take/skip/maybe, got {verdict!r}")
            return 1
        try:
            level = float(level_s)
        except ValueError:
            print(f"  line {n}: {level_s!r} is not a level price")
            return 1
        lines.append((n, coin, level, verdict, note))
    if not lines:
        print("nothing to label")
        return 1

    client = HyperliquidReadOnlyClient()
    by: Dict[str, List[Dict[str, Any]]] = {}
    for c in find_candidates(client, select_universe(client)):
        by.setdefault(str(c["coin"]).upper(), []).append(c)

    done = missed = 0
    for n, coin, level, verdict, note in lines:
        rows = by.get(coin, [])
        match = min(rows, key=lambda e: abs(float(e["level_price"]) - level),
                    default=None)
        # 2% rather than exact: an operator reads a level off a chart, and
        # refusing 142.8 for 142.84 would make the tool useless for the one
        # job it has. Wider than that is a different level, so it is refused
        # rather than silently labelled.
        if match is None or abs(float(match["level_price"]) / level - 1) > 0.02:
            near = (f"nearest {float(match['level_price']):.6g}"
                    if match is not None else "no levels detected")
            print(f"  line {n}: no {coin} level within 2% of {level:g} ({near})")
            missed += 1
            continue
        got = float(match["level_price"])
        off = (got / level - 1) * 100
        if _record(match, match["episode_id"], verdict, note):
            done += 1
            print(f"  recorded {coin:<9}{level:<11g}-> {verdict:<6}"
                  + (f"  [matched {got:.6g}, {off:+.1f}%]" if abs(off) > 0.05 else ""))
        else:
            print(f"  already labelled {coin} {got:.6g}")
    print()
    print(f"{done} recorded, {missed} unmatched")
    print(f"labels: {len(_labelled())} / {TARGET_LABELS}")
    return 1 if missed else 0


def cmd_label(eid: str, verdict: str, note: str):
    """Record the verdict, writing the episode snapshot first so the label has
    something to join to."""
    for prior in _same_level_labels(eid):
        flag = "CONTRADICTS" if prior["verdict"] != verdict else "repeats"
        print(f"  !! {flag} an existing label on this same level: "
              f"{prior['verdict']} at {prior['labelled_at_utc']}"
              + (f" -- {prior['note'][:60]}" if prior.get("note") else ""))
    client = HyperliquidReadOnlyClient()
    universe = select_universe(client)
    match = next((c for c in find_candidates(client, universe) if c["episode_id"] == eid), None)
    if match is None:
        existing = {e["episode_id"] for e in storage.load_episodes()}
        if eid not in existing:
            print(f"no candidate or episode with id {eid}")
            return 1
    ok = _record(match, eid, verdict, note)
    n = len(_labelled())
    print(f"{'recorded' if ok else 'already labelled'}: {eid} -> {verdict}")
    print(f"labels: {n} / {TARGET_LABELS}")
    return 0


def cmd_progress():
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
    ap.add_argument("--maybe", metavar="ID",
                    help="valid level, setup not confirmed yet -- e.g. price is still "
                         "sitting on it and the user wants to see it hold first")
    ap.add_argument("--note", default="")
    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--batch", metavar="FILE",
                    help="label many calls from one scan; one "
                         "'COIN LEVEL VERDICT [note]' per line")
    a = ap.parse_args()
    if a.progress:
        return cmd_progress()
    if a.batch:
        return cmd_batch(a.batch)
    if a.take:
        return cmd_label(a.take, "take", a.note)
    if a.skip:
        return cmd_label(a.skip, "skip", a.note)
    if a.maybe:
        return cmd_label(a.maybe, "maybe", a.note)
    return cmd_list(a)


if __name__ == "__main__":
    raise SystemExit(main())

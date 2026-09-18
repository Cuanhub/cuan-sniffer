"""
Support/resistance level scanner -- logging only.

Watches the funding collector's top-30 universe for the setup the user trades
by hand: a level with a real test history, approached after a run and a
pullback, with the level recently retested. Every gated episode is recorded
with its full feature set, and what price did afterwards is resolved later.

This package places no orders and changes no trading behaviour. It exists to
answer two questions that cannot be answered from the existing logs:

  1. do gated setups make money forward (outcomes.csv)
  2. does the gate agree with the user's own judgement (labels.csv)

Both are needed. The bot's 48 recorded trades sit 7.2pp BELOW the random-walk
baseline for their own RR, while the user's September book returned +4.885%
per closing fill -- so the gap being measured here is between a rule that
demonstrably does not work and a judgement that appears to, on n=9. Guessing
at that gap is what produced four earlier false positives in this pivot.

Usage:
    python -m research.levels.scanner                 # long-lived
    python -m research.levels.scanner --dry-run       # one pass, print, exit
    python -m research.levels.scanner --backfill 90   # replay history first
"""

import argparse
import math
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from research.levels import storage
from research.levels.gate import evaluate, RUN_LOOKBACK
from research.levels.levels import (
    detect_levels, log_sigma, sigma_distance, nearest_level,
)
from research.levels.venue_client import HyperliquidReadOnlyClient, VenueFetchError
from research.levels.universe import select_universe

from research.levels.config import (               # noqa: E402
    POLL_SEC as POLL_INTERVAL_SEC, INTERVAL, HISTORY_DAYS, RESOLVE_MIN_BARS,
    ENTRY_BUFFER_SIGMA, STOP_SIGMA_BELOW, ONSET_MATCH_SIGMA,
)
FAILURE_LOG = Path(__file__).resolve().parent / "data" / "_fetch_failures.log"

_stop = False


def _sigterm(signum, frame):
    global _stop
    _stop = True
    print("[LEVEL_SCANNER] SIGTERM received -- finishing pass, then exiting")


def _log_failure(endpoint: str, attempt: int, error: Exception) -> None:
    FAILURE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(FAILURE_LOG, "a") as f:
        f.write(f"{storage.now_iso()} endpoint={endpoint} attempt={attempt} error={error}\n")
    print(f"[LEVEL_SCANNER][FETCH_FAILURE] {storage.now_iso()} endpoint={endpoint} "
          f"attempt={attempt} error={error}")


def fetch(client, coin: str, days: int = HISTORY_DAYS) -> List[Dict[str, Any]]:
    now_ms = int(time.time() * 1000)
    raw = client.get_candles(coin, INTERVAL, now_ms - days * 86400000, now_ms)
    out = []
    for b in raw:
        try:
            out.append({"t": int(b["t"]), "o": float(b["o"]), "h": float(b["h"]),
                        "l": float(b["l"]), "c": float(b["c"]), "v": float(b.get("v", 0.0))})
        except (KeyError, TypeError, ValueError):
            continue
    return sorted(out, key=lambda x: x["t"])


def _atr_pct(bars: List[Dict[str, Any]], i: int, n: int = 14) -> float:
    seg = bars[max(1, i - n):i + 1]
    if len(seg) < 2:
        return 0.0
    trs = []
    for j in range(1, len(seg)):
        p = seg[j - 1]["c"]
        trs.append(max(seg[j]["h"] - seg[j]["l"], abs(seg[j]["h"] - p), abs(seg[j]["l"] - p)))
    px = bars[i]["c"]
    return (sum(trs) / len(trs) / px * 100) if trs and px > 0 else 0.0


def _fired_previous_bar(bars: List[Dict[str, Any]], i: int, level_price: float,
                        sigma: float) -> bool:
    """Was this same level already gated on the previous bar?

    An episode is the ONSET of a gated condition, not every bar it persists
    for. Logging each bar turned 45 days of backfill into 5,948 rows covering
    far fewer real setups, which both drowns the review queue in duplicates
    and would silently weight any later fit toward whichever setups happened
    to stay gated longest. Levels are re-detected per bar and drift slightly,
    so the match is by proximity in sigma rather than exact price.

    Single-bar path only. It re-runs detection for bar i-1, which is fine for
    a live pass (one bar) but quadratic over a backfill -- a 45-day run had
    not finished after 8h44m. scan_series() below is the linear equivalent
    and is what --backfill uses.
    """
    if i <= 0:
        return False
    prev_close = bars[i - 1]["c"]
    prev_levels = detect_levels(bars, upto=i, kind="support")
    for P in prev_levels:
        # scan_coin only gates levels BELOW price -- these are supports being
        # bought into. Without the same filter here a level sitting above the
        # previous close counts as "already gated", suppressing a genuine
        # onset on the bar where price first drops through it. That made this
        # path disagree with scan_series on 12 of 35 CASHCAT episodes.
        if P.price >= prev_close:
            continue
        if abs(sigma_distance(P.price, level_price, sigma)) <= ONSET_MATCH_SIGMA:
            f = evaluate(bars, i - 1, P.price)
            if f and f["gate_fires"]:
                return True
    return False


def scan_series(coin: str, bars: List[Dict[str, Any]], start: int = 120,
                until_ts: int = None) -> List[Dict[str, Any]]:
    """Walk the whole series once, emitting an episode only where a level's
    gate OPENS. Carries the previous bar's gated levels forward instead of
    recomputing them, which is what makes a backfill finish."""
    rows: List[Dict[str, Any]] = []
    prev_gated: List[float] = []
    for i in range(start, len(bars)):
        if until_ts is not None and bars[i]["t"] < until_ts:
            # still need the gated set carried forward, so evaluate but do not emit
            found = scan_coin(coin, bars, i, onset_only=False)
            prev_gated = [float(r["level_price"]) for r in found]
            continue
        sigma = log_sigma(bars, i)
        found = scan_coin(coin, bars, i, onset_only=False)
        for r in found:
            lp = float(r["level_price"])
            if sigma > 0 and any(abs(sigma_distance(q, lp, sigma)) <= ONSET_MATCH_SIGMA
                                 for q in prev_gated):
                continue     # gate was already open on this level last bar
            rows.append(r)
        prev_gated = [float(r["level_price"]) for r in found]
    return rows


def scan_coin(coin: str, bars: List[Dict[str, Any]], i: int = None,
              onset_only: bool = True) -> List[Dict[str, Any]]:
    """Evaluate every detected support level as of bar i. Returns episode rows
    for levels whose gate fires, by default only where the gate has just
    OPENED rather than where it merely remains open."""
    i = (len(bars) - 1) if i is None else i
    if i < 60:
        return []
    sigma = log_sigma(bars, i)
    if sigma <= 0:
        return []
    supports = detect_levels(bars, upto=i + 1, kind="support")
    resistances = detect_levels(bars, upto=i + 1, kind="resistance")
    px = bars[i]["c"]
    rows = []
    for L in supports:
        if L.price >= px:
            continue
        feats = evaluate(bars, i, L.price)
        if not feats or not feats["gate_fires"]:
            continue
        if onset_only and _fired_previous_bar(bars, i, L.price, sigma):
            continue
        entry = L.price * math.exp(ENTRY_BUFFER_SIGMA * sigma)
        stop = L.price * math.exp(-STOP_SIGMA_BELOW * sigma)
        opp = nearest_level(resistances, px, sigma, below=False)
        tp = opp.price if opp else px * math.exp(8 * sigma)
        rr = ((tp - entry) / (entry - stop)) if entry > stop else 0.0
        rows.append({
            "episode_id": storage.make_episode_id(coin, bars[i]["t"], L.price),
            "coin": coin,
            "detected_at_utc": storage.now_iso(),
            "bar_ts_ms": bars[i]["t"],
            "level_price": round(L.price, 10),
            "level_kind": "support",
            "level_touches": L.touches,
            "level_holds": L.holds,
            "level_hold_rate": round(L.hold_rate, 4),
            "level_pivot_count": L.pivot_count,
            "level_bars_since_last_touch": (i - L.last_touch_idx) if L.last_touch_idx >= 0 else -1,
            "atr_pct": round(_atr_pct(bars, i), 4),
            "entry_buffer_sigma": ENTRY_BUFFER_SIGMA,
            "suggested_entry": round(entry, 10),
            "suggested_stop": round(stop, 10),
            "suggested_tp": round(tp, 10),
            "suggested_rr": round(rr, 3),
            "opposing_level_price": round(opp.price, 10) if opp else "",
            **feats,
        })
    return rows


def resolve_outcomes(client, universe_syms: List[str]) -> int:
    """Write outcomes for episodes old enough to judge."""
    done = storage.resolved_ids()
    pending = [e for e in storage.load_episodes() if e["episode_id"] not in done]
    if not pending:
        return 0
    by_coin: Dict[str, List[Dict[str, Any]]] = {}
    for e in pending:
        by_coin.setdefault(e["coin"], []).append(e)
    written = 0
    for coin, eps in by_coin.items():
        try:
            bars = fetch(client, coin)
        except VenueFetchError:
            continue
        for e in eps:
            try:
                t0 = int(e["bar_ts_ms"]); entry = float(e["suggested_entry"])
                stop = float(e["suggested_stop"]); tp = float(e["suggested_tp"])
                sigma = float(e["sigma_4h"])
            except (TypeError, ValueError):
                continue
            fwd = [b for b in bars if b["t"] > t0]
            if len(fwd) < RESOLVE_MIN_BARS:
                continue
            fill_i = next((k for k, b in enumerate(fwd) if b["l"] <= entry), None)
            row = {"episode_id": e["episode_id"], "coin": coin,
                   "resolved_at_utc": storage.now_iso(), "bars_elapsed": len(fwd),
                   "reached_level": bool(fill_i is not None),
                   "bars_to_reach": fill_i if fill_i is not None else -1}
            for lab, n in (("close_after_24h_pct", 6), ("close_after_72h_pct", 18),
                           ("close_after_168h_pct", 42)):
                row[lab] = round((fwd[n]["c"] - float(e["price"])) / float(e["price"]) * 100, 4) \
                    if len(fwd) > n else ""
            if fill_i is None:
                row.update({"fill_price": "", "mae_pct": "", "mfe_pct": "", "mae_sigma": "",
                            "mfe_sigma": "", "hit_suggested_stop": "", "hit_suggested_tp": "",
                            "bars_to_stop": -1, "bars_to_tp": -1, "outcome_r": "",
                            "level_broke": ""})
            else:
                after = fwd[fill_i:]
                lo = min(b["l"] for b in after); hi = max(b["h"] for b in after)
                s_i = next((k for k, b in enumerate(after) if b["l"] <= stop), None)
                t_i = next((k for k, b in enumerate(after) if b["h"] >= tp), None)
                if t_i is not None and (s_i is None or t_i < s_i):
                    r = (tp - entry) / (entry - stop) if entry > stop else 0.0
                elif s_i is not None:
                    r = -1.0
                else:
                    r = ((after[-1]["c"] - entry) / (entry - stop)) if entry > stop else 0.0
                row.update({
                    "fill_price": round(entry, 10),
                    "mae_pct": round((lo - entry) / entry * 100, 4),
                    "mfe_pct": round((hi - entry) / entry * 100, 4),
                    "mae_sigma": round(sigma_distance(lo, entry, sigma), 3) if sigma > 0 else "",
                    "mfe_sigma": round(sigma_distance(hi, entry, sigma), 3) if sigma > 0 else "",
                    "hit_suggested_stop": bool(s_i is not None),
                    "hit_suggested_tp": bool(t_i is not None),
                    "bars_to_stop": s_i if s_i is not None else -1,
                    "bars_to_tp": t_i if t_i is not None else -1,
                    "outcome_r": round(r, 4),
                    "level_broke": bool(any(b["c"] < float(e["level_price"]) * math.exp(-3 * sigma)
                                            for b in after)),
                })
            if storage.append_outcome(row):
                written += 1
    return written


def run_pass(client, universe, dry_run=False):
    found = []
    for entry in universe:
        coin = entry["symbol"]
        try:
            bars = fetch(client, coin)
        except VenueFetchError:
            continue
        if len(bars) < 120:
            continue
        for row in scan_coin(coin, bars):
            found.append(row)
            if not dry_run:
                storage.append_episode(row)
    return found


def main():
    ap = argparse.ArgumentParser(description="Support/resistance level scanner (logging only)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--backfill", type=int, default=0,
                    help="replay N days of history once before live scanning")
    args = ap.parse_args()

    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGINT, _sigterm)

    client = HyperliquidReadOnlyClient(on_failure=_log_failure)
    universe = select_universe(client)
    print(f"[LEVEL_SCANNER] universe: {len(universe)} symbols")
    print(f"[LEVEL_SCANNER] gate: run>={6.0}s pull>={5.0}s retests>=2 -- PLACEHOLDER thresholds "
          f"fitted to n=2, expected to move once labelled")

    if args.backfill:
        total = 0
        for e in universe:
            coin = e["symbol"]
            try:
                bars = fetch(client, coin)
            except VenueFetchError:
                continue
            if len(bars) < 120:
                continue
            cutoff = int(time.time() * 1000) - args.backfill * 86400000
            n_coin = 0
            for row in scan_series(coin, bars, start=120, until_ts=cutoff):
                if storage.append_episode(row):
                    total += 1
                    n_coin += 1
            print(f"[LEVEL_SCANNER] backfill {coin}: {n_coin} episode(s)", flush=True)
        print(f"[LEVEL_SCANNER] backfill wrote {total} episode(s)")

    if args.dry_run:
        found = run_pass(client, universe, dry_run=True)
        print(f"\n[LEVEL_SCANNER] episode schema: {storage.EPISODE_FIELDNAMES}")
        print(f"[LEVEL_SCANNER] {len(found)} gated episode(s) this pass (not written, dry-run):")
        for r in found:
            print(f"  {r['coin']:10} level={r['level_price']:.6g} hold={r['level_hold_rate']*100:.0f}% "
                  f"run={r['run_sigma']:.1f}s pull={r['pull_sigma']:.1f}s retests={r['retests']} "
                  f"entry={r['suggested_entry']:.6g} rr={r['suggested_rr']:.2f}")
        return

    print(f"[LEVEL_SCANNER] starting long-lived scan, poll={POLL_INTERVAL_SEC}s")
    last_universe_day = datetime.now(timezone.utc).date()
    while not _stop:
        t0 = time.time()
        if datetime.now(timezone.utc).date() != last_universe_day:
            universe = select_universe(client)
            last_universe_day = datetime.now(timezone.utc).date()
        found = run_pass(client, universe)
        n = resolve_outcomes(client, [u["symbol"] for u in universe])
        print(f"[LEVEL_SCANNER] pass complete: {len(found)} gated, {n} outcome(s) resolved "
              f"at {storage.now_iso()}")
        for _ in range(int(max(0.0, POLL_INTERVAL_SEC - (time.time() - t0)))):
            if _stop:
                break
            time.sleep(1)
    print("[LEVEL_SCANNER] stopped cleanly")


if __name__ == "__main__":
    main()

"""
backtest_missed.py
------------------
Backtests every unique missed signal from missed_signals.csv against real
Hyperliquid price data.

Method
------
1. Load missed_signals.csv (13k+ rows)
2. Deduplicate by (coin, side, entry_price, stop_price) — keep first occurrence.
   This collapses 20 identical JTO SHORTs from a 2-min dedup window into one.
3. Only evaluate signals old enough that the full 6H eval window has elapsed.
4. Per coin: fetch 5m candles for the full date range in one bulk request.
5. For each signal, scan candles from signal time:
   - LONG: first bar where low  <= stop_price → LOSS; high >= tp_price → WIN
   - SHORT: first bar where high >= stop_price → LOSS; low  <= tp_price → WIN
   - Neither within 6H → TIMEOUT (unrealized R at final close)
6. Print full breakdown: overall, by reject_reason, by regime family,
   by session, by confidence tier, by coin.

Usage
-----
  python3 tools/research/backtest_missed.py
  python3 tools/research/backtest_missed.py --eval-hours 4
  python3 tools/research/backtest_missed.py --from-date 2026-06-13
"""

import csv
import time
import sys
import argparse
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import requests

MISSED_CSV = "missed_signals.csv"
HYPERLIQUID_URL = "https://api.hyperliquid.xyz/info"
EVAL_HOURS = 6          # evaluation window per signal
MIN_AGE_HOURS = 6       # signal must be this old (eval window elapsed)
API_DELAY = 1.3         # seconds between API requests
MAX_RETRIES = 3
RETRY_BACKOFF = 6


# ── Candle fetch ──────────────────────────────────────────────────────────────

def fetch_candles_range(
    coin: str,
    start_ms: int,
    end_ms: int,
    interval: str = "5m",
) -> List[Dict]:
    """Fetch candles for a single coin over the given ms range. Returns list of bar dicts."""
    payload = {
        "type": "candleSnapshot",
        "req": {
            "coin": coin,
            "interval": interval,
            "startTime": start_ms,
            "endTime": end_ms,
        },
    }
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(HYPERLIQUID_URL, json=payload, timeout=15)
            if resp.status_code == 429:
                w = RETRY_BACKOFF * attempt
                print(f"  [429] rate-limited, waiting {w}s …", flush=True)
                time.sleep(w)
                continue
            resp.raise_for_status()
            data = resp.json()
            bars = []
            for c in data:
                ts_ms = c.get("t") or c.get("T") or 0
                bars.append({
                    "ts_ms": int(ts_ms),
                    "open":  float(c.get("o", 0)),
                    "high":  float(c.get("h", 0)),
                    "low":   float(c.get("l", 0)),
                    "close": float(c.get("c", 0)),
                })
            return bars
        except requests.RequestException as e:
            if attempt == MAX_RETRIES:
                print(f"  [FETCH ERROR] {coin}: {e}", flush=True)
                return []
            time.sleep(RETRY_BACKOFF * attempt)
    return []


# ── Signal loader ─────────────────────────────────────────────────────────────

def load_and_dedup(path: str, from_date: Optional[datetime] = None) -> List[Dict]:
    """
    Load missed_signals.csv, normalise fields, and deduplicate.
    Dedup key: (coin, side, rounded entry, rounded stop).
    Keeps the first occurrence (earliest timestamp per unique setup).
    """
    seen = {}
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            ts_str = r.get("timestamp", "").strip()
            if not ts_str:
                continue
            try:
                ts = datetime.fromisoformat(ts_str)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
            except ValueError:
                continue

            if from_date and ts < from_date:
                continue

            coin = r.get("coin", "").strip().upper()
            side = r.get("side", "").strip().upper()
            try:
                entry = float(r["entry_price"])
                stop  = float(r["stop_price"])
                tp    = float(r["tp_price"])
            except (KeyError, ValueError):
                continue

            if entry <= 0 or stop <= 0 or tp <= 0:
                continue
            if side not in ("LONG", "SHORT"):
                continue

            # Dedup key — round to 7 sig-figs to collapse float noise
            key = (coin, side, round(entry, 7), round(stop, 7))
            if key in seen:
                continue
            seen[key] = True

            stop_dist = abs(entry - stop)
            if stop_dist <= 0:
                continue

            if side == "LONG":
                rr = (tp - entry) / stop_dist
            else:
                rr = (entry - tp) / stop_dist

            rows.append({
                "ts": ts,
                "coin": coin,
                "side": side,
                "entry": entry,
                "stop": stop,
                "tp": tp,
                "stop_dist": stop_dist,
                "rr": round(rr, 3),
                "confidence": float(r.get("confidence") or 0),
                "total_score": float(r.get("total_score") or 0),
                "session": r.get("session", "").strip(),
                "regime": r.get("regime", "").strip(),
                "reject_reason": r.get("reject_reason", "").strip(),
            })

    rows.sort(key=lambda x: x["ts"])
    return rows


# ── Simulation ────────────────────────────────────────────────────────────────

def simulate(
    signal: Dict,
    candles: List[Dict],
    eval_hours: int = EVAL_HOURS,
) -> Tuple[str, float]:
    """
    Walk forward from signal timestamp through 5m bars.
    Returns (outcome, r_multiple).
      outcome: "win" | "loss" | "timeout"
    """
    sig_ms  = int(signal["ts"].timestamp() * 1000)
    end_ms  = sig_ms + eval_hours * 3600 * 1000
    side    = signal["side"]
    entry   = signal["entry"]
    stop    = signal["stop"]
    tp      = signal["tp"]
    stop_d  = signal["stop_dist"]
    rr      = signal["rr"]

    # Find bars that are AFTER the signal time
    for bar in candles:
        bts = bar["ts_ms"]
        if bts < sig_ms:
            continue
        if bts > end_ms:
            break
        h = bar["high"]
        lo = bar["low"]
        cl = bar["close"]

        if side == "LONG":
            # Check SL and TP on same bar — SL takes priority (conservative)
            if lo <= stop and h >= tp:
                return "loss", -1.0
            if lo <= stop:
                return "loss", -1.0
            if h >= tp:
                return "win", round(rr, 3)
            last_close = cl
        else:
            if h >= stop and lo <= tp:
                return "loss", -1.0
            if h >= stop:
                return "loss", -1.0
            if lo <= tp:
                return "win", round(rr, 3)
            last_close = cl
    else:
        last_close = candles[-1]["close"] if candles else entry

    # Timeout: unrealized R
    if side == "LONG":
        unreal = (last_close - entry) / stop_d
    else:
        unreal = (entry - last_close) / stop_d
    return "timeout", round(unreal, 3)


# ── Stats helpers ─────────────────────────────────────────────────────────────

def stats_block(results: List[Tuple[str, float]], label: str, indent: int = 0) -> None:
    if not results:
        return
    pad = " " * indent
    resolved = [(o, r) for o, r in results if o in ("win", "loss")]
    n_total   = len(results)
    n_res     = len(resolved)
    n_win     = sum(1 for o, _ in resolved if o == "win")
    n_loss    = sum(1 for o, _ in resolved if o == "loss")
    n_timeout = n_total - n_res
    win_rate  = n_win / n_res if n_res else 0.0
    r_vals    = [r for _, r in results]
    exp_r     = sum(r_vals) / len(r_vals) if r_vals else 0.0
    win_rs    = [r for o, r in results if o == "win"]
    avg_win   = sum(win_rs) / len(win_rs) if win_rs else 0.0

    print(f"{pad}{label}")
    print(f"{pad}  n={n_total}  resolved={n_res}  win={n_win}({win_rate:.0%})  "
          f"loss={n_loss}  timeout={n_timeout}")
    print(f"{pad}  ExpR={exp_r:+.3f}  AvgWinR={avg_win:.3f}")


def extract_regime_family(regime: str) -> str:
    """reversal|htf_up|macro_up|mkt_strong_trend  →  reversal"""
    parts = regime.split("|")
    return parts[0] if parts else "unknown"


def extract_mkt_regime(regime: str) -> str:
    for part in regime.split("|"):
        if part.startswith("mkt_"):
            return part[4:]  # drop "mkt_" prefix
    return "unknown"


def normalise_reject(reason: str) -> str:
    if "continuation" in reason:
        return "hard_block_continuation"
    if "chop" in reason and "swing_conf" not in reason:
        return "market_regime:chop"
    if "swing_conf" in reason:
        return "swing_conf_gate"
    if "dual_trend" in reason:
        return "reversal_against_dual_trend"
    if "cooldown" in reason:
        return "cooldown"
    return reason[:60]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Backtest missed signals")
    parser.add_argument("--eval-hours", type=int, default=EVAL_HOURS,
                        help="Hours of price data to evaluate each signal (default 6)")
    parser.add_argument("--from-date", type=str, default=None,
                        help="Only test signals from this date onward (YYYY-MM-DD)")
    parser.add_argument("--interval", type=str, default="5m",
                        help="Candle interval for simulation (default 5m)")
    args = parser.parse_args()

    from_date = None
    if args.from_date:
        from_date = datetime.fromisoformat(args.from_date).replace(tzinfo=timezone.utc)

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=args.eval_hours)  # signal must be old enough

    print("=" * 68)
    print("BACKTEST — MISSED SIGNALS")
    print(f"Source : {MISSED_CSV}")
    print(f"Eval   : {args.eval_hours}h window | {args.interval} candles")
    if from_date:
        print(f"Filter : signals from {from_date.date()} onward")
    print("=" * 68)

    # Load & deduplicate
    print("\nLoading & deduplicating …", flush=True)
    signals = load_and_dedup(MISSED_CSV, from_date=from_date)

    # Remove signals too recent to have full eval window
    testable = [s for s in signals if s["ts"] <= cutoff]
    skipped  = len(signals) - len(testable)
    print(f"Unique setups   : {len(signals)}")
    print(f"Testable (≥{args.eval_hours}h old): {len(testable)}  (skipped {skipped} too recent)")

    if not testable:
        print("Nothing to test.")
        return

    # ── Fetch candles per coin ────────────────────────────────────────────────
    coins = list({s["coin"] for s in testable})
    coins.sort()

    # Determine date range across all signals
    earliest_ts = min(s["ts"] for s in testable)
    # Fetch from earliest signal through now + buffer
    range_start_ms = int((earliest_ts - timedelta(minutes=10)).timestamp() * 1000)
    range_end_ms   = int((now + timedelta(hours=1)).timestamp() * 1000)

    candle_cache: Dict[str, List[Dict]] = {}
    print(f"\nFetching {args.interval} candles for {len(coins)} coins …", flush=True)

    interval_min = int(args.interval.rstrip("m"))
    # Hyperliquid returns ≤5000 bars per request; split if range is wide
    range_minutes = (range_end_ms - range_start_ms) / 60_000
    bars_needed = int(range_minutes / interval_min) + 50

    for coin in coins:
        all_bars: List[Dict] = []
        if bars_needed <= 5000:
            print(f"  {coin}: fetching ~{bars_needed} bars …", flush=True)
            bars = fetch_candles_range(coin, range_start_ms, range_end_ms, interval=args.interval)
            all_bars = bars
            time.sleep(API_DELAY)
        else:
            # Chunk into 5000-bar windows (~17 days at 5m)
            chunk_ms = 5000 * interval_min * 60 * 1000
            cur = range_start_ms
            while cur < range_end_ms:
                chunk_end = min(cur + chunk_ms, range_end_ms)
                chunk_bars_needed = (chunk_end - cur) // (interval_min * 60_000)
                print(f"  {coin}: chunk {datetime.fromtimestamp(cur/1000, tz=timezone.utc).date()} (~{chunk_bars_needed} bars)", flush=True)
                bars = fetch_candles_range(coin, cur, chunk_end, interval=args.interval)
                all_bars.extend(bars)
                cur = chunk_end
                time.sleep(API_DELAY)

        candle_cache[coin] = sorted(all_bars, key=lambda b: b["ts_ms"])
        print(f"  {coin}: {len(candle_cache[coin])} bars cached", flush=True)

    # ── Simulate each signal ──────────────────────────────────────────────────
    print(f"\nSimulating {len(testable)} signals …", flush=True)
    outcomes: List[Dict] = []
    for i, sig in enumerate(testable, 1):
        bars = candle_cache.get(sig["coin"], [])
        outcome, r_mult = simulate(sig, bars, eval_hours=args.eval_hours)
        outcomes.append({**sig, "outcome": outcome, "r_mult": r_mult})
        if i % 50 == 0:
            print(f"  … {i}/{len(testable)}", flush=True)

    if not outcomes:
        print("No results.")
        return

    # ── Overall results ───────────────────────────────────────────────────────
    print()
    print("=" * 68)
    print("RESULTS")
    print("=" * 68)

    all_pairs = [(o["outcome"], o["r_mult"]) for o in outcomes]
    stats_block(all_pairs, "OVERALL", indent=0)

    # ── By reject reason ──────────────────────────────────────────────────────
    print()
    print("── By reject reason ──────────────────────────────────")
    by_reject: Dict[str, List] = defaultdict(list)
    for o in outcomes:
        key = normalise_reject(o["reject_reason"])
        by_reject[key].append((o["outcome"], o["r_mult"]))
    for reason, pairs in sorted(by_reject.items(), key=lambda x: -len(x[1])):
        stats_block(pairs, reason, indent=2)

    # ── By regime family (first pipe segment) ──────────────────────────────────
    print()
    print("── By setup family (from regime string) ──────────────")
    by_family: Dict[str, List] = defaultdict(list)
    for o in outcomes:
        key = extract_regime_family(o["regime"])
        by_family[key].append((o["outcome"], o["r_mult"]))
    for fam, pairs in sorted(by_family.items(), key=lambda x: -len(x[1])):
        stats_block(pairs, fam, indent=2)

    # ── By market regime ──────────────────────────────────────────────────────
    print()
    print("── By market regime ──────────────────────────────────")
    by_mkt: Dict[str, List] = defaultdict(list)
    for o in outcomes:
        key = extract_mkt_regime(o["regime"])
        by_mkt[key].append((o["outcome"], o["r_mult"]))
    for mkt, pairs in sorted(by_mkt.items(), key=lambda x: -len(x[1])):
        stats_block(pairs, mkt, indent=2)

    # ── By session ────────────────────────────────────────────────────────────
    print()
    print("── By session ────────────────────────────────────────")
    by_sess: Dict[str, List] = defaultdict(list)
    for o in outcomes:
        by_sess[o["session"]].append((o["outcome"], o["r_mult"]))
    for sess, pairs in sorted(by_sess.items(), key=lambda x: -len(x[1])):
        stats_block(pairs, sess or "(unknown)", indent=2)

    # ── By confidence tier ────────────────────────────────────────────────────
    print()
    print("── By confidence tier ────────────────────────────────")
    tiers = [
        ("≥0.90", lambda c: c >= 0.90),
        ("0.85–0.89", lambda c: 0.85 <= c < 0.90),
        ("0.80–0.84", lambda c: 0.80 <= c < 0.85),
        ("0.70–0.79", lambda c: 0.70 <= c < 0.80),
        ("<0.70",     lambda c: c < 0.70),
    ]
    for label, fn in tiers:
        tier_pairs = [(o["outcome"], o["r_mult"]) for o in outcomes if fn(o["confidence"])]
        if tier_pairs:
            stats_block(tier_pairs, f"conf {label}", indent=2)

    # ── By coin ───────────────────────────────────────────────────────────────
    print()
    print("── By coin ───────────────────────────────────────────")
    by_coin: Dict[str, List] = defaultdict(list)
    for o in outcomes:
        by_coin[o["coin"]].append((o["outcome"], o["r_mult"]))
    for coin, pairs in sorted(by_coin.items(), key=lambda x: -len(x[1])):
        stats_block(pairs, coin, indent=2)

    # ── Key policy questions ──────────────────────────────────────────────────
    print()
    print("=" * 68)
    print("POLICY QUESTIONS")
    print("=" * 68)

    # Q1: continuation signals — should they be unblocked?
    cont = [(o["outcome"], o["r_mult"]) for o in outcomes
            if "continuation" in o["reject_reason"]]
    if cont:
        resolved = [r for o, r in cont if o in ("win", "loss")]
        wins = sum(1 for o, _ in cont if o == "win")
        wr = wins / len(resolved) if resolved else 0
        exp_r = sum(r for _, r in cont) / len(cont)
        print(f"\nQ1  continuation hard-block: should it be relaxed?")
        print(f"    n={len(cont)} signals, WR={wr:.0%}, ExpR={exp_r:+.3f}")
        if exp_r > 0.10 and wr > 0.50:
            print(f"    → POSSIBLE EDGE. Consider regime-gated unblock (strong_trend only).")
        else:
            print(f"    → Block appears JUSTIFIED (low ExpR or WR).")

    # Q2: chop-blocked reversal signals — any edge?
    chop_rev = [(o["outcome"], o["r_mult"]) for o in outcomes
                if "chop" in o["reject_reason"] and "reversal" in o["regime"]]
    if chop_rev:
        resolved = [r for o, r in chop_rev if o in ("win", "loss")]
        wins = sum(1 for o, _ in chop_rev if o == "win")
        wr = wins / len(resolved) if resolved else 0
        exp_r = sum(r for _, r in chop_rev) / len(chop_rev)
        print(f"\nQ2  chop-blocked reversal signals: any edge?")
        print(f"    n={len(chop_rev)} signals, WR={wr:.0%}, ExpR={exp_r:+.3f}")
        if exp_r > 0.0 and wr > 0.50:
            print(f"    → Some edge exists. Consider high-conf override (score≥0.90).")
        else:
            print(f"    → Chop block JUSTIFIED.")

    # Q3: swing_conf_gate signals — the 0.87-0.89 band
    swing_g = [(o["outcome"], o["r_mult"]) for o in outcomes
               if "swing_conf" in o["reject_reason"]]
    if swing_g:
        resolved = [r for o, r in swing_g if o in ("win", "loss")]
        wins = sum(1 for o, _ in swing_g if o == "win")
        wr = wins / len(resolved) if resolved else 0
        exp_r = sum(r for _, r in swing_g) / len(swing_g)
        print(f"\nQ3  swing_conf_gate signals (conf 0.73–0.89): justified?")
        print(f"    n={len(swing_g)} signals, WR={wr:.0%}, ExpR={exp_r:+.3f}")
        conf_band = [(o["outcome"], o["r_mult"]) for o, d in zip(swing_g, [o for o in outcomes if "swing_conf" in o["reject_reason"]])
                     if d["confidence"] >= 0.87]
        # Above is clunky — redo cleanly
        high_conf = [(o["outcome"], o["r_mult"]) for o in outcomes
                     if "swing_conf" in o["reject_reason"] and o["confidence"] >= 0.87]
        if high_conf:
            hc_res = [r for o, r in high_conf if o in ("win", "loss")]
            hc_wins = sum(1 for o, _ in high_conf if o == "win")
            hc_wr = hc_wins / len(hc_res) if hc_res else 0
            hc_expr = sum(r for _, r in high_conf) / len(high_conf)
            print(f"    Conf≥0.87 subset: n={len(high_conf)}, WR={hc_wr:.0%}, ExpR={hc_expr:+.3f}")
        if exp_r > 0.0:
            print(f"    → SWING_MIN_CONFIDENCE=0.87 lowering is SUPPORTED by data.")
        else:
            print(f"    → Keeping SWING_MIN_CONFIDENCE high appears justified.")

    print()
    print("Done.")


if __name__ == "__main__":
    main()

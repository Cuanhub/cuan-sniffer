#!/usr/bin/env python3
"""
Production policy replay — compare engine-only vs production executor.

Runs the shared execution_policy over shadow candidates and compares
four scenarios: engine_only, production_executor, no_stop_redesign,
no_chop_block.

This is the promotion source of truth. engine_only is research;
production_executor is what would actually trade.

Usage:
    python3 tools/research/analyze_production_policy_replay.py
    python3 tools/research/analyze_production_policy_replay.py --days 30
"""

import argparse
import csv
import json
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import requests
except ImportError:
    requests = None

from executor_modules.execution_policy import (
    ExecutionPolicyConfig,
    ExecutionPolicyResult,
    evaluate_execution_policy,
    infer_execution_track,
)

HYPERLIQUID_INFO_URL = "https://api.hyperliquid.xyz/info"

# ── Helpers ──────────────────────────────────────────────────────────

def _sf(val, default=0.0):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _stats(rs):
    if not rs:
        return {"n": 0, "wr": 0, "avg": 0, "med": 0, "total": 0, "pf": 0, "mdd": 0}
    n = len(rs)
    w = sum(1 for r in rs if r > 0)
    gw = sum(r for r in rs if r > 0)
    gl = abs(sum(r for r in rs if r < 0))
    cum = 0; pk = 0; mdd = 0
    for r in rs:
        cum += r
        if cum > pk: pk = cum
        if pk - cum > mdd: mdd = pk - cum
    return {
        "n": n, "wr": w / n * 100, "avg": sum(rs) / n,
        "med": statistics.median(rs), "total": sum(rs),
        "pf": gw / gl if gl > 0 else (999 if gw > 0 else 0), "mdd": mdd,
    }


def _print_stats(label, rs):
    s = _stats(rs)
    print(f"  {label}")
    print(f"    n={s['n']}  WR={s['wr']:.1f}%  AvgR={s['avg']:+.3f}  MedR={s['med']:+.3f}")
    print(f"    TotalR={s['total']:+.2f}  PF={s['pf']:.3f}  MaxDD={s['mdd']:.2f}R")
    return s


def _breakdown(label, rows, key_fn):
    groups = defaultdict(list)
    for r in rows:
        groups[key_fn(r)].append(r["exit_r"])
    print(f"\n  {label}:")
    print(f"  {'Key':<15s} │ {'n':>4s} │ {'WR%':>6s} │ {'AvgR':>7s} │ {'TotalR':>8s} │ {'PF':>6s}")
    print(f"  {'─'*15} ┼ {'─'*4} ┼ {'─'*6} ┼ {'─'*7} ┼ {'─'*8} ┼ {'─'*6}")
    for key in sorted(groups, key=lambda k: -sum(groups[k])):
        s = _stats(groups[key])
        print(f"  {str(key):<15s} │ {s['n']:>4} │ {s['wr']:>5.1f}% │ {s['avg']:>+6.3f} │ {s['total']:>+7.2f} │ {s['pf']:>6.3f}")


# ── Data loading ─────────────────────────────────────────────────────

def load_candidates(csv_path=None):
    candidates = []
    path = Path(csv_path) if csv_path else PROJECT_ROOT / "shadow_research_candidates.csv"
    if not path.exists():
        return candidates
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            e = _sf(row.get("entry_price"))
            s = _sf(row.get("stop_price"))
            t = _sf(row.get("tp_price"))
            side = str(row.get("side", "")).upper()
            if e <= 0 or s <= 0 or t <= 0 or side not in ("LONG", "SHORT"):
                continue
            if side == "LONG" and (e - s <= 0 or t <= e):
                continue
            if side == "SHORT" and (s - e <= 0 or t >= e):
                continue
            candidates.append({
                "ts": row.get("timestamp_utc", ""),
                "sym": row.get("symbol", "").upper(),
                "side": side,
                "entry": e, "stop": s, "tp": t, "risk": abs(e - s),
                "v1": _sf(row.get("score_v1")),
                "v3": _sf(row.get("score_v3")),
                "family": row.get("setup_family", ""),
                "session": row.get("session", ""),
                "mkt": row.get("market_regime", ""),
                "htf": row.get("htf_regime", ""),
                "macro": row.get("macro_regime", ""),
                "stop_method": row.get("stop_method", ""),
                "atr": _sf(row.get("atr")),
                "timeframe": row.get("timeframe", "1h"),
                "confidence": _sf(row.get("confidence_v1")),
            })
    return candidates


_candle_cache: Dict[str, List[Dict]] = {}


def fetch_candles(symbol, start_dt, end_dt):
    key = f"{symbol}_{start_dt}_{end_dt}"
    if key in _candle_cache:
        return _candle_cache[key]
    if requests is None:
        return []
    try:
        resp = requests.post(HYPERLIQUID_INFO_URL, json={
            "type": "candleSnapshot",
            "req": {"coin": symbol, "interval": "1h",
                    "startTime": int(start_dt.timestamp() * 1000),
                    "endTime": int(end_dt.timestamp() * 1000)},
        }, timeout=15)
        data = resp.json()
        candles = sorted([
            {"time": int(c.get("t") or c.get("T")),
             "high": float(c["h"]), "low": float(c["l"]), "close": float(c["c"])}
            for c in data if c.get("t") or c.get("T")
        ], key=lambda x: x["time"])
        _candle_cache[key] = candles
        return candles
    except Exception:
        return []


def replay_trade(entry, stop, tp, side, candles, sig_ms, max_hold=24):
    risk = abs(entry - stop)
    if risk <= 0:
        return None
    idx = next((i for i, c in enumerate(candles) if c["time"] > sig_ms), None)
    if idx is None:
        return None
    for i in range(idx, min(idx + max_hold, len(candles))):
        c = candles[i]
        if side == "LONG":
            if c["low"] <= stop: return -1.0
            if c["high"] >= tp: return (tp - entry) / risk
        else:
            if c["high"] >= stop: return -1.0
            if c["low"] <= tp: return (entry - tp) / risk
    last = candles[min(idx + max_hold - 1, len(candles) - 1)]
    return ((last["close"] - entry) / risk if side == "LONG"
            else (entry - last["close"]) / risk)


def apply_policy(cand, config):
    track = infer_execution_track(
        timeframe=cand["timeframe"],
        setup_family=cand["family"],
    )
    result = evaluate_execution_policy(
        entry=cand["entry"], stop=cand["stop"], tp=cand["tp"],
        side=cand["side"], atr=cand["atr"],
        timeframe=cand["timeframe"],
        session=cand["session"], market_regime=cand["mkt"],
        htf_regime=cand["htf"], macro_regime=cand["macro"],
        setup_family=cand["family"],
        confidence=cand["confidence"],
        config=config, track=track, metadata={},
    )
    return result


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--since", default="")
    parser.add_argument("--until", default="")
    parser.add_argument("--csv", default="", help="Path to shadow_research_candidates.csv")
    args = parser.parse_args()

    candidates = load_candidates(args.csv if args.csv else None)
    if not candidates:
        print("No shadow_research_candidates.csv data.")
        return

    start_dt = (datetime.strptime(args.since, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                if args.since else datetime.now(timezone.utc) - timedelta(days=args.days))
    end_dt = (datetime.strptime(args.until, "%Y-%m-%d").replace(tzinfo=timezone.utc)
              if args.until else datetime.now(timezone.utc) + timedelta(days=1))

    # V3 filter
    v3_candidates = [c for c in candidates if c["v3"] >= 0.80 and c["htf"] != "up"]
    print(f"Candidates: {len(candidates)} total, {len(v3_candidates)} V3-eligible")

    # Fetch candles
    syms = sorted(set(c["sym"] for c in v3_candidates))
    print(f"Fetching candles for {len(syms)} symbols...")
    for sym in syms:
        fetch_candles(sym, start_dt, end_dt)
        time.sleep(0.35)

    # Build configs
    cfg_production = ExecutionPolicyConfig()
    cfg_no_stop = ExecutionPolicyConfig(apply_stop_redesign=False)
    cfg_no_chop = ExecutionPolicyConfig(apply_chop_block=False, apply_dual_chop_block=False)

    # Replay each candidate under each scenario
    results = {"engine_only": [], "production": [], "no_stop_redesign": [], "no_chop_block": []}

    for c in v3_candidates:
        try:
            ts = c["ts"]
            if ts.endswith("Z"): ts = ts[:-1] + "+00:00"
            elif "+" not in ts: ts += "+00:00"
            sig_ms = int(datetime.fromisoformat(ts).timestamp() * 1000)
        except Exception:
            continue

        candles = _candle_cache.get(f"{c['sym']}_{start_dt}_{end_dt}", [])
        if not candles:
            continue

        # Engine-only: use raw entry/stop/tp
        r_raw = replay_trade(c["entry"], c["stop"], c["tp"], c["side"], candles, sig_ms)
        if r_raw is not None:
            results["engine_only"].append({**c, "exit_r": round(r_raw, 4)})

        # Production executor
        res_prod = apply_policy(c, cfg_production)
        if res_prod.approved:
            r_prod = replay_trade(c["entry"], res_prod.redesigned_stop, res_prod.final_tp,
                                  c["side"], candles, sig_ms)
            if r_prod is not None:
                results["production"].append({**c, "exit_r": round(r_prod, 4),
                                              "reject_reason": ""})
        else:
            if r_raw is not None:
                results["production"].append({**c, "exit_r": round(r_raw, 4),
                                              "reject_reason": res_prod.reject_reason or "",
                                              "_rejected": True})

        # No stop redesign
        res_ns = apply_policy(c, cfg_no_stop)
        if res_ns.approved:
            r_ns = replay_trade(c["entry"], res_ns.redesigned_stop, res_ns.final_tp,
                                c["side"], candles, sig_ms)
            if r_ns is not None:
                results["no_stop_redesign"].append({**c, "exit_r": round(r_ns, 4)})

        # No chop block
        res_nc = apply_policy(c, cfg_no_chop)
        if res_nc.approved:
            r_nc = replay_trade(c["entry"], res_nc.redesigned_stop, res_nc.final_tp,
                                c["side"], candles, sig_ms)
            if r_nc is not None:
                results["no_chop_block"].append({**c, "exit_r": round(r_nc, 4)})

    # ── Report ───────────────────────────────────────────────────────
    dates = sorted(set(r["ts"][:10] for scenario in results.values() for r in scenario if r.get("ts")))
    nd = max(1, len(dates))

    print(f"\n{'='*80}")
    print(f"  PRODUCTION POLICY REPLAY REPORT")
    print(f"  {len(v3_candidates)} V3-eligible | {nd} days | {dates[0] if dates else '?'} → {dates[-1] if dates else '?'}")
    print(f"{'='*80}")

    scenario_stats = {}
    for label in ["engine_only", "production", "no_stop_redesign", "no_chop_block"]:
        accepted = [r for r in results[label] if not r.get("_rejected")]
        rs = [r["exit_r"] for r in accepted]
        print(f"\n{'─'*80}")
        s = _print_stats(label.upper(), rs)
        scenario_stats[label] = s

    # ── Breakdowns for production ────────────────────────────────────
    prod_accepted = [r for r in results["production"] if not r.get("_rejected")]
    if prod_accepted:
        _breakdown("Symbol", prod_accepted, lambda r: r["sym"])
        _breakdown("Session", prod_accepted, lambda r: r["session"])
        _breakdown("Family", prod_accepted, lambda r: r["family"])
        _breakdown("Side", prod_accepted, lambda r: r["side"])
        _breakdown("Date", prod_accepted, lambda r: r["ts"][:10])

    # ── Reject reason attribution ────────────────────────────────────
    prod_rejected = [r for r in results["production"] if r.get("_rejected")]
    if prod_rejected:
        print(f"\n{'─'*80}")
        print(f"  REJECT REASON ATTRIBUTION (V3-approved but production-rejected)")
        print(f"{'─'*80}")
        reasons = defaultdict(list)
        for r in prod_rejected:
            reason = r.get("reject_reason", "unknown")
            if "stop_redesign_rr" in reason: reasons["stop_redesign_rr_destroyed"].append(r)
            elif "stop_redesign_too_wide" in reason: reasons["stop_redesign_too_wide"].append(r)
            elif "chop" in reason: reasons["market_regime_block:chop"].append(r)
            elif "tp_cap" in reason: reasons["regime_tp_cap_rr_impossible"].append(r)
            elif "fill_rr" in reason: reasons["fill_rr_below_threshold"].append(r)
            else: reasons[reason[:40]].append(r)

        print(f"\n  {'Reason':<40s} │ {'n':>4s} │ {'AvgR':>7s} │ {'LostR':>8s} │ {'PF':>6s}")
        print(f"  {'─'*40} ┼ {'─'*4} ┼ {'─'*7} ┼ {'─'*8} ┼ {'─'*6}")
        for reason in sorted(reasons, key=lambda k: -sum(r["exit_r"] for r in reasons[k])):
            rs = [r["exit_r"] for r in reasons[reason]]
            s = _stats(rs)
            print(f"  {reason:<40s} │ {s['n']:>4} │ {s['avg']:>+6.3f} │ {s['total']:>+7.2f} │ {s['pf']:>6.3f}")

    # ── ENGINE_ONLY vs PRODUCTION warning ────────────────────────────
    eng = scenario_stats.get("engine_only", {})
    prod = scenario_stats.get("production", {})
    if eng.get("pf", 0) > 1.5 and prod.get("pf", 0) < 1.5:
        print(f"\n  ⚠ ENGINE_ONLY_EDGE_NOT_EXECUTABLE")
        print(f"    Engine-only PF={eng['pf']:.2f} but production PF={prod['pf']:.2f}")
        print(f"    The edge exists in research but is destroyed by executor gates.")

    # ── Promotion criteria ───────────────────────────────────────────
    prod_rs = [r["exit_r"] for r in prod_accepted]
    prod_s = _stats(prod_rs)
    prod_by_sym = defaultdict(list)
    prod_by_day = defaultdict(list)
    for r in prod_accepted:
        prod_by_sym[r["sym"]].append(r["exit_r"])
        prod_by_day[r["ts"][:10]].append(r["exit_r"])

    prof_syms = sum(1 for vals in prod_by_sym.values() if sum(vals) > 0)
    max_sym_pct = (max(sum(v) for v in prod_by_sym.values()) / prod_s["total"] * 100
                   if prod_s["total"] > 0 and prod_by_sym else 0)
    max_day_pct = (max(sum(v) for v in prod_by_day.values()) / prod_s["total"] * 100
                   if prod_s["total"] > 0 and prod_by_day else 0)
    days_pos = sum(1 for v in prod_by_day.values() if sum(v) > 0)

    print(f"\n{'─'*80}")
    print(f"  PROMOTION CRITERIA (production_executor)")
    print(f"{'─'*80}")
    criteria = {
        f"trades >= 100 (have {prod_s['n']})": prod_s["n"] >= 100,
        f"PF > 1.50 (have {prod_s['pf']:.2f})": prod_s["pf"] > 1.50,
        f"Avg R > +0.20 (have {prod_s['avg']:+.3f})": prod_s["avg"] > 0.20,
        f"max DD acceptable ({prod_s['mdd']:.1f}R)": prod_s["mdd"] < 15,
        f"profitable symbols >= 3 (have {prof_syms})": prof_syms >= 3,
        f"max symbol <= 40% (have {max_sym_pct:.0f}%)": max_sym_pct <= 40,
        f"max day <= 35% (have {max_day_pct:.0f}%)": max_day_pct <= 35,
        f"days positive >= 60% ({days_pos}/{len(prod_by_day)})": (
            days_pos / max(1, len(prod_by_day)) >= 0.60 if prod_by_day else False
        ),
    }
    all_pass = all(criteria.values())
    for c, passed in criteria.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {c}")
    print(f"\n  PROMOTE_V3_LIVE = {'YES' if all_pass else 'NO'}")


if __name__ == "__main__":
    main()

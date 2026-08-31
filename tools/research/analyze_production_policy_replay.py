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


def _active_confidence(row: Dict) -> float:
    return (
        _sf(row.get("signal_confidence"))
        or _sf(row.get("active_quality_score"))
        or _sf(row.get("confidence_v1"))
        or _sf(row.get("confidence"))
    )


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
                "confidence": _active_confidence(row),
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
    parser.add_argument(
        "--sessions", default="",
        help="Comma-separated session allow-list to test as an extra scenario on top "
             "of the production policy, e.g. --sessions ny_open,asia_open",
    )
    args = parser.parse_args()
    session_allowlist = (
        {s.strip().lower() for s in args.sessions.split(",") if s.strip()}
        if args.sessions else None
    )

    candidates = load_candidates(args.csv if args.csv else None)
    if not candidates:
        print("No shadow_research_candidates.csv data.")
        return

    # --since/--until/--days select WHICH candidates to replay. They must not
    # also drive the candle-fetch window below: if "now" has drifted far past
    # the candidate data (e.g. the bot's been offline for weeks), a window
    # anchored to now() can end up entirely after every candidate's signal
    # time. replay_trade()'s idx-lookup ("first candle after sig_ms") then
    # silently resolves EVERY trade against candles[0] — an unrelated,
    # unrolated later slice of price action — instead of what actually
    # happened after the signal. This previously produced a wildly wrong
    # (and wildly different run-to-run) profitability read with no error or
    # warning. Fix: derive the fetch window from the candidates actually
    # being replayed, always, regardless of args or wall-clock time.
    def _parse_date_arg(s):
        return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc) if s else None

    since_filter = _parse_date_arg(args.since)
    until_filter = _parse_date_arg(args.until)
    if since_filter is None and until_filter is None:
        since_filter = datetime.now(timezone.utc) - timedelta(days=args.days)

    def _cand_dt(c):
        ts = c["ts"]
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        elif "+" not in ts:
            ts += "+00:00"
        return datetime.fromisoformat(ts)

    if since_filter is not None or until_filter is not None:
        candidates = [
            c for c in candidates
            if (since_filter is None or _cand_dt(c) >= since_filter)
            and (until_filter is None or _cand_dt(c) <= until_filter)
        ]

    # V3 filter — mirrors signal_engine.py's _v3_eligibility_reject_reason:
    # htf=up only blocks non-LONG sides (LONG-in-uptrend is a documented
    # live exception), not every htf=up signal regardless of side.
    v3_candidates = [
        c for c in candidates
        if c["v3"] >= 0.80
        and not (c["htf"] == "up" and str(c.get("side", "")).strip().upper() != "LONG")
    ]
    print(f"Candidates: {len(candidates)} total, {len(v3_candidates)} V3-eligible")

    if not v3_candidates:
        print("No V3-eligible candidates in the selected date range.")
        return

    # Fetch window: earliest signal minus a day, latest signal plus max_hold
    # (24h) plus a day of buffer — always covers every candidate regardless
    # of wall-clock time.
    cand_dts = [_cand_dt(c) for c in v3_candidates]
    start_dt = min(cand_dts) - timedelta(days=1)
    end_dt = max(cand_dts) + timedelta(days=2)

    syms = sorted(set(c["sym"] for c in v3_candidates))
    print(f"Fetching candles for {len(syms)} symbols "
          f"({start_dt.date()} -> {end_dt.date()})...")
    for sym in syms:
        fetch_candles(sym, start_dt, end_dt)
        time.sleep(0.35)

    # Build configs
    cfg_production = ExecutionPolicyConfig()
    cfg_no_stop = ExecutionPolicyConfig(apply_stop_redesign=False)
    cfg_no_chop = ExecutionPolicyConfig(apply_chop_block=False, apply_dual_chop_block=False)

    # Replay each candidate under each scenario
    results = {
        "engine_only": [], "production": [], "no_stop_redesign": [], "no_chop_block": [],
        "session_filtered": [],
    }

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
                # Session-restricted variant: same production policy, plus an
                # additional session allow-list gate on top — tests whether
                # trading only the strongest session(s) beats the full mix.
                if session_allowlist and str(c.get("session", "")).strip().lower() in session_allowlist:
                    results["session_filtered"].append({**c, "exit_r": round(r_prod, 4),
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

    scenario_labels = ["engine_only", "production", "no_stop_redesign", "no_chop_block"]
    if session_allowlist:
        scenario_labels.append("session_filtered")

    scenario_stats = {}
    for label in scenario_labels:
        accepted = [r for r in results[label] if not r.get("_rejected")]
        rs = [r["exit_r"] for r in accepted]
        print(f"\n{'─'*80}")
        title = label.upper() if label != "session_filtered" else f"SESSION_FILTERED ({','.join(sorted(session_allowlist))})"
        s = _print_stats(title, rs)
        scenario_stats[label] = s

    # ── Breakdowns for production ────────────────────────────────────
    prod_accepted = [r for r in results["production"] if not r.get("_rejected")]
    if prod_accepted:
        _breakdown("Symbol", prod_accepted, lambda r: r["sym"])
        _breakdown("Session", prod_accepted, lambda r: r["session"])
        _breakdown("Family", prod_accepted, lambda r: r["family"])
        _breakdown("Side", prod_accepted, lambda r: r["side"])
        _breakdown("Date", prod_accepted, lambda r: r["ts"][:10])

    # ── Breakdowns for session-filtered variant ───────────────────────
    sess_accepted = results["session_filtered"]
    if session_allowlist and sess_accepted:
        print(f"\n{'─'*80}")
        print(f"  SESSION_FILTERED breakdown ({','.join(sorted(session_allowlist))})")
        print(f"{'─'*80}")
        _breakdown("Symbol", sess_accepted, lambda r: r["sym"])
        _breakdown("Family", sess_accepted, lambda r: r["family"])
        _breakdown("Side", sess_accepted, lambda r: r["side"])
        _breakdown("Date", sess_accepted, lambda r: r["ts"][:10])

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
    def _promotion_report(label, accepted_rows):
        rs = [r["exit_r"] for r in accepted_rows]
        s = _stats(rs)
        by_sym = defaultdict(list)
        by_day = defaultdict(list)
        for r in accepted_rows:
            by_sym[r["sym"]].append(r["exit_r"])
            by_day[r["ts"][:10]].append(r["exit_r"])

        prof_syms = sum(1 for vals in by_sym.values() if sum(vals) > 0)
        max_sym_pct = (max(sum(v) for v in by_sym.values()) / s["total"] * 100
                       if s["total"] > 0 and by_sym else 0)
        max_day_pct = (max(sum(v) for v in by_day.values()) / s["total"] * 100
                       if s["total"] > 0 and by_day else 0)
        days_pos = sum(1 for v in by_day.values() if sum(v) > 0)

        print(f"\n{'─'*80}")
        print(f"  PROMOTION CRITERIA ({label})")
        print(f"{'─'*80}")
        criteria = {
            f"trades >= 100 (have {s['n']})": s["n"] >= 100,
            f"PF > 1.50 (have {s['pf']:.2f})": s["pf"] > 1.50,
            f"Avg R > +0.20 (have {s['avg']:+.3f})": s["avg"] > 0.20,
            f"max DD acceptable ({s['mdd']:.1f}R)": s["mdd"] < 15,
            f"profitable symbols >= 3 (have {prof_syms})": prof_syms >= 3,
            f"max symbol <= 40% (have {max_sym_pct:.0f}%)": max_sym_pct <= 40,
            f"max day <= 35% (have {max_day_pct:.0f}%)": max_day_pct <= 35,
            f"days positive >= 60% ({days_pos}/{len(by_day)})": (
                days_pos / max(1, len(by_day)) >= 0.60 if by_day else False
            ),
        }
        all_pass = all(criteria.values())
        for c, passed in criteria.items():
            print(f"  [{'PASS' if passed else 'FAIL'}] {c}")
        print(f"\n  PROMOTE_V3_LIVE = {'YES' if all_pass else 'NO'}")
        return all_pass

    _promotion_report("production_executor", prod_accepted)
    if session_allowlist:
        _promotion_report(f"session_filtered:{','.join(sorted(session_allowlist))}", sess_accepted)


if __name__ == "__main__":
    main()

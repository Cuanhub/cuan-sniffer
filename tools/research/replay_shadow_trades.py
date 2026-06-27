#!/usr/bin/env python3
"""
Shadow trade replay — resolve shadow candidates against real candle data.

Reads shadow logs, fetches historical candles from Hyperliquid, simulates
entry/SL/TP resolution, and produces realized R outcomes for each candidate.

Usage:
    python3 tools/research/replay_shadow_trades.py --days 30
    python3 tools/research/replay_shadow_trades.py --input shadow_research_candidates.csv --output results.csv
    python3 tools/research/replay_shadow_trades.py --since 2026-06-16 --until 2026-06-21

This script is research-only. It never places trades or modifies live state.
"""

import argparse
import csv
import json
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path for imports
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
)

HYPERLIQUID_INFO_URL = "https://api.hyperliquid.xyz/info"
DEFAULT_SHADOW_REPLAY_INPUT = "shadow_research_candidates.csv"

# ── Field mapping ────────────────────────────────────────────────────

FIELD_ALIASES = {
    "timestamp": ["timestamp", "ts", "created_at", "timestamp_utc"],
    "symbol": ["symbol", "coin"],
    "side": ["side"],
    "entry": ["entry", "entry_price", "price"],
    "stop": ["stop", "stop_price"],
    "tp": ["tp", "tp_price", "take_profit"],
    "rr": ["rr", "rr_planned", "planned_rr", "final_rr"],
    "atr": ["atr"],
    "timeframe": ["timeframe"],
    "setup_family": ["setup_family", "regime_local"],
    "session": ["session"],
    "market_regime": ["market_regime"],
    "htf_regime": ["htf_regime", "regime_htf_1h"],
    "macro_regime": ["macro_regime", "regime_macro_4h"],
    "regime": ["regime"],
    "execution_track": ["execution_track", "track"],
    "score_v1": ["score_v1", "score", "total_score"],
    "confidence_v1": ["confidence_v1", "confidence"],
    "score_v2": ["score_v2"],
    "score_v3": ["score_v3"],
    "active_quality_model": ["active_quality_model"],
    "active_quality_score": ["active_quality_score"],
    "signal_confidence": ["signal_confidence", "confidence"],
    "score_v2_tags": ["score_v2_tags"],
    "score_v3_tags": ["score_v3_tags"],
    "live_reject_reason": [
        "live_reject_reason",
        "executor_reject_reason",
        "engine_reject_reason",
        "governance_reason",
        "reject_reason",
    ],
    "metadata": ["metadata"],
}

REQUIRED_FIELDS = {"timestamp", "symbol", "side", "entry", "stop", "tp"}

OUTPUT_FIELDS = [
    "timestamp", "symbol", "side", "timeframe", "entry", "stop", "tp",
    "policy", "policy_reject_reason", "policy_final_stop", "policy_final_tp",
    "policy_final_rr",
    "risk", "exit_time", "exit_price", "exit_reason", "realized_R",
    "bars_held", "same_candle_conflict", "max_hold_bars",
    "setup_family", "session", "market_regime", "htf_regime", "macro_regime",
    "score_v1", "confidence_v1", "score_v2", "score_v3",
    "active_quality_model", "active_quality_score", "signal_confidence",
    "score_v2_tags", "score_v3_tags", "live_reject_reason", "metadata",
]


def _resolve_field(row: Dict[str, str], canonical: str) -> str:
    for alias in FIELD_ALIASES.get(canonical, [canonical]):
        val = row.get(alias, "").strip()
        if val:
            return val
    return ""


def _parse_row(row: Dict[str, str]) -> Optional[Dict[str, Any]]:
    parsed = {}
    for field in FIELD_ALIASES:
        parsed[field] = _resolve_field(row, field)

    missing = [f for f in REQUIRED_FIELDS if not parsed.get(f)]
    if missing:
        return None

    try:
        parsed["entry"] = float(parsed["entry"])
        parsed["stop"] = float(parsed["stop"])
        parsed["tp"] = float(parsed["tp"])
        parsed["atr"] = float(parsed["atr"]) if parsed.get("atr") else 0.0
    except (ValueError, TypeError):
        return None

    parsed["side"] = parsed["side"].upper()
    if parsed["side"] not in ("LONG", "SHORT"):
        return None

    parsed["symbol"] = parsed["symbol"].upper()
    return parsed


# ── Candle fetching ──────────────────────────────────────────────────

_candle_cache: Dict[str, List[Dict]] = {}


def fetch_candles(
    symbol: str,
    interval: str = "1h",
    start_dt: Optional[datetime] = None,
    end_dt: Optional[datetime] = None,
) -> List[Dict]:
    cache_key = f"{symbol}_{interval}_{start_dt}_{end_dt}"
    if cache_key in _candle_cache:
        return _candle_cache[cache_key]

    if requests is None:
        print(f"  [WARN] requests not available, cannot fetch candles for {symbol}")
        return []

    if start_dt is None:
        start_dt = datetime.now(timezone.utc) - timedelta(days=30)
    if end_dt is None:
        end_dt = datetime.now(timezone.utc)

    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    payload = {
        "type": "candleSnapshot",
        "req": {
            "coin": symbol,
            "interval": interval,
            "startTime": start_ms,
            "endTime": end_ms,
        },
    }

    try:
        resp = requests.post(HYPERLIQUID_INFO_URL, json=payload, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  [ERROR] Candle fetch failed for {symbol}: {e}")
        return []

    if not isinstance(data, list):
        return []

    candles = []
    for c in data:
        open_ms = c.get("t") or c.get("T")
        if not open_ms:
            continue
        candles.append({
            "time": int(open_ms),
            "open": float(c["o"]),
            "high": float(c["h"]),
            "low": float(c["l"]),
            "close": float(c["c"]),
            "volume": float(c["v"]),
        })

    candles.sort(key=lambda x: x["time"])
    _candle_cache[cache_key] = candles
    return candles


# ── Replay logic ─────────────────────────────────────────────────────

POLICY_CHOICES = ("engine_only", "production_executor", "no_stop_redesign", "no_chop_block")


def _policy_config(policy: str) -> ExecutionPolicyConfig:
    if policy == "no_stop_redesign":
        return ExecutionPolicyConfig(apply_stop_redesign=False)
    if policy == "no_chop_block":
        return ExecutionPolicyConfig(apply_chop_block=False, apply_dual_chop_block=False)
    return ExecutionPolicyConfig()


def apply_replay_policy(trade: Dict[str, Any], policy: str) -> ExecutionPolicyResult:
    if policy == "engine_only":
        rr = 0.0
        risk = abs(float(trade["entry"]) - float(trade["stop"]))
        if risk > 0:
            rr = abs(float(trade["tp"]) - float(trade["entry"])) / risk
        return ExecutionPolicyResult(
            approved=True,
            reject_reason=None,
            entry=float(trade["entry"]),
            original_stop=float(trade["stop"]),
            redesigned_stop=float(trade["stop"]),
            original_tp=float(trade["tp"]),
            final_tp=float(trade["tp"]),
            original_rr=rr,
            final_rr=rr,
            widen_mult=1.0,
            tp_capped=False,
            metadata={},
        )

    return evaluate_execution_policy(
        entry=float(trade["entry"]),
        stop=float(trade["stop"]),
        tp=float(trade["tp"]),
        side=str(trade["side"]),
        atr=float(trade.get("atr", 0.0) or 0.0),
        timeframe=str(trade.get("timeframe", "")),
        session=str(trade.get("session", "")),
        market_regime=str(trade.get("market_regime", "")),
        htf_regime=str(trade.get("htf_regime", "")),
        macro_regime=str(trade.get("macro_regime", "")),
        setup_family=str(trade.get("setup_family", "")),
        confidence=float(
            trade.get("signal_confidence")
            or trade.get("active_quality_score")
            or trade.get("confidence_v1")
            or 0.0
        ),
        config=_policy_config(policy),
        current_price=float(trade["entry"]),
        regime=str(trade.get("regime", "")),
        track=str(trade.get("execution_track", "")),
        metadata={
            "source": "replay_shadow_trades",
            "shadow_policy": policy,
        },
    )

def replay_trade(
    entry: float,
    stop: float,
    tp: float,
    side: str,
    candles: List[Dict],
    signal_epoch_ms: int,
    max_hold_bars: int = 24,
    tp_first: bool = False,
    entry_mode: str = "next_candle",
) -> Optional[Dict[str, Any]]:
    """
    Simulate a single trade against candle data.

    entry_mode:
      "next_candle" — start checking from the first candle AFTER signal time.
                      Conservative: assumes entry happens at signal time but
                      resolution begins on the next bar.
      "same_candle" — include the candle containing the signal timestamp.
                      More aggressive: immediate resolution possible.
    """

    if side == "LONG":
        risk = entry - stop
        if risk <= 0:
            return None
        if tp <= entry:
            return None
    else:
        risk = stop - entry
        if risk <= 0:
            return None
        if tp >= entry:
            return None

    start_idx = None
    for i, c in enumerate(candles):
        if entry_mode == "same_candle":
            if c["time"] >= signal_epoch_ms:
                start_idx = i
                break
        else:
            if c["time"] > signal_epoch_ms:
                start_idx = i
                break

    if start_idx is None:
        return None

    bars_held = 0
    same_candle_conflict = False

    for i in range(start_idx, min(start_idx + max_hold_bars, len(candles))):
        c = candles[i]
        bars_held += 1

        if side == "LONG":
            sl_hit = c["low"] <= stop
            tp_hit = c["high"] >= tp
        else:
            sl_hit = c["high"] >= stop
            tp_hit = c["low"] <= tp

        if sl_hit and tp_hit:
            same_candle_conflict = True
            if tp_first:
                exit_price = tp
                reason = "tp"
            else:
                exit_price = stop
                reason = "stop"
        elif sl_hit:
            exit_price = stop
            reason = "stop"
        elif tp_hit:
            exit_price = tp
            reason = "tp"
        else:
            continue

        if side == "LONG":
            realized_r = (exit_price - entry) / risk
        else:
            realized_r = (entry - exit_price) / risk

        return {
            "exit_time": datetime.fromtimestamp(c["time"] / 1000, tz=timezone.utc).isoformat(),
            "exit_price": round(exit_price, 8),
            "exit_reason": reason,
            "realized_R": round(realized_r, 4),
            "bars_held": bars_held,
            "same_candle_conflict": same_candle_conflict,
            "risk": round(risk, 8),
        }

    # Max hold — close at last available candle
    last_idx = min(start_idx + max_hold_bars - 1, len(candles) - 1)
    last_close = candles[last_idx]["close"]
    if side == "LONG":
        realized_r = (last_close - entry) / risk
    else:
        realized_r = (entry - last_close) / risk

    return {
        "exit_time": datetime.fromtimestamp(candles[last_idx]["time"] / 1000, tz=timezone.utc).isoformat(),
        "exit_price": round(last_close, 8),
        "exit_reason": "max_hold",
        "realized_R": round(realized_r, 4),
        "bars_held": bars_held,
        "same_candle_conflict": False,
        "risk": round(risk, 8),
    }


# ── Summary generation ───────────────────────────────────────────────

PREFERRED_SYMBOLS = {"FARTCOIN", "JTO", "SOL", "WIF", "SUI"}
NEGATIVE_SYMBOLS = {"ETH", "ZEC", "BNB"}
NEGATIVE_SESSIONS = {"ny_pm", "london_late", "asia_late"}
CHOP_EXCEPTION_BLOCKED_SYMBOLS = {"ETH", "ZEC", "BNB", "NEAR"}
CHOP_EXCEPTION_ALLOWED_SESSIONS = {"ny_open", "asia_open"}


def is_chop_exception_candidate(row: Dict[str, Any]) -> bool:
    reason = str(row.get("live_reject_reason", "") or row.get("reject_reason", "")).lower()
    if "chop" not in reason:
        return False
    symbol = str(row.get("symbol", "")).upper()
    session = str(row.get("session", "")).lower()
    family = str(row.get("setup_family", "")).lower()
    try:
        conf = float(row.get("confidence_v1", 0) or row.get("confidence", 0) or 0)
    except (ValueError, TypeError):
        conf = 0.0
    tags = str(row.get("score_v3_tags", "") or row.get("tags", "")).lower()
    has_fvg = "+fvg" in tags or "fvg" in str(row.get("triggers", "")).lower()

    return (
        conf >= 0.80
        and symbol not in CHOP_EXCEPTION_BLOCKED_SYMBOLS
        and session in CHOP_EXCEPTION_ALLOWED_SESSIONS
        and family == "continuation"
        and has_fvg
    )


def is_v3_full_recipe(row: Dict[str, Any]) -> bool:
    """
    Strict recipe check: ALL conditions must hold.
    Requires: continuation + FVG + v3>=0.70 + preferred symbol + good session.
    """
    family = str(row.get("setup_family", "")).lower()
    symbol = str(row.get("symbol", "")).upper()
    session = str(row.get("session", "")).lower()
    try:
        v3 = float(row.get("score_v3", 0) or 0)
    except (ValueError, TypeError):
        v3 = 0.0

    # FVG detection from tags or trigger fields
    tags = str(row.get("score_v3_tags", "") or row.get("score_v2_tags", "")).lower()
    triggers = str(row.get("triggers", "") or row.get("trigger", "")).lower()
    has_fvg = (
        "+fvg" in tags
        or "fvg" in triggers
        or "v3_full_recipe" in tags
    )

    return (
        family == "continuation"
        and has_fvg
        and v3 >= 0.70
        and symbol in PREFERRED_SYMBOLS
        and session not in NEGATIVE_SESSIONS
        and symbol not in NEGATIVE_SYMBOLS
    )


def generate_summary(results: List[Dict[str, Any]], days: int) -> str:
    lines = ["# Shadow Replay Summary\n"]

    n = len(results)
    if n == 0:
        lines.append("No trades replayed.\n")
        return "\n".join(lines)

    rs = [r["realized_R"] for r in results]
    wins = [r for r in rs if r > 0]
    losses = [r for r in rs if r < 0]
    total_r = sum(rs)
    gw = sum(wins)
    gl = abs(sum(losses))
    pf = gw / gl if gl > 0 else (999 if gw > 0 else 0)
    wr = len(wins) / n * 100
    avg_r = total_r / n
    median_r = statistics.median(rs)

    # Max drawdown
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for r in rs:
        cum += r
        if cum > peak:
            peak = cum
        dd = peak - cum
        if dd > max_dd:
            max_dd = dd

    lines.append(f"## Overall\n")
    lines.append(f"- Trades: {n}")
    lines.append(f"- Win rate: {wr:.1f}%")
    lines.append(f"- Average R: {avg_r:+.4f}")
    lines.append(f"- Median R: {median_r:+.4f}")
    lines.append(f"- Profit factor: {pf:.3f}")
    lines.append(f"- Total R: {total_r:+.2f}")
    lines.append(f"- Max drawdown: {max_dd:.2f}R")
    lines.append(f"- Trades/day: {n / max(1, days):.1f}")
    lines.append("")

    # Bucket analysis helper
    def bucket_section(title, key_fn):
        lines.append(f"## {title}\n")
        lines.append(f"| Bucket | n | WR% | Avg R | PF |")
        lines.append(f"|--------|---|-----|-------|-----|")
        groups = defaultdict(list)
        for r in results:
            groups[key_fn(r)].append(r["realized_R"])
        for bucket in sorted(groups.keys()):
            vals = groups[bucket]
            bn = len(vals)
            bwr = sum(1 for v in vals if v > 0) / bn * 100
            bavg = sum(vals) / bn
            bgw = sum(v for v in vals if v > 0)
            bgl = abs(sum(v for v in vals if v < 0))
            bpf = bgw / bgl if bgl > 0 else (999 if bgw > 0 else 0)
            lines.append(f"| {bucket} | {bn} | {bwr:.1f}% | {bavg:+.3f} | {bpf:.3f} |")
        lines.append("")

    def score_bucket(val):
        try:
            v = float(val)
        except (ValueError, TypeError):
            return "n/a"
        if v < 0.50: return "<0.50"
        if v < 0.60: return "0.50-0.60"
        if v < 0.70: return "0.60-0.70"
        if v < 0.80: return "0.70-0.80"
        if v < 0.90: return "0.80-0.90"
        return ">=0.90"

    bucket_section("By score_v1", lambda r: score_bucket(r.get("score_v1", "")))
    bucket_section("By score_v2", lambda r: score_bucket(r.get("score_v2", "")))
    bucket_section("By score_v3", lambda r: score_bucket(r.get("score_v3", "")))
    bucket_section("By setup_family", lambda r: str(r.get("setup_family", "unknown")))
    bucket_section("By session", lambda r: str(r.get("session", "unknown")))
    bucket_section("By symbol", lambda r: str(r.get("symbol", "?")))
    bucket_section("By market_regime", lambda r: str(r.get("market_regime", "?")))

    # V3 full recipe
    recipe = [r for r in results if is_v3_full_recipe(r)]
    lines.append("## V3 Full Recipe Cohort\n")
    if recipe:
        rr = [r["realized_R"] for r in recipe]
        rn = len(rr)
        rwr = sum(1 for r in rr if r > 0) / rn * 100
        ravg = sum(rr) / rn
        rgw = sum(r for r in rr if r > 0)
        rgl = abs(sum(r for r in rr if r < 0))
        rpf = rgw / rgl if rgl > 0 else 0
        lines.append(f"- Count: {rn}")
        lines.append(f"- WR: {rwr:.1f}%")
        lines.append(f"- Avg R: {ravg:+.4f}")
        lines.append(f"- PF: {rpf:.3f}")
        lines.append(f"- Total R: {sum(rr):+.2f}")
        lines.append(f"- Trades/day: {rn / max(1, days):.1f}")
    else:
        lines.append("No V3 full recipe trades found.\n")

    # Chop exception cohort
    chop_ex = [r for r in results if is_chop_exception_candidate(r)]
    lines.append("## Chop Exception Shadow Lane\n")
    if chop_ex:
        cer = [r["realized_R"] for r in chop_ex]
        cen = len(cer)
        cewr = sum(1 for r in cer if r > 0) / cen * 100
        ceavg = sum(cer) / cen
        cegw = sum(r for r in cer if r > 0)
        cegl = abs(sum(r for r in cer if r < 0))
        cepf = cegw / cegl if cegl > 0 else (999 if cegw > 0 else 0)
        lines.append(f"- Count: {cen}")
        lines.append(f"- WR: {cewr:.1f}%")
        lines.append(f"- Avg R: {ceavg:+.4f}")
        lines.append(f"- PF: {cepf:.3f}")
        lines.append(f"- Total R: {sum(cer):+.2f}")
        lines.append(f"- Trades/day: {cen / max(1, days):.1f}")
        lines.append("")
        lines.append("| Symbol | n | WR% | Avg R | PF |")
        lines.append("|--------|---|-----|-------|-----|")
        ce_syms = defaultdict(list)
        for r in chop_ex:
            ce_syms[r.get("symbol", "")].append(r["realized_R"])
        for sym in sorted(ce_syms.keys()):
            vals = ce_syms[sym]
            sw = sum(1 for v in vals if v > 0)
            sgw = sum(v for v in vals if v > 0)
            sgl = abs(sum(v for v in vals if v < 0))
            spf = sgw / sgl if sgl > 0 else (999 if sgw > 0 else 0)
            lines.append(f"| {sym} | {len(vals)} | {sw/len(vals)*100:.1f}% | {sum(vals)/len(vals):+.3f} | {spf:.3f} |")
        lines.append("")
        lines.append("| Session | n | WR% | Avg R | PF |")
        lines.append("|---------|---|-----|-------|-----|")
        ce_sess = defaultdict(list)
        for r in chop_ex:
            ce_sess[r.get("session", "")].append(r["realized_R"])
        for sess in sorted(ce_sess.keys()):
            vals = ce_sess[sess]
            sw = sum(1 for v in vals if v > 0)
            sgw = sum(v for v in vals if v > 0)
            sgl = abs(sum(v for v in vals if v < 0))
            spf = sgw / sgl if sgl > 0 else (999 if sgw > 0 else 0)
            lines.append(f"| {sess} | {len(vals)} | {sw/len(vals)*100:.1f}% | {sum(vals)/len(vals):+.3f} | {spf:.3f} |")
        lines.append("")
        lines.append("| Date | n | WR% | Total R |")
        lines.append("|------|---|-----|---------|")
        ce_daily = defaultdict(list)
        for r in chop_ex:
            ce_daily[str(r.get("timestamp", ""))[:10]].append(r["realized_R"])
        for dt in sorted(ce_daily.keys()):
            vals = ce_daily[dt]
            lines.append(f"| {dt} | {len(vals)} | {sum(1 for v in vals if v>0)/len(vals)*100:.1f}% | {sum(vals):+.2f} |")
    else:
        lines.append("No chop exception candidates found.\n")

    # Promotion criteria
    lines.append("\n## Promotion Criteria Check\n")
    unique_days = len(set(str(r.get("timestamp", ""))[:10] for r in results))
    recipe_count = len(recipe)
    recipe_pf = rpf if recipe else 0
    recipe_avg = ravg if recipe else 0
    profitable_syms = set()
    if recipe:
        sym_r = defaultdict(list)
        for r in recipe:
            sym_r[r.get("symbol", "")].append(r["realized_R"])
        profitable_syms = {s for s, vals in sym_r.items() if sum(vals) > 0}

    daily_r = defaultdict(float)
    if recipe:
        for r in recipe:
            daily_r[str(r.get("timestamp", ""))[:10]] += r["realized_R"]
    max_day_pct = 0.0
    if daily_r and sum(rr) > 0:
        max_day_pct = max(daily_r.values()) / sum(rr) * 100

    criteria = {
        "days >= 10": unique_days >= 10,
        "recipe trades >= 100": recipe_count >= 100,
        "recipe PF > 1.5": recipe_pf > 1.5,
        "recipe avg R > +0.20": recipe_avg > 0.20,
        "profitable symbols >= 3": len(profitable_syms) >= 3,
        "no day > 35% of total R": max_day_pct <= 35.0,
    }

    all_pass = all(criteria.values())
    for criterion, passed in criteria.items():
        status = "PASS" if passed else "FAIL"
        lines.append(f"- [{status}] {criterion}")

    lines.append(f"\n**PROMOTE_V3_LANE = {'YES' if all_pass else 'NO'}**\n")

    # Chop exception promotion criteria
    lines.append("## Chop Exception Promotion Criteria\n")
    ce_count = len(chop_ex)
    ce_pf = cepf if chop_ex else 0
    ce_avg = ceavg if chop_ex else 0
    ce_profitable_syms = set()
    if chop_ex:
        ce_sym_r = defaultdict(list)
        for r in chop_ex:
            ce_sym_r[r.get("symbol", "")].append(r["realized_R"])
        ce_profitable_syms = {s for s, vals in ce_sym_r.items() if sum(vals) > 0}
    ce_daily_r = defaultdict(float)
    if chop_ex:
        for r in chop_ex:
            ce_daily_r[str(r.get("timestamp", ""))[:10]] += r["realized_R"]
    ce_total = sum(r["realized_R"] for r in chop_ex) if chop_ex else 0
    ce_max_day_pct = 0.0
    if ce_daily_r and ce_total > 0:
        ce_max_day_pct = max(ce_daily_r.values()) / ce_total * 100

    ce_criteria = {
        "chop exception trades >= 100": ce_count >= 100,
        "chop exception PF > 1.5": ce_pf > 1.5,
        "chop exception avg R > +0.20": ce_avg > 0.20,
        "no day > 35% of total R": ce_max_day_pct <= 35.0 if ce_total > 0 else False,
    }
    ce_all_pass = all(ce_criteria.values())
    for criterion, passed in ce_criteria.items():
        status = "PASS" if passed else "FAIL"
        lines.append(f"- [{status}] {criterion}")
    lines.append(f"\n**PROMOTE_CHOP_EXCEPTION = {'YES' if ce_all_pass else 'NO'}**\n")

    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────

def load_shadow_trades(input_path: str) -> List[Dict[str, str]]:
    candidates = []
    seen = set()

    def add_candidate(path: str) -> None:
        if not path:
            return
        normalized = os.path.abspath(os.path.expanduser(path))
        if normalized in seen:
            return
        seen.add(normalized)
        candidates.append(path)

    add_candidate(input_path)
    add_candidate(os.path.join(str(PROJECT_ROOT), DEFAULT_SHADOW_REPLAY_INPUT))
    add_candidate(os.path.join(str(PROJECT_ROOT), "shadow_chop_exception.csv"))

    for path in candidates:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            try:
                with open(path, newline="") as f:
                    rows = list(csv.DictReader(f))
                if rows:
                    print(f"  Loaded {len(rows)} rows from {path}")
                    return rows
            except Exception as e:
                print(f"  [WARN] Failed to read {path}: {e}")
    print("  [WARN] No canonical shadow research candidate file found.")
    return []


def main():
    parser = argparse.ArgumentParser(description="Replay shadow trades against candle data")
    parser.add_argument("--input", default=DEFAULT_SHADOW_REPLAY_INPUT, help="Input CSV path")
    parser.add_argument("--output", default="shadow_replay_results.csv", help="Output CSV path")
    parser.add_argument("--timeframe", default="1h", help="Candle timeframe (default: 1h)")
    parser.add_argument("--days", type=int, default=30, help="Lookback days for candle fetch")
    parser.add_argument("--since", default="", help="Start date YYYY-MM-DD")
    parser.add_argument("--until", default="", help="End date YYYY-MM-DD")
    parser.add_argument("--max-hold-bars", type=int, default=24, help="Max bars to hold (default: 24)")
    parser.add_argument("--tp-first", action="store_true", help="TP wins on same-candle conflict")
    parser.add_argument("--sl-first", action="store_true", help="SL wins on same-candle conflict (default)")
    parser.add_argument("--entry-mode", choices=["next_candle", "same_candle"], default="next_candle",
                        help="next_candle=start checking after signal bar (default); same_candle=include signal bar")
    parser.add_argument("--candle-file", default="", help="JSON file with cached candles (no network). Format: {SYMBOL: [{time,open,high,low,close,volume}]}")
    parser.add_argument("--policy", choices=POLICY_CHOICES, default="engine_only",
                        help="Execution geometry policy to apply before replay (default: engine_only)")
    args = parser.parse_args()

    tp_first = args.tp_first and not args.sl_first
    entry_mode = args.entry_mode

    print("=" * 70)
    print("  SHADOW TRADE REPLAY")
    print("=" * 70)
    print(f"  Input:      {args.input}")
    print(f"  Timeframe:  {args.timeframe}")
    print(f"  Max hold:   {args.max_hold_bars} bars")
    print(f"  Conflict:   {'TP-first' if tp_first else 'SL-first'}")
    print(f"  Entry mode: {entry_mode}")
    print(f"  Policy:     {args.policy}")
    print(f"  Candle src: {'file: ' + args.candle_file if args.candle_file else 'Hyperliquid API'}")
    print()

    # Load trades
    rows = load_shadow_trades(args.input)
    if not rows:
        print("  No data to replay. Exiting.")
        return

    # Parse rows
    parsed = []
    skip_reasons = Counter()
    for row in rows:
        p = _parse_row(row)
        if p is None:
            skip_reasons["invalid_or_missing_fields"] += 1
            continue
        parsed.append({**row, **p})

    print(f"  Parsed: {len(parsed)} tradeable rows, {sum(skip_reasons.values())} skipped")

    if not parsed:
        print("  No valid trades to replay.")
        return

    # Determine date range
    if args.since:
        start_dt = datetime.strptime(args.since, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        start_dt = datetime.now(timezone.utc) - timedelta(days=args.days)
    if args.until:
        end_dt = datetime.strptime(args.until, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        end_dt = datetime.now(timezone.utc)

    # Fetch candles per symbol
    symbols = sorted(set(r["symbol"] for r in parsed))
    print(f"  Symbols: {symbols}")

    if args.candle_file and os.path.exists(args.candle_file):
        print(f"  Loading candles from file: {args.candle_file}")
        with open(args.candle_file) as f:
            file_candles = json.load(f)
        for sym in symbols:
            candles = sorted(file_candles.get(sym, []), key=lambda c: c["time"])
            cache_key = f"{sym}_{args.timeframe}_{start_dt}_{end_dt}"
            _candle_cache[cache_key] = candles
            print(f"    {sym}: {len(candles)} candles (from file)")
    else:
        print(f"  Fetching candles from API...")
        for sym in symbols:
            candles = fetch_candles(sym, args.timeframe, start_dt, end_dt)
            print(f"    {sym}: {len(candles)} candles")
            time.sleep(0.4)

    # Replay
    print(f"\n  Replaying {len(parsed)} trades...")
    results = []
    replay_skipped = Counter()
    policy_rejects = Counter()

    for trade in parsed:
        sym = trade["symbol"]
        candles = _candle_cache.get(f"{sym}_{args.timeframe}_{start_dt}_{end_dt}", [])
        if not candles:
            replay_skipped["no_candles"] += 1
            continue

        # Parse signal timestamp
        ts_str = trade["timestamp"]
        try:
            if "T" in ts_str:
                if ts_str.endswith("Z"):
                    ts_str = ts_str[:-1] + "+00:00"
                elif "+" not in ts_str and not ts_str.endswith("Z"):
                    ts_str += "+00:00"
                sig_epoch = int(datetime.fromisoformat(ts_str).timestamp() * 1000)
            else:
                sig_epoch = int(datetime.strptime(ts_str[:19], "%Y-%m-%d %H:%M:%S").replace(
                    tzinfo=timezone.utc).timestamp() * 1000)
        except Exception:
            replay_skipped["bad_timestamp"] += 1
            continue

        policy_result = apply_replay_policy(trade, args.policy)
        if not policy_result.approved:
            policy_rejects[policy_result.reject_reason or "policy_reject"] += 1
            continue

        outcome = replay_trade(
            entry=policy_result.entry,
            stop=policy_result.redesigned_stop,
            tp=policy_result.final_tp,
            side=trade["side"],
            candles=candles,
            signal_epoch_ms=sig_epoch,
            max_hold_bars=args.max_hold_bars,
            tp_first=tp_first,
            entry_mode=entry_mode,
        )

        if outcome is None:
            replay_skipped["invalid_geometry_or_no_future_candles"] += 1
            continue

        result_row = {
            "timestamp": trade["timestamp"],
            "symbol": trade["symbol"],
            "side": trade["side"],
            "timeframe": trade.get("timeframe", args.timeframe),
            "entry": trade["entry"],
            "stop": trade["stop"],
            "tp": trade["tp"],
            "policy": args.policy,
            "policy_reject_reason": "",
            "policy_final_stop": round(policy_result.redesigned_stop, 8),
            "policy_final_tp": round(policy_result.final_tp, 8),
            "policy_final_rr": round(policy_result.final_rr, 4),
            "risk": outcome["risk"],
            "exit_time": outcome["exit_time"],
            "exit_price": outcome["exit_price"],
            "exit_reason": outcome["exit_reason"],
            "realized_R": outcome["realized_R"],
            "bars_held": outcome["bars_held"],
            "same_candle_conflict": outcome["same_candle_conflict"],
            "max_hold_bars": args.max_hold_bars,
            "setup_family": trade.get("setup_family", ""),
            "session": trade.get("session", ""),
            "market_regime": trade.get("market_regime", ""),
            "htf_regime": trade.get("htf_regime", ""),
            "macro_regime": trade.get("macro_regime", ""),
            "score_v1": trade.get("score_v1", ""),
            "confidence_v1": trade.get("confidence_v1", ""),
            "score_v2": trade.get("score_v2", ""),
            "score_v3": trade.get("score_v3", ""),
            "active_quality_model": trade.get("active_quality_model", ""),
            "active_quality_score": trade.get("active_quality_score", ""),
            "signal_confidence": trade.get("signal_confidence", ""),
            "score_v2_tags": trade.get("score_v2_tags", ""),
            "score_v3_tags": trade.get("score_v3_tags", ""),
            "live_reject_reason": trade.get("live_reject_reason", ""),
            "metadata": trade.get("metadata", ""),
        }
        results.append(result_row)

    print(f"  Replayed: {len(results)}")
    if replay_skipped:
        print(f"  Skipped: {dict(replay_skipped)}")
    if policy_rejects:
        print(f"  Policy rejected: {dict(policy_rejects)}")

    # Write output
    output_path = args.output
    if results:
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
            writer.writeheader()
            for r in results:
                writer.writerow({k: r.get(k, "") for k in OUTPUT_FIELDS})
        print(f"  Output: {output_path} ({len(results)} rows)")

    # Summary
    unique_days = len(set(str(r.get("timestamp", ""))[:10] for r in results))
    days_actual = max(1, unique_days)
    summary = generate_summary(results, days_actual)

    summary_path = output_path.replace(".csv", "_summary.md")
    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"  Summary: {summary_path}")
    print()
    print(summary)


if __name__ == "__main__":
    main()

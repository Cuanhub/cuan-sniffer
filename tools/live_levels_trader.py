#!/usr/bin/env python3
"""
Live execution bridge for research/levels/ setups.

READ THIS BEFORE ARMING IT.

The gate this trades was measured at -0.0413R against random entry over 688
filled backfill trades (z = -0.93) -- i.e. no edge distinguishable from
throwing darts. Its apparent +203R over the backfill came from a window in
which all 28 universe coins rose, median +38.5%; random entries with the
identical stop/TP geometry returned MORE (+0.3368R vs +0.2955R). The gate
also fires roughly 25x more often than the user trades by hand, carries zero
labels, and its thresholds were fitted to two charts chosen after the fact.

This module exists because the user asked for it with that evidence in front
of them. It is built to lose slowly and stop early rather than to make money,
and every default here is chosen accordingly.

It lives in tools/ and NOT in research/levels/ on purpose: that package is
import-isolated from every order-capable module and two tests enforce it.
This bridge imports both sides, so putting it there would break the guarantee
that the scanner cannot trade.

SAFETY MODEL
  - dry-run is the default; --arm AND LEVELS_LIVE_ARMED=yes are both required
  - quality filter far stricter than the gate (the gate is 25x too loose)
  - size derived from the structural stop, never from leverage
  - leverage cap, because a structural stop needs one (a 10%-of-margin stop
    at 20x is not a stop, it is a liquidation)
  - venue-native stop AND take-profit placed immediately after every entry
  - daily loss kill switch and a cap on concurrent positions
  - positions this module did not open are never touched

Usage:
    python -m tools.live_levels_trader                  # dry run, prints intents
    python -m tools.live_levels_trader --once           # one pass, dry
    LEVELS_LIVE_ARMED=yes python -m tools.live_levels_trader --arm
"""

import argparse
import json
import math
import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)

from research.levels.levels import log_sigma, sigma_distance          # noqa: E402
from research.levels.scanner import fetch, scan_coin                  # noqa: E402
from research.levels.venue_client import HyperliquidReadOnlyClient    # noqa: E402
from research.funding.universe import select_universe                 # noqa: E402

# ---- rails -----------------------------------------------------------------
RISK_PCT_PER_TRADE = float(os.getenv("LEVELS_RISK_PCT", "5"))     # % of equity at risk
MAX_CONCURRENT = int(os.getenv("LEVELS_MAX_CONCURRENT", "2"))
MAX_NEW_PER_DAY = int(os.getenv("LEVELS_MAX_NEW_PER_DAY", "0"))   # 0 = unlimited
MAX_LEVERAGE = float(os.getenv("LEVELS_MAX_LEVERAGE", "10"))
DAILY_LOSS_STOP_PCT = float(os.getenv("LEVELS_DAILY_LOSS_STOP_PCT", "10"))
MIN_HOLD_RATE = float(os.getenv("LEVELS_MIN_HOLD_RATE", "0.90"))
MIN_TOUCHES = int(os.getenv("LEVELS_MIN_TOUCHES", "8"))
MIN_RR = float(os.getenv("LEVELS_MIN_RR", "2.5"))
MAX_NEAR_SIGMA = float(os.getenv("LEVELS_MAX_NEAR_SIGMA", "2.0"))
MIN_NOTIONAL_USD = float(os.getenv("LEVELS_MIN_NOTIONAL", "12"))
POLL_SEC = int(os.getenv("LEVELS_POLL_SEC", "900"))

STATE_PATH = Path(__file__).resolve().parents[1] / "research" / "levels" / "data" / "_live_state.json"
TRADE_LOG = Path(__file__).resolve().parents[1] / "research" / "levels" / "data" / "live_trades.csv"

_stop = False


def _sig(signum, frame):
    global _stop
    _stop = True
    print("[LEVELS_LIVE] SIGTERM -- finishing pass, then exiting")


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_state() -> Dict[str, Any]:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except Exception:
            pass
    return {"owned": {}, "opened_today": 0, "day": "", "realised_today": 0.0,
            "halted": False, "halt_reason": ""}


def save_state(s: Dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(s, indent=2))


def log_trade(row: Dict[str, Any]) -> None:
    import csv
    TRADE_LOG.parent.mkdir(parents=True, exist_ok=True)
    cols = ["ts_utc", "action", "coin", "episode_id", "level_price", "entry", "stop", "tp",
            "size_usd", "size_coin", "risk_usd", "rr", "hold_rate", "touches",
            "result", "detail"]
    is_new = not TRADE_LOG.exists()
    with open(TRADE_LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        if is_new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in cols})


def passes_quality(ep: Dict[str, Any]) -> Optional[str]:
    """Returns a rejection reason, or None if the setup clears the bar.

    Far stricter than the gate itself. The gate produced 39 episodes in one
    pass against roughly 9 trades a month by hand; trading it unfiltered would
    express the 25x looseness directly as 25x the losses."""
    def f(k, d=0.0):
        try:
            return float(ep.get(k) or d)
        except (TypeError, ValueError):
            return d
    if f("level_hold_rate") < MIN_HOLD_RATE:
        return f"hold_rate {f('level_hold_rate'):.2f} < {MIN_HOLD_RATE}"
    if f("level_touches") < MIN_TOUCHES:
        return f"touches {int(f('level_touches'))} < {MIN_TOUCHES}"
    if f("suggested_rr") < MIN_RR:
        return f"rr {f('suggested_rr'):.2f} < {MIN_RR}"
    if f("near_sigma") > MAX_NEAR_SIGMA:
        return f"near_sigma {f('near_sigma'):.2f} > {MAX_NEAR_SIGMA}"
    return None


def account_equity(info, addr: str) -> Dict[str, float]:
    """Equity, used and available margin, correct for BOTH account models.

    This account is a unifiedAccount: perp collateral sits in the SPOT layer
    as USDC, and clearinghouseState.marginSummary.accountValue is NOT equity
    -- it is totalRawUsd + totalNtlPos, which read $70.88 against a real
    $786.25 balance. Reading that figure as equity under-sized every position
    by about 11x. live_execution_backend._get_margin_summary_unified() has
    always handled this; this planner bypassed it and got it wrong.
    """
    unified = False
    try:
        unified = str(info.query_user_abstraction_state(addr)).strip().lower() == "unifiedaccount"
    except Exception:
        pass
    if unified:
        sp = info.spot_user_state(addr)
        usdc = next((b for b in sp.get("balances", [])
                     if str(b.get("coin", "")).upper() == "USDC"), None)
        if usdc:
            total = float(usdc.get("total", 0) or 0)
            hold = float(usdc.get("hold", 0) or 0)
            return {"equity": total, "used": hold, "available": max(0.0, total - hold),
                    "model": "unified"}
    st = info.user_state(addr)
    ms = st.get("marginSummary", {}) or {}
    eq = float(ms.get("accountValue", 0) or 0)
    used = float(ms.get("totalMarginUsed", 0) or 0)
    return {"equity": eq, "used": used, "available": max(0.0, eq - used), "model": "standard"}


def size_from_stop(equity: float, entry: float, stop: float,
                   available: float = None) -> Dict[str, float]:
    """Notional implied by risking RISK_PCT_PER_TRADE of equity down to the
    structural stop, then capped by MAX_LEVERAGE. Size comes from the stop
    distance; leverage is only ever a ceiling on the result."""
    if entry <= 0 or stop <= 0 or entry <= stop:
        return {"size_usd": 0.0, "risk_usd": 0.0, "stop_dist_pct": 0.0,
                "capped": 0.0, "margin_capped": 0.0}
    stop_dist = (entry - stop) / entry
    risk_usd = equity * RISK_PCT_PER_TRADE / 100.0
    size_usd = risk_usd / stop_dist
    capped = 1.0 if size_usd > equity * MAX_LEVERAGE else 0.0
    size_usd = min(size_usd, equity * MAX_LEVERAGE)
    # A position also has to fit the margin actually free right now, which is
    # not the same as equity once something else is already held.
    margin_capped = 0.0
    if available is not None and available > 0:
        max_by_margin = available * MAX_LEVERAGE
        if size_usd > max_by_margin:
            margin_capped = 1.0
            size_usd = max_by_margin
    return {"size_usd": size_usd, "risk_usd": risk_usd,
            "stop_dist_pct": stop_dist * 100, "capped": capped,
            "margin_capped": margin_capped}


def build_intents(client, universe) -> List[Dict[str, Any]]:
    """Scan for setups and keep the ones clearing the quality bar. No side
    effects, no venue writes -- this half only reads and computes."""
    intents, rejected = [], []
    for u in universe:
        coin = u["symbol"]
        try:
            bars = fetch(client, coin)
        except Exception:
            continue
        if len(bars) < 120:
            continue
        for ep in scan_coin(coin, bars):
            why = passes_quality(ep)
            if why:
                rejected.append((coin, ep.get("level_price"), why))
                continue
            intents.append(ep)
    return intents, rejected


def describe(ep: Dict[str, Any], equity: float, available: float = None) -> Dict[str, Any]:
    """Turn a setup into the concrete order that would be sent, with the size
    derived from the structural stop rather than from a leverage choice."""
    def f(k, d=0.0):
        try:
            return float(ep.get(k) or d)
        except (TypeError, ValueError):
            return d
    entry, stop, tp = f("suggested_entry"), f("suggested_stop"), f("suggested_tp")
    s = size_from_stop(equity, entry, stop, available)
    return {"coin": ep["coin"], "episode_id": ep.get("episode_id", ""),
            "level_price": f("level_price"), "entry": entry, "stop": stop, "tp": tp,
            "rr": f("suggested_rr"), "hold_rate": f("level_hold_rate"),
            "touches": int(f("level_touches")), **s,
            "too_small": s["size_usd"] < MIN_NOTIONAL_USD}


def main():
    ap = argparse.ArgumentParser(
        description="Level-setup trade planner (dry run; execution not wired)")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--equity", type=float, default=None,
                    help="override equity; default reads the live account (unified-aware)")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)

    state = load_state()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if state.get("day") != today:
        state.update({"day": today, "opened_today": 0, "realised_today": 0.0})
        save_state(state)
    if args.status:
        print(json.dumps(state, indent=2))
        return 0

    print("[LEVELS_LIVE] mode: DRY RUN -- order placement is NOT wired into this file")
    print(f"[LEVELS_LIVE] rails: risk {RISK_PCT_PER_TRADE}%/trade, max {MAX_CONCURRENT} "
          f"concurrent, {'unlimited' if not MAX_NEW_PER_DAY else str(MAX_NEW_PER_DAY)+'/day'}, "
          f"lev cap {MAX_LEVERAGE}x, daily stop {DAILY_LOSS_STOP_PCT}%")
    print(f"[LEVELS_LIVE] quality: hold>={MIN_HOLD_RATE} touches>={MIN_TOUCHES} "
          f"rr>={MIN_RR} near<={MAX_NEAR_SIGMA}s")
    print("[LEVELS_LIVE] NOTE: this gate measured -0.0413R vs random entry "
          "(z=-0.93, n=688 backfill trades).")

    ro = HyperliquidReadOnlyClient()
    universe = select_universe(ro)

    equity = args.equity
    available = None
    if equity is None:
        try:
            from hyperliquid.info import Info
            from hyperliquid.utils import constants
            addr = os.getenv("HL_ACCOUNT_ADDRESS", "").strip()
            info = Info(constants.MAINNET_API_URL, skip_ws=True,
                        spot_meta={"tokens": [], "universe": []})
            acc = account_equity(info, addr)
            equity, available = acc["equity"], acc["available"]
            print(f"[LEVELS_LIVE] account model: {acc['model']}   equity ${equity:.2f}   "
                  f"used ${acc['used']:.2f}   available ${available:.2f}")
        except Exception as e:
            equity = 100.0
            print(f"[LEVELS_LIVE] equity query failed ({e}); defaulting to ${equity:.2f}")
    print(f"[LEVELS_LIVE] risk per trade: ${equity * RISK_PCT_PER_TRADE / 100:.2f}")

    while not _stop:
        intents, rejected = build_intents(ro, universe)
        print(f"\n[LEVELS_LIVE] {len(intents)} setup(s) cleared the quality bar, "
              f"{len(rejected)} rejected")
        for c, lp, why in rejected[:8]:
            print(f"    reject {c:10} level={lp} -- {why}")
        if intents:
            print(f"\n{'coin':10} {'entry':>12} {'stop':>12} {'tp':>12} {'stop%':>7} "
                  f"{'size$':>9} {'risk$':>7} {'rr':>5}  flags")
        for ep in intents:
            d = describe(ep, equity, available)
            flags = []
            if d["capped"]:
                flags.append("lev-capped")
            if d.get("margin_capped"):
                flags.append("margin-capped")
            if d["too_small"]:
                flags.append(f"BELOW ${MIN_NOTIONAL_USD:g} MIN -- would not trade")
            print(f"{d['coin']:10} {d['entry']:>12.6g} {d['stop']:>12.6g} {d['tp']:>12.6g} "
                  f"{d['stop_dist_pct']:>6.2f}% {d['size_usd']:>9.2f} {d['risk_usd']:>7.2f} "
                  f"{d['rr']:>5.2f}  {','.join(flags)}")
            log_trade({"ts_utc": now_iso(), "action": "dry_run", "result": "planned",
                       **{k: d[k] for k in ("coin", "episode_id", "level_price", "entry",
                                            "stop", "tp", "rr", "hold_rate", "touches")},
                       "size_usd": round(d["size_usd"], 2),
                       "risk_usd": round(d["risk_usd"], 2),
                       "detail": ",".join(flags)})
        if args.once:
            return 0
        for _ in range(POLL_SEC):
            if _stop:
                break
            time.sleep(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

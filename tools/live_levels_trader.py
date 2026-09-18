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
import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)

from research.levels.call import find_candidates as _find_candidates  # noqa: E402
from research.levels.venue_client import HyperliquidReadOnlyClient    # noqa: E402
from research.levels.universe import select_universe                 # noqa: E402
from research.levels.trigger import fetch_1h, level_has_held          # noqa: E402

# Rails come from research/levels/config.py -- one definition each, tagged with
# whether they rest on a measurement or on my guess.
from research.levels.config import (                                  # noqa: E402
    RISK_PCT_PER_TRADE, MAX_CONCURRENT, MAX_NEW_PER_DAY, MAX_LEVERAGE,
    DAILY_LOSS_STOP_PCT, MIN_RR, MIN_NOTIONAL_USD, POLL_SEC,
    TRADE_MIN_NEAR_SIGMA, TRADE_MIN_BARS_SINCE_TOUCH, TRADE_MIN_OVERHEAD_SIGMA,
    MAX_ENTRY_DRIFT_PCT,
)
from research.levels.config import summary as config_summary          # noqa: E402

STATE_PATH = Path(__file__).resolve().parents[1] / "research" / "levels" / "data" / "_live_state.json"
TRADE_LOG = Path(__file__).resolve().parents[1] / "research" / "levels" / "data" / "live_trades.csv"

_stop = False

# Injectable so the armed path can be tested without a venue. It defaults to
# None and is built on first use; tests assign a fake. Constructing it at
# import time made the test suite hang -- every armed test tried to fetch 1h
# candles for coins like "AAA" and sat through five retries with backoff.
_RO_CLIENT = None


def _ro_client():
    global _RO_CLIENT
    if _RO_CLIENT is None:
        _RO_CLIENT = HyperliquidReadOnlyClient()
    return _RO_CLIENT


def _sig(signum, frame):
    global _stop
    _stop = True
    print("[LEVELS_LIVE] SIGTERM -- finishing pass, then exiting")


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def iso_to_ms(ts: str) -> Optional[int]:
    """now_iso() back into venue milliseconds. None if unparseable."""
    try:
        dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None
    return int(dt.timestamp() * 1000)


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

    Derived from 18 of the user's own take/skip calls, not invented. Scored on
    those labels this combination keeps 8 of 11 takes and admits 0 of 5 skips.

    What it does NOT check is as important. hold_rate and level_touches were
    the original filters and both point the wrong way: hold_rate passes 4 of 5
    skips against 5 of 11 takes, and takes average FEWER tests than skips
    (10.7 vs 12.2). Filtering on them cost takes and bought nothing.

    Depth, rest and headroom are likely one underlying factor rather than
    three independent ones -- price far from a level has necessarily not
    tested it lately and has more room above. They are checked separately so a
    larger sample can tell them apart.
    """
    def f(k, d=0.0):
        try:
            return float(ep.get(k) or d)
        except (TypeError, ValueError):
            return d
    if f("near_sigma") < TRADE_MIN_NEAR_SIGMA:
        return f"depth {f('near_sigma'):.2f}s < {TRADE_MIN_NEAR_SIGMA}s (price too close to the level)"
    if f("level_bars_since_last_touch") < TRADE_MIN_BARS_SINCE_TOUCH:
        return (f"last tested {int(f('level_bars_since_last_touch'))}b "
                f"< {TRADE_MIN_BARS_SINCE_TOUCH}b (still sitting on it)")
    if not str(ep.get("overhead_sigma", "")).strip():
        return "no resistance mapped above -- no target"
    if f("overhead_sigma") < TRADE_MIN_OVERHEAD_SIGMA:
        return f"headroom {f('overhead_sigma'):.2f}s < {TRADE_MIN_OVERHEAD_SIGMA}s (resistance overhead)"
    if f("suggested_rr") < MIN_RR:
        return f"rr {f('suggested_rr'):.2f} < {MIN_RR}"
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
    effects, no venue writes -- this half only reads and computes.

    Uses the SAME candidate source as the call sheet the user labels. That
    matters more than it looks: the trader previously used scanner.scan_coin,
    which is 4h-only and onset-only, while the user labels candidates from
    call.find_candidates, which scans 4h and 1h and includes flipped
    resistances. Eight of their eleven takes are 1h setups, so the trader
    could not see most of what it was being tuned on, and any filter fitted to
    those labels was being applied to a different population than it was
    measured on.
    """
    intents, rejected = [], []
    for ep in _find_candidates(client, universe):
        why = passes_quality(ep)
        if why:
            rejected.append((ep.get("coin"), ep.get("level_price"), why))
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
    ap = argparse.ArgumentParser(description="Level-setup trader")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--equity", type=float, default=None,
                    help="override equity; default reads the live account (unified-aware)")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--arm", action="store_true",
                    help="place REAL orders. Also requires LEVELS_LIVE_ARMED=yes in the "
                         "environment -- two independent gates, so neither a stray flag "
                         "in a shell history nor a leftover env var can trade on its own.")
    ap.add_argument("--resume", action="store_true",
                    help="clear a daily-loss halt and resume trading")
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
    if args.resume:
        state["halted"] = False
        state["halt_reason"] = ""
        save_state(state)
        print("[LEVELS_LIVE] halt cleared")
        return 0

    env_armed = os.getenv("LEVELS_LIVE_ARMED", "").strip().lower() == "yes"
    armed = bool(args.arm and env_armed)
    if args.arm and not env_armed:
        print("[LEVELS_LIVE] --arm given but LEVELS_LIVE_ARMED is not 'yes' -- staying DRY")
    print(f"[LEVELS_LIVE] mode: {'ARMED -- PLACING REAL ORDERS' if armed else 'DRY RUN'}")
    print(f"[LEVELS_LIVE] rails: risk {RISK_PCT_PER_TRADE}%/trade, max {MAX_CONCURRENT} "
          f"concurrent, {'unlimited' if not MAX_NEW_PER_DAY else str(MAX_NEW_PER_DAY)+'/day'}, "
          f"lev cap {MAX_LEVERAGE}x, daily stop {DAILY_LOSS_STOP_PCT}%")
    print(f"[LEVELS_LIVE] {config_summary()}")
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

    backend = None
    if armed:
        from live_execution_backend import LiveExecutionBackend
        backend = LiveExecutionBackend(debug=True)
        print("[LEVELS_LIVE] live backend up")

    while not _stop:
        if backend is not None:
            reconcile(backend, state)
            why = preflight(backend, state)
            if why:
                print(f"[LEVELS_LIVE] not opening this pass: {why}")
                if args.once:
                    return 0
                for _ in range(POLL_SEC):
                    if _stop:
                        break
                    time.sleep(1)
                continue
            ms = backend.get_margin_summary()
            if ms:
                equity, _used, available = ms
            bad, fatal = venue_sanity(backend, state, equity)
            if bad and fatal:
                state["halted"] = True
                state["halt_reason"] = bad
                save_state(state)
                print(f"[LEVELS_LIVE] !! HALT before trading -- {bad}")
                return 1
            if bad:
                print(f"[LEVELS_LIVE] skipping this pass (transient): {bad}")
                if args.once:
                    return 0
                for _ in range(POLL_SEC):
                    if _stop:
                        break
                    time.sleep(1)
                continue
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
            if backend is None:
                log_trade({"ts_utc": now_iso(), "action": "dry_run", "result": "planned",
                           **{k: d[k] for k in ("coin", "episode_id", "level_price", "entry",
                                                "stop", "tp", "rr", "hold_rate", "touches")},
                           "size_usd": round(d["size_usd"], 2),
                           "risk_usd": round(d["risk_usd"], 2),
                           "detail": ",".join(flags)})
                continue
            if len(state.get("owned", {})) >= MAX_CONCURRENT:
                print("[LEVELS_LIVE] concurrency cap reached, stopping this pass")
                break
            execute_one(backend, ep, equity, available, armed, state)
            # Re-checked after EVERY entry, not once per pass. The runaway
            # opened twelve positions inside a single pass, so a per-pass
            # check would have reported the damage rather than prevented it.
            bad, fatal = venue_sanity(backend, state, equity)
            if bad and fatal:
                state["halted"] = True
                state["halt_reason"] = bad
                save_state(state)
                print(f"[LEVELS_LIVE] !! HALT after entry -- {bad}")
                break
            if bad:
                # Cannot confirm what just happened at the venue, so stop
                # opening anything else this pass rather than trade blind.
                print(f"[LEVELS_LIVE] cannot verify after entry ({bad}) -- pausing this pass")
                break
        if args.once:
            return 0
        for _ in range(POLL_SEC):
            if _stop:
                break
            time.sleep(1)
    return 0


def review_flags(ep: Dict[str, Any], size_usd: float, equity: float) -> List[str]:
    """Reasons a human should look at this trade, even though it passed.

    Passing the filters is not the same as being a normal trade. These are the
    cases where the rails allow something the user may not want, and they are
    surfaced rather than blocked because the filters are fitted to 18 labels
    and the edges of that fit are exactly where it should be questioned.
    """
    def f(k, d=0.0):
        try:
            return float(ep.get(k) or d)
        except (TypeError, ValueError):
            return d
    out = []
    if equity > 0 and size_usd > equity * 1.5:
        out.append(f"notional ${size_usd:,.0f} is {size_usd/equity:.1f}x equity")
    entry, stop = f("suggested_entry"), f("suggested_stop")
    if entry > 0 and stop > 0:
        sd = (entry - stop) / entry * 100
        if sd < 1.5:
            out.append(f"stop only {sd:.2f}% away -- a gap through it loses more than the risk")
    if f("suggested_rr") < MIN_RR * 1.1:
        out.append(f"reward/risk {f('suggested_rr'):.2f} is at the {MIN_RR} floor")
    if f("near_sigma") < TRADE_MIN_NEAR_SIGMA * 1.1:
        out.append(f"depth {f('near_sigma'):.2f}s is at the {TRADE_MIN_NEAR_SIGMA}s floor")
    if f("level_hold_rate") < 0.70:
        out.append(f"level held only {f('level_hold_rate')*100:.0f}% of its {int(f('level_touches'))} tests")
    if ep.get("level_origin") == "flipped_resistance":
        out.append("flipped resistance -- only 1 of the 18 labels was this kind")
    # Time spent capping price, which level_origin misses entirely.
    cap = f("capped_bars")
    since = f("bars_since_capped", -1)
    if cap >= 25 and 0 <= since <= 48:
        out.append(f"was resistance until recently -- capped price {int(cap)}/120 bars, "
                   f"last {int(since)} bars ago; price is still flipping it")
    return out


def entry_is_actionable(backend, coin: str, planned_entry: float, stop: float, tp: float):
    """Refuse to fire unless the live price is where the plan assumed.

    execute_entry sends a MARKETABLE IOC at the current price -- it ignores the
    limit price passed to it. On 2026-09-18 that bought UNI at 8.8125 against a
    planned entry of 7.50: the stop went from 5.66% away to 19.7%, reward/risk
    from 3.27 to 0.04, and the real risk from $23 to $80 on a position sized
    for the plan. Nothing in the module noticed, because everything downstream
    trusted the planned numbers.

    The plan is only valid near the level. This re-derives the geometry from
    the LIVE price and rejects the trade if it no longer holds.
    """
    ref = backend.get_mid_price(coin)
    if not ref or ref <= 0:
        return None, "no midprice"
    drift = (ref - planned_entry) / planned_entry * 100
    if abs(drift) > MAX_ENTRY_DRIFT_PCT:
        return None, (f"price {ref:.6g} is {drift:+.2f}% from the planned entry "
                      f"{planned_entry:.6g} (max {MAX_ENTRY_DRIFT_PCT}%)")
    if ref <= stop:
        return None, f"price {ref:.6g} is already at or below the stop {stop:.6g}"
    if ref >= tp:
        return None, f"price {ref:.6g} is already at or above the target {tp:.6g}"
    live_rr = (tp - ref) / (ref - stop)
    if live_rr < MIN_RR:
        return None, f"reward/risk at the live price is {live_rr:.2f} < {MIN_RR}"
    return ref, None


def execute_one(backend, ep: Dict[str, Any], equity: float, available: float,
                armed: bool, state: Dict[str, Any]) -> bool:
    """Fire one entry at market, then protect it. True if a position opened."""
    def g(k, d=0.0):
        try:
            return float(ep.get(k) or d)
        except (TypeError, ValueError):
            return d
    coin = ep["coin"]
    entry, stop, tp = g("suggested_entry"), g("suggested_stop"), g("suggested_tp")

    # The trigger: has this level actually held? Checked before anything is
    # sent, and only when armed, because it costs an API call per candidate.
    # Arrival alone is not a setup -- every "maybe" the user gave was a level
    # price was sitting on at that moment.
    if armed:
        try:
            bars1h = fetch_1h(_ro_client(), coin)
        except Exception as e:
            print(f"[LEVELS_LIVE] skip {coin}: 1h fetch failed ({str(e)[:60]})")
            return False
        fired, why = level_has_held(bars1h, g("level_price"))
        if not fired:
            log_trade({"ts_utc": now_iso(), "action": "skip", "coin": coin,
                       "episode_id": ep.get("episode_id", ""), "level_price": g("level_price"),
                       "result": "not_triggered", "detail": why})
            print(f"[LEVELS_LIVE] wait {coin} @ {g('level_price'):.6g}: {why}")
            return False
        print(f"[LEVELS_LIVE] TRIGGER {coin} @ {g('level_price'):.6g}: {why}")
        for flag in review_flags(ep, size_from_stop(equity, entry, stop, available)["size_usd"], equity):
            print(f"[LEVELS_LIVE] [REVIEW] {coin}: {flag}")
            log_trade({"ts_utc": now_iso(), "action": "review", "coin": coin,
                       "episode_id": ep.get("episode_id", ""),
                       "level_price": g("level_price"), "result": "flagged",
                       "detail": flag})

    # Size from the LIVE price, not the planned one. Sizing off the plan while
    # filling at market is exactly what made the UNI position carry 3.5x its
    # intended risk.
    if armed:
        ref, why = entry_is_actionable(backend, coin, entry, stop, tp)
        if ref is None:
            log_trade({"ts_utc": now_iso(), "action": "skip", "coin": coin,
                       "episode_id": ep.get("episode_id", ""), "entry": entry,
                       "stop": stop, "tp": tp, "result": "not_actionable",
                       "detail": why})
            print(f"[LEVELS_LIVE] skip {coin}: {why}")
            return False
        entry = ref

    szi = size_from_stop(equity, entry, stop, available)
    size_usd = szi["size_usd"]
    row = {"ts_utc": now_iso(), "coin": coin, "episode_id": ep.get("episode_id", ""),
           "level_price": g("level_price"), "entry": entry, "stop": stop, "tp": tp,
           "size_usd": round(size_usd, 2), "risk_usd": round(szi["risk_usd"], 2),
           "rr": g("suggested_rr"), "hold_rate": g("level_hold_rate"),
           "touches": int(g("level_touches"))}

    if size_usd < MIN_NOTIONAL_USD:
        row.update({"action": "skip", "result": "below_min_notional"})
        log_trade(row)
        return False
    if coin in state.get("owned", {}):
        return False
    if not armed:
        row.update({"action": "dry_run", "result": "not_armed"})
        log_trade(row)
        return False

    # Claim the slot BEFORE sending the order, not after parsing the reply.
    # On 2026-09-18 this recorded the position only after a fill it believed
    # in, and it read success from a "filled_size_coin" attribute FillResult
    # has never had -- so getattr returned 0, every real fill was scored as a
    # rejection, state["owned"] stayed empty, MAX_CONCURRENT never engaged and
    # the pass opened 12 positions across $5,454 of notional on $775 of equity
    # with no stops on any of them. Counting what was SENT means a parsing bug
    # can no longer uncap the loop; the worst case becomes a phantom slot,
    # which reconcile() clears on the next pass.
    state.setdefault("owned", {})[coin] = {
        "episode_id": ep.get("episode_id", ""), "entry": entry, "stop": stop,
        "tp": tp, "size_usd": size_usd, "opened_at": now_iso(), "confirmed": False}
    save_state(state)

    fill = backend.execute_entry(coin=coin, side="LONG", price=entry,
                                 size_usd=size_usd, confidence=0.5)

    # FillResult's own flag, which is what executor.py used for months of live
    # trading. Do not re-derive success from prices or sizes.
    if not getattr(fill, "filled", False):
        state["owned"].pop(coin, None)
        save_state(state)
        row.update({"action": "entry", "result": "rejected",
                    "detail": str(getattr(fill, "reject_reason", ""))[:400]})
        log_trade(row)
        return False

    fill_px = float(getattr(fill, "fill_price", 0) or 0)
    filled_usd = float(getattr(fill, "fill_size_usd", 0) or 0) or size_usd
    size_coin = (filled_usd / fill_px) if fill_px > 0 else 0.0
    if fill_px <= 0 or size_coin <= 0:
        # filled=True but unusable numbers: the position may well exist at the
        # venue, so halt rather than guess. Leaving the slot claimed is
        # deliberate.
        state["halted"] = True
        state["halt_reason"] = f"{coin}: filled=True but fill_price={fill_px} size={size_coin}"
        save_state(state)
        row.update({"action": "entry", "result": "unparseable_fill",
                    "detail": state["halt_reason"]})
        log_trade(row)
        print(f"[LEVELS_LIVE] !! HALT -- {state['halt_reason']}")
        return False

    state["owned"][coin].update({"entry": fill_px, "size_coin": size_coin,
                                 "confirmed": True})
    state["opened_today"] = state.get("opened_today", 0) + 1
    save_state(state)
    row.update({"action": "entry", "result": "filled", "size_coin": size_coin,
                "detail": f"fill {fill_px:.8g}"})
    log_trade(row)

    # Protection is not optional. The account held no resting stops at all when
    # this was written, and every September winner would have been stopped out
    # by a margin-denominated stop -- so what goes in here is the STRUCTURAL
    # stop, at the venue, before anything else happens.
    prot = backend.place_native_protection(coin=coin, side="LONG", size_coin=size_coin,
                                           stop_price=stop, tp_price=tp)
    # 400, not 120. The trade log is the only durable record of what the venue
    # said about a stop, and at 120 the one live protection this account has
    # placed was cut mid-string: the whole tp leg's verdict fell off the end,
    # which is precisely the evidence this row exists to keep. DictWriter
    # quotes the field, so length is free.
    log_trade({**row, "action": "protection", "result": prot.get("status", ""),
               "detail": (f"stop={prot.get('stop_order_id','')} "
                          f"tp={prot.get('tp_order_id','')} "
                          f"verified={prot.get('stop_verified')}/"
                          f"{prot.get('tp_verified')} "
                          f"{prot.get('error','')}")[:400]})
    # place_native_protection returns "protected" / "partial" / "missing" --
    # never "ok". Comparing against "ok" made the check fire on SUCCESS, so
    # the first supervised trade placed a correct stop and take-profit and
    # then immediately emergency-closed the position it had just protected.
    if prot.get("status") != "protected":
        print(f"[LEVELS_LIVE] !! protection FAILED on {coin} -- closing immediately; "
              f"an unprotected position is not the trade")
        exit_fill = backend.execute_exit(coin=coin, side="LONG", price=fill_px,
                                         size_usd=size_usd, reason="protection_failed")

        # Cancel whatever the venue DID accept. Protection can fail with one
        # leg resting, and a leg that failed verification may not be
        # reduce-only -- an order like that, left on a flat account, opens a
        # position rather than closing one: unstopped, and in a coin this
        # module has just stopped tracking.
        for oid in (prot.get("stop_order_id"), prot.get("tp_order_id")):
            if not oid:
                continue
            try:
                if not backend.cancel_order(coin=coin, oid=oid):
                    print(f"[LEVELS_LIVE] !! cancel refused {coin} oid={oid}")
            except Exception as e:
                print(f"[LEVELS_LIVE] !! cancel failed {coin} oid={oid}: {str(e)[:80]}")

        if not getattr(exit_fill, "filled", False):
            # The close did not go through. The position is open, unprotected,
            # and releasing the slot here would drop the only record this
            # module has of it -- venue_sanity would then meet it as an
            # unknown position and halt, but not until the next entry, with
            # the rest of this pass still free to trade. Halt now and KEEP the
            # slot, so reconcile() can retire it once it is genuinely flat.
            state["halted"] = True
            state["halt_reason"] = (
                f"{coin}: protection failed and the emergency exit did not fill "
                f"-- position is open and unprotected")
            save_state(state)
            log_trade({**row, "action": "emergency_exit", "result": "exit_failed",
                       "detail": state["halt_reason"]})
            print(f"[LEVELS_LIVE] !! HALT -- {state['halt_reason']}")
            return False

        note = accrue_close(backend, state, coin, state["owned"].get(coin, {}))
        state["owned"].pop(coin, None)
        save_state(state)
        log_trade({**row, "action": "emergency_exit", "result": "closed",
                   "detail": note})
        return False

    print(f"[LEVELS_LIVE] OPENED {coin} {size_coin} @ {fill_px:.6g} "
          f"stop {stop:.6g} tp {tp:.6g}")
    return True


def venue_sanity(backend, state: Dict[str, Any], equity: float):
    """Compare the venue against what this module believes.

    Returns (reason, fatal). fatal=True is a real disagreement and should
    halt; fatal=False is a transient failure to ask, and should only skip the
    pass. Conflating the two put the bot into a permanent halt on its second
    live pass over a 429 rate limit -- a network hiccup is not a safety
    violation, and treating it as one means any busy minute stops trading
    until a human clears it.

    This is the backstop the 2026-09-18 runaway needed and did not have. Every
    in-process guard shared one assumption -- that state["owned"] reflects
    reality -- so a single bad attribute name defeated all of them at once. A
    check against the venue does not share that assumption.
    """
    try:
        st = backend.info.user_state(backend._get_user_state_address())
    except Exception as e:
        return f"venue query failed: {str(e)[:80]}", False
    positions = [p["position"] for p in st.get("assetPositions", [])]
    live = {p["coin"].upper() for p in positions if abs(float(p.get("szi", 0) or 0)) > 0}
    known = {c.upper() for c in state.get("owned", {})}
    unknown = live - known
    if unknown:
        return (f"venue holds {sorted(unknown)} which this module did not open "
                f"(knows: {sorted(known) or 'nothing'})"), True
    if len(live) > MAX_CONCURRENT:
        return f"venue holds {len(live)} positions, cap is {MAX_CONCURRENT}", True
    notional = sum(abs(float(p.get("positionValue", 0) or 0)) for p in positions)
    ceiling = equity * MAX_LEVERAGE
    if notional > ceiling:
        return f"notional ${notional:,.0f} exceeds {MAX_LEVERAGE}x equity (${ceiling:,.0f})", True
    return None, False


def assumed_loss(info: Dict[str, Any]) -> float:
    """What this trade cost if it hit its stop, from the plan we recorded.

    Used when the venue will not tell us what a close actually realised. The
    daily loss stop exists to react to losses, so the safe assumption when a
    number is missing is the losing one -- an unmeasured close must never be
    cheaper than a measured one.
    """
    try:
        entry = float(info.get("entry") or 0)
        stop = float(info.get("stop") or 0)
        size_coin = float(info.get("size_coin") or 0)
    except (TypeError, ValueError):
        return 0.0
    if entry <= 0 or stop <= 0 or size_coin <= 0:
        return 0.0
    return -abs(entry - stop) * size_coin


def accrue_close(backend, state: Dict[str, Any], coin: str,
                 info: Dict[str, Any]) -> str:
    """Add what a closed position realised to the day's running total.

    EVERY path that closes a position must come through here. A close that
    does not reach realised_today is invisible to the daily loss stop, and the
    emergency-exit path is the one most able to exploit that: it can fire on
    every pass, paying a full round trip in fees and spread each time, while
    realising too little PnL to ever trip a stop it is not reported to.

    When the venue will not say, the close is charged as a stop-out. For an
    emergency exit that overstates the cost -- it closed near the fill price,
    so the true cost is fees and spread -- and that is the direction to be
    wrong in. Halting a strategy with no measured edge one trade early costs
    less than failing to halt it at all.
    """
    since = iso_to_ms(info.get("opened_at", ""))
    pnl = None
    if since is not None:
        try:
            pnl = backend.get_realised_pnl(coin, since)
        except Exception as e:
            print(f"[LEVELS_LIVE] realised PnL query failed for {coin}: {str(e)[:80]}")
    if pnl is None:
        pnl = assumed_loss(info)
        note = f"pnl unavailable -- charged as a stop-out: {pnl:.2f}"
    else:
        note = f"realised {pnl:.2f}"
    state["realised_today"] = state.get("realised_today", 0.0) + pnl
    return note


def reconcile(backend, state: Dict[str, Any]) -> None:
    """Release slots for positions no longer open at the venue -- stopped out,
    took profit, or closed by hand. Without this the module believes it still
    holds something forever and quietly stops trading.

    This is also the only place a closed trade's result is ever observed, so
    it is where realised_today has to be accrued. Until 2026-09-18 nothing
    wrote that field: it was initialised to 0.0, reset to 0.0 each day, and
    read once by the daily loss stop, which therefore compared 0.0 against a
    negative threshold and could not fire. The kill switch the module's own
    SAFETY MODEL advertises was inert for every trade.
    """
    owned = state.get("owned", {})
    for coin in list(owned):
        try:
            sz = backend._get_venue_signed_size(coin)
        except Exception:
            continue
        if sz is not None and abs(sz) < 1e-12:
            info = owned.pop(coin, {})
            note = accrue_close(backend, state, coin, info)
            log_trade({"ts_utc": now_iso(), "action": "closed", "coin": coin,
                       "episode_id": info.get("episode_id", ""), "result": "flat",
                       "detail": f"no longer open at venue; {note}"})
            print(f"[LEVELS_LIVE] {coin} closed at venue -- slot released, {note} "
                  f"(day {state['realised_today']:.2f})")
    save_state(state)


def preflight(backend, state: Dict[str, Any]):
    """Reasons not to open anything this pass. Returns a string, or None."""
    if state.get("halted"):
        return f"halted: {state.get('halt_reason', '')}"
    owned = state.get("owned", {})
    if len(owned) >= MAX_CONCURRENT:
        return f"holding {len(owned)} of max {MAX_CONCURRENT}"
    if MAX_NEW_PER_DAY and state.get("opened_today", 0) >= MAX_NEW_PER_DAY:
        return f"opened {state['opened_today']} today, max {MAX_NEW_PER_DAY}"
    ms = backend.get_margin_summary()
    if ms is None:
        return "margin query failed -- refusing to size blind"
    equity, used, avail = ms
    if equity <= 0:
        return f"equity {equity}"
    lost = state.get("realised_today", 0.0)
    if lost <= -abs(equity) * DAILY_LOSS_STOP_PCT / 100.0:
        state["halted"] = True
        state["halt_reason"] = f"daily loss stop: {lost:.2f} against {equity:.2f} equity"
        save_state(state)
        return state["halt_reason"]
    return None


if __name__ == "__main__":
    raise SystemExit(main())

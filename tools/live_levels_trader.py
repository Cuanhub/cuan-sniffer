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
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)

import notify                                                        # noqa: E402
from research.levels.call import find_candidates as _find_candidates  # noqa: E402
from research.levels.venue_client import HyperliquidReadOnlyClient    # noqa: E402
from research.levels.universe import select_universe                 # noqa: E402
from research.levels.trigger import (                                 # noqa: E402
    fetch_1h, level_has_held, trigger_window,
)
from research.levels.levels import log_sigma, sigma_distance          # noqa: E402

# Rails come from research/levels/config.py -- one definition each, tagged with
# whether they rest on a measurement or on my guess.
from research.levels.config import (                                  # noqa: E402
    MAX_CONCURRENT, MAX_NEW_PER_DAY, MAX_LEVERAGE,
    DAILY_LOSS_STOP_PCT, MIN_RR, MIN_NOTIONAL_USD, POLL_SEC,
    TRADE_MIN_NEAR_SIGMA, TRADE_MIN_BARS_SINCE_TOUCH, TRADE_MIN_OVERHEAD_SIGMA,
    MAX_ENTRY_DRIFT_PCT, WATCH_TTL_HOURS, NEAR_ALERT_SIGMA, FLAT_LEVERAGE,
    COIN_REENTRY_COOLDOWN_HOURS, ACCOUNT_DD_HALT_PCT,
    MARGIN_PCT_PER_TRADE,
    NEAR_ALERT_COOLDOWN_SEC, TRIGGER_MAX_ABOVE_SIGMA, LIMIT_ENTRY_TTL_HOURS,
    TRIGGER_HOLD_HOURS, TRIGGER_HOLD_HOURS_SWING,
)
from research.levels.config import summary as config_summary          # noqa: E402

STATE_PATH = Path(__file__).resolve().parents[1] / "research" / "levels" / "data" / "_live_state.json"
TRADE_LOG = Path(__file__).resolve().parents[1] / "research" / "levels" / "data" / "live_trades.csv"

_stop = False

# The SDK fetches perp metadata inside Info.__init__, so construction can fail
# on a transient 429. Linear backoff: 15s, 30s, 45s, 60s.
BACKEND_START_ATTEMPTS = 5
BACKEND_START_BACKOFF_SEC = 15

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
    """Read the state file. A file that exists but will not parse is FATAL.

    This used to fall through to a fresh empty state, which is the most
    dangerous thing it could do: `halted` would come back False, clearing a
    halt no human cleared; `owned` would come back empty, forgetting live
    positions; `pending` would come back empty, orphaning resting bids at the
    venue. A corrupt file is not a new account, and the difference is the
    whole safety model.

    Returning the empty default is correct only when there is genuinely no
    file -- a first run.
    """
    if STATE_PATH.exists():
        raw = STATE_PATH.read_text()
        try:
            parsed = json.loads(raw)
        except ValueError as e:
            raise SystemExit(
                f"[LEVELS_LIVE] !! state file {STATE_PATH} exists but will not "
                f"parse ({e}). Refusing to start on a blank state: that would "
                f"clear any halt and forget open positions. Inspect it by hand, "
                f"or move it aside if this account is genuinely flat.")
        if not isinstance(parsed, dict):
            raise SystemExit(
                f"[LEVELS_LIVE] !! state file {STATE_PATH} is {type(parsed).__name__}, "
                f"not an object. Refusing to start.")
        return parsed
    return {"owned": {}, "watching": {}, "pending": {}, "opened_today": 0, "day": "",
            "realised_today": 0.0, "halted": False, "halt_reason": ""}


def save_state(s: Dict[str, Any]) -> None:
    """Write the state file atomically.

    write_text truncates and then writes, so a kill mid-write leaves a
    half-written file -- and this process is SIGKILLed routinely by launchd
    and by deploys. Paired with the old load_state, which silently replaced an
    unparseable file with a clean slate, that turned an interrupted write into
    a cleared halt and forgotten positions.

    storage.py has always written its CSVs tmp-then-replace. This is the same
    thing: rename is atomic on POSIX, so a reader sees either the old file or
    the new one, never a fragment.
    """
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(STATE_PATH.suffix + ".tmp")
    tmp.write_text(json.dumps(s, indent=2))
    tmp.replace(STATE_PATH)


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


def passes_quality(ep: Dict[str, Any], watched: bool = False) -> Optional[str]:
    """Returns a rejection reason, or None if the setup clears the bar.

    `watched` means this level already cleared the full bar on an earlier pass
    and is being re-checked as price comes back to it. Depth and rest are then
    skipped, because they are ADMISSION criteria and cannot survive the thing
    they are admitting the level for.

    Both measure distance from the level -- depth wants price >=5 sigma above
    it, rest wants the level untested for >=9 bars -- and the entry trigger
    requires the opposite of each: price within 2 sigma, touched inside 12
    hours. Checking them again at entry is asking a level to be far away at
    the moment it is being bought for being near. That conjunction is empty,
    and it took the trader to 3041 consecutive "not_triggered" decisions with
    the trigger never firing once. This docstring already said they were "one
    underlying factor": price far from a level has necessarily not tested it
    lately.

    Target, headroom and reward/risk are NOT skipped. Those stay true, and
    stay meaningful, with price sitting on the level -- on 2026-09-18 the RR
    floor correctly refused all three USELESS triggers at rr 0.45-0.63.

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
    if not watched:
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
    # Never guess the account model. A failed query used to fall through to
    # `standard`, which on THIS account reads marginSummary.accountValue --
    # $70.88 against a real $786.25, an 11x under-read. Every position would
    # then size off 11x too little equity and fall under MIN_NOTIONAL, so the
    # trader would stop trading entirely and look like a quiet market. The
    # query fails on a 429, which happens routinely.
    try:
        unified = str(info.query_user_abstraction_state(addr)).strip().lower() == "unifiedaccount"
    except Exception as e:
        raise RuntimeError(
            f"cannot determine account model ({str(e)[:120]}); refusing to "
            f"size against a guess") from e
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
                   available: float = None,
                   venue_max_leverage: float = None) -> Dict[str, float]:
    """Flat size: MARGIN_PCT_PER_TRADE of equity, levered FLAT_LEVERAGE times.

    Size was derived from the stop so risk stayed pinned at 3% of equity, and
    that is exactly what made it unpredictable -- a 0.9% stop asked for 3.3x
    equity, a 13% stop for 0.22x. Same idea, wildly different position, purely
    because of where the stop landed.

    Flat sizing trades constant risk for a system that behaves the same way
    every time. risk_usd is still reported, because the trade log and the
    review flags need to say what a position actually risks, but it no longer
    decides anything.

    venue_max_leverage is the per-coin ceiling Hyperliquid enforces -- 3x on
    130 of its 234 perps, including PONS, CASHCAT, VVV and USELESS. Asking for
    5x there is rejected by the venue, so the ask is lowered instead.
    """
    if entry <= 0 or stop <= 0 or entry <= stop:
        return {"size_usd": 0.0, "risk_usd": 0.0, "stop_dist_pct": 0.0,
                "capped": 0.0, "margin_capped": 0.0, "leverage": 0.0}
    stop_dist = (entry - stop) / entry
    lev = min(FLAT_LEVERAGE, MAX_LEVERAGE)
    capped = 0.0
    if venue_max_leverage and venue_max_leverage > 0 and lev > venue_max_leverage:
        lev = float(venue_max_leverage)
        capped = 1.0
    # Margin committed, then levered. NOT equity x leverage: that would put
    # the whole account behind every trade.
    margin_usd = equity * MARGIN_PCT_PER_TRADE / 100.0
    size_usd = margin_usd * lev
    # A position also has to fit the margin actually free right now, which is
    # not the same as equity once something else is already held.
    margin_capped = 0.0
    if available is not None and available > 0:
        max_by_margin = available * MAX_LEVERAGE
        if size_usd > max_by_margin:
            margin_capped = 1.0
            size_usd = max_by_margin
    return {"size_usd": size_usd, "risk_usd": size_usd * stop_dist,
            "stop_dist_pct": stop_dist * 100, "capped": capped,
            "margin_capped": margin_capped, "margin_usd": margin_usd,
            "leverage": lev}


@dataclass(frozen=True)
class Outcome:
    """What execute_one did.

    Truthy when a position was opened AND protected, so `if execute_one(...)`
    keeps its old meaning.

    order_sent is separate because venue_sanity keys off it. That check costs a
    user_state call and must run after anything that could have changed what
    the venue holds -- but it was running after EVERY candidate, ~99% of which
    send nothing. At ~130 candidates a pass that is ~130 extra calls per pass;
    it tripped 429s, and a 429 is transient, so the loop broke and abandoned
    the rest of the pass. 22 passes were truncated that way in nine hours,
    mean coverage 87 of 129 candidates.
    """
    opened: bool = False
    order_sent: bool = False

    def __bool__(self) -> bool:
        return self.opened


def watch_key(ep: Dict[str, Any]) -> str:
    return f"{str(ep.get('coin','')).upper()}|{ep.get('level_price','')}"


def prune_watchlist(watching: Dict[str, Any]) -> int:
    """Drop levels watched longer than WATCH_TTL_HOURS. Returns the count."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=WATCH_TTL_HOURS)
    dropped = 0
    for key in list(watching):
        at = iso_to_ms(watching[key].get("admitted_at", ""))
        if at is None or at < cutoff.timestamp() * 1000:
            watching.pop(key, None)
            dropped += 1
    return dropped


def cooldown_remaining(state: Dict[str, Any], coin: str) -> float:
    """Hours left before `coin` may be traded again, 0.0 when it is free.

    Set by a LOSING close only -- see COIN_REENTRY_COOLDOWN_HOURS for the
    numbers. A level that just stopped us out is a level that broke, and buying
    it again minutes later is buying the break; a level that paid is a level
    that worked.
    """
    if not COIN_REENTRY_COOLDOWN_HOURS:
        return 0.0
    at = iso_to_ms((state.get("cooled") or {}).get(coin.upper(), ""))
    if at is None:
        return 0.0
    elapsed_h = (datetime.now(timezone.utc).timestamp() * 1000 - at) / 3_600_000.0
    return max(0.0, COIN_REENTRY_COOLDOWN_HOURS - elapsed_h)


def start_cooldown(state: Dict[str, Any], coin: str) -> None:
    state.setdefault("cooled", {})[coin.upper()] = now_iso()


def prune_cooldowns(state: Dict[str, Any]) -> int:
    """Drop expired stamps so the state file cannot grow without bound."""
    cooled = state.get("cooled") or {}
    for coin in list(cooled):
        if cooldown_remaining(state, coin) <= 0.0:
            cooled.pop(coin, None)
    return len(cooled)


def seed_watchlist_from_log(watching: Dict[str, Any]) -> int:
    """Admit levels the trade log shows already cleared the full bar.

    The watchlist is new, so it starts empty, and a level only becomes
    eligible once it has been seen while price was still far above it. That
    would leave the levels price has ALREADY come back to -- the ones closest
    to triggering -- ineligible until they drift far away and return again.

    Those admissions did happen and were recorded: a dry_run/planned row is
    written only for a setup that cleared the quality bar, so the log is an
    accurate register of what qualified and when. This replays the ones inside
    the TTL. Returns the number added.
    """
    import csv as _csv
    if not TRADE_LOG.exists():
        return 0
    cutoff = (datetime.now(timezone.utc)
              - timedelta(hours=WATCH_TTL_HOURS)).strftime("%Y-%m-%dT%H:%M:%SZ")
    added = 0
    try:
        with open(TRADE_LOG, newline="") as fh:
            for r in _csv.DictReader(fh):
                if r.get("action") != "dry_run" or r.get("result") != "planned":
                    continue
                if (r.get("ts_utc") or "") < cutoff:
                    continue
                key = f"{str(r.get('coin','')).upper()}|{r.get('level_price','')}"
                if key in watching:
                    continue
                watching[key] = {"coin": r.get("coin"),
                                 "level_price": r.get("level_price"),
                                 "admitted_at": r.get("ts_utc"),
                                 "seeded": True}
                added += 1
    except (OSError, _csv.Error) as e:
        print(f"[LEVELS_LIVE] watchlist seed failed: {str(e)[:100]}")
        return added
    return added


def build_intents(client, universe, watching: Optional[Dict[str, Any]] = None,
                  bars_out: Optional[Dict[str, Any]] = None,
                  tf_cache: Optional[Dict[Any, Any]] = None
                  ) -> Tuple[List[Dict[str, Any]], List[Any]]:
    """Scan for setups and keep the ones clearing the quality bar. No side
    effects, no venue writes -- this half only reads and computes.

    A level clearing the FULL bar is admitted to `watching` and stays eligible
    for WATCH_TTL_HOURS. While it is watched it only has to clear the entry
    criteria -- target, headroom, reward/risk -- because depth and rest were
    already answered at admission and cannot be true again once price has come
    back to the level. Without that, the quality bar and the entry trigger
    contradict each other and nothing can ever be bought (see passes_quality).

    Uses the SAME candidate source as the call sheet the user labels. That
    matters more than it looks: the trader previously used scanner.scan_coin,
    which is 4h-only and onset-only, while the user labels candidates from
    call.find_candidates, which scans 4h and 1h and includes flipped
    resistances. Eight of their eleven takes are 1h setups, so the trader
    could not see most of what it was being tuned on, and any filter fitted to
    those labels was being applied to a different population than it was
    measured on.
    """
    watching = {} if watching is None else watching
    intents, rejected = [], []
    # The scan already pulls a 1h series per coin. Collecting it here and
    # trimming it to the trigger's window is what lets execute_one skip
    # re-fetching the identical bars -- 30 of the pass's ~93 venue calls.
    # Trimmed rather than passed whole because level_has_held derives sigma
    # from the series LENGTH; see trigger.TRIGGER_1H_HOURS.
    raw_1h: Dict[str, Any] = {}
    # One mids request refreshes the forming 4h bar for the whole universe, so
    # the 4h series can be reused between passes without the price everything
    # is measured against going stale with it. Thirty candle calls become one.
    mids: Dict[str, float] = {}
    if tf_cache is not None:
        try:
            mids = client.get_all_mids()
        except Exception as e:
            # No mids means no safe reuse: fall through with an empty cache
            # reference so every series is refetched, which is the old
            # behaviour rather than a stale one.
            print(f"[LEVELS_LIVE] mids fetch failed ({str(e)[:80]}) -- "
                  f"refetching candle series this pass")
            tf_cache = None
    for ep in _find_candidates(client, universe, bars_out=raw_1h,
                               tf_cache=tf_cache, mids=mids):
        key = watch_key(ep)
        strict = passes_quality(ep, watched=False)
        if strict is None:
            # Clears the full bar, so the clock restarts. The TTL means "48h
            # since this last qualified", not "48h since we first noticed it":
            # a level price is still far above goes on qualifying pass after
            # pass, and letting it age out underneath that would drop it for
            # no reason. The window that matters starts when price closes in
            # and the level stops clearing on its own.
            entry = watching.get(key) or {"coin": ep.get("coin"),
                                          "level_price": ep.get("level_price")}
            entry["admitted_at"] = now_iso()
            watching[key] = entry
            intents.append(ep)
            continue
        if key in watching:
            relaxed = passes_quality(ep, watched=True)
            if relaxed is None:
                intents.append({**ep, "_watched": True})
                continue
            rejected.append((ep.get("coin"), ep.get("level_price"),
                             f"watched, but {relaxed}"))
            continue
        rejected.append((ep.get("coin"), ep.get("level_price"), strict))
    if bars_out is not None:
        # Only coins with a live intent: trimming the rest would be work for
        # bars nothing is going to read this pass.
        wanted = {str(e.get("coin", "")) for e in intents}
        for coin, bars in raw_1h.items():
            if coin in wanted:
                bars_out[coin] = trigger_window(bars)
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
    ap.add_argument("--limit-entry", nargs=2, metavar=("COIN", "LEVEL"),
                    help="rest a GTC bid at that level's planned entry, then "
                         "exit. Needs --arm. The bid is protected on fill by "
                         "the normal pass, and pulled after "
                         "LEVELS_LIMIT_ENTRY_TTL_HOURS if it never fills.")
    ap.add_argument("--seed-watchlist", action="store_true",
                    help="admit levels the trade log shows already cleared the "
                         "quality bar, then exit. Run once after the watchlist "
                         "is introduced; it is empty otherwise.")
    args = ap.parse_args()

    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)

    state = load_state()
    roll_day(state)
    if args.status:
        print(json.dumps(state, indent=2))
        return 0
    if args.seed_watchlist:
        watching = state.setdefault("watching", {})
        before = len(watching)
        added = seed_watchlist_from_log(watching)
        prune_watchlist(watching)
        save_state(state)
        print(f"[LEVELS_LIVE] watchlist seeded: {before} -> {len(watching)} "
              f"({added} admitted from the trade log)")
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
    # "margin", not "risk". RISK_PCT_PER_TRADE stopped deciding size when flat
    # sizing landed, and printing it here said 3% while the margin committed
    # was 5% and the actual risk at the stop was 0.44%. Three different numbers
    # under one label is the exact conflation that left the daily loss stop
    # fifty-four losses from firing.
    print(f"[LEVELS_LIVE] rails: margin {MARGIN_PCT_PER_TRADE}%/trade at "
          f"{FLAT_LEVERAGE:g}x, max {MAX_CONCURRENT} "
          f"concurrent, {'unlimited' if not MAX_NEW_PER_DAY else str(MAX_NEW_PER_DAY)+'/day'}, "
          f"lev cap {MAX_LEVERAGE}x, daily stop {DAILY_LOSS_STOP_PCT}%, "
          f"account dd halt {ACCOUNT_DD_HALT_PCT:g}%")
    print(f"[LEVELS_LIVE] {config_summary()}")
    print("[LEVELS_LIVE] NOTE: this gate measured -0.0413R vs random entry "
          "(z=-0.93, n=688 backfill trades).")
    if armed:
        # launchd restarts this on any non-zero exit, so an unannounced
        # restart is indistinguishable from a process that never came back.
        notify.send(
            f"▶️ *Level trader armed*\nmargin {MARGIN_PCT_PER_TRADE}%/trade "
            f"at {FLAT_LEVERAGE:g}x, "
            f"max {MAX_CONCURRENT} concurrent, {MAX_NEW_PER_DAY}/day, "
            f"daily stop {DAILY_LOSS_STOP_PCT}%, "
            f"account dd halt {ACCOUNT_DD_HALT_PCT:g}%",
            key="armed_start", cooldown_sec=300)

    ro = HyperliquidReadOnlyClient()
    universe = select_universe(ro)

    # Lives across passes on purpose, unlike bars_cache. A 4h series is keyed
    # to the bar it belongs to and cannot change until that bar closes, so
    # refetching it every 450s was ~32 requests per bar per coin for identical
    # data. Entries expire by bar boundary rather than by age, and the forming
    # bar is refreshed from live mids each pass, so nothing decision-relevant
    # is ever read stale. Bounded by the universe size: one entry per coin.
    tf_cache: Dict[Any, Any] = {}

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
            if armed:
                # The armed loop re-reads equity from the backend every pass,
                # so this figure is only the banner -- but a banner that
                # invents $100 against a real $768 is the kind of number an
                # operator checks a pass against and believes.
                raise SystemExit(
                    f"[LEVELS_LIVE] !! equity query failed ({str(e)[:120]}) and "
                    f"this process is ARMED. Refusing to start on a guessed "
                    f"balance.")
            equity = 100.0
            print(f"[LEVELS_LIVE] equity query failed ({e}); dry run, "
                  f"assuming ${equity:.2f} to print sizes")
    _margin = equity * MARGIN_PCT_PER_TRADE / 100.0
    print(f"[LEVELS_LIVE] per trade: margin ${_margin:.2f}, "
          f"notional ${_margin * FLAT_LEVERAGE:.2f}, "
          f"risk at a median 1.76% stop ${_margin * 0.088:.2f} "
          f"({_margin * 0.088 / equity * 100:.2f}% of equity)")

    backend = None
    if armed:
        from live_execution_backend import LiveExecutionBackend
        # Retried, because the SDK's Info.__init__ fetches perp metadata
        # itself and a 429 there raises before any of our code runs. Twice on
        # 2026-09-19 that killed the process outright at startup. launchd
        # restarts it, so it self-heals, but a crash loop against a rate limit
        # is the worst possible way to wait -- every restart is another
        # request, which is what caused the limit.
        backend = start_backend(LiveExecutionBackend)
        if backend is None:
            return 0                      # stopped while waiting to retry
        print("[LEVELS_LIVE] live backend up")

    if args.limit_entry:
        if not armed:
            print("[LEVELS_LIVE] --limit-entry places a REAL order and needs "
                  "--arm plus LEVELS_LIVE_ARMED=yes")
            return 1
        return place_resting_bid(backend, ro, universe, args.limit_entry[0],
                                 args.limit_entry[1], equity, available, state)

    while not _stop:
        if roll_day(state):
            # The universe is "top N perps by 24h volume, refreshed daily" --
            # and this loop never refreshed it. scanner.py has always rolled
            # it on the date change; the trader picked it once at startup and
            # ran on it for as long as the process lived, which under launchd
            # is days. A frozen universe keeps trading names that have dropped
            # out for illiquidity and never sees the ones that came in.
            try:
                universe = select_universe(ro)
                print(f"[LEVELS_LIVE] universe refreshed: {len(universe)} symbols")
            except Exception as e:
                print(f"[LEVELS_LIVE] universe refresh failed ({str(e)[:80]}); "
                      f"keeping yesterday's {len(universe)} symbols")
        if backend is not None:
            # Pending first: a bid that filled must become a known, protected
            # position before venue_sanity can meet it as an unknown one.
            reconcile_pending(backend, state)
            reconcile(backend, state)

            # Bookkeeping runs BEFORE the preflight early-return, not after it.
            # preflight continues the loop whenever it refuses a pass, so with
            # these below it they stopped running exactly when the trader was
            # busiest: four positions held for most of 2026-10-02/03 meant the
            # cap refused every pass, and in that window 110 of 976 watched
            # levels aged past the 48h TTL without being dropped -- the oldest
            # 60.4h. That is not cosmetic. build_intents gives a watched level
            # passes_quality(watched=True), which skips the depth and rest
            # admission gates, so a level the TTL had retired stayed eligible.
            #
            # Same shape as the day rollover, the asset-meta load and the
            # universe refresh: something that must happen every cycle, placed
            # where it only happens on some.
            watching = state.setdefault("watching", {})
            expired = prune_watchlist(watching)
            on_cooldown = prune_cooldowns(state)
            if expired or on_cooldown:
                save_state(state)
            if on_cooldown:
                print("[LEVELS_LIVE] cooling off: "
                      + ", ".join(f"{c} {cooldown_remaining(state, c):.1f}h left"
                                  for c in sorted(state.get("cooled") or {})))

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
                halt(state, bad, "before trading")
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
        # Created here, not after the scan, so build_intents can hand the 1h
        # bars it already fetched straight into it. Still one dict per pass:
        # the trigger asks what price did in the last two hours, so bars must
        # never carry across passes.
        bars_cache: Dict[str, Any] = {}
        intents, rejected = build_intents(ro, universe, watching,
                                          bars_out=bars_cache,
                                          tf_cache=tf_cache)
        save_state(state)
        n_watched = sum(1 for e in intents if e.get("_watched"))
        print(f"\n[LEVELS_LIVE] {len(intents)} setup(s) cleared the quality bar "
              f"({n_watched} of them on the watchlist as price came back), "
              f"{len(rejected)} rejected")
        print(f"[LEVELS_LIVE] watching {len(watching)} level(s) for up to "
              f"{WATCH_TTL_HOURS}h"
              + (f"; {expired} expired this pass" if expired else ""))
        for c, lp, why in rejected[:8]:
            print(f"    reject {c:10} level={lp} -- {why}")
        if intents:
            print(f"\n{'coin':10} {'entry':>12} {'stop':>12} {'tp':>12} {'stop%':>7} "
                  f"{'size$':>9} {'risk$':>7} {'rr':>5}  flags")
        watchboard: List[Dict[str, Any]] = []   # how close each near-miss came
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
            if (len(state.get("owned", {}))
                    + len(state.get("pending", {}))) >= MAX_CONCURRENT:
                print("[LEVELS_LIVE] concurrency cap reached, stopping this pass")
                break
            # The daily cap belongs in this guard too, not only in preflight.
            # preflight runs ONCE per pass, so on 2026-09-28 it saw
            # opened_today=4, allowed the pass, and BTC then filled (5) with
            # DOGE filling six seconds later (6) -- one over the cap. The
            # comment below already says this lesson for venue_sanity; the
            # daily cap was simply left out of the same re-check.
            if MAX_NEW_PER_DAY and state.get("opened_today", 0) >= MAX_NEW_PER_DAY:
                print(f"[LEVELS_LIVE] daily cap reached "
                      f"({state['opened_today']}/{MAX_NEW_PER_DAY}), "
                      f"stopping this pass")
                break
            outcome = execute_one(backend, ep, equity, available, armed, state,
                                  bars_cache=bars_cache, watchboard=watchboard)
            # Re-checked after every ORDER, not once per pass: the runaway
            # opened twelve positions inside a single pass, so a per-pass check
            # would have reported the damage rather than prevented it.
            #
            # After every order, though -- not after every candidate. Nearly
            # all of them send nothing, and asking the venue anyway added ~130
            # user_state calls a pass, tripped 429s, and truncated the pass on
            # the transient branch below. The safety property is unchanged:
            # nothing can reach the venue without this running straight after.
            if not outcome.order_sent:
                continue
            bad, fatal = venue_sanity(backend, state, equity)
            if bad and fatal:
                halt(state, bad, "after entry")
                break
            if bad:
                # Cannot confirm what just happened at the venue, so stop
                # opening anything else this pass rather than trade blind.
                print(f"[LEVELS_LIVE] cannot verify after entry ({bad}) -- pausing this pass")
                break
        # Levels price has nearly reached. The board exists for looking at on
        # purpose; this is only so a zone cannot be reached unnoticed while
        # nobody is looking. Rate limited, and capped at five -- alerting on
        # everything eligible would be ~140 lines a pass, which is a mailing
        # list, not a signal.
        near = sorted((w for w in watchboard if w["above"] <= NEAR_ALERT_SIGMA),
                      key=lambda w: w["above"])[:5]
        if near:
            lines = "\n".join(
                f"`{w['coin']}` {float(w['level']):.6g} -- {w['above']:.1f}s above "
                f"(fires <={TRIGGER_MAX_ABOVE_SIGMA}), entry `{float(w['entry']):.6g}`, "
                f"stop `{float(w['stop']):.6g}`, rr {float(w['rr'] or 0):.2f}"
                for w in near)
            sent = notify.send(f"🔵 *Approaching buy zones* ({len(near)} of "
                               f"{len(watchboard)} watched)\n{lines}",
                               key="approaching", cooldown_sec=NEAR_ALERT_COOLDOWN_SEC)
            print(f"[LEVELS_LIVE] {len(near)} level(s) within {NEAR_ALERT_SIGMA} sigma"
                  f" -- digest {'sent' if sent else 'suppressed (cooldown)'}; "
                  f"nearest {near[0]['coin']} at {near[0]['above']:.1f}s")
        else:
            closest = min((w["above"] for w in watchboard), default=None)
            print(f"[LEVELS_LIVE] nothing within {NEAR_ALERT_SIGMA} sigma"
                  + (f"; nearest is {closest:.1f}s" if closest is not None else ""))
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


def cached_1h(coin: str, cache: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """1h bars for `coin`, fetched at most once per pass.

    The trigger needs these per CANDIDATE, but one coin carries several
    levels: 141 candidates came from about 30 coins, so this was fetching the
    same series roughly five times over. That is ~110 wasted calls a pass
    against a venue that returned 52 rate limits in nine hours, and a rate
    limit here is not free -- it truncated the pass.

    As of 2026-10-01 the cache arrives already populated by build_intents,
    which trims the 1h series the scan had in hand anyway, so the common path
    makes NO call at all. The fetch below is the fallback for a coin whose 1h
    leg failed during the scan -- that coin still gets its own chance rather
    than being dropped for an error that belonged to a different request.

    Per pass, deliberately. Bars must not carry between passes: what the
    trigger asks is what price did in the last two hours.
    """
    if cache is None:
        return fetch_1h(_ro_client(), coin)
    if coin not in cache:
        cache[coin] = fetch_1h(_ro_client(), coin)
    return cache[coin]


def execute_one(backend, ep: Dict[str, Any], equity: float, available: float,
                armed: bool, state: Dict[str, Any],
                bars_cache: Optional[Dict[str, Any]] = None,
                watchboard: Optional[List[Dict[str, Any]]] = None) -> "Outcome":
    """Fire one entry at market, then protect it. Truthy if a position opened."""
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
            bars1h = cached_1h(coin, bars_cache)
        except Exception as e:
            # Recorded, not just printed. Every other way a candidate is
            # dropped writes a row, so a setup lost to a rate limit was the
            # one outcome invisible to live_trades.csv -- and the gap between
            # what the gate saw and what it did is the measurement this whole
            # package exists to make.
            detail = f"1h fetch failed: {str(e)[:200]}"
            print(f"[LEVELS_LIVE] skip {coin}: {detail}")
            log_trade({"ts_utc": now_iso(), "action": "skip", "coin": coin,
                       "episode_id": ep.get("episode_id", ""),
                       "level_price": g("level_price"),
                       "result": "fetch_failed", "detail": detail})
            return Outcome()
        # A flipped resistance has to prove the flip by holding a channel;
        # ordinary support taps and sends, and asking it to sit for two hours
        # is asking it to behave like the thing it is not.
        is_flip = str(ep.get("level_origin", "")) == "flipped_resistance"
        hold_h = TRIGGER_HOLD_HOURS if is_flip else TRIGGER_HOLD_HOURS_SWING
        fired, why = level_has_held(bars1h, g("level_price"), hold_hours=hold_h)
        if not fired:
            log_trade({"ts_utc": now_iso(), "action": "skip", "coin": coin,
                       "episode_id": ep.get("episode_id", ""), "level_price": g("level_price"),
                       "result": "not_triggered", "detail": why})
            print(f"[LEVELS_LIVE] wait {coin} @ {g('level_price'):.6g}: {why}")
            # Record how far off it is, so the pass can report the ones that
            # are nearly there. Recomputed from the bars already in hand
            # rather than parsed back out of `why`.
            if watchboard is not None:
                _closed = bars1h[:-1]
                _sig = log_sigma(_closed, len(_closed) - 1) if len(_closed) > 6 else 0.0
                if _sig > 0:
                    watchboard.append({
                        "coin": coin, "level": g("level_price"),
                        "above": sigma_distance(_closed[-1]["c"], g("level_price"), _sig),
                        "why": why, "entry": entry, "stop": stop, "tp": tp,
                        "rr": g("suggested_rr")})
            return Outcome()
        print(f"[LEVELS_LIVE] TRIGGER {coin} @ {g('level_price'):.6g}: {why} "
              f"[{'flip, ' + str(hold_h) + 'h channel hold' if is_flip else 'swing low, tap and reclaim'}]")
        for flag in review_flags(ep, size_from_stop(equity, entry, stop, available,
                                                    venue_max_leverage=None)["size_usd"], equity):
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
            return Outcome()
        entry = ref

    vmax = None
    try:
        vmax = backend.max_leverage_for(coin) if backend is not None else None
    except Exception:
        vmax = None
    szi = size_from_stop(equity, entry, stop, available, venue_max_leverage=vmax)
    size_usd = szi["size_usd"]
    row = {"ts_utc": now_iso(), "coin": coin, "episode_id": ep.get("episode_id", ""),
           "level_price": g("level_price"), "entry": entry, "stop": stop, "tp": tp,
           "size_usd": round(size_usd, 2), "risk_usd": round(szi["risk_usd"], 2),
           "rr": g("suggested_rr"), "hold_rate": g("level_hold_rate"),
           "touches": int(g("level_touches"))}

    if size_usd < MIN_NOTIONAL_USD:
        row.update({"action": "skip", "result": "below_min_notional"})
        log_trade(row)
        return Outcome()
    if coin in state.get("owned", {}):
        return Outcome()
    left = cooldown_remaining(state, coin)
    if left > 0:
        row.update({"action": "skip", "result": "coin_cooldown",
                    "detail": f"stopped out {COIN_REENTRY_COOLDOWN_HOURS - left:.1f}h "
                              f"ago; {left:.1f}h of cooldown left"})
        log_trade(row)
        return Outcome()
    if not armed:
        row.update({"action": "dry_run", "result": "not_armed"})
        log_trade(row)
        return Outcome()

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

    # Hyperliquid enforces leverage per asset, so a 5x-sized order into a coin
    # the account has set to 3x is refused for margin. Set it to match before
    # the order, capped at whatever the venue allows on this coin.
    backend.set_leverage(coin, int(szi.get("leverage") or FLAT_LEVERAGE))

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
        return Outcome(order_sent=True)

    fill_px = float(getattr(fill, "fill_price", 0) or 0)
    filled_usd = float(getattr(fill, "fill_size_usd", 0) or 0) or size_usd
    # Dividing notional by price does not land on a clean coin count: PONS came
    # out 114.99999999999999 for a 115-coin fill, and every consumer of this
    # number rounds DOWN. Snap it to 12 significant places so a float artifact
    # cannot cost a whole unit of size.
    size_coin = round(filled_usd / fill_px, 12) if fill_px > 0 else 0.0
    if fill_px <= 0 or size_coin <= 0:
        # filled=True but unusable numbers: the position may well exist at the
        # venue, so halt rather than guess. Leaving the slot claimed is
        # deliberate.
        halt(state, f"{coin}: filled=True but fill_price={fill_px} size={size_coin}")
        row.update({"action": "entry", "result": "unparseable_fill",
                    "detail": state["halt_reason"]})
        log_trade(row)
        return Outcome(order_sent=True)

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
            halt(state, f"{coin}: protection failed and the emergency exit did "
                        f"not fill -- position is open and unprotected")
            log_trade({**row, "action": "emergency_exit", "result": "exit_failed",
                       "detail": state["halt_reason"]})
            return Outcome(order_sent=True)

        note = accrue_close(backend, state, coin, state["owned"].get(coin, {}))
        state["owned"].pop(coin, None)
        save_state(state)
        log_trade({**row, "action": "emergency_exit", "result": "closed",
                   "detail": note})
        # Real money moved and the position existed unprotected, however
        # briefly. MAX_NEW_PER_DAY bounds how often this can repeat, so it is
        # sent every time rather than rate limited.
        notify.send(
            f"⚠️ *{coin}* opened and closed immediately\n"
            f"protection came back `{prot.get('status', '?')}`\n"
            f"{str(prot.get('error', ''))[:300]}\n{note}")
        return Outcome(order_sent=True)

    print(f"[LEVELS_LIVE] OPENED {coin} {size_coin} @ {fill_px:.6g} "
          f"stop {stop:.6g} tp {tp:.6g}")
    # A filled, protected position is the one event worth interrupting someone
    # for. Everything needed to check it against the chart goes in the message,
    # because the alternative is opening a laptop to find out what was bought.
    risk_usd = abs(fill_px - stop) * size_coin
    reward_usd = abs(tp - fill_px) * size_coin
    notify.send(
        f"🟢 *OPENED {coin}* {size_coin:.6g} @ `{fill_px:.6g}`\n"
        f"stop `{stop:.6g}`  ({(fill_px - stop) / fill_px * 100:.2f}% away, "
        f"risk ${risk_usd:.2f})\n"
        f"tp   `{tp:.6g}`  ({(tp - fill_px) / fill_px * 100:.2f}% away, "
        f"target ${reward_usd:.2f})\n"
        f"size ${size_usd:.2f}   rr {reward_usd / risk_usd if risk_usd else 0:.2f}   "
        f"level {g('level_price'):.6g}")
    return Outcome(opened=True, order_sent=True)


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


def start_backend(factory, sleep=time.sleep):
    """Build the live backend, retrying a transient venue failure.

    The SDK fetches perp metadata inside Info.__init__, so construction itself
    hits the network and a 429 there raises before any of our code runs. Twice
    on 2026-09-19 that killed the process outright at startup. launchd restarts
    it, so it self-heals -- but a crash loop against a rate limit is the worst
    possible way to wait, because every restart is another request, which is
    what caused the limit.

    Returns None if a stop was requested while waiting. Re-raises if the venue
    is still refusing after every attempt: coming up without a backend would
    mean an armed trader that cannot trade and does not say so.
    """
    for attempt in range(1, BACKEND_START_ATTEMPTS + 1):
        try:
            return factory(debug=True)
        except Exception as e:
            if attempt == BACKEND_START_ATTEMPTS:
                print(f"[LEVELS_LIVE] live backend failed to start after "
                      f"{attempt} attempts: {str(e)[:160]}")
                raise
            wait = BACKEND_START_BACKOFF_SEC * attempt
            print(f"[LEVELS_LIVE] backend start attempt {attempt} failed "
                  f"({str(e)[:80]}); retrying in {wait}s")
            for _ in range(wait):
                if _stop:
                    return None
                sleep(1)
    return None


def roll_day(state: Dict[str, Any]) -> bool:
    """Reset the daily counters when the UTC date changes. True if it rolled.

    This ran once, at startup, and the pass loop never called it again -- so a
    process that stays up across midnight keeps yesterday's counters forever.
    Both are safety rails and both fail quietly the longer the process runs:
    opened_today never clears, so after MAX_NEW_PER_DAY fills the trader stops
    opening anything until someone restarts it; realised_today never clears,
    so the daily loss stop starts summing losses across days and halts on a
    drawdown that was never a day's worth. This trader is built to run for
    days under launchd, which is exactly the case it got wrong.
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if state.get("day") == today:
        return False
    was = state.get("day")
    state.update({"day": today, "opened_today": 0, "realised_today": 0.0})
    save_state(state)
    print(f"[LEVELS_LIVE] new UTC day {today} (was {was}) -- "
          f"daily counters reset")
    return True


def halt(state: Dict[str, Any], reason: str, where: str = "") -> str:
    """Stop trading and tell a human. Returns the reason, for the caller.

    Every halt goes through here. A halt is the one condition this module
    cannot resolve on its own -- preflight refuses on the persisted flag until
    someone clears it with --resume -- so it is the one condition where
    printing into an unwatched log file is not enough. It is also the only
    alert that would repeat forever if left unkeyed: the flag persists, the
    trader polls every 15 minutes, and the halt would be re-announced on every
    pass until it stopped meaning anything.
    """
    already = bool(state.get("halted"))
    state["halted"] = True
    state["halt_reason"] = reason
    save_state(state)
    print(f"[LEVELS_LIVE] !! HALT{' ' + where if where else ''} -- {reason}")
    if not already:
        notify.send(
            f"🛑 *Level trader halted*\n{reason}\n\n"
            f"It will not open anything until it is cleared with --resume.",
            key="halt", cooldown_sec=3600)
    return reason


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
    # Stamped here because this is the single path every close comes through,
    # and it is the only place that knows the realised result -- which is what
    # decides whether a cooldown is warranted at all.
    if pnl <= 0:
        start_cooldown(state, coin)
        if COIN_REENTRY_COOLDOWN_HOURS:
            note += (f"; {coin} on cooldown for "
                     f"{COIN_REENTRY_COOLDOWN_HOURS:g}h")
    return note


def place_resting_bid(backend, client, universe, coin: str, level: str,
                      equity: float, available: float, state: Dict[str, Any]) -> int:
    """Rest a bid at a level's planned entry. One order, then exit.

    Sized from the PLANNED entry and stop, never from the live price -- the
    whole point is that the live price is somewhere else. Everything the
    normal path checks still applies: reward/risk, minimum notional, the
    concurrency cap and the daily cap.
    """
    coin = coin.upper()
    try:
        want = float(level)
    except (TypeError, ValueError):
        print(f"[LEVELS_LIVE] '{level}' is not a level price")
        return 1

    ep = next((e for e in _find_candidates(client, universe)
               if str(e.get("coin", "")).upper() == coin
               and abs(float(e.get("level_price") or 0) - want) < 1e-9), None)
    if ep is None:
        print(f"[LEVELS_LIVE] no current candidate for {coin} at {want:g}")
        return 1

    why = passes_quality(ep, watched=True)
    if why:
        print(f"[LEVELS_LIVE] {coin} {want:g} does not clear the entry bar: {why}")
        return 1
    if coin in state.get("owned", {}) or coin in state.get("pending", {}):
        print(f"[LEVELS_LIVE] already holding or bidding on {coin}")
        return 1
    left = cooldown_remaining(state, coin)
    if left > 0:
        print(f"[LEVELS_LIVE] {coin} is on cooldown for another {left:.1f}h "
              f"after a losing close")
        return 1
    if len(state.get("owned", {})) + len(state.get("pending", {})) >= MAX_CONCURRENT:
        print(f"[LEVELS_LIVE] at the concurrency cap ({MAX_CONCURRENT})")
        return 1
    if MAX_NEW_PER_DAY and state.get("opened_today", 0) >= MAX_NEW_PER_DAY:
        print(f"[LEVELS_LIVE] daily cap reached ({MAX_NEW_PER_DAY})")
        return 1

    entry = float(ep["suggested_entry"]); stop = float(ep["suggested_stop"])
    tp = float(ep["suggested_tp"])
    vmax = None
    try:
        vmax = backend.max_leverage_for(coin)
    except Exception:
        vmax = None
    d = size_from_stop(equity, entry, stop, available, venue_max_leverage=vmax)
    backend.set_leverage(coin, int(d.get("leverage") or FLAT_LEVERAGE))
    if d["size_usd"] < MIN_NOTIONAL_USD:
        print(f"[LEVELS_LIVE] size ${d['size_usd']:.2f} below the "
              f"${MIN_NOTIONAL_USD:g} minimum")
        return 1

    print(f"[LEVELS_LIVE] resting bid {coin} @ {entry:.6g} "
          f"(stop {stop:.6g}, tp {tp:.6g}, rr {float(ep.get('suggested_rr') or 0):.2f}, "
          f"${d['size_usd']:.2f})")
    res = backend.place_limit_entry(coin=coin, side="LONG", price=entry,
                                    size_usd=d["size_usd"])
    if not res.get("oid"):
        print(f"[LEVELS_LIVE] bid REJECTED: {res.get('error')}")
        log_trade({"ts_utc": now_iso(), "action": "entry", "coin": coin,
                   "episode_id": ep.get("episode_id", ""), "entry": entry,
                   "stop": stop, "tp": tp, "result": "bid_rejected",
                   "detail": str(res.get("error"))[:400]})
        return 1

    state.setdefault("pending", {})[coin] = {
        "oid": res["oid"], "price": res["price"], "size_coin": res["size_coin"],
        "stop": stop, "tp": tp, "episode_id": ep.get("episode_id", ""),
        "placed_at": now_iso()}
    save_state(state)
    log_trade({"ts_utc": now_iso(), "action": "entry", "coin": coin,
               "episode_id": ep.get("episode_id", ""), "entry": res["price"],
               "stop": stop, "tp": tp, "size_usd": round(d["size_usd"], 2),
               "size_coin": res["size_coin"], "rr": ep.get("suggested_rr"),
               "result": "bid_resting", "detail": f"oid={res['oid']}"})
    notify.send(
        f"🟡 *BID RESTING {coin}* {res['size_coin']:.6g} @ `{res['price']:.6g}`\n"
        f"stop `{stop:.6g}`   tp `{tp:.6g}`   "
        f"rr {float(ep.get('suggested_rr') or 0):.2f}   ${d['size_usd']:.2f}\n"
        f"protected on fill; pulled after {LIMIT_ENTRY_TTL_HOURS}h if unfilled")
    print(f"[LEVELS_LIVE] bid resting, oid={res['oid']}")
    return 0


def reconcile_pending(backend, state: Dict[str, Any]) -> None:
    """Turn a filled resting bid into a protected position, or retire it.

    A resting bid is the only path here that can put on a position while
    nothing is watching, so this runs FIRST in the pass -- before
    venue_sanity, which would otherwise meet the new position as one this
    module did not open and halt.

    The gap between fill and protection is one poll interval. That is the
    honest cost of resting a bid instead of sending a marketable order, and
    it is why the position is protected here before anything else happens,
    and closed immediately if protection will not attach.
    """
    pending = state.setdefault("pending", {})
    for coin in list(pending):
        p = pending[coin]
        try:
            szi = backend._get_venue_signed_size(coin)
        except Exception:
            continue
        if szi is None:
            continue

        if abs(szi) > 1e-12:
            size_coin = abs(float(szi))
            entry = float(p.get("price") or 0)
            stop, tp = float(p.get("stop") or 0), float(p.get("tp") or 0)
            size_usd = size_coin * entry
            print(f"[LEVELS_LIVE] resting bid FILLED {coin} {size_coin} @ ~{entry:.6g}")
            state.setdefault("owned", {})[coin] = {
                "episode_id": p.get("episode_id", ""), "entry": entry, "stop": stop,
                "tp": tp, "size_usd": size_usd, "size_coin": size_coin,
                "opened_at": p.get("placed_at") or now_iso(), "confirmed": True}
            state["opened_today"] = state.get("opened_today", 0) + 1
            pending.pop(coin, None)
            save_state(state)
            log_trade({"ts_utc": now_iso(), "action": "entry", "coin": coin,
                       "episode_id": p.get("episode_id", ""), "entry": entry,
                       "stop": stop, "tp": tp, "size_usd": round(size_usd, 2),
                       "size_coin": size_coin, "result": "filled",
                       "detail": f"resting bid oid={p.get('oid','')}"})

            prot = backend.place_native_protection(coin=coin, side="LONG",
                                                   size_coin=size_coin,
                                                   stop_price=stop, tp_price=tp)
            log_trade({"ts_utc": now_iso(), "action": "protection", "coin": coin,
                       "episode_id": p.get("episode_id", ""),
                       "result": prot.get("status", ""),
                       "detail": (f"stop={prot.get('stop_order_id','')} "
                                  f"tp={prot.get('tp_order_id','')} "
                                  f"verified={prot.get('stop_verified')}/"
                                  f"{prot.get('tp_verified')} "
                                  f"{prot.get('error','')}")[:400]})
            if prot.get("status") != "protected":
                print(f"[LEVELS_LIVE] !! protection FAILED on {coin} -- closing")
                backend.execute_exit(coin=coin, side="LONG", price=entry,
                                     size_usd=size_usd, reason="protection_failed")
                for oid in (prot.get("stop_order_id"), prot.get("tp_order_id")):
                    if oid:
                        try:
                            backend.cancel_order(coin=coin, oid=oid)
                        except Exception:
                            pass
                note = accrue_close(backend, state, coin, state["owned"].get(coin, {}))
                state["owned"].pop(coin, None)
                save_state(state)
                notify.send(f"⚠️ *{coin}* resting bid filled but protection came back "
                            f"`{prot.get('status','?')}` -- closed immediately\n{note}")
                continue
            notify.send(
                f"🟢 *OPENED {coin}* (resting bid) {size_coin:.6g} @ `{entry:.6g}`\n"
                f"stop `{stop:.6g}`   tp `{tp:.6g}`   size ${size_usd:.2f}")
            continue

        # Still resting. Pull it once the setup is too old to still be the setup.
        placed = iso_to_ms(p.get("placed_at", ""))
        age_h = ((datetime.now(timezone.utc).timestamp() * 1000) - placed) / 3600_000 \
            if placed else 0
        # Say so every pass. A resting bid that is being watched and one that
        # was silently dropped from state look identical in a log that only
        # speaks up when something changes -- and this is the one path that
        # can open a position while nobody is looking.
        try:
            mid = backend.get_mid_price(coin) or 0.0
        except Exception:
            mid = 0.0
        away = ((mid - float(p.get("price") or 0)) / mid * 100) if mid > 0 else 0.0
        print(f"[LEVELS_LIVE] bid resting {coin} @ {float(p.get('price') or 0):.6g}"
              + (f", mid {mid:.6g} ({away:+.2f}% away)" if mid > 0 else "")
              + f", {age_h:.1f}h old of {LIMIT_ENTRY_TTL_HOURS}h")
        if placed and age_h >= LIMIT_ENTRY_TTL_HOURS:
            ok = False
            try:
                ok = backend.cancel_order(coin=coin, oid=p.get("oid", ""))
            except Exception as e:
                print(f"[LEVELS_LIVE] cancel failed {coin}: {str(e)[:80]}")
            print(f"[LEVELS_LIVE] resting bid on {coin} expired after "
                  f"{age_h:.1f}h -- cancel {'ok' if ok else 'FAILED'}")
            log_trade({"ts_utc": now_iso(), "action": "skip", "coin": coin,
                       "episode_id": p.get("episode_id", ""), "result": "bid_expired",
                       "detail": f"unfilled after {age_h:.1f}h, cancel={'ok' if ok else 'failed'}"})
            pending.pop(coin, None)
            save_state(state)


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
        believed = 0.0
        try:
            believed = abs(float(owned[coin].get("size_coin") or 0))
        except (TypeError, ValueError):
            believed = 0.0

        # A partial close is not a close, and it was invisible here. reconcile
        # only ever asked "is it flat?", so a take-profit that filled 114 of
        # 115 PONS on 2026-09-25 left one coin behind: the slot stayed
        # occupied, the recorded size stayed at 115, the protection left
        # resting covered a position that no longer existed, and the $5.04 the
        # trade actually made was never accrued. venue_sanity could not help
        # either -- it looks for positions this module does NOT know about,
        # never for ones whose size has moved.
        if (sz is not None and abs(sz) > 1e-12 and believed > 0
                and abs(sz) < believed * 0.99):
            entry = float(owned[coin].get("entry") or 0)
            residual_usd = abs(sz) * entry
            print(f"[LEVELS_LIVE] {coin} partially closed at venue: holding "
                  f"{abs(sz):g} of {believed:g} ({residual_usd:.2f} USD left)")
            if residual_usd < MIN_NOTIONAL_USD:
                # Dust. Not worth a concurrency slot, and too small to
                # re-protect -- close it and book the trade.
                print(f"[LEVELS_LIVE] {coin} residual is below the "
                      f"${MIN_NOTIONAL_USD:g} minimum -- closing it out")
                try:
                    backend.execute_exit(coin=coin, side="LONG", price=entry,
                                         size_usd=residual_usd,
                                         reason="residual_after_partial_close")
                except Exception as e:
                    print(f"[LEVELS_LIVE] residual exit failed: {str(e)[:100]}")
                for oid in (owned[coin].get("stop_oid"), owned[coin].get("tp_oid")):
                    if oid:
                        try:
                            backend.cancel_order(coin=coin, oid=oid)
                        except Exception:
                            pass
                info = owned.pop(coin, {})
                note = accrue_close(backend, state, coin, info)
                save_state(state)
                log_trade({"ts_utc": now_iso(), "action": "closed", "coin": coin,
                           "episode_id": info.get("episode_id", ""),
                           "result": "flat", "detail": f"partial close, residual "
                           f"{abs(sz):g} closed out; {note}"})
                print(f"[LEVELS_LIVE] {coin} slot released, {note} "
                      f"(day {state['realised_today']:.2f})")
                notify.send(f"⚪ *CLOSED {coin}* (partial fill, residual closed)\n{note}\n"
                            f"day realised ${state.get('realised_today', 0.0):.2f}")
                continue
            # Big enough to keep: correct the record and re-protect the size
            # that is actually there.
            owned[coin]["size_coin"] = abs(sz)
            save_state(state)
            prot = backend.place_native_protection(
                coin=coin, side="LONG", size_coin=abs(sz),
                stop_price=float(owned[coin].get("stop") or 0),
                tp_price=float(owned[coin].get("tp") or 0))
            print(f"[LEVELS_LIVE] {coin} re-protected at {abs(sz):g}: "
                  f"{prot.get('status')}")
            log_trade({"ts_utc": now_iso(), "action": "protection", "coin": coin,
                       "episode_id": owned[coin].get("episode_id", ""),
                       "result": prot.get("status", ""),
                       "detail": f"re-protected after partial close at {abs(sz):g}"})
            continue

        if sz is not None and abs(sz) < 1e-12:
            info = owned.pop(coin, {})
            note = accrue_close(backend, state, coin, info)
            log_trade({"ts_utc": now_iso(), "action": "closed", "coin": coin,
                       "episode_id": info.get("episode_id", ""), "result": "flat",
                       "detail": f"no longer open at venue; {note}"})
            print(f"[LEVELS_LIVE] {coin} closed at venue -- slot released, {note} "
                  f"(day {state['realised_today']:.2f})")
            # Closed by a venue stop, a take-profit, or by hand -- this module
            # only ever learns after the fact, so the alert is how anyone finds
            # out a position is gone and what it cost.
            pnl = state.get("realised_today", 0.0)
            entry = info.get("entry")
            notify.send(
                f"⚪ *CLOSED {coin}*\n{note}\n"
                + (f"entry was `{float(entry):.6g}`, stop `{float(info.get('stop') or 0):.6g}`, "
                   f"tp `{float(info.get('tp') or 0):.6g}`\n" if entry else "")
                + f"day realised ${pnl:.2f}")
    save_state(state)


def preflight(backend, state: Dict[str, Any]):
    """Reasons not to open anything this pass. Returns a string, or None."""
    if state.get("halted"):
        return f"halted: {state.get('halt_reason', '')}"
    owned = state.get("owned", {})
    pending = state.get("pending", {})
    # A resting bid is a position the account has already committed to; it
    # just has not been filled yet. Counting only `owned` would let the bot
    # open its full allowance alongside bids that then fill on top.
    if len(owned) + len(pending) >= MAX_CONCURRENT:
        return (f"holding {len(owned)} and bidding on {len(pending)}, "
                f"max {MAX_CONCURRENT}")
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
        return halt(state, f"daily loss stop: {lost:.2f} against "
                           f"{equity:.2f} equity")
    # Account drawdown from the high-water mark. The daily stop resets at
    # 00:00 UTC, so a slow bleed never trips it -- the three days to
    # 2026-09-30 lost $19.99 across 17 trades without any single day coming
    # close. This is the rail that watches the trajectory instead of the day.
    #
    # The peak is persisted, so it survives restarts; load_state refuses to
    # start on a corrupt file rather than silently handing back an empty one,
    # which is what stops a lost state file from quietly resetting the mark
    # and forgiving a drawdown already incurred.
    peak = float(state.get("equity_peak") or 0.0)
    if equity > peak:
        state["equity_peak"] = equity
        save_state(state)
        peak = equity
    if ACCOUNT_DD_HALT_PCT and peak > 0:
        dd_pct = (peak - equity) / peak * 100.0
        if dd_pct >= ACCOUNT_DD_HALT_PCT:
            return halt(state, f"account drawdown {dd_pct:.1f}% "
                               f"({equity:.2f} against a {peak:.2f} peak), "
                               f"limit {ACCOUNT_DD_HALT_PCT:g}%")
    return None


if __name__ == "__main__":
    raise SystemExit(main())

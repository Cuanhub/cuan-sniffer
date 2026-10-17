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
from core import execute, protect, safety, streak                     # noqa: E402
from core.meta import MetaStore, TradeMeta                            # noqa: E402
from core.venue import Venue                                          # noqa: E402
from research.levels import storage                                   # noqa: E402
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
    MIN_RR, MIN_NOTIONAL_USD, POLL_SEC,
    TRADE_MIN_NEAR_SIGMA, TRADE_MIN_BARS_SINCE_TOUCH, TRADE_MIN_OVERHEAD_SIGMA,
    MAX_ENTRY_DRIFT_PCT, WATCH_TTL_HOURS, NEAR_ALERT_SIGMA, FLAT_LEVERAGE,
    ENTRY_MAX_BREAK_FRAC, ENTRY_MAX_BASIS_BPS, ENTRY_MAX_PULL_OVER_RUN,
    COIN_REENTRY_COOLDOWN_HOURS,
    ENTRY_SPACING_MINUTES,
    MARGIN_PCT_PER_TRADE,
    NEAR_ALERT_COOLDOWN_SEC, TRIGGER_MAX_ABOVE_SIGMA, LIMIT_ENTRY_TTL_HOURS,
    TRIGGER_HOLD_HOURS, TRIGGER_HOLD_HOURS_SWING,
    FLIP_CAPPED_BARS_MIN, FLIP_CAPPED_RECENT_BARS,
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


# Keys from designs that have been replaced. Dropped on load, because the
# state dict is loaded once and re-saved whole: editing the FILE while the
# trader is up is futile, since the running process still holds the old keys
# in memory and writes them back on its next save. That is exactly what
# happened on 2026-10-17 -- these three were deleted from the file in the
# morning and were back within a pass.
#
# day_open_equity / _equity_for_day_open*: the per-day drawdown reference,
# replaced by the rolling equity_samples window.
RETIRED_STATE_KEYS = {
    "day_open_equity",
    "_equity_for_day_open",
    "_equity_for_day_open_day",
}


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
        for dead in sorted(k for k in RETIRED_STATE_KEYS if k in parsed):
            parsed.pop(dead, None)
            print(f"[LEVELS_LIVE] dropped retired state key {dead!r}")
        return parsed
    # Strategy state only. What the account HOLDS is not in here: that is
    # read from the venue, and the halt and the loss limits live in
    # core.safety. See core.meta for why this file can no longer express a
    # position -- it is REVIEW.md findings 5, 6, 8, 9 and 11 in one line.
    return {"watching": {}, "opened_today": 0, "day": ""}


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


# One list, read by log_trade and by the startup migration. New columns go on
# the END: widen_header_if_needed migrates positionally and refuses a header
# that is not a prefix of this, rather than guessing.
TRADE_LOG_COLS = ["ts_utc", "action", "coin", "episode_id", "level_price",
                  "entry", "stop", "tp", "size_usd", "size_coin", "risk_usd",
                  "rr", "hold_rate", "touches", "result", "detail",
                  "pull_over_run"]


def migrate_trade_log() -> None:
    """Bring live_trades.csv up to TRADE_LOG_COLS. Called ONCE, at startup.

    Not from log_trade: widen_header_if_needed reads the whole file before it
    can tell whether anything needs doing, the trade log is 54MB and 429k rows
    at 0.43s a read, and log_trade fires hundreds of times a pass for skips.
    That would have cost minutes per pass to discover there was nothing to do.
    """
    try:
        if storage.widen_header_if_needed(TRADE_LOG, TRADE_LOG_COLS):
            print(f"[LEVELS_LIVE] trade log widened to "
                  f"{len(TRADE_LOG_COLS)} columns")
    except Exception as e:
        # A failed migration must not stop the trader starting. log_trade
        # writes the wider row either way; the worst case is a trailing
        # column the old header does not name, which readers tolerate.
        print(f"[LEVELS_LIVE] trade log migration skipped: {str(e)[:160]}")


def log_trade(row: Dict[str, Any]) -> None:
    import csv
    TRADE_LOG.parent.mkdir(parents=True, exist_ok=True)
    cols = TRADE_LOG_COLS
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


def size_from_stop(equity: float, entry: float, stop: float,
                   available: Optional[float] = None,
                   venue_max_leverage: Optional[float] = None
                   ) -> Dict[str, float]:
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


def spacing_remaining(state: Dict[str, Any]) -> float:
    """Minutes left before another position may be opened, 0.0 when free.

    Applies across coins, unlike cooldown_remaining. The open loop fills every
    free slot in one pass, so without this four triggers in a single scan
    became four positions in the same minute -- and in a universe correlating
    0.47 pairwise, that is one bet sized four times. See
    ENTRY_SPACING_MINUTES for the numbers.
    """
    if not ENTRY_SPACING_MINUTES:
        return 0.0
    at = iso_to_ms(state.get("last_entry_at", ""))
    if at is None:
        return 0.0
    mins = (datetime.now(timezone.utc).timestamp() * 1000 - at) / 60_000.0
    return max(0.0, ENTRY_SPACING_MINUTES - mins)


def mark_entry(state: Dict[str, Any]) -> None:
    """Stamp the clock the spacing rule reads. Called on every fill, including
    a resting bid's -- a bid filling still adds correlated exposure at a moment
    nobody chose, even though it cannot be refused at that point."""
    state["last_entry_at"] = now_iso()


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
                  tf_cache: Optional[Dict[Any, Any]] = None,
                  ctx_out: Optional[Dict[str, Any]] = None
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
    # get_perp_contexts rather than get_all_mids: it carries mark AND oracle
    # alongside the mid, so the basis veto costs no extra request. Still one
    # call per pass.
    mids: Dict[str, float] = {}
    try:
        cx = client.get_perp_contexts()
        if ctx_out is not None:
            ctx_out.update(cx)
        mids = {k: v["mid_px"] for k, v in cx.items() if v.get("mid_px")}
    except Exception as e:
        # No mids means no safe reuse of the 4h series: fall through with an
        # empty cache reference so every series is refetched, which is the old
        # behaviour rather than a stale one. The basis veto then has nothing to
        # read and abstains, which is the right direction for a veto.
        print(f"[LEVELS_LIVE] context fetch failed ({str(e)[:80]}) -- "
              f"refetching candle series this pass, basis veto inactive")
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


def describe(ep: Dict[str, Any], equity: float,
             available: Optional[float] = None) -> Dict[str, Any]:
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
    ap.add_argument("--clear-standdown", action="store_true",
                    help="lift the loss-streak stand-down and reset the run")
    ap.add_argument("--seed-watchlist", action="store_true",
                    help="admit levels the trade log shows already cleared the "
                         "quality bar, then exit. Run once after the watchlist "
                         "is introduced; it is empty otherwise.")
    args = ap.parse_args()

    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)

    state = load_state()
    migrate_trade_log()
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
    halt_state = safety.HaltState()
    eqlog = safety.EquityLog()
    if args.resume:
        was = halt_state.active()
        halt_state.clear()
        print(f"[LEVELS_LIVE] halt cleared"
              + (f" (was: {was.get('reason')})" if was else " (none was active)"))
        return 0
    if args.clear_standdown:
        # One timestamp, not a flag: it silences the run that caused the
        # stand-down without disabling the rail, so a fresh loss afterwards
        # starts a new run. See core.streak.
        safety.StandDownOverride().clear_now()
        print("[LEVELS_LIVE] stand-down cleared for the current losing run")
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
          f"lev cap {MAX_LEVERAGE}x, daily stop {safety.DAILY_LOSS_PCT:g}%, "
          f"drawdown halt {safety.DRAWDOWN_PCT:g}%")
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
            f"daily stop {safety.DAILY_LOSS_PCT:g}%, "
            f"drawdown halt {safety.DRAWDOWN_PCT:g}%",
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

    # One Venue, built whether or not we are armed, so the banner equity and
    # the equity a pass sizes against come from the same place. They used to
    # come from two different code paths and could disagree.
    venue = None
    try:
        venue = start_venue(Venue)
    except Exception as e:
        if armed:
            raise SystemExit(
                f"[LEVELS_LIVE] !! venue unavailable ({str(e)[:120]}) and this "
                f"process is ARMED. Refusing to start.")
        print(f"[LEVELS_LIVE] venue unavailable ({str(e)[:90]}); dry run")
    if venue is None and armed and not _stop:
        return 0                      # stopped while waiting to retry

    equity = args.equity
    available = None
    if equity is None and venue is not None:
        eq = venue.equity()
        if eq is not None:
            equity, available = eq.usdc_total, eq.available
            print(f"[LEVELS_LIVE] equity ${equity:.2f}   "
                  f"available ${available:.2f}"
                  + (f"   (incl. unrealised {eq.unrealized_pnl:+.2f})"
                     if eq.unrealized_pnl else ""))
        elif armed:
            raise SystemExit("[LEVELS_LIVE] !! equity query failed and this "
                             "process is ARMED. Refusing to start on a "
                             "guessed balance.")
    if equity is None:
        equity = 100.0
        print(f"[LEVELS_LIVE] no equity reading; dry run, assuming "
              f"${equity:.2f} to print sizes")
    _margin = equity * MARGIN_PCT_PER_TRADE / 100.0
    print(f"[LEVELS_LIVE] per trade: margin ${_margin:.2f}, "
          f"notional ${_margin * FLAT_LEVERAGE:.2f}, "
          f"risk at a median 1.76% stop ${_margin * 0.088:.2f} "
          f"({_margin * 0.088 / equity * 100:.2f}% of equity)")

    meta = MetaStore()
    if armed and venue is not None:
        # Before the first pass, make sure everything the venue holds has a
        # stop -- including anything this module did not open, which is what
        # a timeout-after-fill leaves behind. The old startup sweep halted on
        # such a position; this one protects it and carries on.
        for r in protect.run_once(venue, meta):
            print(f"[LEVELS_LIVE] startup protection {r.coin}: {r.action}")

    if args.limit_entry:
        if not armed or venue is None:
            print("[LEVELS_LIVE] --limit-entry places a REAL order and needs "
                  "--arm plus LEVELS_LIVE_ARMED=yes")
            return 1
        coin, level = args.limit_entry[0].upper(), args.limit_entry[1]
        return rest_one_bid(venue, meta, ro, universe, coin, level,
                            equity, available, state)

    while not _stop:
        # Before anything else in the pass, so a queued halt takes effect on
        # this pass rather than the next one.
        drain_control_queue(state, venue, halt_state)
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
        if venue is not None:
            # Protection first, every pass. There is nothing to "reconcile":
            # the venue is read directly, so a bid that filled is simply a
            # position this cycle finds and stops. The old pair of functions
            # existed to keep a state file in step with the venue, and that
            # step is what findings 2, 5, 6, 8, 9 and 11 all lived in.
            for r in protect.run_once(venue, meta):
                if not r.protected:
                    print(f"[LEVELS_LIVE] protection {r.coin}: {r.action} "
                          f"-- {r.detail[:100]}")
            for coin in persist_filled_bids(venue, meta):
                print(f"[LEVELS_LIVE] {coin}: resting bid filled -- episode "
                      f"persisted")

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

            why = preflight(venue, state, halt_state, eqlog)
            if why:
                print(f"[LEVELS_LIVE] not opening this pass: {why}")
                if args.once:
                    return 0
                for _ in range(POLL_SEC):
                    if _stop:
                        break
                    time.sleep(1)
                continue
            eq = venue.equity()
            if eq is not None:
                equity, available = eq.usdc_total, eq.available
        # Created here, not after the scan, so build_intents can hand the 1h
        # bars it already fetched straight into it. Still one dict per pass:
        # the trigger asks what price did in the last two hours, so bars must
        # never carry across passes.
        bars_cache: Dict[str, Any] = {}
        # Filled by build_intents from the one contexts call it already makes,
        # then read by the basis veto at entry. No extra request.
        ctx_cache: Dict[str, Any] = {}
        intents, rejected = build_intents(ro, universe, watching,
                                          bars_out=bars_cache,
                                          tf_cache=tf_cache,
                                          ctx_out=ctx_cache)
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
            if venue is None or not armed:
                log_trade({"ts_utc": now_iso(), "action": "dry_run", "result": "planned",
                           **{k: d[k] for k in ("coin", "episode_id", "level_price", "entry",
                                                "stop", "tp", "rr", "hold_rate", "touches")},
                           "size_usd": round(d["size_usd"], 2),
                           "risk_usd": round(d["risk_usd"], 2),
                           "detail": ",".join(flags)})
                continue
            # Re-read from the VENUE, not from a counter. This is the guard
            # the 2026-09-18 runaway defeated: it asked state["owned"], one
            # parsing bug left that empty, and a single pass opened twelve
            # positions across $5,454 of notional on $775 of equity with no
            # stops on any of them. A reading of the venue cannot be emptied
            # by a bug in our own parsing, because there is no parsing.
            held = venue.positions()
            if held is None:
                print("[LEVELS_LIVE] venue unreadable mid-pass -- stopping here")
                break
            resting = venue.open_orders()
            bids = {o.coin for o in (resting or []) if o.is_entry}
            if len(held) + len(bids) >= MAX_CONCURRENT:
                print(f"[LEVELS_LIVE] concurrency cap reached "
                      f"({len(held)} held, {len(bids)} bid), stopping this pass")
                break
            if d["coin"] in held or d["coin"] in bids:
                continue
            # The daily cap belongs in this guard too, not only in preflight.
            # preflight runs ONCE per pass, so on 2026-09-28 it saw
            # opened_today=4, allowed the pass, and BTC then filled (5) with
            # DOGE filling six seconds later (6) -- one over the cap.
            if MAX_NEW_PER_DAY and state.get("opened_today", 0) >= MAX_NEW_PER_DAY:
                print(f"[LEVELS_LIVE] daily cap reached "
                      f"({state['opened_today']}/{MAX_NEW_PER_DAY}), "
                      f"stopping this pass")
                break
            # Spacing belongs in this guard above all: the loop filling every
            # free slot in one pass is exactly what it exists to stop, and
            # checking only inside execute_one would write a skip row for
            # every remaining candidate instead of ending the pass.
            _gap = spacing_remaining(state)
            if _gap > 0:
                print(f"[LEVELS_LIVE] entry spacing: {_gap:.0f}m until another "
                      f"position may open, stopping this pass")
                break
            outcome = execute_one(venue, meta, ep, equity, available, armed,
                                  state, ctx_cache=ctx_cache,
                                  bars_cache=bars_cache, watchboard=watchboard)
            # Protect what just opened, immediately -- not on the next 60s
            # cycle. An entry and its stop belong together, and the gap
            # between them is the only window in which this system holds an
            # unprotected position.
            if not outcome.order_sent:
                continue
            for r in protect.run_once(venue, meta):
                if not r.protected:
                    print(f"[LEVELS_LIVE] !! post-entry protection {r.coin}: "
                          f"{r.action} -- {r.detail[:120]}")
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


def preflight(venue, state: Dict[str, Any], halt: "safety.HaltState",
              eqlog: "safety.EquityLog"):
    """Reasons not to open anything this pass. Returns a string, or None.

    Everything it consults is either the venue or a derived rail. Nothing is
    read from a counter this process maintains, which is what the 1,043 lines
    this replaces got wrong.
    """
    active = halt.active()
    if active:
        return f"halted ({active.get('kind')}): {active.get('reason', '')}"

    # Equity-based limits. They flatten and halt on a breach themselves.
    verdicts = safety.enforce_limits(venue, halt, eqlog)
    if any(v.breached for v in verdicts):
        return "; ".join(v.detail for v in verdicts if v.breached)

    sd = safety.standing_down(venue)
    if sd is not None:
        return f"standing down {sd.hours_left:.1f}h more -- {sd.reason}"

    # Concurrency from the VENUE, plus anything resting that could become a
    # position. state["owned"] used to answer this and a single parsing bug
    # emptied it, which is how one pass opened twelve positions.
    positions = venue.positions()
    if positions is None:
        safety.record_error("positions", "preflight could not read the venue")
        return "venue unreadable -- not opening blind"
    orders = venue.open_orders()
    bidding = {o.coin for o in (orders or []) if o.is_entry}
    if len(positions) + len(bidding) >= MAX_CONCURRENT:
        return (f"holding {len(positions)} and bidding on {len(bidding)}, "
                f"max {MAX_CONCURRENT}")

    if MAX_NEW_PER_DAY and state.get("opened_today", 0) >= MAX_NEW_PER_DAY:
        return f"opened {state['opened_today']} today, max {MAX_NEW_PER_DAY}"

    eq = venue.equity()
    if eq is None:
        safety.record_error("equity", "preflight could not read equity")
        return "equity query failed -- refusing to size blind"
    if eq.usdc_total <= 0:
        return f"equity {eq.usdc_total}"
    return None


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
    if behaves_like_flip(ep) and str(ep.get("level_origin", "")) != "flipped_resistance":
        out.append(f"was resistance until recently -- capped price {int(cap)}/120 bars, "
                   f"last {int(since)} bars ago; price is still flipping it, so it "
                   f"must clear the {TRIGGER_HOLD_HOURS}h channel hold")
    return out


def behaves_like_flip(ep: Dict[str, Any]) -> bool:
    """True when a level is still acting as resistance, however it was labelled.

    level_origin marks only what the resistance detector happened to find, so a
    level can be scored swing_low while plainly still capping price -- and a
    swing_low is asked for the easy 1h hold rather than the 2h channel hold a
    flip has to prove. ARB 0.186655 was bought on 2026-10-13 having capped
    price for 66 of the last 120 bars, most recently 6 bars earlier, on a 1h
    hold. SOL 101.72 was the same shape at 95 of 120 bars, last 12 bars ago.

    Used by review_flags AND by the hold decision, from one definition, so the
    advisory line and the rule can never disagree about what a flip is.
    """
    def f(k, d=0.0):
        try:
            return float(ep.get(k) or d)
        except (TypeError, ValueError):
            return d
    if str(ep.get("level_origin", "")) == "flipped_resistance":
        return True
    since = f("bars_since_capped", -1)
    return f("capped_bars") >= FLIP_CAPPED_BARS_MIN and 0 <= since <= FLIP_CAPPED_RECENT_BARS


def persist_traded_episode(ep: Dict[str, Any]) -> bool:
    """Write the episode we actually traded, with its positioning context.

    Without this the research loop cannot reach the trades. The trader computes
    a full episode through find_candidates, acts on it, and discarded it: on
    2026-10-07, 0 of 39 traded episode ids existed in episodes.csv, so none of
    the 654 rows of open interest, funding and book depth could be joined to a
    single real outcome. The two paths also mint incompatible ids -- the
    scanner uses make_episode_id(coin, ...) and this path coin + "-" + tf --
    so they could never have matched even if both had been written.

    All eight fields resolve_outcomes needs are already on the row (coin,
    bar_ts_ms, the three suggested prices and sigma_4h), so a persisted episode
    is picked up by the scanner's resolver on its own and acquires an outcome
    about seven days later. Nothing else has to change for the join to work.

    Context is fetched HERE rather than per pass: it costs one request and only
    on the passes that actually trade, so a quiet pass stays exactly as cheap
    as it was. Returns whether a row was written; never raises, because losing
    a research row must not disturb a position that is already open.
    """
    try:
        row = {k: v for k, v in ep.items() if not str(k).startswith("_")}
        coin = str(row.get("coin", "")).upper()
        try:
            cx = (_ro_client().get_perp_contexts() or {}).get(coin) or {}
        except Exception as e:
            print(f"[LEVELS_LIVE] context unavailable for the traded episode "
                  f"({str(e)[:60]}) -- persisting without it")
            cx = {}
        if cx:
            row["ctx_captured_at_utc"] = storage.now_iso()
            for k in ("oi_base", "oi_notional_usd", "funding_rate", "premium",
                      "mark_px", "oracle_px", "impact_bid", "impact_ask",
                      "impact_spread_bps", "day_ntl_vlm", "prev_day_px"):
                v = cx.get(k)
                row[k] = "" if v is None else v
        return bool(storage.append_episode(row))
    except Exception as e:
        safety.record_error("persist_episode", e)
        return False


def pull_over_run(ep: Dict[str, Any]) -> Optional[float]:
    """How much of the run that created the level has been given back.

    pull_sigma / run_sigma, both already measured by the scanner, so this
    costs nothing at entry. Returns None when either is missing or the run is
    non-positive -- a veto that cannot read its input must not invent a
    refusal.
    """
    def f(k):
        try:
            v = float(ep.get(k))
        except (TypeError, ValueError):
            return None
        return v
    run, pull = f("run_sigma"), f("pull_sigma")
    if run is None or pull is None or run <= 0:
        return None
    return pull / run


def entry_is_actionable(venue, coin: str, planned_entry: float, stop: float,
                        tp: float, level: float = 0.0,
                        bars_1h: Optional[List[Dict[str, Any]]] = None,
                        ctx: Optional[Dict[str, Any]] = None,
                        ep: Optional[Dict[str, Any]] = None):
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
    ref = venue.mid(coin)
    if not ref or ref <= 0:
        return None, "no midprice"
    drift = (ref - planned_entry) / planned_entry * 100
    if abs(drift) > MAX_ENTRY_DRIFT_PCT:
        return None, (f"price {ref:.6g} is {drift:+.2f}% from the planned entry "
                      f"{planned_entry:.6g} (max {MAX_ENTRY_DRIFT_PCT}%)")
    if ref <= stop:
        return None, f"price {ref:.6g} is already at or below the stop {stop:.6g}"
    # The same rule, asked of the BAR rather than of the mid. The check above
    # only sees where price is this instant; a dip through the stop that
    # bounced back before the pass looked at it passes it untouched. That is
    # how eleven entries were taken after the hour had already traded past the
    # stop -- not one of them won.
    #
    # level_has_held cannot catch this: it judges CLOSED bars and excludes the
    # forming one by design, so the level can be breaking as we buy it. The
    # operator read exactly this off the STRK chart as "no double bottom had
    # been created yet".
    if ENTRY_MAX_BREAK_FRAC and level > stop > 0 and bars_1h:
        low = min((b.get("l", 0.0) for b in bars_1h[-1:]), default=0.0)
        if low > 0:
            broken = (level - low) / (level - stop)
            if broken > ENTRY_MAX_BREAK_FRAC:
                return None, (f"this hour has already traded {broken*100:.0f}% of the "
                              f"way from the level {level:.6g} to the stop "
                              f"{stop:.6g} (low {low:.6g}, max "
                              f"{ENTRY_MAX_BREAK_FRAC*100:.0f}%)")
    # Perp richness to spot. Mark above oracle means leveraged longs are paying
    # a premium to be here, and buying support into that is buying a crowded
    # trade. Over 248 resolved episodes, perp-above-oracle averaged -0.524R
    # against +0.044R at or below (p = 0.0035) -- the first feature in this
    # project to reach significance, where all nineteen price features failed.
    # Abstains when the reading is missing: a veto that cannot read its input
    # must not invent a refusal.
    if ENTRY_MAX_BASIS_BPS >= 0 and ctx:
        mark, oracle = ctx.get("mark_px"), ctx.get("oracle_px")
        if mark and oracle and oracle > 0:
            basis = (mark / oracle - 1) * 10_000
            if basis > ENTRY_MAX_BASIS_BPS:
                return None, (f"perp is {basis:+.1f}bps rich to oracle "
                              f"(mark {mark:.6g} vs {oracle:.6g}, max "
                              f"{ENTRY_MAX_BASIS_BPS:+.1f}) -- longs are paying up")
    # How much of the run that created this level has already been given back.
    # PULL_SIGMA_MIN demands a minimum pullback and never capped it against
    # the run, so a level tested after surrendering most of its run passed
    # every gate. Across 492 episodes the probability of reaching target falls
    # from 42.9% to 9.1% as this ratio rises, and the stop rate climbs from
    # 76.5% to 94.9% -- holding the day and the reward/risk fixed.
    #
    # Abstains when either leg is missing, for the same reason the basis veto
    # does: an unreadable input is not evidence of a bad trade.
    if ENTRY_MAX_PULL_OVER_RUN > 0 and ep is not None:
        por = pull_over_run(ep)
        if por is not None and por > ENTRY_MAX_PULL_OVER_RUN:
            return None, (f"pullback is {por:.2f} of the run that made the "
                          f"level (max {ENTRY_MAX_PULL_OVER_RUN:.2f}) -- the "
                          f"run is gone, this is distribution not a dip")
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


def start_venue(factory, sleep=time.sleep):
    """Build the Venue, retrying a transient venue failure.

    The SDK fetches perp metadata inside Info.__init__, so construction hits
    the network and a 429 there raises before any of our code runs. Twice on
    2026-09-19 that killed the process outright at startup. launchd restarts
    it, so it self-heals -- but a crash loop against a rate limit is the worst
    possible way to wait, because every restart is another request, which is
    what caused the limit.

    Returns None if a stop was requested while waiting. Re-raises if the venue
    is still refusing after every attempt: coming up without a venue would
    mean an armed trader that cannot trade and does not say so.
    """
    for attempt in range(1, BACKEND_START_ATTEMPTS + 1):
        try:
            return factory(debug=False)
        except Exception as e:
            if attempt == BACKEND_START_ATTEMPTS:
                print(f"[LEVELS_LIVE] venue failed to start after "
                      f"{attempt} attempts: {str(e)[:160]}")
                raise
            wait = BACKEND_START_BACKOFF_SEC * attempt
            print(f"[LEVELS_LIVE] venue start attempt {attempt} failed "
                  f"({str(e)[:80]}); retrying in {wait}s")
            for _ in range(int(wait)):
                if _stop:
                    return None
                sleep(1)
    return None


def persist_filled_bids(venue, meta: "MetaStore") -> List[str]:
    """Persist the episode behind any resting bid that has since filled.

    This is the "on fill, not on placement" half of the rule above. There is
    no fill event to hook -- the venue is polled -- so the trigger is simply
    observing a position for a coin whose plan still carries an unpersisted
    episode. Clearing the field afterwards is what makes it once-only.
    """
    held = venue.positions() if venue is not None else None
    if not held:
        return []
    done = []
    for coin in sorted(held):
        row = meta.get(coin)
        if row is None or not row.pending_episode:
            continue
        try:
            ep = json.loads(row.pending_episode)
        except ValueError:
            ep = None
        if isinstance(ep, dict) and persist_traded_episode(ep):
            done.append(coin)
        row.pending_episode = ""
        meta.put(coin, row)
    return done


def rest_one_bid(venue, meta, client, universe, coin: str, level: str,
                 equity: float, available: float,
                 state: Optional[Dict[str, Any]] = None) -> int:
    """Rest a GTC bid at one level's planned entry, then exit.

    The plan goes into metadata and the order goes to the venue. Nothing else
    is recorded: the bid is visible in open_orders, carries the venue's own
    timestamp for expiry, and the protection loop stops whatever it fills.
    The old version kept an oid and a plan in state["pending"], and a partial
    fill popped that record while the remainder stayed live -- REVIEW.md
    finding 6.
    """
    cands, _ = build_intents(client, [coin], None)
    want = f"{float(level):.10g}"
    ep = next((e for e in cands
               if f"{float(e.get('level_price') or 0):.10g}" == want), None)
    if ep is None:
        print(f"[LEVELS_LIVE] no live candidate for {coin} at {level} -- "
              f"the scanner has to see the level before a bid can rest on it")
        return 1

    def g(k, d=0.0):
        try:
            return float(ep.get(k) or d)
        except (TypeError, ValueError):
            return d

    gap = spacing_remaining(state) if state is not None else 0.0
    if gap > 0:
        # A bid commits future exposure even though it cannot be refused once
        # it fills, so the spacing rail applies at placement.
        print(f"[LEVELS_LIVE] entry spacing: {gap:.0f}m until another "
              f"position may be committed")
        return 1

    entry, stop, tp = g("suggested_entry"), g("suggested_stop"), g("suggested_tp")
    vmax = venue.max_leverage_for(coin)
    szi = size_from_stop(equity, entry, stop, available, venue_max_leverage=vmax)
    size_usd = szi["size_usd"]
    if size_usd < MIN_NOTIONAL_USD:
        print(f"[LEVELS_LIVE] {coin} sizes to ${size_usd:.2f}, below the "
              f"${MIN_NOTIONAL_USD:g} minimum")
        return 1

    venue.set_leverage(coin, int(szi.get("leverage") or FLAT_LEVERAGE))
    # The episode is carried, not persisted: a bid that is never hit is not
    # a trade, and writing its features now would put setups we did not take
    # into the set used to judge the ones we did.
    meta.put(coin, TradeMeta(reason=f"resting bid at {entry:.6g}",
                             planned_stop=stop, planned_tp=tp,
                             opened_at=now_iso(),
                             pending_episode=json.dumps(ep)))
    order = execute.rest_bid(venue, coin, usd=size_usd, px=entry)
    if order is not None and state is not None:
        # Stamped at placement, not on fill: the commitment is made here, and
        # a fill can happen while nothing is watching.
        mark_entry(state)
        save_state(state)
    if order is None:
        meta.drop(coin)
        print(f"[LEVELS_LIVE] {coin}: no bid rested")
        return 1
    log_trade({"ts_utc": now_iso(), "action": "entry", "coin": coin,
               "episode_id": ep.get("episode_id", ""), "entry": order.limit_px,
               "stop": stop, "tp": tp, "size_usd": round(size_usd, 2),
               "size_coin": order.size, "result": "bid_resting",
               "detail": f"GTC oid={order.oid}, expires in "
                         f"{protect.BID_TTL_HOURS:g}h"})
    notify.send(f"🔵 *Bid resting on {coin}*\n{order.size:g} @ "
                f"`{order.limit_px:.6g}`\nstop `{stop:.6g}`  tp `{tp:.6g}`\n"
                f"Pulled after {protect.BID_TTL_HOURS:g}h if it never fills.")
    print(f"[LEVELS_LIVE] bid resting {coin} {order.size:g} @ "
          f"{order.limit_px:.6g} (oid {order.oid})")
    return 0


def execute_one(venue, meta, ep: Dict[str, Any], equity: float,
                available: float, armed: bool, state: Dict[str, Any],
                bars_cache: Optional[Dict[str, Any]] = None,
                watchboard: Optional[List[Dict[str, Any]]] = None,
                ctx_cache: Optional[Dict[str, Any]] = None) -> "Outcome":
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
        # Asks how the level BEHAVES, not only how it was labelled. A level
        # still capping price gets the channel hold even when level_origin
        # called it a swing low.
        is_flip = behaves_like_flip(ep)
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
        ref, why = entry_is_actionable(venue, coin, entry, stop, tp,
                                       level=g("level_price"),
                                       bars_1h=bars1h if armed else None,
                                       ctx=(ctx_cache or {}).get(coin.upper()),
                                       ep=ep)
        if ref is None:
            log_trade({"ts_utc": now_iso(), "action": "skip", "coin": coin,
                       "episode_id": ep.get("episode_id", ""), "entry": entry,
                       "stop": stop, "tp": tp, "result": "not_actionable",
                       "detail": why, "pull_over_run": pull_over_run(ep)})
            print(f"[LEVELS_LIVE] skip {coin}: {why}")
            return Outcome()
        entry = ref

    vmax = venue.max_leverage_for(coin) if venue is not None else None
    szi = size_from_stop(equity, entry, stop, available, venue_max_leverage=vmax)
    size_usd = szi["size_usd"]
    row = {"ts_utc": now_iso(), "coin": coin, "episode_id": ep.get("episode_id", ""),
           "pull_over_run": pull_over_run(ep),
           "level_price": g("level_price"), "entry": entry, "stop": stop, "tp": tp,
           "size_usd": round(size_usd, 2), "risk_usd": round(szi["risk_usd"], 2),
           "rr": g("suggested_rr"), "hold_rate": g("level_hold_rate"),
           "touches": int(g("level_touches"))}

    if size_usd < MIN_NOTIONAL_USD:
        row.update({"action": "skip", "result": "below_min_notional"})
        log_trade(row)
        return Outcome()

    left = streak.coins_on_cooldown(venue, COIN_REENTRY_COOLDOWN_HOURS
                                    ).get(coin, 0.0) if venue else 0.0
    if left > 0:
        row.update({"action": "skip", "result": "coin_cooldown",
                    "detail": f"stopped out {COIN_REENTRY_COOLDOWN_HOURS - left:.1f}h "
                              f"ago; {left:.1f}h of cooldown left"})
        log_trade(row)
        return Outcome()
    gap = spacing_remaining(state)
    if gap > 0:
        row.update({"action": "skip", "result": "entry_spacing",
                    "detail": f"last entry {ENTRY_SPACING_MINUTES - gap:.0f}m ago; "
                              f"{gap:.0f}m of spacing left"})
        log_trade(row)
        return Outcome()
    if not armed:
        row.update({"action": "dry_run", "result": "not_armed"})
        log_trade(row)
        return Outcome()

    # Record the PLAN before sending, then send, then look at the venue.
    #
    # There is no slot to claim any more. The old code claimed one in
    # state["owned"] because the concurrency cap was counted from that dict,
    # and on 2026-09-18 a parsing bug left the dict empty and one pass opened
    # twelve positions across $5,454 of notional on $775 of equity with no
    # stops. The cap is now counted from venue.positions(), which no parsing
    # bug of ours can empty -- so what gets written here is only the plan, and
    # losing it costs a planned stop, not a tracked position.
    meta.put(coin, TradeMeta(reason=f"level {g('level_price'):.6g}",
                             planned_stop=stop, planned_tp=tp,
                             opened_at=now_iso()))
    mark_entry(state)
    save_state(state)
    # Persisted on the way IN, not on close: the features describe the setup as
    # it was when we decided, and by the time it closes the bars have moved.
    persist_traded_episode(ep)

    # Hyperliquid enforces leverage per asset, so a 5x-sized order into a coin
    # the account has set to 3x is refused for margin. Set it to match before
    # the order, capped at whatever the venue allows on this coin.
    venue.set_leverage(coin, int(szi.get("leverage") or FLAT_LEVERAGE))

    out = execute.enter(venue, coin, usd=size_usd, is_long=True)

    if out.unknown:
        # An order went out and the venue cannot be read. This is neither a
        # fill nor a reject, and the old code had no third answer: it recorded
        # a reject and released the slot, leaving a live unprotected position
        # untracked for up to a full poll interval. The plan stays written, so
        # the next protection cycle -- 60s away, not 450s -- finds whatever is
        # there and stops it at the price we planned.
        row.update({"action": "entry", "result": "unknown",
                    "detail": "order sent, venue unreadable afterwards; the "
                              "protection loop will adopt whatever filled"})
        log_trade(row)
        notify.send(f"⚠️ *{coin}: order sent, result unknown*\nThe venue could "
                    f"not be read after sending. The protection loop will "
                    f"stop whatever filled at `{stop:.6g}` within a minute.",
                    key=f"unknown_fill_{coin}", cooldown_sec=300)
        return Outcome(order_sent=True)

    if not out.filled:
        meta.drop(coin)
        row.update({"action": "entry", "result": "rejected",
                    "detail": out.note[:400]})
        log_trade(row)
        return Outcome(order_sent=True)

    fill_px, size_coin = out.avg_px, out.filled_size
    state["opened_today"] = state.get("opened_today", 0) + 1
    save_state(state)
    row.update({"action": "entry", "result": "filled", "size_coin": size_coin,
                "detail": f"fill {fill_px:.8g}"})
    log_trade(row)

    # Protection is not optional, and it is one call now: the loop sizes the
    # stop to what the venue actually holds, so a partial fill needs no
    # special case here. The old path placed protection for the size it
    # believed it had filled and had no way to correct it afterwards.
    results = protect.run_once(venue, meta)
    mine = next((r for r in results if r.coin == coin), None)
    log_trade({**row, "action": "protection",
               "result": mine.action if mine else "no_result",
               "detail": (mine.detail if mine else "position not found")[:400]})
    if mine is None or not mine.protected:
        # protect.run_once already flattened it if no stop would rest -- an
        # unprotected position is not the trade. Say so and stop; there is no
        # slot to release and nothing left to unwind.
        print(f"[LEVELS_LIVE] !! {coin} could not be protected "
              f"({mine.action if mine else 'not found'})")
        notify.send(
            f"⚠️ *{coin}* opened but could not be protected\n"
            f"`{(mine.detail if mine else 'position not found')[:300]}`\n"
            f"The protection loop closes what it cannot stop.")
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
    # No day-open equity is recorded here any more. The drawdown reference is
    # a rolling ACCOUNT_DD_WINDOW_HOURS window sampled every pass by
    # record_equity_sample, so there is nothing for a date boundary to do but
    # reset the daily counters.
    state.update({"day": today, "opened_today": 0})
    save_state(state)
    print(f"[LEVELS_LIVE] new UTC day {today} (was {was}) -- "
          f"daily counters reset")
    return True


CONTROL_QUEUE = STATE_PATH.parent / "_control_queue.json"


def drain_control_queue(state: Dict[str, Any], venue,
                        halt_state: "safety.HaltState") -> List[str]:
    """Apply queued Telegram commands. Returns what was done, for the log.

    The control process never writes _live_state.json -- it appends here and
    this is the only thing that applies it, so the state file keeps exactly
    one writer. The queue is REPLACED with [] rather than edited, so a
    command cannot be applied twice even if the controller appends mid-drain:
    the worst case is a command arriving a pass late.

    Anything unrecognised is dropped with a note rather than retried, because
    a command this build does not understand will not become understood by
    being tried again on the next pass.
    """
    try:
        raw = json.loads(CONTROL_QUEUE.read_text())
    except (OSError, ValueError):
        return []
    if not isinstance(raw, list) or not raw:
        return []
    try:
        CONTROL_QUEUE.write_text("[]")
    except OSError as e:
        # Could not clear it, so do NOT act: acting without clearing would
        # replay every command on every pass.
        print(f"[LEVELS_LIVE] control queue unclearable ({e}) -- ignoring it")
        return []

    # Collapse repeats before acting. A real drain found four identical
    # `halt` commands waiting, and each one ran its own flatten_all -- four
    # venue sweeps for one intent. Commands are idempotent by nature here
    # (halt, resume, pause), so the last of each verb is the one that counts.
    collapsed: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    skipped = 0
    for item in raw:
        if not isinstance(item, dict):
            continue
        verb = str(item.get("cmd") or "").lower()
        if verb in collapsed:
            skipped += 1
        else:
            order.append(verb)
        collapsed[verb] = item
    if skipped:
        print(f"[LEVELS_LIVE] control: collapsed {skipped} repeated "
              f"command(s); acting on {order}")

    done: List[str] = []
    for verb in order:
        item = collapsed[verb]
        cmd = verb
        arg = str(item.get("arg") or "")
        if cmd == "halt":
            # The venue is passed in, so this halt can verify and flatten.
            # Every queued halt used to run with backend=None, which made its
            # protection check a silent no-op -- REVIEW.md finding 4, on the
            # one path whose whole job is reacting to an unprotected position.
            if venue is None:
                done.append("halt requested but no venue -- NOT applied")
                continue
            safety.halt(venue, halt_state,
                        f"halted from Telegram: {arg}" if arg
                        else "halted from Telegram", kind=safety.MANUAL)
            done.append("halted by request")
        elif cmd == "resume":
            halt_state.clear()
            keys = state.get("alert_keys")
            if isinstance(keys, dict):
                keys.pop("halt", None)
            done.append("halt cleared by request")
        elif cmd == "pause":
            try:
                hours = float(arg)
            except ValueError:
                done.append(f"ignored pause '{arg}': not a number")
                continue
            if hours <= 0:
                # `pause 0` means "stop pausing". One verb for the dial, with
                # zero as the obvious way to turn it off.
                safety.StandDownOverride().clear_now()
                done.append("stand-down cleared for the current losing run")
                continue
            # A timed pause is a halt that clears itself, which is exactly
            # what AUTO means -- so it is one, rather than a third state with
            # its own timestamp and its own way of getting stuck.
            if venue is None:
                done.append("pause requested but no venue -- NOT applied")
                continue
            safety.halt(venue, halt_state,
                        f"paused {hours:g}h from Telegram", kind=safety.AUTO,
                        flatten=False)
            done.append(f"paused by request (clears on the next UTC day)")
        else:
            done.append(f"ignored unknown command '{cmd}'")
    if done:
        save_state(state)
        for line in done:
            print(f"[LEVELS_LIVE] control: {line}")
    return done


if __name__ == "__main__":
    raise SystemExit(main())

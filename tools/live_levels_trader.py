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
from typing import Any, Dict, List, Optional, Set, Tuple

from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)

import notify                                                        # noqa: E402
from core import books, config_audit, execute, protect, safety, streak  # noqa: E402
from core.meta import MetaStore, TradeMeta                            # noqa: E402
from core.venue import Venue                                          # noqa: E402
from research.levels import storage                                   # noqa: E402
from research.levels.venue_client import HyperliquidReadOnlyClient    # noqa: E402
from research.levels.universe import select_universe                 # noqa: E402
from research.levels.trigger import fetch_1h                          # noqa: E402
from research.levels import zone_pass                                 # noqa: E402
from research.levels import zone_log                                  # noqa: E402
from research.levels import zones                                     # noqa: E402

# Rails come from research/levels/config.py -- one definition each, tagged with
# whether they rest on a measurement or on my guess.
from research.levels.config import (                                  # noqa: E402
    MAX_CONCURRENT, MAX_NEW_PER_DAY, MAX_LEVERAGE,
    MIN_RR, MIN_NOTIONAL_USD, POLL_SEC,
    MAX_ENTRY_DRIFT_PCT, FLAT_LEVERAGE,
    ENTRY_MAX_BREAK_FRAC, ENTRY_MAX_BASIS_BPS, ENTRY_MAX_PULL_OVER_RUN,
    COIN_REENTRY_COOLDOWN_HOURS,
    ENTRY_SPACING_MINUTES,
    MARGIN_PCT_PER_TRADE,
    LIMIT_ENTRY_TTL_HOURS,
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
# replaced by a rolling window.
#
# owned / pending: what the account holds. Read from the venue now. Leaving
# them in the file is actively misleading -- the whole point is that this
# file cannot express a position -- so they are stripped on load.
#
# halted / halt_reason / halt_kind: core.safety.HaltState owns the halt. A
# stale `halted: true` in here looks like a live halt and is not one.
#
# realised_today / equity_samples / equity_peak / loss_streak: the loss
# limits read equity and the losing run is derived from fills, so none of
# these counters exists any more.
RETIRED_STATE_KEYS = {
    # The watchlist went with the level scan. A band is either in reach this
    # pass or it is not, and a band price has not reached yet has a RESTING
    # BID on it rather than a watchlist entry -- the venue holds that, with
    # its own timestamp and its own expiry, so there is nothing for this file
    # to remember. Retired rather than merely unused so an old state file does
    # not carry 976 stale entries forward in silence.
    "watching",
    "day_open_equity",
    "_equity_for_day_open",
    "_equity_for_day_open_day",
    "owned",
    "pending",
    "halted",
    "halt_reason",
    "halt_kind",
    "realised_today",
    "equity_samples",
    "equity_peak",
    "loss_streak",
    "streak_cooldown_until",
    "forced",
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
    return {"opened_today": 0, "day": ""}


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
    args = ap.parse_args()

    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)

    state = load_state()
    migrate_trade_log()
    roll_day(state)
    if args.status:
        print(json.dumps(state, indent=2))
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
    # A setting that looks live and does nothing is a control the operator
    # will one day reach for and find is not there. Refuse to trade while
    # .env describes one; DRY still protects positions and expires bids.
    why_not = config_audit.refusal(config_audit.dead_keys())
    if why_not:
        print(f"[LEVELS_LIVE] !! {why_not}")
        if armed:
            armed = False
            print("[LEVELS_LIVE] !! refusing to ARM until .env matches the code")
            notify.send(f"⛔ *Trader NOT armed*\n{why_not}\n\nRunning DRY: positions "
                        f"stay protected and bids still expire; nothing new opens.",
                        key="dead_config", cooldown_sec=3600)
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
    universe = start_universe(ro)
    if universe is None:
        return 1

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
    # Risk per trade is NOT a constant any more, so this no longer pretends
    # it is. It used to read "risk at a median 1.76% stop $3.10", with both
    # the 1.76% and the 0.088 that derived it hardcoded -- and when the stop
    # became max(0.25%, ZONE_STOP_ATR_MULT x ATR) the median moved to about
    # 3.37% while the line kept reporting the old figure. A number the
    # operator reads to know what the bot is risking has to come from the
    # same place the bot gets it, or not be printed.
    _notional = _margin * FLAT_LEVERAGE
    print(f"[LEVELS_LIVE] per trade: margin ${_margin:.2f}, "
          f"notional ${_notional:.2f}; risk = notional x the stop distance, "
          f"which is max({zones.ZONE_STOP_BUFFER_PCT}%, "
          f"{zones.ZONE_STOP_ATR_MULT:g}x ATR) under the band low -- so "
          f"${_notional * 0.02:.2f} at a 2% stop, "
          f"${_notional * 0.04:.2f} at 4%")

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

    last_record = 0.0                 # when bands were last recorded
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
            # Armed only. A dry run reads the same real fills, so its
            # alerts would be truthful -- but it shares this state file, and
            # advancing the watermark there would silence the armed process
            # on the next real close.
            if armed:
                announce_closes(venue, state)

            # Bookkeeping runs BEFORE the preflight early-return, not after it.
            # preflight continues the loop whenever it refuses a pass, so with
            # Same shape as the day rollover, the asset-meta load and the
            # universe refresh: something that must happen every cycle, placed
            # where it only happens on some.
            on_cooldown = prune_cooldowns(state)
            if on_cooldown:
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
                if should_record_while_blocked(why, last_record, time.time()):
                    scan_and_record(ro, universe, venue)
                    last_record = time.time()
                # Self-heal. Refusing to open blind is correct and it is what
                # the line above does, but refusing for ever is not a
                # recovery: on 2026-10-19 this process lost DNS after 25
                # hours and sat doing exactly that for half an hour while
                # `dig` resolved the host instantly. A dead resolver does not
                # come back by waiting, and a fresh process gets a new one.
                #
                # The plist has KeepAlive SuccessfulExit=false, so a NON-ZERO
                # exit is what restarts it, after ThrottleInterval (60s).
                # Nothing is dropped by going: the state file is on disk and
                # every stop is resting at the venue.
                dead = safety.give_up_if_unreadable("Trader")
                if dead:
                    print(f"[LEVELS_LIVE] exiting so launchd can restart "
                          f"me: {dead}")
                    return 1
                for _ in range(POLL_SEC):
                    if _stop:
                        break
                    time.sleep(1)
                continue
            eq = venue.equity()
            if eq is not None:
                equity, available = eq.usdc_total, eq.available
        intents, rejected, found, bars_cache, ctx_cache, mids = scan_and_record(
            ro, universe, venue)
        last_record = time.time()
        save_state(state)
        # The candidate table, every pass, before anything is sent. The
        # operator asked for it and the reason is in the measurement: no
        # selection rule reproduced his own band choice better than 9 of 12,
        # so showing only the winner would hide what it was chosen over.
        print()
        print(zone_pass.render_candidates(found, MIN_RR))
        n_now = sum(1 for e in intents if e.get("zone_immediate"))
        print(f"\n[LEVELS_LIVE] {len(intents)} band(s) tradeable "
              f"({n_now} with price in the band, {len(intents) - n_now} "
              f"waiting on a bid), {len(rejected)} rejected")
        for c, lp, why in rejected[:8]:
            print(f"    reject {c:10} band_low={lp} -- {why}")
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
            # The trend book's coin while it is armed (core/books.py). One
            # net position per coin per account: a band entry here would net
            # against the trend book's position, not sit beside it.
            if d["coin"] in books.band_excludes():
                continue
            # Re-read from the VENUE, not from a counter. This is the guard
            # the 2026-09-18 runaway defeated: it asked state["owned"], one
            # parsing bug left that empty, and a single pass opened twelve
            # positions across $5,454 of notional on $775 of equity with no
            # stops on any of them. A reading of the venue cannot be emptied
            # by a bug in our own parsing, because there is no parsing.
            slots = slots_in_use(venue)
            if slots is None:
                print("[LEVELS_LIVE] venue unreadable mid-pass -- stopping here")
                break
            held, bids = slots
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
            # In the band, or waiting for it. Two different orders, and the
            # difference is the whole strategy: price inside the band is the
            # setup and there is nothing to wait for, while price above the
            # band gets a bid AT the band rather than a market order here.
            #
            # That second branch is what the 2026-10-17 failure needed. ARB
            # was bought at market 0.18482 while its band was 0.1705-0.175;
            # a bid at the band price cannot fill above the band.
            if ep.get("zone_immediate"):
                outcome = execute_one(venue, meta, ep, equity, available,
                                      armed, state, ctx_cache=ctx_cache,
                                      bars_cache=bars_cache,
                                      watchboard=watchboard)
            else:
                rc = place_resting_bid(venue, meta, ep, equity, available,
                                       state)
                if rc == BID_UNKNOWN:
                    # Whether it rested is unknown, so the slot count is too.
                    break
                # A bid is a commitment, not a fill. Nothing to protect yet --
                # the protection loop stops whatever it fills, and expire_bids
                # pulls it after BID_TTL_HOURS if it never does.
                if rc == 0:
                    continue
                outcome = Outcome()
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
        # The near-level digest is gone with the watchlist. A band price has
        # not reached yet is not a near miss waiting to be announced -- it has
        # a RESTING BID on it, and that bid sends its own alert when it is
        # placed and its own fill when it is hit. Announcing the same band a
        # second time as "approaching" was one message per pass for a thing
        # already committed to.
        if args.once:
            return 0
        for _ in range(POLL_SEC):
            if _stop:
                break
            time.sleep(1)
    return 0


# Bands change only when an hour closes, so while the pass is blocked from
# opening anything, recording once an hour sees every band there is to see.
RECORD_EVERY_SEC = 3600


def should_record_while_blocked(why: str, last_record: float, now: float) -> bool:
    """Record bands on a blocked pass? Hourly, and never while the venue is
    unreadable -- scanning 45 coins then only adds load to a rate limit.

    Before 2026-10-27 a blocked pass recorded nothing, and the pass is
    blocked most of the time (348 of the last 379, mostly a full book). So
    the band record sampled about 8% of the hours, and only the hours the
    bot had room -- which bands got recorded depended on the bot's own state.
    """
    if "unreadable" in why:
        return False
    return now - last_record >= RECORD_EVERY_SEC


def scan_and_record(ro, universe, venue):
    """Plan every coin's bands and record every band seen, taken or refused.

    Returns (intents, rejected, found, bars_cache, ctx_cache, mids). Records
    only: it decides nothing and sends nothing, and zone_log swallows its own
    failures -- a recorder that can break the pass is worse than none.

    Refusals are recorded too, because "why are we losing" needs the
    counterfactual, and nothing else here can say whether a refusal was right.
    """
    # The 1h bars the scan already fetched, handed straight on so the entry
    # checks make no second request.
    bars_cache: Dict[str, Any] = {}
    # Read by the basis veto at entry. One contexts call per scan.
    ctx_cache: Dict[str, Any] = {}
    mids: Dict[str, float] = {}
    try:
        cx = ro.get_perp_contexts()
        ctx_cache.update(cx)
        mids = {k: v["mid_px"] for k, v in cx.items() if v.get("mid_px")}
    except Exception as e:
        # A veto that cannot read its input must abstain, not refuse, and the
        # scan falls back to each series' own last close for price.
        print(f"[LEVELS_LIVE] context fetch failed ({str(e)[:80]}) -- "
              f"basis veto inactive this pass")
    intents, rejected, found = zone_pass.plans_by_coin(
        ro, universe, min_rr=MIN_RR, bars_out=bars_cache, mids=mids)
    _rec = zone_log.record_pass(
        found, MIN_RR, bars_cache, mids=mids, universe=universe,
        leverage={c: lv for c in found
                  if (lv := venue.max_leverage_for(c)) is not None}
        if venue is not None else None,
        ctx=ctx_cache)
    if _rec:
        print(f"[ZONE_LOG] recorded {_rec} new band episode(s)")
    return intents, rejected, found, bars_cache, ctx_cache, mids


def slots_in_use(venue) -> Optional[Tuple[Dict[str, Any], Set[str]]]:
    """(positions, coins with a resting entry bid), or None if EITHER could
    not be read.

    Both are needed to count slots, and a book that cannot be read is not an
    empty one. On 2026-11-03, in a burst of 429s, the open-orders read failed
    mid-pass, `(resting or [])` counted that as no bids, and the pass rested a
    fifth commitment against MAX_CONCURRENT 4.
    """
    positions = venue.positions()
    if positions is None:
        return None
    orders = venue.open_orders()
    if orders is None:
        return None
    # The trend book's coins are not this book's slots (core/books.py).
    other = books.band_excludes()
    return ({c: p for c, p in positions.items() if c not in other},
            {o.coin for o in orders if o.is_entry and o.coin not in other})


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
    # The real outcome into the readability watchdog. None could not be read;
    # an empty dict is a flat account, which read fine.
    safety.note_read(positions is not None)
    if positions is None:
        safety.record_error("positions", "preflight could not read the venue")
        return "venue unreadable -- not opening blind"
    orders = venue.open_orders()
    if orders is None:
        # Not "no bids". See slots_in_use.
        safety.record_error("open_orders", "preflight could not read the book")
        return "open orders unreadable -- not opening blind"
    # The trend book's ten positions are not this book's slots; counted, they
    # would hold MAX_CONCURRENT full for as long as it is armed.
    other = books.band_excludes()
    positions = {c: p for c, p in positions.items() if c not in other}
    bidding = {o.coin for o in orders if o.is_entry and o.coin not in other}
    if len(positions) + len(bidding) >= MAX_CONCURRENT:
        return (f"holding {len(positions)} and bidding on {len(bidding)}, "
                f"max {MAX_CONCURRENT}")

    if MAX_NEW_PER_DAY and state.get("opened_today", 0) >= MAX_NEW_PER_DAY:
        return f"opened {state['opened_today']} today, max {MAX_NEW_PER_DAY}"

    eq = venue.equity()
    safety.note_read(eq is not None)
    if eq is None:
        safety.record_error("equity", "preflight could not read equity")
        return "equity query failed -- refusing to size blind"
    if eq.usdc_total <= 0:
        return f"equity {eq.usdc_total}"
    return None


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


def start_universe(ro, sleep=time.sleep) -> Optional[List[Dict[str, Any]]]:
    """The universe at startup, retrying a transient venue failure.

    This call was unguarded, and it is the first thing after the venue comes
    up. During the 2026-10-22 outage it raised VenueFetchError straight out of
    main() 58 times -- a raw traceback to stderr and a dead process each time,
    restarted by launchd at its 60s ThrottleInterval. 77KB of identical
    tracebacks for "DNS is down".

    The in-loop refresh has always been guarded and falls back to the previous
    universe. At startup there IS no previous one, so the choice is retry or
    exit -- and exiting on the first failure is the crash loop. Same five
    attempts and linear backoff as start_venue, for the same reason its
    docstring gives: every restart is another request.

    Returns None when a stop was requested, or after every attempt failed. The
    caller exits non-zero, which is what launchd restarts on, so a genuinely
    long outage still gets a fresh process -- just every ~3 minutes rather
    than every 60 seconds, and without a traceback.
    """
    for attempt in range(1, BACKEND_START_ATTEMPTS + 1):
        try:
            u = select_universe(ro)
            if u:
                return u
            last = "the venue returned an empty universe"
        except Exception as e:                                 # noqa: BLE001
            last = str(e)[:160]
        if attempt == BACKEND_START_ATTEMPTS:
            print(f"[LEVELS_LIVE] universe failed to load after {attempt} "
                  f"attempts: {last} -- exiting for launchd to retry")
            return None
        wait = BACKEND_START_BACKOFF_SEC * attempt
        print(f"[LEVELS_LIVE] universe load attempt {attempt} failed "
              f"({last[:80]}); retrying in {wait}s")
        for _ in range(int(wait)):
            if _stop:
                return None
            sleep(1)
    return None


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
        # ALERT HERE. A resting bid filling is a position opening, and it was
        # silent: the "OPENED" notification lives in execute_one, which is the
        # IMMEDIATE-entry path, and the band strategy almost never takes that
        # one -- on a live 45-coin pass 19 of 21 plans were bids waiting. So
        # every position this system has actually opened arrived without a
        # message. XRP 135 @ 1.2967 was found open by hand.
        #
        # Once-only for the same reason the persist is: pending_episode is
        # cleared below, and this is the only place that clears it.
        _announce_fill(venue, coin, held.get(coin), row)
        row.pending_episode = ""
        meta.put(coin, row)
    return done


def _announce_fill(venue, coin: str, pos, plan) -> None:
    """Tell the operator a bid became a position, with what it risks.

    Everything needed to check it against a chart goes in the message,
    because the alternative is opening a laptop to find out what was bought.
    Reads the STOP AND TARGET FROM THE VENUE rather than from the plan: the
    plan says what was intended, and the only thing worth alerting on is what
    is actually resting.
    """
    if pos is None:
        return
    try:
        orders = venue.open_orders(coin) if venue is not None else None
        # None means the book could not be read. An unread book is not an
        # empty one: calling a protected position unprotected because of a
        # 429 is the mistake that cost NEAR $5.44 in protect.py, and an
        # alert that cries wolf gets ignored exactly when it matters.
        readable = orders is not None
        stops = [o for o in (orders or []) if o.is_protective and o.is_stop]
        tps = [o for o in (orders or []) if o.is_protective and not o.is_stop]
        stop_px = stops[0].trigger_px if stops else (plan.planned_stop or 0.0)
        tp_px = tps[0].trigger_px if tps else (plan.planned_tp or 0.0)
        entry, size = pos.entry_px, pos.size
        risk = abs(entry - stop_px) * size if stop_px else 0.0
        reward = abs(tp_px - entry) * size if tp_px else 0.0
        lines = [f"🟢 *FILLED {coin}* {size:g} @ `{entry:.6g}`",
                 f"notional ${entry * size:,.2f}   "
                 f"now `{pos.mark_px:.6g}` ({pos.unrealized_pnl:+,.2f})"]
        if stop_px:
            lines.append(f"stop `{stop_px:.6g}`  "
                         f"({(entry - stop_px) / entry * 100:.2f}% away, "
                         f"risk ${risk:.2f})"
                         + ("" if stops or not readable
                            else "  ⚠️ PLANNED ONLY, not resting"))
        else:
            lines.append("🚨 *NO STOP* -- the protection loop should place one")
        if tp_px:
            lines.append(f"tp   `{tp_px:.6g}`  "
                         f"({(tp_px - entry) / entry * 100:.2f}% away, "
                         f"target ${reward:.2f})"
                         + ("" if tps or not readable else "  ⚠️ planned only"))
        if risk:
            lines.append(f"rr {reward / risk:.2f}")
        notify.send("\n".join(lines), key=f"filled_{coin}", cooldown_sec=60)
    except Exception as exc:                                   # noqa: BLE001
        # An alert that can break the pass is worse than a missing alert.
        print(f"[LEVELS_LIVE] fill alert for {coin} failed: {str(exc)[:120]}")


def announce_closes(venue, state: Dict[str, Any]) -> int:
    """Tell the operator when a trade ENDS, not only when it starts.

    The other half of the silence the operator found. A stop or a target
    fills at the venue; no code of ours runs. protect notices the position
    is gone, prints "is gone at the venue -- metadata dropped" into a log
    nobody reads, and sends nothing. So the run of seven consecutive
    stop-outs that cost $20.31 happened entirely in the dark, and the only
    way to learn a trade had ended was to notice the equity line move.

    The result comes from `core.streak`, which groups the venue's own fills
    into round trips: net of fees, computed fresh every time, never a number
    this process incremented and believed. A trade whose opening fill is
    outside the lookback window is skipped by that grouping -- its entry
    fees are missing, so its result would be wrong -- and skipping it is
    better than announcing a figure that is off by an unknown amount.

    The watermark is the last close already announced. On a first run there
    is none, and this announces NOTHING: it stamps the watermark at the
    newest close it can see and starts from the next one. Otherwise the
    first armed pass after a deploy would fire two weeks of history at the
    operator's phone, and an alert channel that does that once gets muted.
    """
    since = int(time.time() * 1000) - int(streak.LOOKBACK_HOURS * 3_600_000)
    fills = venue.fills(since)
    if fills is None:
        return 0                       # unreadable: say nothing, keep the mark
    # The trend book's round trips are its own to report (core/books.py).
    other = books.band_excludes()
    trades = [t for t in streak.trades_from_fills(fills) if t.coin not in other]
    if not trades:
        return 0
    mark = int(state.get("announced_close_ms") or 0)
    if not mark:
        state["announced_close_ms"] = max(t.closed_ms for t in trades)
        save_state(state)
        print(f"[LEVELS_LIVE] close alerts start from now "
              f"({len(trades)} earlier trade(s) left unannounced)")
        return 0

    fresh = [t for t in trades if t.closed_ms > mark]
    for t in fresh:
        _announce_close(t)
    if fresh:
        state["announced_close_ms"] = max(t.closed_ms for t in fresh)
        save_state(state)
    return len(fresh)


def _announce_close(t: "streak.Trade") -> None:
    """One finished round trip, as the venue accounts for it.

    Deliberately does NOT claim which order closed it. A loss is almost
    always the stop and a win almost always the target, but a hand `/close`
    reads as either, and a message that names the wrong cause is worse than
    one that names none.
    """
    try:
        word = "won" if not t.lost else ("lost" if t.pnl < 0 else "scratched")
        log_trade({"ts_utc": now_iso(), "action": "closed", "coin": t.coin,
                   "result": word,
                   "detail": f"net {t.pnl:+.4f}, {t.fills} fill(s), "
                             f"closed_ms {t.closed_ms}"})
        held_h = max(0.0, (t.closed_ms - t.opened_ms) / 3_600_000)
        head = "🎯" if not t.lost else "🔴"
        notify.send(
            f"{head} *{t.coin} closed* -- {word} *${t.pnl:+,.2f}* net of fees\n"
            f"held {held_h:.1f}h, {t.fills} fill(s)",
            key=f"closed_{t.coin}_{t.closed_ms}", cooldown_sec=0)
    except Exception as exc:                                   # noqa: BLE001
        print(f"[LEVELS_LIVE] close alert for {t.coin} failed: {str(exc)[:120]}")


def rest_one_bid(venue, meta, client, universe, coin: str, level: str,
                 equity: float, available: float,
                 state: Optional[Dict[str, Any]] = None) -> int:
    """`--limit-entry COIN LEVEL`: rest a bid on one band, by hand.

    Finds the band whose LOW is `level` among the coin's live plans, then
    hands it to place_resting_bid -- the same function the pass uses, so a
    bid placed by hand and a bid placed by the loop are the same order with
    the same metadata and the same expiry.
    """
    if coin.upper() in books.band_excludes():
        print(f"[LEVELS_LIVE] {coin} belongs to the trend book while it is "
              f"armed -- no band bid (core/books.py)")
        return 1
    intents, _, _ = zone_pass.plans_by_coin(client, [{"symbol": coin}],
                                            min_rr=MIN_RR)
    want = f"{float(level):.10g}"
    ep = next((e for e in intents
               if f"{float(e.get('level_price') or 0):.10g}" == want), None)
    if ep is None:
        have = ", ".join(f"{float(e['level_price']):.6g}" for e in intents)
        print(f"[LEVELS_LIVE] no live band on {coin} with low {level}"
              + (f" -- bands in reach: {have}" if have else
                 " -- no band in reach with a target"))
        return 1
    return place_resting_bid(venue, meta, ep, equity, available, state)


# place_resting_bid sent the bid and could not learn whether it rested: the
# caller must stop the pass, because the slot count is unknown too.
BID_UNKNOWN = 2


def place_resting_bid(venue, meta, ep: Dict[str, Any], equity: float,
                      available: float,
                      state: Optional[Dict[str, Any]] = None) -> int:
    """Rest a GTC bid at this plan's entry, then exit. 0 on success.

    The plan goes into metadata and the order goes to the venue. Nothing else
    is recorded: the bid is visible in open_orders, carries the venue's own
    timestamp for expiry, and the protection loop stops whatever it fills.
    The old version kept an oid and a plan in state["pending"], and a partial
    fill popped that record while the remainder stayed live -- REVIEW.md
    finding 6.

    This is the normal path now, not the exception. A band price has not
    reached yet is the common case -- on a live 45-coin scan 21 of 23 plans
    were bids waiting -- and a bid at the band price is also what makes the
    2026-10-17 failure impossible: the order's price IS a band price, so it
    cannot fill above the band the way a market order did.
    """
    coin = str(ep["coin"])

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
    unknown = False
    if order is None:
        # rest_bid answers None both when nothing rested and when it SENT the
        # order but could not read the book back. Only the first may drop the
        # plan. On 2026-11-03 an ETH bid rested in a burst of 429s while this
        # logged "no bid rested" and dropped its stop and target, so a fill
        # would have been adopted with the 3% default stop and no target.
        book = venue.open_orders(coin)
        if book is None:
            unknown = True
            print(f"[LEVELS_LIVE] {coin}: bid sent, book unreadable -- keeping "
                  f"its plan and stopping this pass")
        else:
            order = next((o for o in book if o.is_entry), None)
            if order is not None:
                print(f"[LEVELS_LIVE] {coin}: bid confirmed on a second read")
    if (order is not None or unknown) and state is not None:
        # Stamped at placement, not on fill: the commitment is made here, and
        # a fill can happen while nothing is watching. A bid that may have
        # rested is a commitment too.
        mark_entry(state)
        save_state(state)
    if unknown:
        return BID_UNKNOWN
    if order is None:
        # The plan stays: this read can lag the order too. The protection loop
        # forgets it after its grace period if nothing ever appears.
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


def _plan_from(ep: Dict[str, Any]) -> "zones.Plan":
    """Rebuild the Plan from the episode dict, for the zone review flags.

    The episode is what crosses the boundary into execution -- it is what the
    log and the meta store hold, and it survives a restart where an in-memory
    Plan would not. Carrying the object alongside it would give two sources of
    truth for the same trade, which is the class of bug this rebuild removed.
    """
    def f(k, d=0.0):
        try:
            return float(ep.get(k) or d)
        except (TypeError, ValueError):
            return d
    coin = str(ep.get("coin") or "")
    band = zones.Zone(coin=coin, lo=f("zone_lo"), hi=f("zone_hi"),
                      base_i=int(f("zone_base_i")), bos_i=int(f("zone_base_i")))
    target = zones.Zone(coin=coin, lo=f("target_lo"), hi=f("target_hi"),
                        base_i=0, bos_i=0, kind=zones.SUPPLY)
    return zones.Plan(zone=band, target=target, entry=f("suggested_entry"),
                      stop=f("suggested_stop"), tp=f("suggested_tp"),
                      immediate=bool(ep.get("zone_immediate")))


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
    # A band carries its own trigger: price being INSIDE it is the setup, and
    # there is nothing left to wait for. The level trigger that used to stand
    # here asked a different question -- has price tapped this price and then
    # held above it for some hours -- which a band answers by construction,
    # because an entry inside the band is only reachable when price is in the
    # band. It is deleted rather than bypassed; with the pass trading bands it
    # had no reachable caller, and dead code on the entry path is the kind
    # that gets re-enabled by accident.
    #
    # The checks below are NOT skipped. entry_is_actionable is generic and it
    # is the one that matters most here: it re-derives the geometry from the
    # live price and refuses an entry that has drifted off the plan, which is
    # the guard the UNI fill at 8.8125 against a planned 7.50 defeated.
    bars1h: Optional[List[Dict[str, Any]]] = None
    if armed:
        bars1h = (bars_cache or {}).get(coin)
        for flag in zone_pass.review_flags(
                _plan_from(ep),
                size_from_stop(equity, entry, stop, available,
                               venue_max_leverage=None)["size_usd"],
                equity, bars=bars1h):
            print(f"[LEVELS_LIVE] [REVIEW] {coin}: {flag}")
            log_trade({"ts_utc": now_iso(), "action": "review", "coin": coin,
                       "episode_id": ep.get("episode_id", ""),
                       "level_price": g("level_price"), "result": "flagged",
                       "detail": flag})
        print(f"[LEVELS_LIVE] IN BAND {coin} "
              f"{g('zone_lo'):.6g}-{g('zone_hi'):.6g} -- entry {entry:.6g}")

    # Size from the LIVE price, not the planned one. Sizing off the plan while
    # filling at market is exactly what made the UNI position carry 3.5x its
    # intended risk.
    if armed:
        ref, why = entry_is_actionable(venue, coin, entry, stop, tp,
                                       level=g("level_price"),
                                       bars_1h=bars1h,
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
        # The plan stays, for the protection loop to forget after its grace
        # period. A "no fill" can be a read that lags the fill: ENA on
        # 2026-11-05 lost its plan this way and was adopted at the 3% default.
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


# How old a queued command may be and still be applied.
#
# The controller already refuses to ENQUEUE a replayed Telegram update, but
# that cannot help a command which is already in the queue -- from before
# that check existed, or from while the trader was down. On 2026-10-18 a
# `halt` enqueued at 00:14:19Z was still waiting to fire hours later, which
# is the same hazard arriving by a different door.
#
# Deliberately more generous than the controller's own window: a queued
# command is one the operator has already been told applies "on the trader's
# next pass", and a pass is 450s. This only has to be shorter than the gap
# that makes a command no longer mean what it said.
CONTROL_COMMAND_MAX_AGE_SEC = 1800


def _command_age_sec(item: Dict[str, Any]) -> float:
    """How long ago a queued command was written. 0.0 if it does not say.

    An entry with no `at` is obeyed. The field was added after the queue
    format was, and refusing to act on a command because it predates the
    timestamp would be a worse failure than the one this guards against.
    """
    raw = str(item.get("at") or "")
    if not raw:
        return 0.0
    try:
        when = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return max(0.0, (datetime.now(timezone.utc) - when).total_seconds())


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
    stale = 0
    for item in raw:
        if not isinstance(item, dict):
            continue
        if _command_age_sec(item) > CONTROL_COMMAND_MAX_AGE_SEC:
            stale += 1
            continue
        verb = str(item.get("cmd") or "").lower()
        if verb in collapsed:
            skipped += 1
        else:
            order.append(verb)
        collapsed[verb] = item
    if stale:
        print(f"[LEVELS_LIVE] control: dropped {stale} command(s) older "
              f"than {CONTROL_COMMAND_MAX_AGE_SEC // 60} min")
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
            # flatten=False, because this halt must mean what the
            # operator was told it means. `/halt`'s own reply says "blocks
            # new entries only -- open positions keep their stops. Use
            # /closeall to flatten", and HELP says "stop opening anything".
            #
            # It flattened anyway, and the regression is traceable: every
            # queued halt used to run with backend=None (REVIEW.md finding
            # 4), so safety.halt's flatten was a silent no-op and the
            # promise held by accident. Passing the venue in to fix the
            # protection check ARMED a flatten that had never once fired --
            # and on 2026-10-17 it market-closed a live ARB position off a
            # /halt Telegram had REPLAYED from an unacknowledged backlog.
            #
            # `/pause` already got this right one branch below. Flattening
            # stays where it is asked for by name: /closeall, /close, and
            # the equity rails in core.safety, whose halts DO flatten
            # because exposure is the thing they are breached on.
            safety.halt(venue, halt_state,
                        f"halted from Telegram: {arg}" if arg
                        else "halted from Telegram", kind=safety.MANUAL,
                        flatten=False)
            done.append("halted by request (positions keep their stops)")
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

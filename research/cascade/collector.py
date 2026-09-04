"""
Cascade-fade forward collector -- runs the same fixed detection rule as
backtest_cascade.py, forward, in real time, alongside the funding
collector. Purpose: history for 1m candles on this venue only goes back
~3.5 days (see DECISIONS.md), which made the historical backtest
inconclusive either way (n=3 at 1m, n=21 at 5m with a strong single-month
concentration). Collection only -- this does not place, size, or
recommend trades. It logs what a mechanical fade would have done, so
that in a few weeks there's a real, unseen forward sample to check
instead of another thin historical slice.

Universe: reuses research.funding.universe.select_universe (same top-30-
by-volume set the funding collector already tracks) via a read-only
import of that one pure function -- no shared state, no risk to the
already-running funding collector.

Stateless-per-cycle design: each ~60s cycle, fetch a fresh ~70-minute 1m
candle window per symbol (rather than maintaining an in-memory rolling
buffer) -- simpler and safe across restarts, at the cost of one extra
small API call per symbol per minute, which is cheap at this universe
size.

Two-phase logging (events now, resolutions once known), matching this
codebase's existing shadow_research_candidates/outcomes pattern. Pending
events survive a restart via a small persisted state file (same pattern
as perp_sentiment.py's OI history persistence).

Usage:
    python -m research.cascade.collector
    python -m research.cascade.collector --dry-run
"""

import argparse
import json
import signal
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from research.cascade import storage
from research.cascade.detector import detect_cascade, try_fill_and_resolve, VOL_LOOKBACK_BARS, FILL_WINDOW_BARS, TIME_STOP_BARS
from research.cascade.venue_client import HyperliquidReadOnlyClient, VenueFetchError
from research.funding.universe import select_universe

POLL_INTERVAL_SEC = 60
CANDLE_WARMUP_MINUTES = VOL_LOOKBACK_BARS + 5
RESOLUTION_CHECK_MARGIN_MINUTES = FILL_WINDOW_BARS + TIME_STOP_BARS + 2
FAILURE_LOG_PATH = Path(__file__).resolve().parent / "data" / "_fetch_failures.log"
PENDING_STATE_PATH = Path(__file__).resolve().parent / "data" / "_pending_state.json"

_stop_requested = False


def _handle_sigterm(signum, frame):
    global _stop_requested
    print("[CASCADE_COLLECTOR] SIGTERM received -- finishing current cycle, then exiting")
    _stop_requested = True


def _log_failure(endpoint: str, error: str, attempt: int) -> None:
    line = f"{storage.now_iso()} endpoint={endpoint} attempt={attempt} error={error}"
    print(f"[CASCADE_COLLECTOR][FETCH_FAILURE] {line}")
    FAILURE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FAILURE_LOG_PATH, "a") as f:
        f.write(line + "\n")


def _load_pending() -> dict:
    if not PENDING_STATE_PATH.exists():
        return {}
    try:
        with open(PENDING_STATE_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_pending(pending: dict) -> None:
    PENDING_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = f"{PENDING_STATE_PATH}.tmp"
    with open(tmp, "w") as f:
        json.dump(pending, f)
    Path(tmp).replace(PENDING_STATE_PATH)


def _fetch_symbol_candles(client, symbol, start_ms, end_ms):
    try:
        raw = client.get_candles(symbol, "1m", start_ms, end_ms)
    except VenueFetchError:
        return []
    return [{"ts_ms": int(c["t"]), "open": float(c["o"]), "high": float(c["h"]),
              "low": float(c["l"]), "close": float(c["c"]), "volume": float(c["v"])}
            for c in raw]


def run_cycle(client, universe, pending: dict, dry_run: bool = False):
    now = datetime.now(timezone.utc)
    now_ms = int(now.timestamp() * 1000)
    detected_this_cycle = []

    # --- 1. scan for new cascades ---
    for entry in universe:
        symbol = entry["symbol"]
        start_ms = now_ms - CANDLE_WARMUP_MINUTES * 60_000
        candles = _fetch_symbol_candles(client, symbol, start_ms, now_ms)
        event = detect_cascade(symbol, candles)
        if event is None:
            continue
        event_id = storage.make_event_id(symbol, event.detected_ts_ms)
        row = {
            "event_id": event_id, "symbol": symbol,
            "detected_at_utc": datetime.fromtimestamp(event.detected_ts_ms / 1000, tz=timezone.utc)
                .strftime("%Y-%m-%dT%H:%M:%SZ"),
            "cascade_move_pct": event.cascade_move_pct, "cascade_vol_ratio": event.cascade_vol_ratio,
            "fade_side": event.fade_side, "entry_price": event.entry_price,
            "stop_price": event.stop_price, "tp1_price": event.tp1_price, "tp2_price": event.tp2_price,
        }
        detected_this_cycle.append(row)
        if not dry_run:
            is_new = storage.append_event(row)
            if is_new:
                pending[event_id] = {"symbol": symbol, "detected_ts_ms": event.detected_ts_ms}

    # --- 2. check pending events for resolution ---
    resolved_ids = []
    if not dry_run:
        for event_id, meta in list(pending.items()):
            detected_ts_ms = meta["detected_ts_ms"]
            symbol = meta["symbol"]
            elapsed_min = (now_ms - detected_ts_ms) / 60_000
            if elapsed_min < FILL_WINDOW_BARS:
                continue  # too soon even for the fill-window check
            after = _fetch_symbol_candles(client, symbol, detected_ts_ms + 1,
                                           min(now_ms, detected_ts_ms + RESOLUTION_CHECK_MARGIN_MINUTES * 60_000))
            event_row = storage.load_event(event_id)
            if not event_row:
                resolved_ids.append(event_id)  # orphaned pending entry, drop it
                continue
            from research.cascade.detector import CascadeEvent
            ev = CascadeEvent(
                symbol=symbol, detected_ts_ms=detected_ts_ms,
                cascade_move_pct=float(event_row["cascade_move_pct"]),
                cascade_vol_ratio=float(event_row["cascade_vol_ratio"]),
                fade_side=event_row["fade_side"], entry_price=float(event_row["entry_price"]),
                stop_price=float(event_row["stop_price"]), tp1_price=float(event_row["tp1_price"]),
                tp2_price=float(event_row["tp2_price"]),
            )
            resolution = try_fill_and_resolve(ev, after)
            if resolution is None:
                continue  # not enough elapsed time yet
            res_row = {
                "event_id": event_id, "symbol": symbol, "resolved_at_utc": storage.now_iso(),
                **{k: resolution.get(k, "") for k in
                   ("filled", "exit_reason", "avg_exit_price", "gross_pnl_pct",
                    "fees_pct", "net_pnl_pct", "pnl_r", "bars_held")},
            }
            storage.append_resolution(res_row)
            resolved_ids.append(event_id)

    for eid in resolved_ids:
        pending.pop(eid, None)

    return detected_this_cycle, pending


def main():
    parser = argparse.ArgumentParser(description="Hyperliquid cascade-fade forward collector")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    client = HyperliquidReadOnlyClient(on_failure=_log_failure)
    universe = select_universe(client)
    print(f"[CASCADE_COLLECTOR] universe: {len(universe)} symbols "
          f"({', '.join(u['symbol'] for u in universe[:10])}{'...' if len(universe) > 10 else ''})")

    if args.dry_run:
        detected, _ = run_cycle(client, universe, {}, dry_run=True)
        print(f"\n[CASCADE_COLLECTOR] schema (events): {storage.EVENT_FIELDNAMES}")
        print(f"[CASCADE_COLLECTOR] schema (resolutions): {storage.RESOLUTION_FIELDNAMES}")
        print(f"[CASCADE_COLLECTOR] {len(detected)} cascade(s) detected this cycle (not written, dry-run):")
        for row in detected:
            print(" ", row)
        return

    pending = _load_pending()
    last_universe_refresh_date = datetime.now(timezone.utc).date()
    print(f"[CASCADE_COLLECTOR] starting long-lived collection, poll_interval={POLL_INTERVAL_SEC}s, "
          f"universe_size={len(universe)}, resuming {len(pending)} pending event(s)")

    while not _stop_requested:
        cycle_start = time.time()
        now = datetime.now(timezone.utc)

        if now.date() != last_universe_refresh_date:
            universe = select_universe(client)
            last_universe_refresh_date = now.date()
            print(f"[CASCADE_COLLECTOR] universe refreshed for {now.date()}: {len(universe)} symbols")

        detected, pending = run_cycle(client, universe, pending, dry_run=False)
        _save_pending(pending)
        if detected:
            print(f"[CASCADE_COLLECTOR] {len(detected)} new cascade(s) detected: "
                  f"{[(d['symbol'], d['fade_side']) for d in detected]}")
        print(f"[CASCADE_COLLECTOR] cycle complete: pending={len(pending)} at {storage.now_iso()}")

        elapsed = time.time() - cycle_start
        sleep_for = max(0.0, POLL_INTERVAL_SEC - elapsed)
        for _ in range(int(sleep_for)):
            if _stop_requested:
                break
            time.sleep(1)

    print("[CASCADE_COLLECTOR] stopped cleanly")


if __name__ == "__main__":
    main()

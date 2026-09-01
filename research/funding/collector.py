"""
Funding-data research collector -- Phase 3 of the 2026-09-01 data-only
pivot (see PIVOT_STATUS.md). Collection only: no strategy, no signal, no
backtest, no expected-return model. If you find yourself computing an
expected carry here, stop -- that's Phase 2 of the research plan and it
needs this data first.

Two cadences in one schema, per symbol:
  - Every SNAPSHOT_INTERVAL_SEC (default 300s / 5min): funding_rate (the
    venue's live, continuously-updated rate), mark/oracle price, open
    interest, best_bid/best_ask/mid/spread, 24h volume.
  - Once per funding settlement (hourly, Hyperliquid's actual interval):
    realized_funding_rate, the actual settled amount for the period that
    just closed, from fundingHistory -- ground truth, distinct from the
    live-sampled funding_rate above.

Usage:
    python -m research.funding.collector            # long-lived collection
    python -m research.funding.collector --dry-run   # one cycle, print, exit
"""

import argparse
import json
import signal
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from research.funding import storage
from research.funding.universe import select_universe
from research.funding.venue_client import HyperliquidReadOnlyClient, VenueFetchError

SNAPSHOT_INTERVAL_SEC = 300
FUNDING_INTERVAL_HOURS = 1.0  # Hyperliquid settles hourly -- not exposed as
                              # a per-asset field on any endpoint checked;
                              # documented assumption, not a magic number.
FAILURE_LOG_PATH = Path(__file__).resolve().parent / "data" / "_fetch_failures.log"

_stop_requested = False


def _handle_sigterm(signum, frame):
    global _stop_requested
    print("[FUNDING_COLLECTOR] SIGTERM received -- finishing current cycle, then exiting")
    _stop_requested = True


def _log_failure(endpoint: str, error: str, attempt: int) -> None:
    line = f"{storage.now_iso()} endpoint={endpoint} attempt={attempt} error={error}"
    print(f"[FUNDING_COLLECTOR][FETCH_FAILURE] {line}")
    FAILURE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FAILURE_LOG_PATH, "a") as f:
        f.write(line + "\n")


def _build_row(symbol: str, ctx: dict, book: dict, realized_funding_rate) -> dict:
    try:
        mark_px = float(ctx.get("markPx", 0.0) or 0.0)
    except (TypeError, ValueError):
        mark_px = 0.0
    try:
        oracle_px = float(ctx.get("oraclePx", 0.0) or 0.0)
    except (TypeError, ValueError):
        oracle_px = 0.0
    try:
        open_interest = float(ctx.get("openInterest", 0.0) or 0.0)
    except (TypeError, ValueError):
        open_interest = 0.0
    open_interest_usd = open_interest * mark_px if mark_px > 0 else 0.0

    best_bid = best_ask = mid = spread_bps = ""
    levels = book.get("levels") if isinstance(book, dict) else None
    if isinstance(levels, list) and len(levels) == 2 and levels[0] and levels[1]:
        try:
            best_bid = float(levels[0][0]["px"])
            best_ask = float(levels[1][0]["px"])
            mid = (best_bid + best_ask) / 2.0
            spread_bps = (best_ask - best_bid) / mid * 10_000.0 if mid > 0 else ""
        except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError):
            best_bid = best_ask = mid = spread_bps = ""

    try:
        funding_rate = float(ctx.get("funding", 0.0) or 0.0)
    except (TypeError, ValueError):
        funding_rate = ""
    try:
        volume_24h_usd = float(ctx.get("dayNtlVlm", 0.0) or 0.0)
    except (TypeError, ValueError):
        volume_24h_usd = ""

    return {
        "timestamp_utc": storage.now_iso(),
        "symbol": symbol,
        "funding_rate": funding_rate,
        "realized_funding_rate": realized_funding_rate if realized_funding_rate is not None else "",
        "funding_interval_hours": FUNDING_INTERVAL_HOURS,
        "mark_price": mark_px,
        "oracle_price": oracle_px,
        "open_interest": open_interest,
        "open_interest_usd": open_interest_usd,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "mid": mid,
        "spread_bps": spread_bps,
        "volume_24h_usd": volume_24h_usd,
        "raw_response_json": json.dumps({"ctx": ctx, "book_top": levels[:1] if isinstance(levels, list) else None}),
    }


def _fetch_realized_funding(client: HyperliquidReadOnlyClient, symbol: str, now: datetime):
    """Realized rate for the most recently closed hourly window, or None on failure."""
    end_ms = int(now.timestamp() * 1000)
    start_ms = int((now - timedelta(hours=2)).timestamp() * 1000)
    try:
        history = client.get_funding_history(symbol, start_ms, end_ms)
    except VenueFetchError:
        return None
    if not history:
        return None
    latest = max(history, key=lambda r: r.get("time", 0))
    try:
        return float(latest.get("fundingRate", 0.0))
    except (TypeError, ValueError):
        return None


def run_cycle(client: HyperliquidReadOnlyClient, universe, fetch_realized: bool, dry_run: bool):
    try:
        data = client.get_meta_and_asset_ctxs()
    except VenueFetchError as e:
        print(f"[FUNDING_COLLECTOR] cycle aborted -- {e}")
        return []
    universe_obj, ctxs = data
    idx_by_symbol = {
        str(a.get("name", "")).upper(): i for i, a in enumerate(universe_obj.get("universe", []))
    }

    now = datetime.now(timezone.utc)
    rows = []
    for entry in universe:
        symbol = entry["symbol"]
        idx = idx_by_symbol.get(symbol)
        if idx is None or idx >= len(ctxs):
            _log_failure(f"ctx_lookup:{symbol}", "symbol not found in current universe response", 1)
            continue
        ctx = ctxs[idx]
        try:
            book = client.get_l2_book(symbol)
        except VenueFetchError as e:
            _log_failure(f"l2Book:{symbol}", str(e), 1)
            book = {}

        realized = _fetch_realized_funding(client, symbol, now) if fetch_realized else None
        row = _build_row(symbol, ctx, book, realized)
        rows.append(row)

        if not dry_run:
            storage.append_row(row)

    return rows


def main():
    parser = argparse.ArgumentParser(description="Hyperliquid funding-data research collector")
    parser.add_argument("--dry-run", action="store_true", help="Fetch one cycle, print schema, exit")
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    client = HyperliquidReadOnlyClient(on_failure=_log_failure)
    universe = select_universe(client)
    print(f"[FUNDING_COLLECTOR] universe: {len(universe)} symbols "
          f"({', '.join(u['symbol'] for u in universe[:10])}{'...' if len(universe) > 10 else ''})")

    if args.dry_run:
        rows = run_cycle(client, universe, fetch_realized=True, dry_run=True)
        print(f"\n[FUNDING_COLLECTOR] schema ({len(storage.FIELDNAMES)} columns):")
        print("  " + ", ".join(storage.FIELDNAMES))
        print(f"\n[FUNDING_COLLECTOR] {len(rows)} rows fetched this cycle, not written (dry-run):")
        for row in rows[:5]:
            print(" ", {k: row[k] for k in ("timestamp_utc", "symbol", "funding_rate", "mark_price", "spread_bps")})
        return

    last_universe_refresh_date = datetime.now(timezone.utc).date()
    last_hourly_fetch = None
    print(f"[FUNDING_COLLECTOR] starting long-lived collection, "
          f"snapshot_interval={SNAPSHOT_INTERVAL_SEC}s, universe_size={len(universe)}")

    while not _stop_requested:
        cycle_start = time.time()
        now = datetime.now(timezone.utc)

        if now.date() != last_universe_refresh_date:
            universe = select_universe(client)
            last_universe_refresh_date = now.date()
            print(f"[FUNDING_COLLECTOR] universe refreshed for {now.date()}: {len(universe)} symbols")

        do_hourly = last_hourly_fetch is None or (now - last_hourly_fetch) >= timedelta(hours=1)
        rows = run_cycle(client, universe, fetch_realized=do_hourly, dry_run=False)
        if do_hourly:
            last_hourly_fetch = now
        print(f"[FUNDING_COLLECTOR] cycle complete: {len(rows)} rows written "
              f"(hourly_fetch={do_hourly}) at {storage.now_iso()}")

        elapsed = time.time() - cycle_start
        sleep_for = max(0.0, SNAPSHOT_INTERVAL_SEC - elapsed)
        for _ in range(int(sleep_for)):
            if _stop_requested:
                break
            time.sleep(1)

    print("[FUNDING_COLLECTOR] stopped cleanly")


if __name__ == "__main__":
    main()

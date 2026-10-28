"""Fetch the fresh 1h and 15m snapshots the rr_floor and ltf_15m tests use.

    python -m research.backtest.experiments.fetch_fresh

Paced (common.fetch_paced): one request per 10 seconds, because the live bot
shares this IP's request budget with whatever runs here.
"""

from __future__ import annotations

from research.backtest import common as C
from research.backtest.experiments.ltf_15m import SNAPSHOT_15M
from research.backtest.experiments.rr_floor import SNAPSHOT


def main() -> int:
    from research.levels.universe import select_universe
    from research.levels.venue_client import HyperliquidReadOnlyClient
    client = HyperliquidReadOnlyClient()
    coins = [str(r["symbol"]).upper() for r in select_universe(client)]
    print(f"[FETCH] live universe: {len(coins)} coins")
    one = C.fetch_paced(coins, "1h", client=client)
    one.save(SNAPSHOT)
    print(f"[FETCH] 1h: {len(one.bars)} coins -> {SNAPSHOT}")
    fifteen = C.fetch_paced(sorted(one.bars), "15m", client=client)
    fifteen.save(SNAPSHOT_15M)
    print(f"[FETCH] 15m: {len(fifteen.bars)} coins -> {SNAPSHOT_15M}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

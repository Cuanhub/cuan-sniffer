"""
Symbol universe selection for the level scanner.

Moved here from research/funding/universe.py when the funding collector was
retired (2026-09-18). The rule is unchanged: the top N perps by 24h notional
volume, refreshed daily. It originated in the funding research and outlived
it because the level scanner needs the same thing -- a liquid, bounded set of
symbols to watch -- and keeping it in a retired package would have left the
whole of research/funding alive to serve one pure function.

Decision (2026-09-01, see DECISIONS.md): volume-filtered top-N rather than
the full ~233-perp venue listing or the handful of coins then being traded.
Broad enough that no single symbol dominates a cross-sectional check,
filtered so the illiquid tail -- which could not support a real position
anyway -- does not add noise and API load.

Exclusion list (2026-09-16): symbols the venue ranks by volume but cannot
actually serve data for are dropped BEFORE truncation, so the universe stays
top_n wide and the freed slot goes to the next genuinely liquid symbol.
KPEPE is excluded on that basis -- the venue 500s on candleSnapshot,
fundingHistory and the L2 book for it, so 2,088 collected rows carried zero
realized funding and zero spread while its retries cost ~16s per cycle.
"""

import os
from typing import Any, Dict, List

from research.levels.venue_client import HyperliquidReadOnlyClient

# 45, raised from 30 on 2026-10-01, to be reviewed after three days of
# trading.
#
# The top 30 by 24h volume is a list that barely moves -- BTC and ETH are not
# going to drop out of it -- so the same names recurred on every board while
# 204 of the venue's 234 perps were never looked at. The cut also fell straight
# through setups that had been called by hand: CASHCAT, FARTCOIN, USELESS and
# BNB all sat at ranks 31-45.
#
# What made this affordable rather than merely desirable is the request budget.
# At 30 coins a pass cost 90 venue calls; after removing the duplicate 1h fetch
# and caching the 4h series between bar boundaries it costs 31. So 45 coins now
# runs at about 46 calls a pass -- roughly half the load that was producing
# rate limits at half the universe size.
#
# The floor moves from $10.3M to $4.9M of 24h notional. At a $119 position that
# is not a liquidity constraint; what it does buy is thinner, more erratic
# names, which is exactly the thing three days of trading is meant to measure.
# Review against the pre-change book: 26 closed trades, 8W/18L, +$28.94.
UNIVERSE_TOP_N = int(os.getenv("LEVELS_UNIVERSE_TOP_N",
                               os.getenv("FUNDING_UNIVERSE_TOP_N", "45")))

# KBONK joins KPEPE on 2026-10-01: the venue returns a 500 from
# candleSnapshot for it, consistently and on every retry, so it entered the
# universe at rank 43 and then burned five retries per timeframe per pass --
# ten failed requests a pass in both the trader and the scanner, for a coin
# that can never produce a bar. Excluded rather than tolerated.
UNIVERSE_EXCLUDE = {
    s.strip().upper()
    for s in os.getenv("LEVELS_UNIVERSE_EXCLUDE",
                       os.getenv("FUNDING_UNIVERSE_EXCLUDE",
                                 "KPEPE,KBONK")).split(",")
    if s.strip()
}


def select_universe(client: HyperliquidReadOnlyClient,
                    top_n: int = UNIVERSE_TOP_N) -> List[Dict[str, Any]]:
    """
    Return the top_n perps by 24h notional volume (dayNtlVlm), each as
    {"symbol": str, "day_ntl_vlm": float}, sorted descending by volume.
    """
    data = client.get_meta_and_asset_ctxs()
    universe_obj, ctxs = data
    universe = universe_obj.get("universe", [])

    rows = []
    for asset, ctx in zip(universe, ctxs):
        symbol = str(asset.get("name", "")).upper()
        if not symbol or not isinstance(ctx, dict):
            continue
        try:
            vlm = float(ctx.get("dayNtlVlm", 0.0) or 0.0)
        except (TypeError, ValueError):
            vlm = 0.0
        if symbol in UNIVERSE_EXCLUDE:
            continue
        rows.append({"symbol": symbol, "day_ntl_vlm": vlm})

    rows.sort(key=lambda r: r["day_ntl_vlm"], reverse=True)
    return rows[:top_n]

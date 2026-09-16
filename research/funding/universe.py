"""
Symbol universe selection for the funding-data collector.

Decision (2026-09-01, see DECISIONS.md): volume-filtered top-N by 24h
notional volume, not the full ~233-perp venue listing and not just the
8 currently-traded coins. Broader than 8 for real cross-sectional
breadth in the eventual HYPOTHESES.md independence check; filtered by
volume to exclude the illiquid tail that couldn't support a real carry
position and would only add noise and API load.

Exclusion list (2026-09-16, see DECISIONS.md): symbols the venue lists
and ranks by volume but cannot actually serve data for are dropped
before truncation, so the universe stays top_n wide and the freed slot
goes to the next genuinely-liquid symbol.
"""

import os
from typing import Any, Dict, List

from research.funding.venue_client import HyperliquidReadOnlyClient

UNIVERSE_TOP_N = int(os.getenv("FUNDING_UNIVERSE_TOP_N", "30"))

# Symbols excluded regardless of volume rank. KPEPE (2026-09-16): the venue
# ranks it top-30 by volume but 500s on candleSnapshot, fundingHistory and
# the L2 book -- 2,088 collected rows carried zero realized funding and zero
# spread, while its retries cost ~16s per funding cycle and ~32s per cascade
# cycle. Excluded as unservable, not as a judgment about the asset.
UNIVERSE_EXCLUDE = {
    s.strip().upper()
    for s in os.getenv("FUNDING_UNIVERSE_EXCLUDE", "KPEPE").split(",")
    if s.strip()
}


def select_universe(client: HyperliquidReadOnlyClient, top_n: int = UNIVERSE_TOP_N) -> List[Dict[str, Any]]:
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

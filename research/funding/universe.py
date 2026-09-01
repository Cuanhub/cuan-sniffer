"""
Symbol universe selection for the funding-data collector.

Decision (2026-09-01, see DECISIONS.md): volume-filtered top-N by 24h
notional volume, not the full ~233-perp venue listing and not just the
8 currently-traded coins. Broader than 8 for real cross-sectional
breadth in the eventual HYPOTHESES.md independence check; filtered by
volume to exclude the illiquid tail that couldn't support a real carry
position and would only add noise and API load.
"""

import os
from typing import Any, Dict, List

from research.funding.venue_client import HyperliquidReadOnlyClient

UNIVERSE_TOP_N = int(os.getenv("FUNDING_UNIVERSE_TOP_N", "30"))


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
        rows.append({"symbol": symbol, "day_ntl_vlm": vlm})

    rows.sort(key=lambda r: r["day_ntl_vlm"], reverse=True)
    return rows[:top_n]

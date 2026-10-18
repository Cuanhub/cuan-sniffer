#!/usr/bin/env python3
"""Which bands are closest to a fill, and what each one would trade.

The point of this project is the gap between what the rules fire on and what
the operator would actually trade, and that gap can only be measured by
putting live setups in front of a human. This is that board.

It applies the trader's own band logic -- the same zone_pass the loop uses --
so what it shows is what the bot would do, not a second opinion computed a
different way. Read-only: it places nothing and writes nothing.

    python -m tools.near_firing              # the 10 closest
    python -m tools.near_firing --all
    python -m tools.near_firing --coin ENA
    python -m tools.near_firing --min-rr 1.5 # try a different floor

Rewritten when the pass moved from scored levels to bands. The old board read
the trader's watchlist and called passes_quality, and both went with the level
scan. "Closest to firing" also means something simpler now: a band price is
inside is firing, and a band above price is as far away as the bid is.
"""

import argparse
from typing import Any, Dict, List

from dotenv import load_dotenv

load_dotenv()

from research.levels import zone_pass                            # noqa: E402
from research.levels.config import MIN_RR                        # noqa: E402
from research.levels.universe import select_universe             # noqa: E402
from research.levels.venue_client import (                       # noqa: E402
    HyperliquidReadOnlyClient,
)


def collect(coin_filter: str = "", min_rr: float = MIN_RR) -> List[Dict[str, Any]]:
    """Every band in reach, nearest to a fill first."""
    ro = HyperliquidReadOnlyClient()
    universe = select_universe(ro)
    if coin_filter:
        universe = [u for u in universe
                    if str(u.get("symbol", "")).upper() == coin_filter.upper()]
        if not universe:
            universe = [{"symbol": coin_filter.upper()}]
    try:
        cx = ro.get_perp_contexts()
        mids = {k: v["mid_px"] for k, v in cx.items() if v.get("mid_px")}
    except Exception:
        mids = {}
    _, _, found = zone_pass.plans_by_coin(ro, universe, min_rr=min_rr,
                                          mids=mids)
    rows = []
    for coin, plans in found.items():
        px = float(mids.get(coin) or 0.0)
        for p in plans:
            # How far price has to fall for this to fill. Zero if it is in the
            # band -- that is what "firing" means for a band.
            away = 0.0 if p.immediate or px <= 0 else (px - p.entry) / px * 100.0
            rows.append({
                "coin": coin, "px": px, "away": away, "plan": p,
                "block": ("IN BAND -- fires now" if p.immediate
                          else f"bid {away:.2f}% below price"),
                "why_not": zone_pass.rejection(p, min_rr) or "",
            })
    rows.sort(key=lambda r: (bool(r["why_not"]), r["away"]))
    return rows


def render(rows: List[Dict[str, Any]], limit: int) -> str:
    if not rows:
        return "no band in reach on any coin"
    out = [f"{'coin':<10}{'band':>24}{'entry':>11}{'stop':>11}{'tp':>11}"
           f"{'risk%':>7}{'rr':>6}  status"]
    for r in rows[:limit]:
        p = r["plan"]
        status = r["why_not"] and f"refused: {r['why_not']}" or r["block"]
        out.append(
            f"{r['coin']:<10}{f'{p.zone.lo:.6g}-{p.zone.hi:.6g}':>24}"
            f"{p.entry:11.6g}{p.stop:11.6g}{p.tp:11.6g}"
            f"{p.risk_pct:7.2f}{p.rr:6.2f}  {status}")
    if len(rows) > limit:
        out.append(f"... and {len(rows) - limit} more")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="every band, not 10")
    ap.add_argument("--coin", default="", help="one coin only")
    ap.add_argument("--min-rr", type=float, default=MIN_RR,
                    help=f"reward/risk floor to apply (default {MIN_RR})")
    args = ap.parse_args()
    rows = collect(args.coin, args.min_rr)
    print(render(rows, 10_000 if args.all else 10))
    firing = sum(1 for r in rows if not r["why_not"] and r["plan"].immediate)
    waiting = sum(1 for r in rows if not r["why_not"] and not r["plan"].immediate)
    print(f"\n{firing} firing now, {waiting} waiting on a bid, "
          f"{sum(1 for r in rows if r['why_not'])} refused "
          f"(rr floor {args.min_rr})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

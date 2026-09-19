#!/usr/bin/env python3
"""Which watched levels are closest to firing, and what is holding each back.

The point of this project is the gap between what the rules fire on and what
the user would actually trade, and that gap can only be measured if the setups
are put in front of a human while they are still live. This is that board: the
levels currently eligible to trade, ranked by how close they are, each with
the chart link and the exact command to label it.

It reads the trader's own watchlist and applies the trader's own filters, so
what it shows is what the bot would do -- not a second opinion computed a
different way. Read-only: it places nothing, and writes nothing except when
you run one of the label commands it prints.

    python -m tools.near_firing              # top 10
    python -m tools.near_firing --all        # everything eligible
    python -m tools.near_firing --coin ENA   # one coin
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)

import tools.live_levels_trader as T                                  # noqa: E402
from research.levels.venue_client import HyperliquidReadOnlyClient    # noqa: E402
from research.levels.universe import select_universe                  # noqa: E402
from research.levels.levels import log_sigma, sigma_distance          # noqa: E402
from research.levels.trigger import fetch_1h                          # noqa: E402
from research.levels.config import (                                  # noqa: E402
    TRIGGER_ZONE_SIGMA, TRIGGER_HOLD_HOURS, TRIGGER_TOUCH_LOOKBACK_HOURS,
    TRIGGER_MAX_ABOVE_SIGMA, MAX_ENTRY_DRIFT_PCT,
)


def load_watchlist() -> Dict[str, Any]:
    try:
        return json.loads(T.STATE_PATH.read_text()).get("watching", {})
    except (OSError, ValueError):
        return {}


def sigma_to_fire(above: float, touched: bool, below: bool, broke: int) -> float:
    """How far price must still FALL, in sigma, before the trigger can fire.

    Ranking on `above` alone is wrong, and misleadingly so. A level at 1.9
    sigma that has not been touched is nearer than one at 4.6 that has: the
    first needs a 0.9 sigma dip to register a touch, the second needs 2.6 to
    get back under the ceiling. Both then need the 2h hold, which is time
    rather than distance and so is not counted here.

    An untouched level is the binding case: price must reach within
    TRIGGER_ZONE_SIGMA of the level to count as a touch at all, and that is a
    deeper requirement than the ceiling it also satisfies on the way.
    """
    if below or broke:
        return float("inf")     # price is under the level; not a buy setup yet
    if not touched:
        return max(0.0, above - TRIGGER_ZONE_SIGMA)
    return max(0.0, above - TRIGGER_MAX_ABOVE_SIGMA)


def blocking_reason(px: float, lvl: float, above: float, touched: bool,
                    broke: int) -> str:
    """The single condition standing between this level and an entry."""
    if px <= lvl:
        return "price is BELOW the level"
    if broke:
        return f"closed below it {broke}x in {TRIGGER_HOLD_HOURS}h"
    if not touched:
        return f"has not reached it in {TRIGGER_TOUCH_LOOKBACK_HOURS}h"
    if above > TRIGGER_MAX_ABOVE_SIGMA:
        return f"{above:.1f}s above (needs <={TRIGGER_MAX_ABOVE_SIGMA})"
    return "FIRING NOW"


def collect(coin_filter: str = "") -> List[Dict[str, Any]]:
    watching = load_watchlist()
    ro = HyperliquidReadOnlyClient()
    rows, cache = [], {}
    for ep in T._find_candidates(ro, select_universe(ro)):
        coin = str(ep.get("coin", ""))
        if coin_filter and coin.upper() != coin_filter.upper():
            continue
        # Eligible exactly as the trader judges it: admitted now, or watched.
        if not ((T.passes_quality(ep, watched=False) is None)
                or (T.watch_key(ep) in watching)):
            continue
        # Target, headroom and reward/risk still have to hold.
        if T.passes_quality(ep, watched=True) is not None:
            continue
        if coin not in cache:
            try:
                cache[coin] = fetch_1h(ro, coin)
            except Exception:
                cache[coin] = []
        bars = cache[coin]
        if len(bars) < TRIGGER_HOLD_HOURS + 2:
            continue
        closed = bars[:-1]
        sigma = log_sigma(closed, len(closed) - 1)
        if sigma <= 0:
            continue
        lvl = float(ep["level_price"])
        px = closed[-1]["c"]
        above = sigma_distance(px, lvl, sigma)
        touched = any(abs(sigma_distance(b["l"], lvl, sigma)) <= TRIGGER_ZONE_SIGMA
                      or b["l"] <= lvl
                      for b in closed[-TRIGGER_TOUCH_LOOKBACK_HOURS:])
        broke = sum(1 for b in closed[-TRIGGER_HOLD_HOURS:] if b["c"] < lvl)
        entry = float(ep.get("suggested_entry") or 0)
        rows.append({
            "coin": coin, "lvl": lvl, "px": px, "above": above,
            "touched": touched,
            "to_fire": sigma_to_fire(above, touched, px <= lvl, broke),
            "block": blocking_reason(px, lvl, above, touched, broke),
            "rr": float(ep.get("suggested_rr") or 0), "entry": entry,
            "stop": float(ep.get("suggested_stop") or 0),
            "tp": float(ep.get("suggested_tp") or 0),
            "eid": ep.get("episode_id", ""),
            "origin": ep.get("level_origin", "?"),
            "tf": "1H" if "-1H_" in str(ep.get("episode_id", "")) else "4H",
            "buffer": float(ep.get("entry_buffer_sigma") or 0),
            "since_touch": int(float(ep.get("level_bars_since_last_touch") or 0)),
            "retests": int(float(ep.get("retests") or 0)),
            "touches": int(float(ep.get("level_touches") or 0)),
            "hold_rate": float(ep.get("level_hold_rate") or 0),
            "gap_pct": (px - lvl) / lvl * 100 if lvl else 0.0,
            "drift_pct": abs(px - entry) / entry * 100 if entry else 0.0,
        })
    rows.sort(key=lambda r: (r["block"] != "FIRING NOW", r["to_fire"], r["above"]))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Levels closest to firing")
    ap.add_argument("--all", action="store_true", help="show every eligible level")
    ap.add_argument("--coin", default="", help="restrict to one coin")
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    rows = collect(args.coin)
    watching = load_watchlist()
    print(f"{len(rows)} tradeable level(s); {len(watching)} on the watchlist")
    print(f"trigger needs: touched within {TRIGGER_TOUCH_LOOKBACK_HOURS}h, held "
          f"{TRIGGER_HOLD_HOURS}h, and <= {TRIGGER_MAX_ABOVE_SIGMA} sigma above "
          f"the level")
    print(f"entry also needs live price within {MAX_ENTRY_DRIFT_PCT}% of the "
          f"planned entry\n")

    for i, r in enumerate(rows if args.all else rows[:args.top], 1):
        flag = " [FLIPPED resistance -> support]" if r["origin"] == "flipped_resistance" else ""
        dist = ("-" if r["to_fire"] == float("inf")
                else f"{r['to_fire']:.1f}s further down")
        print(f"{i}. {r['coin']} {r['tf']}  level {r['lvl']:.6g}   "
              f"now {r['px']:.6g}  ({r['gap_pct']:+.1f}%, {r['above']:.1f}s above){flag}")
        print(f"     needs : {r['block']}   [{dist}]")
        # Lead with the level; label the entry as derived from it. Quoting
        # the entry as though it were the level produced a wrong skip on
        # USELESS .25817 -- the entry .260 was not recognised, because there
        # is no level there, only the 0.35 sigma buffer above one.
        print(f"     entry : {r['entry']:.6g} (= level +{r['buffer']:g}s buffer)"
              f"   stop {r['stop']:.6g}   tp {r['tp']:.6g}   rr {r['rr']:.2f}")
        print(f"     drift : {r['drift_pct']:.2f}% from plan "
              f"({'fillable' if r['drift_pct'] <= MAX_ENTRY_DRIFT_PCT else 'too far to fill'})")
        # Staleness is shown but NOT filtered on. STRK 0.03796 reached this
        # board 655 bars after its last touch -- 109 days on 4h -- with zero
        # retests, and nothing in the trading path rejects that. Until enough
        # labels say where the boundary is, the judgement is the operator's,
        # which means the number has to be in front of them.
        hours = r["since_touch"] * (1 if r["tf"] == "1H" else 4)
        age = f"{hours}h" if hours < 48 else f"{hours/24:.0f}d"
        print(f"     tested: last touched {r['since_touch']}b ago ({age}), "
              f"{r['touches']} touches, {r['hold_rate']*100:.0f}% held, "
              f"{r['retests']} retests")
        print(f"     chart : https://app.hyperliquid.xyz/trade/{r['coin']}")
        # call.py, not review.py: review requires the episode to already exist
        # in episodes.csv, which only the scanner writes and which is 4h-only.
        # call.py records the episode snapshot itself before labelling, which
        # is how every 1h label in the file got there.
        print(f"     label : python -m research.levels.call --take {r['eid']} "
              f"--note \"...\"")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

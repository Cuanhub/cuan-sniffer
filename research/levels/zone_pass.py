"""The trading pass, expressed as bands.

One idea: for each coin, find the recent demand bands price can reach, take
the most recent, and buy inside it with the stop under its low and the target
at the first supply band overhead. research/levels/zones.py has the
definitions and the evidence behind them; this module is the glue between
that and the trader -- the fetch, the gate, the episode dict the executor
already speaks, and the table a human reads before anything is sent.

Why this replaces the levels scan
---------------------------------
levels.py scored the PRICE a pivot sat at, with filters fitted to 18 labels:
depth in sigma, bars since the last touch, hold rate, overhead sigma. None of
them separated the operator's own take/skip calls (`level_hold_rate` by
+0.019), and on 2026-10-17 the pass bought ARB at 0.18482 while his band was
0.1705-0.175 -- 5.6% above it. A band has a top and a bottom, so an entry can
sit INSIDE it, and that is a guarantee rather than a score.

So the gate here is small on purpose. plans_for has already required a recent
band, in reach, with a target above the market. The only thing left to refuse
is reward/risk, and that floor is the trader's existing MIN_RR rather than a
new knob.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from research.levels import zones as Z
from research.levels.zones import Plan

# Bars of 1h history to pull per coin. ZONE_MIN_BARS is 800 and a band's base
# can sit far back, so this has to clear it with room: 1200 is 50 days.
ZONE_PASS_1H_HOURS = 1200


def fetch_bars(client, coin: str, hours: int = ZONE_PASS_1H_HOURS
               ) -> List[Dict[str, Any]]:
    """1h bars for one coin, normalised to the keys zones.py reads.

    Its own fetch rather than trigger.fetch_1h, which pulls only the trigger's
    short window -- a band's base can be 700 bars back and would simply not be
    visible in it.
    """
    now_ms = int(time.time() * 1000)
    raw = client.get_candles(coin, "1h", now_ms - hours * 3600_000, now_ms)
    out: List[Dict[str, Any]] = []
    for b in raw or []:
        try:
            out.append({"t": int(b["t"]), "o": float(b["o"]),
                        "h": float(b["h"]), "l": float(b["l"]),
                        "c": float(b["c"])})
        except (KeyError, TypeError, ValueError):
            continue
    return out


def touches(plan: Plan, bars: List[Dict[str, Any]]) -> int:
    """How many bars have reached into the band since it formed.

    Informational. His bands sit at the 8th percentile of this, so a high
    count is a reason for a human to look rather than a reason to refuse --
    the evidence is not strong enough to gate on.
    """
    return sum(1 for b in bars[plan.zone.bos_i:] if b["l"] <= plan.zone.hi)


def rejection(plan: Plan, min_rr: float) -> Optional[str]:
    """Why this plan is not tradeable, or None.

    Deliberately one check. Band recency, reach, side and the existence of a
    target are structural and already settled by plans_for; a plan that got
    here is a real band with a real target, and the only remaining question
    is whether the trade pays for its own risk.
    """
    if plan.rr < min_rr:
        return f"rr {plan.rr:.2f} < {min_rr}"
    return None


def first_tradeable(plans: List[Plan], min_rr: float) -> Optional[Plan]:
    """The plan the live pass takes for one coin: the first that passes.

    `plans` arrives most recent band first (plans_for's order), so this is
    the most recent band whose trade pays for its risk. Factored out of
    plans_by_coin so research/backtest replays the SAME selection the live
    pass makes rather than a reimplementation of it -- the previous scratch
    backtest took plans[0] and skipped the coin when that one failed, which
    is not what trades.
    """
    for p in plans:
        if rejection(p, min_rr) is None:
            return p
    return None


def review_flags(plan: Plan, size_usd: float, equity: float,
                 bars: Optional[List[Dict[str, Any]]] = None) -> List[str]:
    """Reasons a human should look at this trade even though it passed.

    Not the level-based flags: `near_sigma` and `level_hold_rate` do not exist
    for a band, and reading them through a defaulting helper produced
    nonsense like "level held only 0% of its 0 tests".
    """
    out: List[str] = []
    if equity > 0 and size_usd > equity * 1.5:
        out.append(f"notional ${size_usd:,.0f} is {size_usd/equity:.1f}x equity")
    if plan.risk_pct < 1.0:
        out.append(f"stop only {plan.risk_pct:.2f}% away -- a gap through it "
                   f"loses more than the risk")
    if plan.zone.width_pct > 5.0:
        out.append(f"band is {plan.zone.width_pct:.1f}% wide -- the stop is "
                   f"far from the entry")
    if plan.immediate and plan.entry < plan.zone.mid:
        out.append("price is in the LOWER half of the band -- it is being "
                   "tested, not held")
    if bars is not None:
        t = touches(plan, bars)
        if t > 20:
            out.append(f"band has been reached into {t} times since it formed")
    return out


def to_episode(plan: Plan, bars: Optional[List[Dict[str, Any]]] = None
               ) -> Dict[str, Any]:
    """The plan as the episode dict the executor and the meta store speak.

    Keeping that contract is what lets sizing, execution, the stop loop and
    the trade log stay exactly as they are -- the rebuild is in what gets
    chosen, not in how it is sent.

    `level_price` is the band LOW: it is the price that invalidates the band
    and the one the stop is derived from, so it is the number worth logging
    next to a fill.
    """
    z = plan.zone
    return {
        "coin": z.coin,
        "episode_id": f"zone:{z.coin}:{z.base_i}:{z.lo:.10g}",
        "level_price": z.lo,
        "level_origin": "demand_zone",
        "suggested_entry": plan.entry,
        "suggested_stop": plan.stop,
        "suggested_tp": plan.tp,
        "suggested_rr": plan.rr,
        # band shape, for the log and the table
        "zone_lo": z.lo,
        "zone_hi": z.hi,
        "zone_width_pct": z.width_pct,
        "zone_base_i": z.base_i,
        "zone_immediate": plan.immediate,
        "target_lo": plan.target.lo,
        "target_hi": plan.target.hi,
        "level_touches": touches(plan, bars) if bars is not None else 0,
    }


def plans_by_coin(client, universe, min_rr: float,
                  bars_out: Optional[Dict[str, List[Dict[str, Any]]]] = None,
                  mids: Optional[Dict[str, float]] = None,
                  ) -> Tuple[List[Dict[str, Any]], List[Tuple[str, float, str]],
                             Dict[str, List[Plan]]]:
    """Scan the universe for band plans. Reads only; sends nothing.

    Returns (intents, rejected, all_plans). `intents` are episode dicts in
    trader order -- best first per coin, coins in universe order. `rejected`
    is (coin, level, reason) for the log. `all_plans` is every plan found,
    including refused ones, so the candidate table can show what was passed
    over rather than only what was taken.
    """
    mids = mids or {}
    intents: List[Dict[str, Any]] = []
    rejected: List[Tuple[str, float, str]] = []
    found: Dict[str, List[Plan]] = {}
    for row in universe:
        coin = str(row.get("symbol") or row.get("coin") or "").upper()
        if not coin:
            continue
        try:
            bars = fetch_bars(client, coin)
        except Exception as e:
            rejected.append((coin, 0.0, f"bars unavailable: {str(e)[:60]}"))
            continue
        if not bars:
            rejected.append((coin, 0.0, "no bars"))
            continue
        px = float(mids.get(coin) or bars[-1]["c"] or 0.0)
        if px <= 0:
            rejected.append((coin, 0.0, "no price"))
            continue
        if bars_out is not None:
            bars_out[coin] = bars
        plans = Z.plans_for(bars, px, coin=coin)
        if not plans:
            rejected.append((coin, 0.0, "no recent band in reach with a target"))
            continue
        found[coin] = plans
        # ONE intent per coin. plans_for returns every reachable band most
        # recent first, and admitting all of them would open a position per
        # band: BCH had five, so five BCH positions for one idea. The most
        # recent qualifying band is the selection rule -- it agrees with the
        # operator's own choice on 9 of 12, against 6 of 12 for the deepest
        # and 4 of 12 for the nearest.
        chosen = first_tradeable(plans, min_rr)
        if chosen is not None:
            intents.append(to_episode(chosen, bars))
        for p in plans:
            why = rejection(p, min_rr)
            if why is not None:
                rejected.append((coin, p.zone.lo, why))
    return intents, rejected, found


def render_candidates(found: Dict[str, List[Plan]], min_rr: float,
                      limit: int = 40) -> str:
    """The candidate table, for a human to read before anything is sent.

    Every band in reach, taken or not, with the reason when not. The operator
    asked to see this: no selection rule reproduced his own band choice better
    than 9 of 12, so the list is the honest interface -- it shows what was
    chosen AND what it was chosen over.
    """
    rows = []
    for coin in sorted(found):
        for p in found[coin]:
            rows.append((coin, p))
    if not rows:
        return "[ZONES] no band in reach on any coin this pass"
    rows.sort(key=lambda r: (r[0], -r[1].zone.base_i))
    out = [f"{'coin':<10}{'band':>24}{'w%':>6}{'entry':>11}{'stop':>11}"
           f"{'tp':>11}{'risk%':>7}{'rr':>6}  when / why not"]
    for coin, p in rows[:limit]:
        why = rejection(p, min_rr)
        when = "NOW (in band)" if p.immediate else "bid, waiting"
        out.append(
            f"{coin:<10}{f'{p.zone.lo:.6g}-{p.zone.hi:.6g}':>24}"
            f"{p.zone.width_pct:6.1f}{p.entry:11.6g}{p.stop:11.6g}"
            f"{p.tp:11.6g}{p.risk_pct:7.2f}{p.rr:6.2f}  "
            f"{when if why is None else 'refused: ' + why}")
    if len(rows) > limit:
        out.append(f"... and {len(rows) - limit} more")
    return "\n".join(out)

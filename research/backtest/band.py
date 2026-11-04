"""The band strategy, replayed bar by bar the way the live pass decides.

At each bar close the planner sees ONLY the trailing ZONE_PASS_1H_HOURS
closed bars -- the window production fetches -- and the choice is made by
`zone_pass.first_tradeable`, the same function the live pass calls. So what
this replays is the strategy that trades, not a description of it.

Mirrored from production
  - one position or resting bid per coin; a coin with either is skipped
  - in the band -> market entry at the decision close; above it -> a bid at
    the band entry resting PROTECT_BID_TTL_HOURS (12) bars
  - the 6h re-entry cooldown after a losing close. Today it is checked only
    on the MARKET path (execute_one) and not when a bid is placed, so that
    is what is mirrored; `cooldown_on_bids=True` models the rail as its own
    comment describes it.

The short-side mirror (side=SHORT; PREREG_short_mirror.md)
  The same planner run on the bars turned upside down (`invert`), each order
  mapped back and simulated as a short on the real bars. Its entries are
  therefore `zones.supply_zones` bands, its stops sit above them and its
  targets below -- every rule mirrored by construction rather than by a second
  copy of the strategy that could drift from the first.

NOT modelled, and why it matters
  - the portfolio rails: max 4 concurrent, 20/day, 60 min between entries.
    Each coin is simulated alone, so this takes more trades than the live
    account could. They select by TIMING, not by any feature, so they thin
    every arm of a comparison alike -- but absolute totals here are larger
    than the live account would have produced.
  - the 3-loss stand-down (core.streak), for the same reason.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from research.backtest.costs import Costs, MEASURED
from research.backtest.engine import (LIMIT, LONG, MARKET_CLOSE, SHORT, Order,
                                      Trade, simulate)
from research.levels import zone_pass
from research.levels import zones as Z
from research.levels.config import COIN_REENTRY_COOLDOWN_HOURS, MIN_RR

WINDOW = zone_pass.ZONE_PASS_1H_HOURS
# core.protect.BID_TTL_HOURS. Not imported: research/ may not reach core/,
# which places orders. test_backtest_engine asserts the two agree.
BID_TTL_BARS = 12

FeatureFn = Callable[[str, List[Dict[str, Any]], int, Any], Dict[str, Any]]


def order_at(coin: str, bars: List[Dict[str, Any]], i: int,
             min_rr: float = MIN_RR, window: int = WINDOW,
             ttl: int = BID_TTL_BARS,
             features: Optional[FeatureFn] = None) -> Optional[Order]:
    """The order the live pass would make at the close of bar i, or None."""
    lo = max(0, i - window + 1)
    hist = bars[lo:i + 1]
    if len(hist) < Z.ZONE_MIN_BARS:
        return None
    px = hist[-1]["c"]
    p = zone_pass.first_tradeable(Z.plans_for(hist, px, coin=coin), min_rr)
    if p is None:
        return None
    feats: Dict[str, Any] = {
        "rr": p.rr, "risk_pct": p.risk_pct, "atr_pct": p.atr_pct,
        "stop_atr_x": p.stop_atr_mult, "immediate": p.immediate,
        "zone_lo": p.zone.lo, "zone_hi": p.zone.hi,
        "zone_width_pct": p.zone.width_pct,
        # Absolute bar index of the band's base, so features can find it.
        "zone_base_abs": lo + p.zone.base_i,
        "target_lo": p.target.lo,
    }
    if features is not None:
        feats.update(features(coin, bars, i, p))
    if p.immediate:
        return Order(coin, i, MARKET_CLOSE, p.stop, p.tp, tag="band",
                     features=feats)
    return Order(coin, i, LIMIT, p.stop, p.tp, limit=p.entry, ttl=ttl,
                 tag="band", features=feats)


def invert_k(bars: List[Dict[str, Any]]) -> float:
    """The constant `invert` divides into: the first close squared, so the
    inverted series starts at the same price and nothing scale-sensitive moves."""
    return float(bars[0]["c"]) ** 2 if bars else 1.0


def invert(bars: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The same bars upside down: every price p becomes K/p.

    A high becomes a low and a fall becomes a rise, so a demand band found on
    these is a supply band on the real bars, a stop under it is a stop above,
    and a target over it is a target below. Inverting twice gives the bars
    back. Volume and timestamps are untouched.
    """
    k = invert_k(bars)
    return [{**b, "o": k / b["o"], "h": k / b["l"], "l": k / b["h"], "c": k / b["c"]}
            for b in bars]


def mirror_order(o: Order, k: float) -> Order:
    """An order planned on inverted bars, as the SHORT it is on the real ones.

    The planner's own numbers (rr, risk_pct, atr_pct, stop_atr_x) are kept as
    it computed them -- they are what decided the trade -- and the band edges
    are mapped back so they read as real prices.
    """
    f = dict(o.features)
    if "zone_lo" in f and "zone_hi" in f:
        f["zone_lo"], f["zone_hi"] = k / f["zone_hi"], k / f["zone_lo"]
    if "target_lo" in f:
        f["target_near"] = k / f.pop("target_lo")    # the top edge of the demand band below
    return Order(o.coin, o.decided_i, o.kind, k / o.stop, k / o.tp,
                 limit=k / o.limit if o.kind == LIMIT else 0.0, ttl=o.ttl,
                 tag=o.tag + "_short", entry_fee=o.entry_fee, features=f, side=SHORT)


def run(coin: str, bars: List[Dict[str, Any]], costs: Costs = MEASURED,
        min_rr: float = MIN_RR, window: int = WINDOW, ttl: int = BID_TTL_BARS,
        cooldown_h: float = COIN_REENTRY_COOLDOWN_HOURS,
        cooldown_on_bids: bool = False,
        features: Optional[FeatureFn] = None,
        start_i: Optional[int] = None,
        gate: Optional[Callable[[Order], bool]] = None,
        side: str = LONG) -> List[Trade]:
    """Every trade the strategy takes on one coin, in time order.

    `gate`, when given, may refuse an order the strategy would have made --
    a candidate filter, applied where the live pass would apply it: at the
    decision, before anything is sent. A refused order leaves the coin free.

    side=SHORT trades the mirror: orders are planned on `invert(bars)` and
    simulated, mapped back, on `bars`. A `features` function then sees the
    inverted bars, since those are what the planner saw.
    """
    if side not in (LONG, SHORT):
        raise ValueError(f"unknown side {side!r}")
    plan_bars = invert(bars) if side == SHORT else bars
    k = invert_k(bars)
    trades: List[Trade] = []
    n = len(bars)
    i = max(Z.ZONE_MIN_BARS - 1, start_i or 0)
    cooled_until = -1               # first bar index a market entry may use
    while i < n - 1:
        o = order_at(coin, plan_bars, i, min_rr, window, ttl, features)
        if o is not None and side == SHORT:
            o = mirror_order(o, k)
        if o is None or (gate is not None and not gate(o)):
            i += 1
            continue
        on_cooldown = i < cooled_until and (o.kind == MARKET_CLOSE
                                            or cooldown_on_bids)
        if on_cooldown:
            i += 1
            continue
        tr = simulate(bars, o, costs)
        if tr is None:
            # A bid that never filled held the coin for its whole life.
            i += o.ttl if o.kind == LIMIT else 1
            continue
        trades.append(tr)
        if tr.exit_i is None:
            break                   # still open at the end of the data
        if tr.r_net <= 0 and cooldown_h > 0:
            cooled_until = tr.exit_i + int(round(cooldown_h))
        i = max(tr.exit_i, i + 1)
    return trades

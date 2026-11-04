"""Fills and exits on real bars, resolved conservatively.

A 1h bar says where price went, not in what order. Every rule below picks the
reading that is WORSE for the trade whenever the bar cannot say which came
first, because a backtest that resolves ambiguity in its own favour reports
an edge that only exists in the resolver.

Entry
  MARKET_CLOSE      fills at the decision bar's close (taker). The first bar
                    that can stop it out is the NEXT one -- the decision
                    bar's own high and low happened before the close, so
                    letting them resolve the trade would use price action
                    from before the entry. The previous scratch harness did
                    exactly that; it is the first thing this file fixes.
  MARKET_NEXT_OPEN  fills at the next bar's open (taker). That whole bar is
                    after the entry, so it can resolve the trade.
  LIMIT             a resting bid at `limit` for bars i+1 .. i+ttl (maker),
                    filled at the limit the first time a bar's low reaches
                    it. Production's PROTECT_BID_TTL_HOURS is 12.
  MARKET_AT         a market entry at a KNOWN price (`limit`) somewhere inside
                    bar i -- a live entry recorded mid-hour (taker). Bar i is
                    not used to resolve the exit: part of its range happened
                    before the entry, and the order within it is unknown.

Exit (stop and target are both trigger -> market orders: taker)
  - a bar that reaches the stop and the target is a LOSS;
  - on the bar a LIMIT fills, the stop counts (price reached the bid on its
    way down, so a low under the stop came after the fill) but the target
    does NOT: the high may well have printed before the dip that filled the
    bid. Those trades are flagged `fill_bar_tp` so the cost of this caution
    can be counted rather than guessed;
  - a bar that OPENS through the stop exits at that open, not at the stop;
  - stop exits pay the measured stop slippage on top (see costs.py).

R is measured against the PLANNED stop distance from the actual fill, net of
fees, so a gap through the stop reads as worse than -1R, as it is.

Shorts (side=SHORT) mirror every rule: the stop sits above the entry and the
target below; a resting sell fills at its limit the first time a bar's HIGH
reaches it; on that fill bar the stop counts and the target does not (the
low may have printed before the rally that filled it); a bar that opens
through the stop exits at that open; slippage makes every taker price worse
for the short -- a lower sell, a higher buy-back.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from research.backtest.costs import Costs, MEASURED

MARKET_CLOSE = "market_close"
MARKET_NEXT_OPEN = "market_next_open"
LIMIT = "limit"
MARKET_AT = "market_at"
LONG = "long"
SHORT = "short"


@dataclass(frozen=True)
class Order:
    coin: str
    decided_i: int          # the bar whose CLOSE the decision was made at
    kind: str
    stop: float
    tp: float
    limit: float = 0.0      # LIMIT only
    ttl: int = 12           # LIMIT only
    tag: str = ""
    # Charge the entry as maker/taker regardless of `kind`. Only the
    # matched-random controls use it, so a control pays the same entry fee as
    # the trade it is a control for.
    entry_fee: Optional[str] = None
    features: Mapping[str, Any] = field(default_factory=dict)
    side: str = LONG


@dataclass
class Trade:
    coin: str
    tag: str
    kind: str
    decided_i: int
    entry_i: int
    entry_t: int
    entry_px: float
    stop: float
    tp: float
    exit_i: Optional[int]   # None while still open at the end of the data
    exit_px: float
    outcome: str            # "tp" | "stop" | "open"
    r_gross: float
    cost_r: float
    r_net: float
    ambiguous: bool = False     # one bar reached both stop and target
    fill_bar_tp: bool = False   # target reached on the fill bar, not counted
    features: Dict[str, Any] = field(default_factory=dict)
    side: str = LONG

    @property
    def resolved(self) -> bool:
        return self.outcome != "open"

    @property
    def planned_rr(self) -> float:
        s = 1.0 if self.side == LONG else -1.0
        risk = s * (self.entry_px - self.stop)
        return s * (self.tp - self.entry_px) / risk if risk > 0 else 0.0

    @property
    def risk_pct(self) -> float:
        s = 1.0 if self.side == LONG else -1.0
        return s * (self.entry_px - self.stop) / self.entry_px * 100.0


def _fill(bars: List[Dict[str, Any]], o: Order, costs: Costs):
    """(entry_i, entry_px, entry_fee_pct, first_exit_bar) or None."""
    i, n = o.decided_i, len(bars)
    long = o.side == LONG
    slip = 1 + (1 if long else -1) * costs.market_entry_slip_pct / 100   # a worse price either way
    if o.kind == MARKET_CLOSE:
        px = bars[i]["c"] * slip
        fee = costs.taker_fee_pct
        entry_i, first = i, i + 1
    elif o.kind == MARKET_NEXT_OPEN:
        if i + 1 >= n:
            return None
        px = bars[i + 1]["o"] * slip
        fee = costs.taker_fee_pct
        entry_i, first = i + 1, i + 1
    elif o.kind == MARKET_AT:
        px = o.limit * slip
        fee = costs.taker_fee_pct
        entry_i, first = i, i + 1
    elif o.kind == LIMIT:
        for j in range(i + 1, min(i + 1 + o.ttl, n)):
            if (bars[j]["l"] <= o.limit) if long else (bars[j]["h"] >= o.limit):
                px, fee, entry_i, first = o.limit, costs.maker_fee_pct, j, j
                break
        else:
            return None
    else:
        raise ValueError(f"unknown order kind {o.kind!r}")
    if o.entry_fee == "maker":
        fee = costs.maker_fee_pct
    elif o.entry_fee == "taker":
        fee = costs.taker_fee_pct
    return entry_i, px, fee, first


def simulate(bars: List[Dict[str, Any]], o: Order,
             costs: Costs = MEASURED) -> Optional[Trade]:
    """The trade this order produces, or None if it never fills or is
    invalid at its fill price (entry not strictly between stop and target)."""
    got = _fill(bars, o, costs)
    if got is None:
        return None
    entry_i, entry_px, entry_fee, first = got
    long = o.side == LONG
    s = 1.0 if long else -1.0
    if not (s * (entry_px - o.stop) > 0 and s * (o.tp - entry_px) > 0):
        return None
    risk = s * (entry_px - o.stop)

    exit_i: Optional[int] = None
    exit_px = bars[-1]["c"]
    outcome = "open"
    ambiguous = fill_bar_tp = False
    for j in range(first, len(bars)):
        b = bars[j]
        hit_stop = b["l"] <= o.stop if long else b["h"] >= o.stop
        hit_tp = b["h"] >= o.tp if long else b["l"] <= o.tp
        if o.kind == LIMIT and j == entry_i:
            if hit_tp and not hit_stop:
                fill_bar_tp = True
            hit_tp = False
        if hit_stop:
            ambiguous = ambiguous or hit_tp
            raw = min(o.stop, b["o"]) if long else max(o.stop, b["o"])
            exit_px = raw * (1 - s * costs.stop_slip_pct / 100)
            outcome, exit_i = "stop", j
            break
        if hit_tp:
            raw = max(o.tp, b["o"]) if long else min(o.tp, b["o"])
            exit_px = raw * (1 - s * costs.tp_slip_pct / 100)
            outcome, exit_i = "tp", j
            break

    exit_fee = costs.taker_fee_pct if outcome != "open" else 0.0
    r_gross = s * (exit_px - entry_px) / risk
    cost_r = (entry_fee / 100 * entry_px + exit_fee / 100 * exit_px) / risk
    return Trade(coin=o.coin, tag=o.tag, kind=o.kind, decided_i=o.decided_i,
                 entry_i=entry_i, entry_t=int(bars[entry_i].get("t", 0)),
                 entry_px=entry_px, stop=o.stop, tp=o.tp, exit_i=exit_i,
                 exit_px=exit_px, outcome=outcome, r_gross=r_gross,
                 cost_r=cost_r, r_net=r_gross - cost_r, ambiguous=ambiguous,
                 fill_bar_tp=fill_bar_tp, features=dict(o.features), side=o.side)


def atr_pct_series(bars: List[Dict[str, Any]], n: int = 14) -> List[float]:
    """zones.atr_pct(bars[:t+1], n) for every t, in one pass.

    Same definition exactly -- the mean true range over the last min(n, t)
    bars as a percent of bar t's close -- and test_backtest_engine checks the
    two agree on real bars, so a stop computed here is the stop production
    would compute.
    """
    out: List[float] = []
    trs: List[float] = []
    window = 0.0
    for t, b in enumerate(bars):
        if t == 0:
            out.append(0.0)
            continue
        prev = bars[t - 1]
        tr = max(b["h"] - b["l"], abs(b["h"] - prev["c"]), abs(b["l"] - prev["c"]))
        trs.append(tr)
        window += tr
        if len(trs) > n:
            window -= trs[-n - 1]
        k = min(n, len(trs))
        out.append(window / k / b["c"] * 100.0 if b["c"] > 0 else 0.0)
    return out

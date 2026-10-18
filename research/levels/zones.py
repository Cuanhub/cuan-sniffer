"""Demand zones: the band an impulse launched from.

One idea, and the whole module is that idea:

    A demand zone is the base candle of an impulse that CLOSED ABOVE a prior
    swing high. The band runs from that candle's LOW to its CLOSE.

That is the whole definition. There is deliberately no expiry -- see
"Why a zone does not die" below, which is the one thing about this module
most likely to look like a missing feature.

The low is where the band ends, so it is also where the stop belongs. The
band has a top and a bottom, so an entry can sit INSIDE it rather than
above it.

Why this replaces levels.py's scoring
-------------------------------------
levels.py finds the PRICE a pivot sat at. That is a different object from the
band a move launched from, and the difference is what put real money in the
wrong place: on 2026-10-17 the trader bought ARB at 0.18482 while the operator's
band was 0.1705-0.1750 -- 5.6% above it. Across eight setups the operator
judged by hand, band position predicted every verdict: every skip was an entry
above the band, every take was inside it or 0.44% above. No feature levels.py
computes separated them at all (`level_hold_rate` separated takes from skips
by +0.019).

Fitted, not guessed
-------------------
The definition was chosen by reproducing seven bands the operator drew on his
own charts. Searching pivot window x zone-top definition, "close" beat "high",
"body", "open" and "mid":

    top=close   edge error 0.73%   mean width 2.97%   (operator's: 2.96%)
    top=body    edge error 1.65%   mean width 5.48%
    top=high    edge error 1.72%   mean width 5.77%

The width agreeing to 0.01pp is what says this found the same object rather
than something nearby. Pivot window is insensitive -- 2 and 3 are identical,
4 and 5 marginally worse -- so there is no fragile knob here.

Why a zone does not die
-----------------------
The first cut dropped any band price had since CLOSED below, on the reasoning
that a broken level is not a level. That rule was the single thing standing
between this module and the operator's own charts: it found 9 of his 16 bands.
Without it the detector finds 16 of 16, inside 2.5% on every edge and 0.97%
mean edge error -- and detection was never the problem.

Seven of his sixteen bands had closed below their low and he kept all seven.
The depths say the rule is not merely mistimed but wrong in kind:

    ETH 0.72%   VVV 1.18%   WLD 2.06%   AERO 3.09%
    MON 3.58%   NEAR 3.78%  APT 8.47%

He still marks APT as a buy band after price closed 8.47% under its low. No
tolerance threshold reproduces that; only having no threshold does. (A 3%
cutoff was measured and reached 12 of 16, and it also contradicts the 0.25%
stop buffer, which says a quarter of a percent below the low is already out.)

So bands persist, and what keeps a dead one from becoming a trade is not a
detection filter:
  - `live_zones` only offers bands price can presently reach, so a band price
    has collapsed through is out of scope until price returns to it;
  - the stop sits just under the band low, which is what the operator said it
    is for. A band that fails again fails for one stop.

tests/fixtures/zones/ holds those bands and the bars, and
test_zones.py keeps the definition pinned to them -- including a test that
asserts a band price closed under is STILL returned, so this cannot be
quietly reinstated as a bug fix.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from research.levels.levels import find_pivots

# A pivot needs this many bars either side. 2 and 3 give identical results
# against the operator's bands; the search is flat here on purpose.
ZONE_PIVOT_WINDOW = int(os.getenv("LEVELS_ZONE_PIVOT_WINDOW", "3"))
# How far under the band low the stop sits. The zone is invalidated by a close
# below the low, so the stop only needs to clear the noise of the touch
# itself -- not a volatility multiple.
ZONE_STOP_BUFFER_PCT = float(os.getenv("LEVELS_ZONE_STOP_BUFFER_PCT", "0.25"))
# Bands narrower than this are a single tight candle rather than a base worth
# trading; wider than this and the stop is too far for the position to size.
ZONE_MIN_WIDTH_PCT = float(os.getenv("LEVELS_ZONE_MIN_WIDTH_PCT", "0.3"))
ZONE_MAX_WIDTH_PCT = float(os.getenv("LEVELS_ZONE_MAX_WIDTH_PCT", "8.0"))
# A zone's base can be far back, and the detector only sees the bars it is
# given. Measured on the operator's seven: six bases sat 50-61 bars back, but
# HYPE's sat 740 back -- a band price had not closed under in 31 days, and the
# one he drew. Below the horizon the detector does not fail loudly, it finds a
# THINNER, more recent base instead: at 400 bars HYPE came out 77.44-77.675,
# 0.30% wide, which is a 0.40% stop that noise takes out before the idea is
# tested. So short history returns NOTHING rather than a band that happens to
# fit inside it.
#
# 800 bars of 1h is ~33 days. Raising this is cheap; a zone older than the
# window is simply invisible, and no guard can recover it.
ZONE_MIN_BARS = int(os.getenv("LEVELS_ZONE_MIN_BARS", "800"))
# How far below price a band can sit and still be this pass's business. Not a
# quality judgement -- a band further away is not worse, it is just not
# reachable yet, and the watchlist is where it waits.
ZONE_REACH_PCT = float(os.getenv("LEVELS_ZONE_REACH_PCT", "5.0"))
# How far back a band's base may sit and still be a candidate to TRADE.
#
# Detection keeps every band for ever (see "Why a zone does not die"), which
# leaves a median of 23 per coin over 60 days of 1h bars -- bands roughly 1%
# apart, so price is inside one essentially always and "inside a band" stops
# meaning anything. The operator draws one or two per chart.
#
# What separates the sixteen he drew from the other ~22 was measured: his sit
# at the 94th percentile of recency (11 of 16 in the top quartile) and the 8th
# percentile of touch count. Age is the single feature that separates them, so
# it is the single knob here. At 240 bars (10 days of 1h) the median drops
# from 23 bands to 4 and every band of his that price can still reach
# survives -- the four it loses were already out of reach.
ZONE_RECENT_BARS = int(os.getenv("LEVELS_ZONE_RECENT_BARS", "240"))


DEMAND = "demand"
SUPPLY = "supply"


@dataclass(frozen=True)
class Zone:
    """One band. `lo` and `hi` are the whole of it.

    `kind` says which side it is. A DEMAND band is a place to buy and carries
    `entry_px`/`stop_px`; a SUPPLY band is only ever read as a target here, so
    it has no entry of its own -- see `supply_zones`.
    """
    coin: str
    lo: float
    hi: float
    base_i: int          # bar the band is built on
    bos_i: int           # bar whose close broke the prior swing level
    kind: str = DEMAND

    @property
    def mid(self) -> float:
        return (self.lo + self.hi) / 2.0

    @property
    def width_pct(self) -> float:
        return (self.hi - self.lo) / self.lo * 100.0 if self.lo > 0 else 0.0

    def contains(self, px: float) -> bool:
        return self.lo <= px <= self.hi

    def entry_px(self, px_now: Optional[float] = None) -> float:
        """Where to buy. Never above the band, and never above the market.

        The operator's rule: "entry can be executed anywhere inside the band,
        middle to upper would be ok to make sure we dont miss the band
        fully". So:

          price INSIDE the band  -> that price. Already in the band, and a
            price in the lower half is a better fill than the midpoint, not a
            worse one. Taking it also keeps the order from ever being priced
            above the market, which a midpoint bid would be whenever price
            sits in the lower half.
          price ABOVE the band   -> the midpoint, as a resting bid. Middle
            rather than the low so a partial penetration still fills, which
            is the reason he gave; middle rather than the high because the
            high is the worst price in the band.

        What this CANNOT return is a price above `hi`. That is the whole
        point: on 2026-10-17 the trader bought ARB at 0.18482 at market while
        the band was 0.1705-0.175, 5.6% below. An entry that is always a band
        price makes that arithmetically impossible, which is a stronger
        guarantee than any filter that merely scores the setup.
        """
        if px_now is not None and self.lo <= px_now <= self.hi:
            return px_now
        return self.mid

    def stop_px(self, buffer_pct: float = ZONE_STOP_BUFFER_PCT) -> float:
        """Just under the band low -- the price that invalidates the zone.

        The definition supplies the stop. A close below the low is what kills
        the zone, so there is nothing to tune: no sigma multiple, no next
        support, no structural guess.
        """
        return self.lo * (1.0 - buffer_pct / 100.0)


@dataclass(frozen=True)
class Plan:
    """One concrete trade: a demand band to buy, and the supply band above it.

    Everything comes from the two bands. There is no scoring step and nothing
    to tune: the entry is a band price, the stop is just under the band low
    because that is what invalidates it, and the target is the near edge of
    the first supply band overhead.
    """
    zone: Zone
    target: Zone
    entry: float
    stop: float
    tp: float
    immediate: bool      # price is in the band now, rather than above it

    @property
    def coin(self) -> str:
        return self.zone.coin

    @property
    def risk_pct(self) -> float:
        return ((self.entry - self.stop) / self.entry * 100.0
                if self.entry > 0 else 0.0)

    @property
    def rr(self) -> float:
        risk = self.entry - self.stop
        return (self.tp - self.entry) / risk if risk > 0 else 0.0


def plans_for(bars: List[Dict[str, Any]], px: float, coin: str = "",
              window: int = ZONE_PIVOT_WINDOW,
              recent_bars: Optional[int] = None,
              min_bars: Optional[int] = None) -> List[Plan]:
    """Tradeable plans for one coin, most recent band first.

    Three things decide a plan, and none of them is a score:

      1. the band must be RECENT (ZONE_RECENT_BARS). Age is the one feature
         measured to separate the bands the operator draws from the ~22 others
         per coin -- his sit at the 94th percentile of recency.
      2. price must be able to REACH it (`live_zones`): inside the band, or
         above it by no more than ZONE_REACH_PCT. A band above price is not a
         buy, and a band far below is not this pass's business.
      3. there must be a TARGET above the entry, or there is no tp and no
         reward/risk and so no plan, however good the band.

    Ordered by the band's own recency, NOT by reward/risk. Ordering by rr was
    tried and it systematically picks the DEEPEST band in reach, because a
    deeper entry sits further below the same target -- the mirror of the
    shallow-band bias that `nearest_below` had. Recency is the ordering the
    evidence supports: among the bands price can reach, picking the most
    recent agrees with the operator's own choice on 9 of 12, against 6 of 12
    for the deepest and 4 of 12 for the nearest.

    What this does NOT claim
    ------------------------
    It does not reproduce his take/skip calls. Scored on the eight he judged
    by hand -- "never offer an entry above the band he drew" -- no setting of
    the recency window or the width floor beat 5 of 8, so no setting is
    presented as if it had. Which genuine band to prefer is a judgement those
    eight labels cannot settle.

    The protection is structural instead of statistical: every entry here is a
    price INSIDE a real band, with its stop under that band's low. The failure
    on 2026-10-17 was an entry 5.6% ABOVE the band at market, and `entry_px`
    cannot return such a price.
    """
    px = float(px)
    if px <= 0:
        return []
    horizon = ZONE_RECENT_BARS if recent_bars is None else recent_bars
    zs = demand_zones(bars, coin=coin, window=window, min_bars=min_bars)
    n = len(bars)
    if horizon > 0:
        zs = [z for z in zs if (n - z.base_i) <= horizon]
    demand = live_zones(zs, px)
    if not demand:
        return []
    supply = supply_zones(bars, coin=coin, window=window, min_bars=min_bars)
    out: List[Plan] = []
    for z in demand:
        if px < z.lo:
            continue                   # band is above price: not a buy
        entry = z.entry_px(px_now=px)
        stop = z.stop_px()
        if not entry > stop > 0:
            continue
        # Measured from the MARKET, not from the entry. A supply band below
        # the current price has already been traded through, so it is not
        # resistance any more -- and taking it as the target crushes the
        # reward on exactly the trades worth having. WLD scored rr 0.08 that
        # way: a bid at 0.46969 against a "target" of 0.4705 while price was
        # 0.49138, already above it.
        t = target_above(supply, max(entry, px))
        if t is None:
            continue                   # nothing overhead: no target, no plan
        out.append(Plan(zone=z, target=t, entry=entry, stop=stop, tp=t.lo,
                        immediate=z.contains(px)))
    out.sort(key=lambda pl: -pl.zone.base_i)
    return out


def demand_zones(bars: List[Dict[str, Any]], coin: str = "",
                 window: int = ZONE_PIVOT_WINDOW,
                 min_bars: Optional[int] = None) -> List[Zone]:
    """Every demand zone in `bars`, oldest first.

    All of them. There is no aliveness filter and that is not an omission:
    dropping bands price had closed below found 9 of the operator's 16, and
    keeping them finds 16 of 16. He himself keeps a band price closed 8.47%
    under. See "Why a zone does not die" in the module docstring, and
    `test_a_band_price_closed_under_is_still_a_band`.

    Reachability is `live_zones`'s job, not this function's.
    """
    floor = ZONE_MIN_BARS if min_bars is None else min_bars
    if len(bars) < max(floor, window * 2 + 2):
        return []
    highs, _ = find_pivots(bars, window)
    out: List[Zone] = []
    for hi_i, hi_px in highs:
        bos = _first_close_above(bars, hi_i + 1, hi_px)
        if bos is None:
            continue
        base = min(range(hi_i, bos + 1), key=lambda k: bars[k]["l"])
        lo, hi = float(bars[base]["l"]), float(bars[base]["c"])
        if hi <= lo:
            continue                       # base closed at its low: no band
        z = Zone(coin=coin.upper(), lo=lo, hi=hi, base_i=base, bos_i=bos)
        if not ZONE_MIN_WIDTH_PCT <= z.width_pct <= ZONE_MAX_WIDTH_PCT:
            continue
        out.append(z)
    out.sort(key=lambda z: z.base_i)
    return _merge_nested(_dedupe(out))


def supply_zones(bars: List[Dict[str, Any]], coin: str = "",
                 window: int = ZONE_PIVOT_WINDOW,
                 min_bars: Optional[int] = None) -> List[Zone]:
    """Every supply band in `bars`, oldest first. The mirror of demand.

    A supply zone is the base candle of an impulse that CLOSED BELOW a prior
    swing LOW. The band runs from that candle's CLOSE up to its HIGH -- the
    reflection of demand's low-to-close.

    NOT FITTED, and that is deliberate rather than an omission: the operator
    drew demand bands only, so there is nothing to fit supply against. What
    makes it safe to use anyway is how narrow its job is. A supply band is
    never entered. It is read once, as the target above a demand band, and it
    only has to be roughly where resistance is for the reward/risk number to
    mean something. An entry band has to be right to the edge because money
    is placed on its low; a target band that is 1% out moves rr by about a
    tenth.

    If supply bands ever become something to SELL, they need the same
    treatment demand got -- bands the operator drew, and a fit measured
    against them -- and this docstring is the warning that no such evidence
    exists yet.
    """
    floor = ZONE_MIN_BARS if min_bars is None else min_bars
    if len(bars) < max(floor, window * 2 + 2):
        return []
    _, lows = find_pivots(bars, window)
    out: List[Zone] = []
    for lo_i, lo_px in lows:
        bos = _first_close_below(bars, lo_i + 1, lo_px)
        if bos is None:
            continue
        base = max(range(lo_i, bos + 1), key=lambda k: bars[k]["h"])
        lo, hi = float(bars[base]["c"]), float(bars[base]["h"])
        if hi <= lo:
            continue                       # base closed at its high: no band
        z = Zone(coin=coin.upper(), lo=lo, hi=hi, base_i=base, bos_i=bos,
                 kind=SUPPLY)
        if not ZONE_MIN_WIDTH_PCT <= z.width_pct <= ZONE_MAX_WIDTH_PCT:
            continue
        out.append(z)
    out.sort(key=lambda z: z.base_i)
    return _merge_nested(_dedupe(out))


def target_above(supplies: List[Zone], px: float) -> Optional[Zone]:
    """The nearest supply band wholly above `px`, or None.

    Its LOW is the target, not its middle: the band is where sellers are, so
    the trade takes profit as price arrives at the near edge rather than
    hoping to cut through it. Nearest rather than best, because a target
    further away is not more likely to be reached -- the first band overhead
    is the one that decides whether the move gets there at all.
    """
    above = [z for z in supplies if z.lo > px]
    return min(above, key=lambda z: z.lo) if above else None


def live_zones(zones: List[Zone], px: float,
               max_above_pct: float = ZONE_REACH_PCT) -> List[Zone]:
    """Every band price could reasonably reach, nearest first.

    No choosing. A coin with three live bands under it offers three candidate
    trades, each with its own entry, stop and -- once a target is attached --
    its own reward/risk. Which one is worth taking is a question the rr gate
    can answer from numbers; which one is NEAREST is not the same question,
    and answering the second while pretending to answer the first is what put
    money 5.6% above the band it meant to buy.

    This replaced a `nearest_below` that returned the single highest band
    under price. Scored against the sixteen bands the operator drew by hand,
    that rule agreed with ten -- and it is biased by construction, because
    price approaches support from above, so the nearest base is always the
    shallowest one. It is deleted rather than deprecated: a function with a
    warning in its docstring is still a function somebody can call.

    A band price is already inside comes first; the rest follow by distance.
    Bands more than `max_above_pct` below price are dropped -- not because
    they are wrong, but because price has to travel too far for them to be
    this pass's business.
    """
    out = [z for z in zones
           if z.contains(px) or (z.hi < px <= z.hi * (1 + max_above_pct / 100.0))
           or z.lo <= px <= z.hi]
    reach = [z for z in zones
             if z.hi <= px and (px - z.hi) / z.hi * 100.0 <= max_above_pct]
    seen = {id(z) for z in out}
    out += [z for z in reach if id(z) not in seen]
    return sorted(out, key=lambda z: (not z.contains(px), abs(px - z.hi)))


def _first_close_above(bars: List[Dict[str, Any]], start: int,
                       level: float) -> Optional[int]:
    """Break of structure: the first bar to CLOSE above `level`.

    The close, not the wick. A wick through a swing high is the failure this
    is meant to exclude -- it is what a sweep looks like, not what a break
    looks like.
    """
    for j in range(start, len(bars)):
        if bars[j]["c"] > level:
            return j
    return None


def _first_close_below(bars: List[Dict[str, Any]], start: int,
                       level: float) -> Optional[int]:
    """Break of structure, downward: the first bar to CLOSE below `level`.

    The close, not the wick, for the same reason as `_first_close_above`: a
    wick under a swing low is a sweep, not a break.
    """
    for j in range(start, len(bars)):
        if bars[j]["c"] < level:
            return j
    return None


def _merge_nested(zones: List[Zone]) -> List[Zone]:
    """Drop a band that sits wholly INSIDE another.

    A tight base can sit inside the wider base of the same move. Both are
    genuine impulse bases, but they are one zone, and taking the inner one
    puts the stop inside the band instead of under it -- the one thing a stop
    must never be.

    Containment, not overlap. Merging anything that merely touches chains
    adjacent bands into one runaway zone: the first cut of this turned HYPE's
    77.03-78.63 into 75.15-78.63, a 4.6% band spanning two separate bases.
    Two bands that only overlap at the edges are two bands.
    """
    out: List[Zone] = []
    for z in sorted(zones, key=lambda z: (z.lo - z.hi)):      # widest first
        if any(o.lo <= z.lo and z.hi <= o.hi for o in out):
            continue
        out.append(z)
    return sorted(out, key=lambda z: z.base_i)


def _dedupe(zones: List[Zone]) -> List[Zone]:
    """Collapse zones built on the same base bar.

    Several swing highs can be broken by one impulse, which yields the same
    band several times over. This is the same duplication that gave ARB
    seventeen "levels" where there was one -- caught at the source here
    rather than patched downstream.
    """
    seen: Dict[int, Zone] = {}
    for z in zones:
        prev = seen.get(z.base_i)
        if prev is None or z.bos_i < prev.bos_i:
            seen[z.base_i] = z
    return [seen[k] for k in sorted(seen)]

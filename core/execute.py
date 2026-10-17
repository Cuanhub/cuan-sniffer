"""Place an order, then look at the venue to see what happened.

Rule 2. There is no response parsing here and no `FillResult`: a fill is a
CHANGE IN THE VENUE'S POSITION, measured by reading before and after. That makes
the four cases that used to need separate handling into one case.

  - full fill, partial fill, zero fill: the delta says which.
  - timeout after the venue filled: the delta still says so. The request failing
    tells us nothing about whether the order landed, which is what REVIEW.md
    finding 1 was.
  - a venue that cannot be read afterwards: `Outcome.unknown`, which is neither
    a fill nor a reject, and the caller must not treat it as either.

Entries come two ways: `enter` sends a marketable IOC, and `rest_bid` rests a
GTC bid at a level. The second is kept because the evidence behind it is
one-sided (see `rest_bid`); what was dangerous about it was the `pending`
bookkeeping, not the order, and there is none of that here.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

from core.venue import Order, Venue

# How long to wait after sending before asking the venue what it holds. An IOC
# either fills or is cancelled immediately, so this is propagation time, not
# order lifetime.
SETTLE_SEC = float(os.getenv("EXEC_SETTLE_SEC", "1.5"))
# Marketable band for an IOC. Wide enough to cross the spread, bounded so a
# thin book cannot fill us arbitrarily far from the mid.
IOC_BPS = float(os.getenv("EXEC_IOC_BPS", "25"))
EXIT_IOC_BPS = float(os.getenv("EXEC_EXIT_IOC_BPS", "60"))


@dataclass(frozen=True)
class Outcome:
    """What the venue did, measured rather than parsed."""
    coin: str
    filled_size: float          # coin units, always >= 0
    avg_px: float               # venue entryPx for an entry; mark for an exit
    unknown: bool = False       # the venue could not be read afterwards
    note: str = ""

    @property
    def filled(self) -> bool:
        return self.filled_size > 0

    @property
    def notional(self) -> float:
        return self.filled_size * self.avg_px


def _size_of(venue: Venue, coin: str, urgent: bool = False) -> Optional[float]:
    pos = venue.positions(urgent=urgent)
    if pos is None:
        return None
    p = pos.get(coin.upper())
    return p.szi if p else 0.0


def enter(venue: Venue, coin: str, usd: float, is_long: bool = True) -> Outcome:
    """Open or add to a position with one IOC. Reports the observed delta."""
    coin = coin.upper()
    if usd <= 0:
        return Outcome(coin, 0.0, 0.0, note="non_positive_size")
    if not venue.load_meta():
        # Without szDecimals every size and price is rounded to a guessed
        # precision. Refusing is the only safe answer.
        return Outcome(coin, 0.0, 0.0, note="no_asset_metadata")

    mid = venue.mid(coin)
    if not mid or mid <= 0:
        return Outcome(coin, 0.0, 0.0, note="no_price")

    before = _size_of(venue, coin)
    if before is None:
        # Cannot establish the starting point, so a delta would be meaningless.
        return Outcome(coin, 0.0, 0.0, note="venue_unreadable_before_send")

    size = venue.round_size(coin, usd / mid)
    if size <= 0:
        return Outcome(coin, 0.0, 0.0, note="size_rounds_to_zero")
    limit = venue.round_price(
        coin, mid * (1 + IOC_BPS / 10_000) if is_long
        else mid * (1 - IOC_BPS / 10_000),
        direction="up" if is_long else "down")
    if limit <= 0:
        return Outcome(coin, 0.0, 0.0, note="price_not_representable")

    venue.send_ioc(coin, is_buy=is_long, size=size, limit_px=limit,
                   reduce_only=False)
    return _observe(venue, coin, before, "entry")


def close(venue: Venue, coin: str, size: Optional[float] = None) -> Outcome:
    """Reduce a position with one IOC. `size` None means all of it.

    Side and size come from the VENUE, never from a caller's belief, so this
    works on a position nothing in this process opened and cannot send an order
    that would ADD to one. The old code took `side="LONG"` from its caller at
    every emergency call site.
    """
    coin = coin.upper()
    pos = venue.positions(urgent=True)
    if pos is None:
        return Outcome(coin, 0.0, 0.0, unknown=True,
                       note="venue_unreadable_before_send")
    held = pos.get(coin)
    if held is None:
        return Outcome(coin, 0.0, 0.0, note="already_flat")

    want = venue.round_size(coin, min(abs(size), held.size) if size is not None
                            else held.size)
    if want <= 0:
        return Outcome(coin, 0.0, 0.0, note="size_rounds_to_zero")

    # Price from the position's own mark, so a throttled or dead price feed
    # cannot stop an exit. REVIEW.md finding 10.
    mark = held.mark_px or venue.mid(coin) or 0.0
    if mark <= 0:
        return Outcome(coin, 0.0, 0.0, note="no_price")
    close_is_buy = not held.is_long
    limit = venue.round_price(
        coin, mark * (1 + EXIT_IOC_BPS / 10_000) if close_is_buy
        else mark * (1 - EXIT_IOC_BPS / 10_000),
        direction="up" if close_is_buy else "down")
    if limit <= 0:
        return Outcome(coin, 0.0, 0.0, note="price_not_representable")

    venue.send_ioc(coin, is_buy=close_is_buy, size=want, limit_px=limit,
                   reduce_only=True)
    return _observe(venue, coin, held.szi, "exit", px_hint=mark)


def _observe(venue: Venue, coin: str, before: float, what: str,
             px_hint: float = 0.0) -> Outcome:
    """The only thing that decides whether an order filled.

    `px_hint` is the price the order was sent against, and it is what a FULL
    exit reports: once the position is gone the venue has no entryPx or mark
    left to read, so without it a complete close would report a notional of
    zero. It is a reported price, never a decision input.
    """
    time.sleep(SETTLE_SEC)
    pos = venue.positions(urgent=True)
    if pos is None:
        # An order was sent and we cannot see the result. Saying "rejected"
        # here is what left a live, unprotected position untracked for a full
        # poll interval; saying "filled" would invent one.
        print(f"[EXEC] {what} {coin}: venue unreadable after send -- UNKNOWN")
        return Outcome(coin, 0.0, 0.0, unknown=True, note="unreadable_after_send")
    held = pos.get(coin)
    after = held.szi if held else 0.0
    delta = abs(after - before)
    if held is None:
        px = px_hint
    elif what == "entry":
        px = held.entry_px
    else:
        px = held.mark_px or px_hint
    if delta <= 0:
        print(f"[EXEC] {what} {coin}: position unchanged at {before:g} -- no fill")
        return Outcome(coin, 0.0, px, note="no_fill")
    print(f"[EXEC] {what} {coin}: {before:g} -> {after:g} "
          f"(filled {delta:g} @ {px:g})")
    return Outcome(coin, delta, px, note=what)


def rest_bid(venue: Venue, coin: str, usd: float, px: float,
             is_long: bool = True) -> Optional[Order]:
    """Rest a GTC bid at `px`. Returns the resting order, or None.

    Kept, not removed, because the evidence is one-sided: across 139 tradeable
    watched levels on 2026-09-19 not one was inside the 1.5% entry drift guard
    and the nearest was 3.10% away -- every tap resolved between polls. A
    marketable IOC cannot catch a level that taps and leaves; a bid sitting at
    the level is filled BY the tap.

    What made this dangerous before was not the order, it was the bookkeeping:
    a `pending` dict held the oid and the plan, and a partial fill popped the
    dict while the remainder stayed live at the venue, unprotected and
    untracked. There is no `pending` here. The bid is visible in open_orders,
    the position is visible in positions, and the protection loop sizes the
    stop to whatever the venue currently holds -- so a partial fill is just a
    smaller position with a correctly sized stop, and the remainder filling
    later is just a bigger one.

    Confirmed by reading the book back, like every other order here.
    """
    coin = coin.upper()
    if usd <= 0 or px <= 0:
        return None
    if not venue.load_meta():
        return None
    limit = venue.round_price(coin, px, direction="down" if is_long else "up")
    if limit <= 0:
        return None
    size = venue.round_size(coin, usd / limit)
    if size <= 0:
        return None

    before = {o.oid for o in (venue.open_orders(coin) or [])}
    venue.send_limit(coin, is_buy=is_long, size=size, limit_px=limit)
    after = venue.open_orders(coin)
    if after is None:
        print(f"[EXEC] bid {coin}: sent but the book could not be read")
        return None
    fresh = [o for o in after if o.oid not in before and o.is_entry]
    if not fresh:
        print(f"[EXEC] bid {coin}: nothing rested at {limit:g}")
        return None
    print(f"[EXEC] bid {coin}: resting {size:g} @ {limit:g}")
    return fresh[0]

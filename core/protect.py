"""Every venue position carries exactly one stop, sized to the whole position.

Runs every 60 seconds, independent of the signal loop. That independence is the
point: protection that is only asserted at entry is protection that is wrong
the moment the position changes size, and the old code had no path at all for a
position that had GROWN (REVIEW.md finding 6).

Three things make this much smaller than what it replaces:

  - The stop's size comes from the venue's current position size, every cycle.
    There is no stored size to drift and no "re-protect after a partial close"
    special case; a changed position is just this cycle's reading.
  - Order ids are re-read from the venue every cycle and never stored. The old
    code never persisted them at all, so its only cancel call site was a
    permanent no-op (finding 7).
  - There is no hard/soft verification taxonomy. A loop that replaces whatever
    is wrong within 60s does not need to grade how wrong it was.

Replacement order is place-then-cancel, never the reverse: the new stop must be
resting at the venue before the old one is cancelled, so no sequence of failures
can leave a live position with no stop at all.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import notify
from core import safety
from core.meta import MetaStore, TradeMeta, now_iso
from core.venue import Order, Position, Venue

# A stop within these tolerances of what we want is left alone. Without a
# tolerance the loop would replace every stop every cycle on float noise.
SIZE_TOLERANCE = float(os.getenv("PROTECT_SIZE_TOLERANCE", "0.01"))    # 1%
PRICE_TOLERANCE = float(os.getenv("PROTECT_PRICE_TOLERANCE", "0.002"))  # 0.2%
# The stop given to a position with no recorded plan -- a C-1 orphan, or one
# whose metadata was lost. Distance from the current mark, per rule 1.
MAX_RISK_PCT = float(os.getenv("PROTECT_MAX_RISK_PCT", "3.0"))
MAX_PLACE_ATTEMPTS = 2


@dataclass
class Result:
    coin: str
    action: str          # "ok" | "placed" | "replaced" | "flattened" | "unknown"
    detail: str = ""

    @property
    def protected(self) -> bool:
        return self.action in ("ok", "placed", "replaced")


def desired_stop(venue: Venue, pos: Position, meta: Optional[TradeMeta]
                 ) -> tuple[float, str]:
    """Where this position's stop belongs, and where that came from.

    A planned stop is used only if it is on the correct side of the mark: a
    stop at or above the mark on a long is not a stop, it is a market order
    waiting to fire, and a stale plan after a big move is exactly how one
    arrives. Anything unusable falls back to MAX_RISK from the mark.
    """
    mark = pos.mark_px
    if mark <= 0:
        return 0.0, "no mark price"

    def usable(px: float) -> bool:
        """A stop must be on the losing side of the mark after ROUNDING.

        Checked on the rounded price, not the requested one. Rounding a long's
        stop UP is correct -- it triggers slightly earlier -- but at coarse
        precision it can cross the mark, and a stop above the mark on a long is
        not a stop, it is a market order waiting to fire. Found by
        test_a_stop_price_that_cannot_be_represented_is_reported_not_guessed:
        a 3% fallback below a mark of 1e-9 rounds up to 0.1 on BTC.
        """
        return px > 0 and ((px < mark) if pos.is_long else (px > mark))

    direction = "up" if pos.is_long else "down"
    planned = meta.planned_stop if meta else 0.0
    if planned > 0:
        px = venue.round_price(pos.coin, planned, direction=direction)
        if usable(px):
            return px, "planned"

    risk = MAX_RISK_PCT / 100.0
    fallback = mark * (1 - risk) if pos.is_long else mark * (1 + risk)
    px = venue.round_price(pos.coin, fallback, direction=direction)
    why = "default" if planned <= 0 else "planned stop unusable"
    if not usable(px):
        return 0.0, (f"{why}, and a {MAX_RISK_PCT:g}% stop from {mark:g} "
                     f"has no valid representation for {pos.coin}")
    return px, f"{why} ({MAX_RISK_PCT:g}% from {mark:g})"


def _matches(order: Order, size: float, px: float) -> bool:
    if order.trigger_px is None or order.size <= 0:
        return False
    return (abs(order.size - size) <= max(size * SIZE_TOLERANCE, 1e-12)
            and abs(order.trigger_px - px) <= max(px * PRICE_TOLERANCE, 1e-12))


def ensure_stop(venue: Venue, pos: Position, meta: Optional[TradeMeta]
                ) -> Result:
    """Guarantee exactly one correctly sized stop on `pos`."""
    coin = pos.coin
    orders = venue.open_orders(coin)
    if orders is None:
        # Cannot see the book, so cannot know whether a stop is resting.
        # Replacing blind could double up; calling it unprotected would
        # flatten a protected position. Report and wait for the next cycle.
        safety.record_error("open_orders", coin)
        return Result(coin, "unknown", "order book unreadable")

    want_px, why = desired_stop(venue, pos, meta)
    if want_px <= 0:
        # No stop price exists for this position, so it can never be
        # protected. Rule 1: a position that cannot be given a default stop is
        # flattened. Carrying it unstopped while alerting every cycle is worse.
        return _give_up(venue, coin, f"no usable stop price -- {why}")

    stops = [o for o in orders if o.is_protective and o.is_stop]
    good = [o for o in stops if _matches(o, pos.size, want_px)]
    if len(stops) == 1 and good:
        return Result(coin, "ok", f"stop {want_px:g} ({why})")

    placed = _place_stop(venue, pos, want_px)
    if placed is None:
        return _give_up(venue, coin,
                        f"no stop would rest after {MAX_PLACE_ATTEMPTS} attempts")

    # Only now that a correct stop is resting does the old one go. The reverse
    # order leaves a window -- and a failure in that window leaves a live
    # position with no stop at all.
    stale = [o for o in stops if o.oid != placed.oid]
    for o in stale:
        resp = venue.cancel(coin, o.oid)
        ok = isinstance(resp, dict) and str(resp.get("status")) == "ok"
        if not ok:
            # Harmless in itself: two reduce-only stops cannot open anything,
            # and the venue cancels both when the position closes. Worth
            # knowing about, not worth closing a protected position over.
            safety.record_error("cancel_stale_stop", f"{coin} oid={o.oid}")
    action = "placed" if not stale else "replaced"
    print(f"[PROTECT] {coin}: {action} stop at {want_px:g} for {pos.size:g} "
          f"({why}){f', cancelled {len(stale)} stale' if stale else ''}")
    return Result(coin, action, f"stop {want_px:g} ({why})")


def _give_up(venue: Venue, coin: str, why: str) -> Result:
    """Close a position that cannot be protected. An unprotected position is
    not the trade.
    """
    print(f"[PROTECT] !! {coin}: {why} -- flattening")
    res = safety.flatten_all(venue, reason=f"{coin}: {why}")
    notify.send(
        f"🚨 *{coin} could not be protected*\n{why}, so the position was "
        f"closed.\n\nFlat: `{res.flat}`  closed: `{res.closed or '{}'}`",
        key=f"protect_failed_{coin}", cooldown_sec=900)
    return Result(coin, "flattened", f"{why}; {res.detail}")


def _place_stop(venue: Venue, pos: Position, px: float) -> Optional[Order]:
    """Place a stop and confirm it by reading the book, not by parsing a
    response. Returns the resting order, or None after MAX_PLACE_ATTEMPTS.
    """
    coin = pos.coin
    size = venue.round_size(coin, pos.size)
    if size <= 0:
        return None
    for attempt in range(1, MAX_PLACE_ATTEMPTS + 1):
        before = {o.oid for o in (venue.open_orders(coin) or [])}
        venue.send_trigger(coin, is_buy=not pos.is_long, size=size,
                           trigger_px=px, is_stop=True)
        after = venue.open_orders(coin)
        if after is None:
            safety.record_error("confirm_stop", f"{coin} attempt {attempt}")
            continue
        fresh = [o for o in after
                 if o.oid not in before and o.is_protective and o.is_stop]
        if fresh:
            return fresh[0]
        safety.record_error("place_stop",
                            f"{coin} attempt {attempt}: nothing rested")
    return None


def ensure_tp(venue: Venue, pos: Position, meta: Optional[TradeMeta]) -> None:
    """Keep a take-profit if one was planned. Best effort by design: a missing
    TP costs upside, not safety, so it never flattens and never blocks.
    """
    planned = meta.planned_tp if meta else 0.0
    if planned <= 0 or pos.mark_px <= 0:
        return
    if (pos.is_long and planned <= pos.mark_px) or \
            (not pos.is_long and planned >= pos.mark_px):
        return                                  # already through it
    px = venue.round_price(pos.coin, planned, direction="nearest")
    if px <= 0:
        return
    orders = venue.open_orders(pos.coin)
    if orders is None:
        return
    tps = [o for o in orders if o.is_protective and not o.is_stop]
    if len(tps) == 1 and _matches(tps[0], pos.size, px):
        return
    size = venue.round_size(pos.coin, pos.size)
    if size <= 0:
        return
    before = {o.oid for o in tps}
    venue.send_trigger(pos.coin, is_buy=not pos.is_long, size=size,
                       trigger_px=px, is_stop=False)
    for o in [t for t in tps if t.oid in before]:
        venue.cancel(pos.coin, o.oid)
    print(f"[PROTECT] {pos.coin}: tp at {px:g} for {size:g}")


def run_once(venue: Venue, meta: Optional[MetaStore] = None) -> List[Result]:
    """One protection cycle over every position the venue reports."""
    store = meta or MetaStore()
    positions = venue.positions()
    if positions is None:
        safety.record_error("positions", "venue unreadable")
        print("[PROTECT] venue unreadable this cycle")
        return []
    if not positions:
        print("[PROTECT] flat")
        _forget_closed(store, set())
        return []

    results: List[Result] = []
    for coin in sorted(positions):
        pos = positions[coin]
        row = store.get(coin)
        if row is None:
            # A position with no plan: a timeout-after-fill orphan, or one
            # whose metadata was lost to a restart. It gets a default stop
            # rather than a halt, and is recorded so the next cycle is quiet.
            print(f"[PROTECT] {coin}: no metadata -- adopting with a "
                  f"{MAX_RISK_PCT:g}% stop")
            notify.send(
                f"⚠️ *{coin} has no recorded plan*\nThe venue holds "
                f"{pos.size:g} that this system did not record. Adopting it "
                f"with a {MAX_RISK_PCT:g}% stop from {pos.mark_px:g}.",
                key=f"adopted_{coin}", cooldown_sec=3600)
        res = ensure_stop(venue, pos, row)
        results.append(res)
        if res.protected:
            if row is None:
                px, _ = desired_stop(venue, pos, None)
                store.put(coin, TradeMeta(reason="adopted", planned_stop=px,
                                          opened_at=now_iso()))
            else:
                ensure_tp(venue, pos, row)
        elif res.action == "unknown":
            notify.send(
                f"⚠️ *{coin}: protection could not be verified*\n{res.detail}\n"
                f"The venue holds {pos.size:g}. Retrying next cycle.",
                key=f"protect_unknown_{coin}", cooldown_sec=1800)

    _forget_closed(store, set(positions))
    return results


def _forget_closed(store: MetaStore, live: set[str]) -> None:
    """Drop metadata for coins the venue no longer holds.

    Metadata is not a position record, so this is tidying, not reconciliation
    -- nothing decides anything from its presence. It exists so an adopted
    position's stop does not get reused on a later, unrelated trade.
    """
    for coin in sorted(store.coins() - live):
        store.drop(coin)
        print(f"[PROTECT] {coin} is gone at the venue -- metadata dropped")

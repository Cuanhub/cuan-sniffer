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
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

import notify
from core import books, safety
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
# A stop the venue accepted can take seconds to appear in frontendOpenOrders.
# Reading once, immediately, is what closed the whole book on 2026-11-05 with
# correct stops resting: see _place_stop.
CONFIRM_READS = 4
CONFIRM_WAIT_SEC = float(os.getenv("PROTECT_CONFIRM_WAIT_SEC", "1.5"))
# A plan younger than this survives even with nothing visible behind it. The
# trader writes the plan BEFORE sending, so a young plan with no position and
# no bid usually means the venue's read has not caught up with the order.
PLAN_GRACE_SEC = float(os.getenv("PROTECT_PLAN_GRACE_SEC", "600"))
# How long a resting entry bid may sit before it is pulled. A level that has
# not been tapped in this long is no longer the setup that justified the bid.
BID_TTL_HOURS = float(os.getenv("PROTECT_BID_TTL_HOURS", "12"))


@dataclass
class Result:
    coin: str
    # "ok" | "placed" | "replaced" | "flattened" | "unknown", or for the trend
    # book's positions "trend" / "trend_unstopped" (core/books.py)
    action: str
    detail: str = ""

    @property
    def protected(self) -> bool:
        # "trend" is not this loop's stop, but nothing here is owed: the trend
        # pass keeps it. Counted, it would be logged as a failure every pass.
        return self.action in ("ok", "placed", "replaced", "trend")


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

    # UNKNOWN precision is not an unprotectable position. Without this the
    # rounders' refusal reaches desired_stop as "no valid representation",
    # ensure_stop reads that as unprotectable, and _give_up tries to flatten
    # -- which ALSO needs round_size, so the flatten fails too. The observed
    # result was an alert saying "could not be protected, so the position was
    # closed" with Flat: False and the position still open. A lie, a failed
    # exit and a false alarm, for a metadata fetch that will probably succeed
    # on the next cycle.
    #
    # So it lands where the unreadable order book lands: report, wait, retry.
    # precision_known retries the metadata load itself, and the restart
    # watchdog is the backstop if it never comes back.
    if not venue.precision_known(coin):
        safety.record_error("precision", f"{coin} szDecimals unknown")
        return Result(coin, "unknown",
                      f"{coin} szDecimals unknown -- cannot price a stop, "
                      f"retrying next cycle")

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
    if good:
        # A correct stop is already resting, so tidy up around it. This used
        # to place yet another stop to replace the duplicates, and a failed
        # read of that one then flattened a position holding two good stops.
        keep = good[0]
        cancelled = _cancel_stale(venue, coin,
                                  [o for o in stops if o.oid != keep.oid])
        print(f"[PROTECT] {coin}: kept stop at {want_px:g}, cancelled "
              f"{cancelled} other(s)")
        return Result(coin, "replaced", f"stop {want_px:g} ({why})")

    placed, book_read = _place_stop(venue, pos, want_px, {o.oid for o in orders})
    if placed is None and not book_read:
        # Same answer as the unreadable book above: an order may be resting
        # and nothing here can tell.
        return Result(coin, "unknown",
                      "stop sent, order book unreadable afterwards")
    if placed is None and any(o.size >= pos.size * (1 - SIZE_TOLERANCE)
                              for o in stops):
        # The new stop did not appear, but the old one still covers the whole
        # position. Wrong price, not no protection: nothing here justifies
        # closing it, and the next cycle tries again.
        return Result(coin, "unknown",
                      "replacement stop not visible, existing stop kept")
    if placed is None:
        return _give_up(venue, coin,
                        f"no stop would rest after {MAX_PLACE_ATTEMPTS} attempts")

    # Only now that a correct stop is resting does the old one go. The reverse
    # order leaves a window -- and a failure in that window leaves a live
    # position with no stop at all.
    stale = [o for o in stops if o.oid != placed.oid]
    _cancel_stale(venue, coin, stale)
    action = "placed" if not stale else "replaced"
    print(f"[PROTECT] {coin}: {action} stop at {want_px:g} for {pos.size:g} "
          f"({why}){f', cancelled {len(stale)} stale' if stale else ''}")
    return Result(coin, action, f"stop {want_px:g} ({why})")


def _cancel_stale(venue: Venue, coin: str, orders: List[Order]) -> int:
    done = 0
    for o in orders:
        resp = venue.cancel(coin, o.oid)
        if isinstance(resp, dict) and str(resp.get("status")) == "ok":
            done += 1
        else:
            # Harmless in itself: two reduce-only stops cannot open anything,
            # and the venue cancels both when the position closes. Worth
            # knowing about, not worth closing a protected position over.
            safety.record_error("cancel_stale_stop", f"{coin} oid={o.oid}")
    return done


def _give_up(venue: Venue, coin: str, why: str) -> Result:
    """Close a position that cannot be protected, and only that position. An
    unprotected position is not the trade, but the others keep their own stops.
    If the venue is refusing every stop, each coin reaches here in its own turn.
    """
    print(f"[PROTECT] !! {coin}: {why} -- closing {coin}")
    res = safety.flatten_coin(venue, coin, reason=f"{coin}: {why}")
    outcome = (f"so {coin} was closed. Other positions were left alone."
               if res.flat else
               f"and closing {coin} did not finish: `{res.still_open}`. "
               f"Other positions were left alone.")
    notify.send(
        f"🚨 *{coin} could not be protected*\n{why}, {outcome}\n\n"
        f"Closed: `{res.closed or '{}'}`",
        key=f"protect_failed_{coin}", cooldown_sec=900)
    return Result(coin, "flattened", f"{why}; {res.detail}")


def _place_stop(venue: Venue, pos: Position, px: float, before: set[int]
                ) -> tuple[Optional[Order], bool]:
    """Place a stop and confirm it by reading the book, not by parsing a
    response. Returns the resting order or None, and whether the book could be
    read at all after sending. `before` is the book the caller decided from.

    The read waits, and repeats, before anything is sent again. Reading once
    straight after sending cost real money on 2026-11-05: the venue accepted
    ETH stops at 15:09:36, :37, 15:10:41 and :43 while this logged "nothing
    rested" each time, and the book was flattened with all of them resting.
    The same path doubled stops on ENA (10-28), ETH and XPL (10-29).
    """
    coin = pos.coin
    size = venue.round_size(coin, pos.size)
    if size <= 0:
        return None, True
    for attempt in range(1, MAX_PLACE_ATTEMPTS + 1):
        venue.send_trigger(coin, is_buy=not pos.is_long, size=size,
                           trigger_px=px, is_stop=True)
        read = False
        for _ in range(CONFIRM_READS):
            time.sleep(CONFIRM_WAIT_SEC)
            after = venue.open_orders(coin)
            if after is None:
                continue
            read = True
            fresh = [o for o in after
                     if o.oid not in before and o.is_protective and o.is_stop]
            if fresh:
                return fresh[0], True
        if not read:
            # Sending again blind could only stack stops.
            safety.record_error("confirm_stop", f"{coin} attempt {attempt}")
            return None, False
        safety.record_error("place_stop",
                            f"{coin} attempt {attempt}: nothing rested")
    return None, True


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
    # Feed the readability watchdog with the real outcome. None means the read
    # could not be MADE; an empty dict means a flat account, which reads fine
    # and must keep the watchdog alive.
    safety.note_read(positions is not None)
    if positions is None:
        safety.record_error("positions", "venue unreadable")
        print("[PROTECT] venue unreadable this cycle")
        return []
    expire_bids(venue)
    resting = venue.open_orders()
    # An UNREADABLE order book is not evidence that a bid is gone.
    #
    # This read used to be `(resting or [])`, so a 429 on it produced an empty
    # `bidding` set -- and _forget_closed then deleted the plan for every
    # resting bid, because nothing appeared to be holding one. The bid was
    # still at the venue, so it filled later with no metadata, was adopted
    # with a default MAX_RISK_PCT stop, and got no take-profit at all because
    # an adopted plan carries planned_tp 0.0.
    #
    # It cost real money on 2026-10-18: NEAR's plan was dropped that way, it
    # took a 3% default stop instead of its 1.1% band stop, and lost $5.44
    # against a planned $2.02. WLD's plan went the same way and its position
    # then ran straight through its target with no order to take it. ZEC, the
    # one position whose plan survived the window, got both its real stop and
    # its TP and behaved exactly as designed.
    #
    # So forgetting is now conditional on having actually READ the book.
    # Deferring cleanup one cycle costs nothing; forgetting a live plan is
    # how a planned trade silently becomes an unplanned one.
    readable = resting is not None
    safety.note_read(readable)
    if not readable:
        safety.record_error("open_orders", "venue unreadable -- keeping plans")
    bidding = {o.coin for o in (resting or []) if o.is_entry}
    if not positions:
        print("[PROTECT] flat" + (f", bidding on {sorted(bidding)}"
                                  if bidding else "")
              + ("" if readable else " (order book unreadable -- plans kept)"))
        if readable:
            _forget_closed(store, bidding)
        return []

    results: List[Result] = []
    for coin in sorted(positions):
        pos = positions[coin]
        row = store.get(coin)
        if books.trend_owns(coin, row):
            # The trend book's (core/books.py). Its strategy has no stop of
            # this kind, and its disaster stop is kept by the trend pass.
            # Adopting it here would give it a 3% stop and close it on an
            # ordinary day. It is still reported, so this log keeps moving
            # while only trend positions are open, and a missing disaster
            # stop is said out loud rather than left to the next trend pass.
            stopped = any(o.coin == coin and o.is_protective and o.is_stop
                          for o in (resting or []))
            if stopped or not readable:
                results.append(Result(coin, "trend", "the trend book's own stop"))
            else:
                results.append(Result(coin, "trend_unstopped",
                                      "no disaster stop resting"))
                notify.send(
                    f"⚠️ *{coin}: the trend book's position has no stop*\n"
                    f"The venue holds {pos.size:g} with no disaster stop "
                    f"resting. The trend pass restores it on its next hourly "
                    f"run (com.cuansniffer.trendlive).",
                    key=f"trend_unstopped_{coin}", cooldown_sec=3600)
            continue
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

    if readable:
        _forget_closed(store, set(positions) | bidding)
    return results


def expire_bids(venue: Venue) -> List[str]:
    """Cancel resting entry bids older than BID_TTL_HOURS.

    Age comes from the order's own venue timestamp, so this needs no record of
    having placed it -- which is why a resting bid carries no `pending` state
    here and cannot be orphaned by one. A bid that partially filled is simply
    a smaller resting order next to a position, and both are handled by
    reading them.
    """
    orders = venue.open_orders()
    if orders is None:
        return []
    pulled = []
    for o in orders:
        if not o.is_entry:
            continue
        age = o.age_hours()
        # Unknown age is treated as expired. Cancelling an entry bid costs an
        # opportunity at most; leaving one of unknown age resting risks a fill
        # in a market that no longer resembles the one that justified it.
        if age is not None and age < BID_TTL_HOURS:
            continue
        resp = venue.cancel(o.coin, o.oid)
        ok = isinstance(resp, dict) and str(resp.get("status")) == "ok"
        print(f"[PROTECT] bid {o.coin} @ {o.limit_px:g} "
              + (f"expired after {age:.1f}h" if age is not None
                 else "has no venue timestamp -- treated as expired")
              + f" -- cancel {'ok' if ok else 'FAILED'}")
        if ok:
            pulled.append(o.coin)
        else:
            safety.record_error("cancel_expired_bid", f"{o.coin} oid={o.oid}")
    return pulled


def _plan_age_sec(row: TradeMeta) -> float:
    """Seconds since the plan was written. Unreadable counts as old, so a
    damaged timestamp cannot keep a plan alive forever."""
    try:
        at = datetime.fromisoformat(row.opened_at.replace("Z", "+00:00"))
    except ValueError:
        return float("inf")
    if at.tzinfo is None:
        at = at.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - at).total_seconds()


def _forget_closed(store: MetaStore, live: set[str]) -> None:
    """Drop metadata for coins the venue no longer holds.

    Metadata is not a position record, so this is tidying, not reconciliation
    -- nothing decides anything from its presence. It exists so an adopted
    position's stop does not get reused on a later, unrelated trade.

    `live` includes coins with a resting BID as well as a position: the plan
    for a bid that has not filled yet is the plan the stop will need the
    moment it does.

    A young plan is left alone: the trader saves it before sending, so its
    order may simply not be visible yet. Dropping one turned a filled ENA
    entry into a 3% adoption on 2026-11-05, and left a BTC bid planless.
    """
    for coin in sorted(store.coins() - live):
        row = store.get(coin)
        if row is not None and _plan_age_sec(row) < PLAN_GRACE_SEC:
            continue
        store.drop(coin)
        print(f"[PROTECT] {coin} is gone at the venue -- metadata dropped")

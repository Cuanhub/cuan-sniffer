"""The only place this system talks to Hyperliquid.

Two rules shape every function here:

  1. The venue is the only truth. `positions()` and `open_orders()` return what
     the venue says, and nothing in this module remembers an answer.
  2. Order responses are not interpreted. `send_*` returns the raw response for
     the log and nothing else. Callers learn what happened by reading the venue
     again -- see `core.execute`.

The second rule is why the 400 lines of response parsing this replaces are
gone. They were not merely complicated: measured against 2,000 real order
records from this account, they read `sz` (the REMAINING size) as the filled
size and tested only the orderStatus envelope, which reads "order" for all nine
of the inner statuses the venue actually emits. A fully filled order therefore
parsed as a reject, and 32 recorded zero-fill rejects parsed as full fills.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING, ROUND_DOWN, ROUND_FLOOR, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

from eth_account import Account
from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from hyperliquid.utils import constants
from hyperliquid.utils.types import Cloid

HL_TESTNET = os.getenv("HL_TESTNET", "true").lower() == "true"
HL_ACCOUNT_ADDRESS = os.getenv("HL_ACCOUNT_ADDRESS", "").strip()
HL_SECRET_KEY = os.getenv("HL_SECRET_KEY", "").strip()
HL_VAULT_ADDRESS = os.getenv("HL_VAULT_ADDRESS", "").strip() or None

FLAT_EPSILON = 1e-9
# The 429 brake, kept deliberately small. It is the one rail from
# live_data_guard.py worth keeping: without it a rate-limit storm becomes a
# crash loop, and every restart is another request. `urgent` callers skip it --
# the kill switch must never be throttled by the thing it is trying to escape.
RATE_LIMIT_SLEEP_SEC = float(os.getenv("VENUE_RATE_LIMIT_SLEEP_SEC", "2.0"))
_EMPTY_SPOT_META: Dict[str, Any] = {"tokens": [], "universe": []}


@dataclass(frozen=True)
class Position:
    """One open perp position, exactly as the venue reports it."""
    coin: str
    szi: float               # signed: >0 long, <0 short
    entry_px: float
    mark_px: float           # positionValue / |szi|; no extra request needed
    unrealized_pnl: float

    @property
    def size(self) -> float:
        return abs(self.szi)

    @property
    def is_long(self) -> bool:
        return self.szi > 0


@dataclass(frozen=True)
class Order:
    """One resting order. `None` means the payload did not say."""
    coin: str
    oid: int
    size: float
    is_buy: Optional[bool]
    reduce_only: bool
    is_trigger: bool
    trigger_px: Optional[float]
    is_stop: bool            # "Stop *" order types; False for take-profits
    limit_px: float = 0.0
    timestamp_ms: int = 0    # when the venue accepted it

    @property
    def is_protective(self) -> bool:
        """A resting order that can only ever reduce this position."""
        return self.reduce_only and self.is_trigger

    @property
    def is_entry(self) -> bool:
        """A resting order that can OPEN a position: a bid at a level.

        The venue carries its own age (`timestamp`), so expiring one needs no
        stored record of having placed it. That is the whole reason resting
        bids are safe here and were not before: the old code kept the oid and
        the plan in a `pending` dict, and a partial fill popped the dict while
        the remainder stayed live at the venue.
        """
        return not self.reduce_only and not self.is_trigger

    def age_hours(self, now_ms: Optional[int] = None) -> float:
        if self.timestamp_ms <= 0:
            return 0.0
        ref = now_ms if now_ms is not None else int(time.time() * 1000)
        return max(0.0, (ref - self.timestamp_ms) / 3_600_000.0)


@dataclass(frozen=True)
class Equity:
    """Account value, from the spot layer.

    `usdc_total` ALREADY INCLUDES unrealised perp PnL. Established from 265
    equity readings in the live trader's own log (recorded as
    tests/fixtures/hyperliquid/equity_hold_series.txt): across consecutive
    readings with a position open, the median of d(total)/d(hold) is exactly
    5.0000 -- the flat leverage this system trades at -- so `total` moves by
    the full change in position notional, not by the margin alone. It marks to
    market.

    So `usdc_total` is the equity figure, on its own. Adding `unrealized_pnl`
    to it double-counts the open position, which is why there is no
    `with_unrealized` property to reach for; the field is kept for reporting.
    """
    usdc_total: float
    unrealized_pnl: float    # informational only -- already inside usdc_total


class Venue:
    def __init__(self, debug: bool = False) -> None:
        if not HL_ACCOUNT_ADDRESS:
            raise ValueError("[VENUE] HL_ACCOUNT_ADDRESS not set")
        if not HL_SECRET_KEY:
            raise ValueError("[VENUE] HL_SECRET_KEY not set")
        self.debug = debug
        self.address = (HL_VAULT_ADDRESS or HL_ACCOUNT_ADDRESS).lower()
        base = constants.TESTNET_API_URL if HL_TESTNET else constants.MAINNET_API_URL
        wallet = Account.from_key(HL_SECRET_KEY)
        self.info = Info(base, skip_ws=True, spot_meta=_EMPTY_SPOT_META)
        self.exchange = Exchange(
            wallet, base, account_address=HL_ACCOUNT_ADDRESS,
            vault_address=HL_VAULT_ADDRESS, spot_meta=_EMPTY_SPOT_META)
        self._sz_decimals: Dict[str, int] = {}
        self._cloid_seq = 0
        self.load_meta()
        print(f"[VENUE] {'TESTNET' if HL_TESTNET else 'MAINNET'} "
              f"addr={self.address} assets={len(self._sz_decimals)}")

    # ── metadata and rounding ──────────────────────────────────────────────

    def load_meta(self) -> bool:
        """Load szDecimals. Retried on demand: a 429 at startup used to leave
        every coin rounded to a guessed precision for the life of the process.
        """
        if self._sz_decimals:
            return True
        try:
            for asset in (self.info.meta() or {}).get("universe", []):
                name = str(asset.get("name", "")).upper()
                if name:
                    self._sz_decimals[name] = int(asset.get("szDecimals", 3))
        except Exception as e:
            print(f"[VENUE] meta load failed: {str(e)[:120]}")
        return bool(self._sz_decimals)

    def round_size(self, coin: str, size: float) -> float:
        if size <= 0:
            return 0.0
        q = Decimal("1").scaleb(-self._sz_decimals.get(coin.upper(), 3))
        return float(Decimal(str(size)).quantize(q, rounding=ROUND_DOWN))

    def round_price(self, coin: str, px: float, direction: str = "nearest") -> float:
        """5 significant figures, then (6 - szDecimals) decimals, per the venue.

        `direction` is applied at BOTH steps. Applying it only at the second
        step lets "down" return a price above the input: 1.23456 rounds to
        1.2346 at five significant figures, and flooring that to four decimals
        leaves it there. A stop one tick on the wrong side of its level is the
        bug class this whole rebuild is about, so the direction has to survive
        every step.

        Returns 0.0 when the price cannot be expressed at this coin's
        precision, which the caller must treat as a refusal. BTC allows one
        decimal place, so 0.00012345 has no representation there; the code this
        replaces substituted the smallest increment (0.1) instead, quietly
        turning a stop into one three orders of magnitude away. Refusing is the
        only safe answer, and callers already handle 0.0 because a
        non-positive input returns it too.
        """
        if px <= 0:
            return 0.0
        mode = {"up": ROUND_CEILING, "down": ROUND_FLOOR}.get(direction,
                                                              ROUND_HALF_UP)
        d = Decimal(str(px))
        d = d.quantize(Decimal(1).scaleb(d.adjusted() - 4), rounding=mode)
        decimals = max(0, 6 - int(self._sz_decimals.get(coin.upper(), 3)))
        d = d.quantize(Decimal(1).scaleb(-decimals), rounding=mode)
        return float(d) if d > 0 else 0.0

    def new_cloid(self) -> Cloid:
        """A client order id, for the log only. Nothing branches on it."""
        self._cloid_seq += 1
        return Cloid.from_int(((time.time_ns() << 16) | (self._cloid_seq & 0xFFFF))
                              & ((1 << 128) - 1))

    # ── reads ──────────────────────────────────────────────────────────────

    def _get(self, what: str, call: Any, urgent: bool = False) -> Optional[Any]:
        """One read, with the 429 brake. None means 'could not ask'."""
        try:
            return call()
        except Exception as e:
            print(f"[VENUE] {what} failed: {str(e)[:120]}")
            if not urgent and "429" in str(e):
                time.sleep(RATE_LIMIT_SLEEP_SEC)
            return None

    def positions(self, urgent: bool = False) -> Optional[Dict[str, Position]]:
        """Every open position. None means the venue could not be read, which is
        never the same as flat -- reporting an unreadable venue as flat is what
        orphans a live position.
        """
        state = self._get("user_state", lambda: self.info.user_state(self.address),
                          urgent)
        if not isinstance(state, dict):
            return None
        out: Dict[str, Position] = {}
        for item in state.get("assetPositions") or []:
            p = (item or {}).get("position") or {}
            coin = str(p.get("coin", "")).upper()
            szi = _f(p.get("szi"))
            if not coin or szi is None or abs(szi) <= FLAT_EPSILON:
                continue
            value = abs(_f(p.get("positionValue")) or 0.0)
            out[coin] = Position(
                coin=coin, szi=szi, entry_px=_f(p.get("entryPx")) or 0.0,
                mark_px=(value / abs(szi)) if abs(szi) > 0 else 0.0,
                unrealized_pnl=_f(p.get("unrealizedPnl")) or 0.0)
        return out

    def open_orders(self, coin: str = "", urgent: bool = False
                    ) -> Optional[List[Order]]:
        """Resting orders, from frontendOpenOrders -- the only endpoint that
        reports reduceOnly, isTrigger and triggerPx. None means unreadable.
        """
        raw = self._get("frontend_open_orders",
                        lambda: self.info.frontend_open_orders(self.address), urgent)
        if not isinstance(raw, list):
            return None
        out: List[Order] = []
        want = coin.upper()
        for o in raw:
            if not isinstance(o, dict):
                continue
            c = str(o.get("coin", "")).upper()
            if want and c != want:
                continue
            oid = o.get("oid")
            if oid is None:
                continue
            kind = str(o.get("orderType") or "")
            side = str(o.get("side") or "").strip().lower()
            out.append(Order(
                coin=c, oid=int(oid), size=_f(o.get("sz")) or 0.0,
                # Hyperliquid encodes side as "B" (bid/buy) or "A" (ask/sell).
                is_buy=(True if side == "b" else False if side == "a" else None),
                reduce_only=bool(o.get("reduceOnly")),
                is_trigger=bool(o.get("isTrigger")),
                trigger_px=_f(o.get("triggerPx")),
                is_stop=kind.startswith("Stop"),
                limit_px=_f(o.get("limitPx")) or 0.0,
                timestamp_ms=int(_f(o.get("timestamp")) or 0)))
        return out

    def equity(self, urgent: bool = False) -> Optional[Equity]:
        """Account value from the spot layer.

        This account is a `unifiedAccount`: its perp clearinghouse reports
        accountValue "0.0" even when funded, so marginSummary cannot be used.
        Verified against the live account.

        `usdc_total` is the figure the limits use. `unrealized_pnl` is summed
        and carried alongside so the logs can show both while a position is
        open -- whether `total` already moves with unrealised PnL decides which
        is right, and that is settled by observation, not assumption.
        """
        spot = self._get("spot_user_state",
                         lambda: self.info.spot_user_state(self.address), urgent)
        if not isinstance(spot, dict):
            return None
        usdc = next((b for b in spot.get("balances") or []
                     if str(b.get("coin", "")).upper() == "USDC"), None)
        if usdc is None:
            print("[VENUE] no USDC balance in spot state")
            return None
        total = _f(usdc.get("total"))
        if total is None:
            return None
        pos = self.positions(urgent=urgent)
        unreal = sum(p.unrealized_pnl for p in pos.values()) if pos else 0.0
        return Equity(usdc_total=total, unrealized_pnl=unreal)

    def mid(self, coin: str) -> Optional[float]:
        """Mid price. Entry sizing only; the kill switch uses Position.mark_px,
        so a throttled or failing price feed can never block a flatten.
        """
        mids = self._get("all_mids", self.info.all_mids)
        if not isinstance(mids, dict):
            return None
        return _f(mids.get(coin.upper()))

    # ── writes. Every one returns the raw response, for the log only. ───────

    def send_ioc(self, coin: str, is_buy: bool, size: float, limit_px: float,
                 reduce_only: bool) -> Any:
        return self._send(f"IOC {coin} {'BUY' if is_buy else 'SELL'} {size}",
                          lambda: self.exchange.order(
                              coin, is_buy, size, limit_px,
                              {"limit": {"tif": "Ioc"}}, reduce_only=reduce_only,
                              cloid=self.new_cloid()))

    def send_limit(self, coin: str, is_buy: bool, size: float,
                   limit_px: float) -> Any:
        """Rest a GTC limit order. Does not wait for a fill.

        A marketable IOC is the right shape for "enter now" and the wrong one
        for support that taps and leaves. Measured 2026-09-19: across 139
        tradeable watched levels not one was inside the 1.5% entry drift
        guard, the nearest 3.10% away -- every tap had resolved before a poll
        could see it. A bid sitting at the level is filled BY the tap.
        """
        return self._send(f"LIMIT {coin} {'BUY' if is_buy else 'SELL'} "
                          f"{size}@{limit_px}",
                          lambda: self.exchange.order(
                              coin, is_buy, size, limit_px,
                              {"limit": {"tif": "Gtc"}}, reduce_only=False,
                              cloid=self.new_cloid()))

    def send_trigger(self, coin: str, is_buy: bool, size: float,
                     trigger_px: float, is_stop: bool) -> Any:
        """Place a reduce-only trigger. The caller confirms it is resting by
        reading open_orders, not by parsing this response.
        """
        return self._send(
            f"TRIGGER {coin} {'sl' if is_stop else 'tp'} {size}@{trigger_px}",
            lambda: self.exchange.order(
                coin, is_buy, size, trigger_px,
                {"trigger": {"triggerPx": trigger_px, "isMarket": True,
                             "tpsl": "sl" if is_stop else "tp"}},
                reduce_only=True))

    def cancel(self, coin: str, oid: int) -> Any:
        return self._send(f"CANCEL {coin} {oid}",
                          lambda: self.exchange.cancel(coin, int(oid)))

    def _send(self, what: str, call: Any) -> Any:
        try:
            resp = call()
        except Exception as e:
            print(f"[VENUE] {what} raised: {str(e)[:160]}")
            return {"status": "err", "response": str(e)[:200]}
        if self.debug:
            print(f"[VENUE] {what} -> {str(resp)[:200]}")
        return resp


def _f(v: Any) -> Optional[float]:
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None

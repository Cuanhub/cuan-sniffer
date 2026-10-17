"""A fake Hyperliquid built from the recorded payloads in tests/fixtures.

Every payload this serves is a real recorded one, loaded from disk and edited
only in the fields a scenario needs (coin, size, price, oid). Nothing here
constructs a payload shape from scratch. That restriction is the whole point:
the previous suite's fake returned the REQUESTED size as `sz` and hardcoded the
inner status to "filled", so it could not express a zero-fill reject, and a test
asserting the opposite of the venue's real behaviour passed.

One thing this fake deliberately does NOT model carefully: the body of an order
response. Under rule 2 nothing reads it, so `order()` returns a minimal envelope
and the tests drive outcomes by changing what the venue HOLDS. If a future
change starts depending on a response body, these tests stop covering it -- and
`test_core_venue.py::test_nothing_reads_an_order_response_body` fails on
purpose to say so.
"""

from __future__ import annotations

import copy
import json
import pathlib
import time
from typing import Any, Dict, List, Optional

FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures/hyperliquid"


def load(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text())


def order_record(inner_status: str) -> Dict[str, Any]:
    """A real recorded order record with the given inner status."""
    for row in load("order_status_by_inner_status.json"):
        if str(row.get("status")) == inner_status:
            return copy.deepcopy(row)
    raise KeyError(f"no recorded order with inner status {inner_status!r}; "
                   f"re-run tests/record_fixtures.py")


def order_envelope(inner_status: str) -> Dict[str, Any]:
    """The real orderStatus envelope for the given inner status."""
    for row in load("order_status_envelopes.json"):
        if str(row.get("_inner_status")) == inner_status:
            return copy.deepcopy(row["payload"])
    raise KeyError(f"no recorded envelope for {inner_status!r}")


def _position_template() -> Dict[str, Any]:
    return copy.deepcopy(load("clearinghouse_state_with_position.json")
                         ["assetPositions"][0])


def _trigger_template(is_stop: bool) -> Dict[str, Any]:
    want = "Stop Market" if is_stop else "Take Profit Market"
    for o in load("frontend_open_orders_protected.json"):
        if o.get("orderType") == want:
            return copy.deepcopy(o)
    raise KeyError(f"no recorded {want} order")


class Transport(Exception):
    """What the SDK raises when a request does not complete."""


class FakeHL:
    """Serves both the Info and Exchange surfaces the Venue uses."""

    def __init__(self, positions: Optional[Dict[str, float]] = None,
                 mid: float = 100.0, usdc: float = 1000.0,
                 unrealized: Dict[str, float] | None = None) -> None:
        self.pos: Dict[str, float] = dict(positions or {})
        self.unrealized: Dict[str, float] = dict(unrealized or {})
        self.mid_px = mid
        self.usdc = usdc
        self.resting: List[Dict[str, Any]] = []
        self.sent: List[Dict[str, Any]] = []
        self.cancelled: List[int] = []
        self._next_oid = 900_000
        # Scripted failures: a count of how many of the next calls raise.
        self.fail_info = 0
        self.fail_order = 0
        self.fail_cancel = 0
        self.fail_info_always = False
        # Real venue positions are quantized to the asset's szDecimals, so a
        # sub-precision residual cannot exist. Modelling that matters: without
        # it, repeated partial closes converge to dust no order can express and
        # a flatten loop has nothing to work with.
        self.sz_decimals = 2
        self.order_fills = True      # does a marketable IOC move the position?
        # The venue's clock, which stamps every resting order. Real `now` by
        # default so a freshly placed bid reads as fresh; a test that wants a
        # stale one moves this back.
        self.now_ms = int(time.time() * 1000)
        self.fill_fraction = 1.0
        self.raise_after_fill = False

    # ---- Info surface ----------------------------------------------------
    def _maybe_fail_info(self) -> None:
        if self.fail_info_always:
            raise Transport("429 Too Many Requests")
        if self.fail_info > 0:
            self.fail_info -= 1
            raise Transport("429 Too Many Requests")

    def _quantize(self, coin: str) -> None:
        """Positions are held to szDecimals, as the venue holds them."""
        held = self.pos.get(coin)
        if held is None:
            return
        step = 10 ** -self.sz_decimals
        q = round(held / step) * step
        if abs(q) < step / 2:
            self.pos.pop(coin, None)
        else:
            self.pos[coin] = round(q, self.sz_decimals)

    def user_state(self, address: str, dex: str = "") -> Dict[str, Any]:
        self._maybe_fail_info()
        state = copy.deepcopy(load("clearinghouse_state_flat.json"))
        for coin, szi in self.pos.items():
            if abs(szi) <= 0:
                continue
            item = _position_template()
            p = item["position"]
            p["coin"] = coin
            p["szi"] = str(szi)
            p["positionValue"] = str(abs(szi) * self.mid_px)
            p["entryPx"] = str(self.mid_px)
            p["unrealizedPnl"] = str(self.unrealized.get(coin, 0.0))
            state["assetPositions"].append(item)
        return state

    def spot_user_state(self, address: str) -> Dict[str, Any]:
        self._maybe_fail_info()
        state = copy.deepcopy(load("spot_user_state.json"))
        for b in state["balances"]:
            if str(b.get("coin")).upper() == "USDC":
                b["total"] = str(self.usdc)
        return state

    def frontend_open_orders(self, address: str, dex: str = "") -> List[Dict[str, Any]]:
        self._maybe_fail_info()
        return copy.deepcopy(self.resting)

    def all_mids(self, dex: str = "") -> Dict[str, str]:
        self._maybe_fail_info()
        coins = set(self.pos) | {"HYPE", "ZRO", "BTC", "NIL"}
        return {c: str(self.mid_px) for c in coins}

    def meta(self, dex: str = "") -> Dict[str, Any]:
        self._maybe_fail_info()
        return copy.deepcopy(load("meta.json"))

    def user_fills_by_time(self, address: str, start: int,
                           end: Optional[int] = None) -> List[Dict[str, Any]]:
        self._maybe_fail_info()
        return copy.deepcopy(load("user_fills_30d.json"))

    # ---- Exchange surface ------------------------------------------------
    def order(self, name: str, is_buy: bool, sz: float, limit_px: float,
              order_type: Dict[str, Any], reduce_only: bool = False,
              cloid: Any = None, builder: Any = None) -> Dict[str, Any]:
        if self.fail_order > 0:
            self.fail_order -= 1
            raise Transport("Read timed out")
        rec = {"coin": name, "is_buy": is_buy, "sz": sz, "px": limit_px,
               "reduce_only": reduce_only, "type": order_type,
               "cloid": cloid.to_raw() if cloid is not None else None}
        self.sent.append(rec)

        tif = str((order_type.get("limit") or {}).get("tif") or "")
        if tif == "Gtc":
            # A GTC limit below the market RESTS; it does not fill at the mid.
            # Modelling that matters: it is the whole reason a bid at a level
            # is filled by the tap instead of chasing it.
            self.rest_plain_order(name, sz, limit_px, is_buy=is_buy)
            return {"status": "ok"}

        if "trigger" in order_type:
            t = order_type["trigger"]
            o = _trigger_template(is_stop=(t.get("tpsl") == "sl"))
            self._next_oid += 1
            o.update({"coin": name, "sz": str(sz), "origSz": str(sz),
                      "oid": self._next_oid, "side": "B" if is_buy else "A",
                      "triggerPx": str(t["triggerPx"]),
                      "limitPx": str(t["triggerPx"]), "reduceOnly": True})
            self.resting.append(o)
            return {"status": "ok"}

        if self.order_fills:
            filled = sz * self.fill_fraction
            held = self.pos.get(name, 0.0)
            if reduce_only:
                delta = -min(filled, held) if held > 0 else \
                    (min(filled, -held) if held < 0 else 0.0)
            else:
                delta = filled if is_buy else -filled
            self.pos[name] = held + delta
            self._quantize(name)
            if abs(self.pos.get(name, 0.0)) <= 1e-12:
                self.pos.pop(name, None)
                # The venue cancels reduce-only orders when the position
                # closes. Observed 286 times in the recorded history.
                self.resting = [r for r in self.resting
                                if not (r["coin"] == name and r.get("reduceOnly"))]
        if self.raise_after_fill:
            raise Transport("Read timed out")
        return {"status": "ok"}

    def cancel(self, name: str, oid: int) -> Dict[str, Any]:
        if self.fail_cancel > 0:
            self.fail_cancel -= 1
            raise Transport("429 Too Many Requests")
        before = len(self.resting)
        self.resting = [r for r in self.resting if int(r["oid"]) != int(oid)]
        if len(self.resting) < before:
            self.cancelled.append(int(oid))
            return {"status": "ok"}
        return {"status": "err", "response": "unknown oid"}

    def update_leverage(self, leverage: int, name: str,
                        is_cross: bool = True) -> Dict[str, Any]:
        return {"status": "ok"}

    # ---- helpers for tests ----------------------------------------------
    def rest_protection(self, coin: str, size: float, stop_px: float,
                        tp_px: Optional[float] = None) -> None:
        """Put real recorded trigger orders on the book for `coin`."""
        for is_stop, px in ((True, stop_px), (False, tp_px)):
            if px is None:
                continue
            o = _trigger_template(is_stop=is_stop)
            self._next_oid += 1
            o.update({"coin": coin, "sz": str(size), "origSz": str(size),
                      "oid": self._next_oid, "triggerPx": str(px),
                      "limitPx": str(px), "side": "A", "reduceOnly": True})
            self.resting.append(o)

    def rest_plain_order(self, coin: str, size: float, px: float,
                         is_buy: bool = True) -> None:
        """A non-protective resting order -- a GTC entry bid.

        Built from a real recorded `open` Limit record. This is also the shape
        that made the old has_protection() report an unprotected position as
        protected, because any order for the coin counted.
        """
        o = copy.deepcopy(order_record("open")["order"])
        self._next_oid += 1
        o.update({"coin": coin, "sz": str(size), "origSz": str(size),
                  "oid": self._next_oid, "limitPx": str(px),
                  "side": "B" if is_buy else "A",
                  "reduceOnly": False, "isTrigger": False, "triggerPx": "0.0",
                  "orderType": "Limit", "timestamp": self.now_ms})
        self.resting.append(o)


def build_venue(fake: FakeHL, debug: bool = False) -> Any:
    """A real core.venue.Venue wired to the fake. No network, no keys."""
    from core.venue import Venue
    v = object.__new__(Venue)
    v.debug = debug
    v.address = "0x" + "11" * 20
    v.info = fake
    v.exchange = fake
    v._sz_decimals = {}
    v._cloid_seq = 0
    v.load_meta()
    return v

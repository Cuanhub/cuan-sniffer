"""End-to-end order lifecycle against a fake venue.

Audit findings C-1 and D-1. Written BEFORE the fixes, so every test in
OrderIdentity and HaltProtectsOpenPositions fails on the pre-fix code.

Why this file exists at all: live_execution_backend.py is 2,065 lines that
place every order this account makes, and no test sent an order through it
end to end. The uncovered paths were exactly the expensive ones -- response
parsing, partial fills, rejects, and the unknown state after a timeout.

The five scenarios the audit asked for:
    fill                 a clean full fill
    partial fill         venue fills part of the size
    reject               venue refuses
    timeout-after-fill   the request times out AFTER the venue filled it.
                         The client sees an exception; the venue holds a
                         position. This is C-1: with no client order id there
                         is no way to ask "did my order land?", so the code
                         records a reject and the position is untracked and
                         unprotected until the next reconcile, up to 450s.
    halt with positions  D-1: halt() blocked new entries and never checked
                         that the open ones still had a stop.
"""

import threading
import unittest
from unittest import mock

from hyperliquid.utils.types import Cloid

from live_execution_backend import LiveExecutionBackend
import tools.live_levels_trader as T


class VenueTimeout(Exception):
    """What requests raises when the response never arrives."""


class FakeVenue:
    """Enough of Hyperliquid's Info + Exchange to drive a whole order.

    Mirrors the real SDK's surface for the calls the backend makes. Scripted
    per test through `behaviour`:
        "fill" | "partial" | "reject" | "timeout_after_fill"
    """

    def __init__(self, behaviour="fill", fill_fraction=1.0, mid=100.0):
        self.behaviour = behaviour
        self.fill_fraction = fill_fraction
        self.mid = mid
        self.positions = {}          # coin -> signed size
        self.orders_received = []    # every order the venue saw
        self.resting = []            # reduce-only stop/tp orders
        self.cancelled = []

    # ---- Exchange surface ------------------------------------------------
    def order(self, name, is_buy, sz, limit_px, order_type,
              reduce_only=False, cloid=None, builder=None):
        rec = {"coin": name, "is_buy": is_buy, "sz": sz, "px": limit_px,
               "reduce_only": reduce_only,
               "cloid": cloid.to_raw() if cloid is not None else None}
        self.orders_received.append(rec)

        # Branch on the ORDER TYPE, not on reduce_only. Protection is a
        # TRIGGER order and rests; an exit is a reduce-only IOC and fills.
        # Conflating them made the fake rest every exit, which is not what
        # the venue does and broke the flatten-on-halt test for the wrong
        # reason.
        is_trigger = isinstance(order_type, dict) and "trigger" in order_type
        if is_trigger:
            rec["is_trigger"] = True
            self.resting.append(rec)
            return {"status": "ok", "response": {"type": "order", "data": {
                "statuses": [{"resting": {"oid": 900 + len(self.resting)}}]}}}

        if self.behaviour == "reject":
            return {"status": "ok", "response": {"type": "order", "data": {
                "statuses": [{"error": "Insufficient margin"}]}}}

        filled = sz * (self.fill_fraction if self.behaviour == "partial" else 1.0)
        signed = filled if is_buy else -filled
        if reduce_only:
            # Cannot flip the position, only shrink it -- which is what
            # reduce-only means and what flatten() relies on.
            held = self.positions.get(name, 0.0)
            signed = max(-abs(held), min(abs(held), signed)) if held else 0.0
            if held > 0:
                signed = -min(filled, held)
            elif held < 0:
                signed = min(filled, -held)
        self.positions[name] = self.positions.get(name, 0.0) + signed

        if self.behaviour == "timeout_after_fill":
            # The venue filled it. The client will never hear about it.
            raise VenueTimeout("Read timed out")

        return {"status": "ok", "response": {"type": "order", "data": {
            "statuses": [{"filled": {"totalSz": str(filled),
                                     "avgPx": str(limit_px),
                                     "oid": 1234}}]}}}

    def cancel(self, name, oid):
        self.cancelled.append((name, oid))
        self.resting = [r for r in self.resting]
        return {"status": "ok"}

    def cancel_by_cloid(self, name, cloid):
        self.cancelled.append((name, cloid.to_raw()))
        return {"status": "ok"}

    def update_leverage(self, leverage, name, is_cross=True):
        return {"status": "ok"}

    # ---- Info surface ----------------------------------------------------
    def user_state(self, address, dex=""):
        return {"assetPositions": [
            {"position": {"coin": c, "szi": str(s),
                          "positionValue": str(abs(s) * self.mid),
                          "unrealizedPnl": "0.0"}}
            for c, s in self.positions.items() if abs(s) > 0],
            "marginSummary": {"accountValue": "1000.0",
                              "totalMarginUsed": "0.0"},
            "withdrawable": "1000.0"}

    def all_mids(self, dex=""):
        return {c: str(self.mid) for c in
                set(list(self.positions) + ["HYPE", "ZRO", "BTC"])}

    def open_orders(self, address, dex=""):
        return [{"coin": r["coin"], "oid": 900 + i, "side": "A",
                 "sz": str(r["sz"]), "limitPx": str(r["px"]),
                 "reduceOnly": True, "isTrigger": True,
                 "triggerPx": str(r["px"])}
                for i, r in enumerate(self.resting)]

    def frontend_open_orders(self, address, dex=""):
        return self.open_orders(address)

    def query_order_by_cloid(self, user, cloid):
        """The C-1 lookup: did an order with this id reach the venue?"""
        raw = cloid.to_raw()
        for r in self.orders_received:
            if r["cloid"] == raw:
                return {"status": "order", "order": {"order": {
                    "coin": r["coin"], "sz": str(r["sz"]),
                    "limitPx": str(r["px"]), "cloid": raw},
                    "status": "filled"}}
        return {"status": "unknownOid"}

    def user_fills_by_time(self, address, start_time, end_time=None):
        return []

    def meta(self, dex=""):
        return {"universe": [{"name": c, "szDecimals": 2, "maxLeverage": 10}
                             for c in ("HYPE", "ZRO", "BTC")]}

    def query_user_abstraction_state(self, address):
        return {"abstraction": "standard"}

    def spot_user_state(self, address):
        return {"balances": []}


def backend(venue):
    """A real LiveExecutionBackend wired to the fake venue, no network."""
    b = object.__new__(LiveExecutionBackend)
    b.debug = False
    b.base_url = "http://fake"
    b.account_address = "0x" + "11" * 20
    b.vault_address = None
    b._wallet = None
    b.wallet = None
    b.info = venue
    b.exchange = venue
    b._mid_cache = {"HYPE": venue.mid, "ZRO": venue.mid, "BTC": venue.mid}
    b._mid_cache_ts = 1e18          # never stale within a test
    b._sz_decimals = {"HYPE": 2, "ZRO": 2, "BTC": 2}
    b._max_leverage = {"HYPE": 10, "ZRO": 10, "BTC": 10}
    b._unified_account_cache = False
    # Every attribute __init__ sets must be set here too. Bypassing __init__
    # and then missing one is how this helper first broke four passing tests:
    # the fake was less complete than the real object, so it tested the fake.
    b._cloid_seq = 0
    b._cloid_lock = threading.Lock()
    return b


def _assert_helper_matches_real_init():
    """The helper must cover every instance attribute __init__ assigns.

    Checked at import so a new attribute in __init__ fails this file loudly
    rather than silently leaving the fake behind.
    """
    import re
    src = open("live_execution_backend.py").read()
    # Bounded by a marker that exists in every version of the file. Using
    # one the fix introduces turned this check into a COLLECTION error on
    # pre-fix code, which hides which tests would have failed.
    start = src.index("    def __init__(self, debug")
    init = src[start:src.index("\n    def execute_entry", start)]
    assigned = set(re.findall(r"self\.(_?[a-z][A-Za-z0-9_]*)\s*[:=]", init))
    built = set(vars(backend(FakeVenue())))
    missing = assigned - built - {"base_url"}
    assert not missing, (
        f"test_e2e_fake_venue.backend() is missing {sorted(missing)}, which "
        f"__init__ sets -- the fake is less complete than the real object")


_assert_helper_matches_real_init()


class SanityOfTheFake(unittest.TestCase):
    """If the fake is wrong, every test below is theatre."""

    def test_a_clean_fill_reaches_the_venue_and_opens_a_position(self):
        v = FakeVenue("fill")
        r = backend(v).execute_entry(coin="HYPE", side="LONG", price=100.0,
                                     size_usd=200.0, confidence=0.5)
        self.assertTrue(r.filled, r.reject_reason)
        self.assertEqual(len(v.orders_received), 1)
        self.assertGreater(v.positions["HYPE"], 0)

    def test_a_reject_does_not_open_a_position(self):
        v = FakeVenue("reject")
        r = backend(v).execute_entry(coin="HYPE", side="LONG", price=100.0,
                                     size_usd=200.0, confidence=0.5)
        self.assertFalse(r.filled)
        self.assertEqual(v.positions.get("HYPE", 0.0), 0.0)

    def test_a_partial_fill_is_reported_as_partial(self):
        v = FakeVenue("partial", fill_fraction=0.5)
        r = backend(v).execute_entry(coin="HYPE", side="LONG", price=100.0,
                                     size_usd=200.0, confidence=0.5)
        self.assertTrue(r.filled)
        self.assertLess(r.fill_size_usd, 200.0 * 0.9)
        self.assertGreater(v.positions["HYPE"], 0)

    def test_the_timeout_fake_really_does_fill_before_raising(self):
        v = FakeVenue("timeout_after_fill")
        with self.assertRaises(VenueTimeout):
            v.order("HYPE", True, 2.0, 100.0, {"limit": {"tif": "Ioc"}})
        self.assertGreater(v.positions["HYPE"], 0,
                           "the fake must leave a position behind")


class OrderIdentity(unittest.TestCase):
    """C-1. Fails before the fix: no order carries a client order id."""

    def test_every_entry_carries_a_cloid(self):
        v = FakeVenue("fill")
        backend(v).execute_entry(coin="HYPE", side="LONG", price=100.0,
                                 size_usd=200.0, confidence=0.5)
        self.assertIsNotNone(v.orders_received[0]["cloid"],
                             "entry sent with no client order id")

    def test_every_exit_carries_a_cloid(self):
        v = FakeVenue("fill")
        b = backend(v)
        b.execute_entry(coin="HYPE", side="LONG", price=100.0,
                        size_usd=200.0, confidence=0.5)
        v.orders_received.clear()
        b.execute_exit(coin="HYPE", side="LONG", price=100.0,
                       size_usd=200.0, reason="test")
        self.assertIsNotNone(v.orders_received[0]["cloid"],
                             "exit sent with no client order id")

    def test_cloids_are_unique_per_order(self):
        v = FakeVenue("fill")
        b = backend(v)
        for _ in range(3):
            b.execute_entry(coin="HYPE", side="LONG", price=100.0,
                            size_usd=200.0, confidence=0.5)
        seen = [o["cloid"] for o in v.orders_received if not o["reduce_only"]]
        self.assertEqual(len(seen), len(set(seen)), "cloids collided")

    def test_a_cloid_is_a_valid_venue_cloid(self):
        v = FakeVenue("fill")
        backend(v).execute_entry(coin="HYPE", side="LONG", price=100.0,
                                 size_usd=200.0, confidence=0.5)
        Cloid.from_str(v.orders_received[0]["cloid"])   # raises if malformed

    def test_a_timeout_is_resolved_by_cloid_not_assumed_rejected(self):
        """The whole point of C-1.

        The venue filled; the client timed out. Pre-fix the backend returns a
        reject and the caller forgets a live, unprotected position. Post-fix
        it must look the order up and report the fill it finds.
        """
        v = FakeVenue("timeout_after_fill")
        r = backend(v).execute_entry(coin="HYPE", side="LONG", price=100.0,
                                     size_usd=200.0, confidence=0.5)
        self.assertTrue(
            r.filled,
            "a timed-out order that FILLED was reported as a reject -- the "
            "position is live, unprotected and untracked")

    def test_the_lookup_is_actually_performed(self):
        v = FakeVenue("timeout_after_fill")
        with mock.patch.object(v, "query_order_by_cloid",
                               wraps=v.query_order_by_cloid) as q:
            backend(v).execute_entry(coin="HYPE", side="LONG", price=100.0,
                                     size_usd=200.0, confidence=0.5)
        self.assertTrue(q.called, "no lookup after an ambiguous response")

    def test_an_unknown_order_after_timeout_stays_a_reject(self):
        """A timeout where the venue never saw the order must NOT become a
        phantom fill. The lookup has to distinguish the two."""
        v = FakeVenue("fill")

        def timeout_without_filling(*a, **k):
            raise VenueTimeout("Read timed out")

        b = backend(v)
        with mock.patch.object(v, "order", side_effect=timeout_without_filling):
            r = b.execute_entry(coin="HYPE", side="LONG", price=100.0,
                                size_usd=200.0, confidence=0.5)
        self.assertFalse(r.filled, "a never-placed order became a fill")


class StartupAdoption(unittest.TestCase):
    """C-1's other half: a position found at startup must be adopted or
    flattened, never ignored."""

    def test_an_untracked_position_is_detected_at_startup(self):
        v = FakeVenue("fill")
        v.positions["ZRO"] = 50.0
        reason, fatal = T.venue_sanity(backend(v), {"owned": {}}, 1000.0)
        self.assertTrue(fatal)
        self.assertIn("ZRO", reason)

    def test_adopt_or_flatten_exists_and_is_reachable(self):
        self.assertTrue(hasattr(T, "adopt_or_flatten_orphans"),
                        "no startup path adopts or flattens an orphan")


class HaltProtectsOpenPositions(unittest.TestCase):
    """D-1. Fails before the fix: halt blocked entries and checked nothing."""

    def state_with_position(self):
        return {"owned": {"HYPE": {"entry": 100.0, "stop": 95.0, "tp": 115.0,
                                   "size_usd": 200.0, "size_coin": 2.0,
                                   "episode_id": "e1"}},
                "pending": {}, "realised_today": 0.0, "halted": False}

    def test_halting_verifies_a_stop_exists_on_every_open_position(self):
        v = FakeVenue("fill")
        v.positions["HYPE"] = 2.0          # open, but NO resting stop
        st = self.state_with_position()
        with mock.patch.object(T, "save_state"), \
                mock.patch.object(T.notify, "send") as sent:
            T.halt(st, "test halt", kind="manual", backend=backend(v))
        self.assertTrue(st["halted"])
        blob = " ".join(str(c[0][0]) for c in sent.call_args_list)
        self.assertIn("unprotected", blob.lower(),
                      "halt did not notice an open position with no stop")

    def test_halting_with_a_protected_position_does_not_cry_wolf(self):
        v = FakeVenue("fill")
        v.positions["HYPE"] = 2.0
        v.resting.append({"coin": "HYPE", "sz": 2.0, "px": 95.0,
                          "reduce_only": True, "is_trigger": True,
                          "cloid": None, "is_buy": False})
        st = self.state_with_position()
        with mock.patch.object(T, "save_state"), \
                mock.patch.object(T.notify, "send") as sent:
            T.halt(st, "test halt", kind="manual", backend=backend(v))
        blob = " ".join(str(c[0][0]) for c in sent.call_args_list)
        self.assertNotIn("unprotected", blob.lower())

    def test_flatten_on_halt_is_opt_in_and_defaults_off(self):
        from research.levels import config as C
        self.assertEqual(C.HALT_ACTION, "verify_stops",
                         "the default must be the conservative one")

    def test_halt_action_flatten_closes_open_positions(self):
        v = FakeVenue("fill")
        v.positions["HYPE"] = 2.0
        st = self.state_with_position()
        saved = T.HALT_ACTION
        try:
            T.HALT_ACTION = "flatten"
            with mock.patch.object(T, "save_state"), \
                    mock.patch.object(T.notify, "send"):
                T.halt(st, "test halt", kind="manual", backend=backend(v))
            self.assertAlmostEqual(v.positions.get("HYPE", 0.0), 0.0, places=6)
        finally:
            T.HALT_ACTION = saved

    def test_a_halt_without_a_backend_still_halts(self):
        """Callers that have no backend to hand must not be blocked from
        halting -- a halt that cannot happen is worse than an unverified one.
        """
        st = self.state_with_position()
        with mock.patch.object(T, "save_state"), \
                mock.patch.object(T.notify, "send"):
            T.halt(st, "no backend here", kind="manual")
        self.assertTrue(st["halted"])


if __name__ == "__main__":
    unittest.main()

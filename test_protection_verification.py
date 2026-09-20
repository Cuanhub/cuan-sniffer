"""
Tests for stop/take-profit verification in live_execution_backend.

This file exists because of the only live protection this account has ever
placed. On 2026-09-18 a UNI long filled and its protection was recorded:

    protection,UNI,...,protected,stop=548611271723 tp=548611279065
      stop:verify_warn:not_trigger_order;wrong_tpsl:none;
           missing_trigger_price;missing_side

Four verification failures, and a status of "protected" anyway. Two separate
defects had to line up to produce that line, and each is pinned below:

  1. get_open_orders preferred the plain openOrders endpoint, which returns
     only coin/limitPx/oid/side/sz/timestamp. No reduceOnly, no isTrigger, no
     triggerPx -- so verification could not confirm anything it was asked to,
     and said so on every field. frontendOpenOrders carries those fields and
     is now queried first.
  2. status was computed from OID presence alone and verification was
     explicitly advisory, so none of those four failures reached the one
     field the caller acts on.

Separately, the side check could never pass: Hyperliquid encodes side as "A"
(ask/sell) or "B" (bid/buy), and "a" was missing from the sell set, so every
stop on a LONG -- which closes SELL -- reported missing_side.

The fix has to hold two lines at once. A venue CONTRADICTION must downgrade
status, or the check is decorative. An UNREPORTED field must not, or we
re-create the other 2026-09-18 failure, where a correctly protected position
was emergency-closed seconds after entry by a check that fired on success.
Both directions are tested here.
"""

import unittest

from live_execution_backend import LiveExecutionBackend as Backend


def frontend_row(oid="S1", side="A", order_type="Stop Market", trigger_px="95.0",
                 sz="46.3", reduce_only=True, is_trigger=True, coin="UNI"):
    """A row shaped like a real frontendOpenOrders entry."""
    return {"coin": coin, "side": side, "limitPx": trigger_px, "sz": sz,
            "origSz": sz, "oid": oid, "timestamp": 1789000000000,
            "triggerCondition": f"Price below {trigger_px}",
            "isTrigger": is_trigger, "triggerPx": trigger_px,
            "isPositionTpsl": False, "reduceOnly": reduce_only,
            "orderType": order_type, "tif": None, "cloid": None, "children": []}


class NormalizerTest(unittest.TestCase):
    """Whether the payload can be read at all."""

    def norm(self, **kw):
        return Backend._normalize_open_order_item(frontend_row(**kw))

    def test_side_a_is_sell(self):
        # The regression. A stop on a LONG closes SELL, which HL sends as "A".
        self.assertIs(self.norm(side="A")["is_buy"], False)

    def test_side_b_is_buy(self):
        self.assertIs(self.norm(side="B")["is_buy"], True)

    def test_tpsl_recovered_from_order_type(self):
        # frontendOpenOrders has no tpsl field; orderType is the only signal.
        self.assertEqual(self.norm(order_type="Stop Market")["tpsl"], "sl")
        self.assertEqual(self.norm(order_type="Stop Limit")["tpsl"], "sl")
        self.assertEqual(self.norm(order_type="Take Profit Market")["tpsl"], "tp")
        self.assertEqual(self.norm(order_type="Take Profit Limit")["tpsl"], "tp")

    def test_plain_order_type_yields_no_tpsl(self):
        self.assertEqual(self.norm(order_type="Limit")["tpsl"], "")

    def test_absent_fields_are_unknown_not_false(self):
        # A plain openOrders row. Nothing here denies anything; it is silent.
        row = Backend._normalize_open_order_item(
            {"coin": "UNI", "limitPx": "8.35", "oid": "S1", "side": "A", "sz": "46.3"})
        self.assertIsNone(row["is_trigger"])
        self.assertIsNone(row["reduce_only"])
        self.assertIsNone(row["trigger_px"])

    def test_venue_denial_is_false_not_unknown(self):
        # Here the venue does deny it, and that must stay distinguishable.
        row = self.norm(is_trigger=False, reduce_only=False, order_type="Limit")
        self.assertIs(row["is_trigger"], False)
        self.assertIs(row["reduce_only"], False)

    def test_explicit_tpsl_beats_order_type(self):
        row = Backend._normalize_open_order_item(
            {"order": {"coin": "UNI", "side": "A", "sz": "46.3", "oid": "S1",
                       "reduceOnly": True,
                       "orderType": {"trigger": {"tpsl": "sl", "triggerPx": "95.0",
                                                 "isMarket": True}}}})
        self.assertEqual(row["tpsl"], "sl")
        self.assertIs(row["is_trigger"], True)


class SingleOrderVerificationTest(unittest.TestCase):
    """Whether a discrepancy is classified as a contradiction or an unknown."""

    def setUp(self):
        self.b = Backend.__new__(Backend)
        self.b._sz_decimals = {"UNI": 1}

    def check(self, row):
        return self.b._verify_single_protection_order(
            row, expected_tpsl="sl", expected_size_coin=46.3,
            expected_trigger_px=95.0, expected_is_buy=False, coin="UNI")

    def test_correct_stop_is_clean(self):
        hard, soft = self.check(Backend._normalize_open_order_item(frontend_row()))
        self.assertEqual(hard, [])
        self.assertEqual(soft, [])

    def test_order_not_yet_visible_is_unknown(self):
        hard, soft = self.check(None)
        self.assertEqual(hard, [])
        self.assertIn("order_not_found_in_open_orders", soft)

    def test_field_poor_payload_yields_no_contradictions(self):
        row = Backend._normalize_open_order_item(
            {"coin": "UNI", "oid": "S1", "side": "A", "sz": "46.3"})
        hard, soft = self.check(row)
        self.assertEqual(hard, [], "silence must never be read as a contradiction")
        self.assertTrue(soft)

    def test_wrong_side_is_contradiction(self):
        hard, _ = self.check(Backend._normalize_open_order_item(frontend_row(side="B")))
        self.assertTrue(any(i.startswith("wrong_side") for i in hard), hard)

    def test_stop_resting_as_take_profit_is_contradiction(self):
        # The dangerous confusion: a TP rests on the opposite side of price.
        hard, _ = self.check(Backend._normalize_open_order_item(
            frontend_row(order_type="Take Profit Market")))
        self.assertTrue(any(i.startswith("wrong_tpsl") for i in hard), hard)

    def test_trigger_price_mismatch_is_contradiction(self):
        hard, _ = self.check(Backend._normalize_open_order_item(
            frontend_row(trigger_px="88.0")))
        self.assertTrue(any(i.startswith("trigger_price_mismatch") for i in hard), hard)

    def test_size_mismatch_is_contradiction(self):
        hard, _ = self.check(Backend._normalize_open_order_item(frontend_row(sz="12.0")))
        self.assertTrue(any(i.startswith("size_mismatch") for i in hard), hard)

    def test_not_reduce_only_is_contradiction(self):
        # Worse than no stop: it can open a position instead of closing one.
        hard, _ = self.check(Backend._normalize_open_order_item(
            frontend_row(reduce_only=False)))
        self.assertIn("not_reduce_only", hard)

    def test_not_a_trigger_order_is_contradiction(self):
        hard, _ = self.check(Backend._normalize_open_order_item(
            frontend_row(is_trigger=False, order_type="Limit")))
        self.assertIn("not_trigger_order", hard)

    def test_trigger_price_tolerance_allows_rounding(self):
        hard, _ = self.check(Backend._normalize_open_order_item(
            frontend_row(trigger_px="95.001")))
        self.assertEqual(hard, [])


class StubBackend(Backend):
    """Real rounding, real relation checks, real verification -- no network.

    Only the four methods that actually touch Hyperliquid are replaced, so the
    status computation under test is the one that runs in production.
    """

    def __init__(self, resting, stop_oid="S1", tp_oid="T1"):
        self._sz_decimals = {"UNI": 1}
        self._max_leverage = {"UNI": 10}
        self._resting = resting          # None models an unreachable endpoint
        self._stop_oid, self._tp_oid = stop_oid, tp_oid
        self.debug = False

    def get_mid_price(self, coin):
        return 100.0

    def _get_venue_signed_size(self, coin):
        return 46.3

    def _place_trigger_reduce_only(self, coin, close_is_buy, size_coin,
                                   trigger_price, tpsl, trigger_is_market,
                                   allow_limit_fallback):
        return (self._stop_oid if tpsl == "sl" else self._tp_oid), ""

    def get_open_orders(self, coin=""):
        if self._resting is None:
            return None
        return [Backend._normalize_open_order_item(r) for r in self._resting]

    def protect(self):
        return self.place_native_protection(coin="UNI", side="LONG", size_coin=46.3,
                                            stop_price=95.0, tp_price=115.0)


def good_stop(**kw):
    kw.setdefault("oid", "S1")
    kw.setdefault("trigger_px", "95.0")
    kw.setdefault("order_type", "Stop Market")
    return frontend_row(**kw)


def good_tp(**kw):
    kw.setdefault("oid", "T1")
    kw.setdefault("trigger_px", "115.0")
    kw.setdefault("order_type", "Take Profit Market")
    return frontend_row(**kw)


class ProtectionStatusTest(unittest.TestCase):
    """Whether the verdict reaches the only field the caller acts on.

    tools/live_levels_trader closes any position whose protection status is
    not exactly "protected", so status IS the safety decision.
    """

    def test_both_legs_verified_is_protected(self):
        self.assertEqual(StubBackend([good_stop(), good_tp()]).protect()["status"],
                         "protected")

    def test_contradicted_stop_is_not_protected(self):
        # The headline regression: before this fix, an OID was enough.
        for label, row in [
            ("wrong side", good_stop(side="B")),
            ("resting as a take-profit", good_stop(order_type="Take Profit Market")),
            ("trigger 8% away", good_stop(trigger_px="88.0")),
            ("not reduce-only", good_stop(reduce_only=False)),
            ("not a trigger order", good_stop(is_trigger=False, order_type="Limit")),
        ]:
            with self.subTest(label):
                res = StubBackend([row, good_tp()]).protect()
                self.assertNotEqual(res["status"], "protected")
                self.assertIs(res["stop_verified"], False)
                self.assertIn("verify_fail", res["error"])

    def test_contradicted_tp_is_not_protected(self):
        res = StubBackend([good_stop(), good_tp(trigger_px="150.0")]).protect()
        self.assertNotEqual(res["status"], "protected")
        self.assertIs(res["tp_verified"], False)

    def test_orders_not_yet_visible_stays_protected(self):
        # The opposite failure mode, and the reason this is not a blanket
        # downgrade: on 2026-09-18 a correctly protected position was
        # emergency-closed by a check that fired on success. A snapshot that
        # lags the ack must not close a real position.
        res = StubBackend([]).protect()
        self.assertEqual(res["status"], "protected")
        self.assertIn("verify_warn", res["error"])
        self.assertNotIn("verify_fail", res["error"])

    def test_unreachable_open_orders_stays_protected(self):
        res = StubBackend(None).protect()
        self.assertEqual(res["status"], "protected")
        self.assertIn("open_orders_verification_unavailable", res["error"])

    def test_field_poor_payload_stays_protected(self):
        # What the plain openOrders endpoint returns. It cannot confirm the
        # stop, but it cannot impeach it either.
        poor = {"coin": "UNI", "oid": "S1", "side": "A", "sz": "46.3"}
        poor_tp = {"coin": "UNI", "oid": "T1", "side": "A", "sz": "46.3"}
        res = StubBackend([poor, poor_tp]).protect()
        self.assertEqual(res["status"], "protected")
        self.assertIn("verify_warn", res["error"])

    def test_oids_survive_a_contradiction(self):
        # They are the only handle on cancelling the bad order.
        res = StubBackend([good_stop(side="B"), good_tp()]).protect()
        self.assertEqual(res["stop_order_id"], "S1")
        self.assertEqual(res["tp_order_id"], "T1")

    def test_early_return_still_reports_verification_keys(self):
        # Callers read these unconditionally; the bail-out paths place nothing,
        # so they must still answer "not verified" rather than raise.
        res = StubBackend([]).place_native_protection(
            coin="UNI", side="LONG", size_coin=0.0, stop_price=95.0, tp_price=115.0)
        self.assertEqual(res["status"], "missing")
        self.assertIs(res["stop_verified"], False)
        self.assertIs(res["tp_verified"], False)

    def test_rejected_stop_is_not_protected(self):
        res = StubBackend([good_tp()], stop_oid=None).protect()
        self.assertNotEqual(res["status"], "protected")


class StopOnlyStatusTest(unittest.TestCase):
    """place_stop_only had no verification at all -- an OID was enough.

    Nothing on the levels path calls it today, but it is a public,
    order-capable method that reports the same "protected" the caller treats
    as permission to keep a position open.
    """

    def stop_only(self, resting, stop_oid="S1"):
        b = StubBackend(resting, stop_oid=stop_oid)
        return b.place_stop_only(coin="UNI", side="LONG", size_coin=46.3,
                                 stop_price=95.0)

    def test_verified_stop_is_protected(self):
        self.assertEqual(self.stop_only([good_stop()])["status"], "protected")

    def test_contradicted_stop_is_not_protected(self):
        res = self.stop_only([good_stop(side="B")])
        self.assertEqual(res["status"], "missing")
        self.assertIs(res["stop_verified"], False)
        self.assertIn("verify_fail", res["error"])

    def test_unreported_fields_stay_protected(self):
        res = self.stop_only([{"coin": "UNI", "oid": "S1", "side": "A", "sz": "46.3"}])
        self.assertEqual(res["status"], "protected")
        self.assertIn("verify_warn", res["error"])

    def test_rejected_order_is_missing(self):
        res = self.stop_only([], stop_oid=None)
        self.assertEqual(res["status"], "missing")
        self.assertIs(res["stop_verified"], False)


class AssetMetadataTest(unittest.TestCase):
    """_load_meta ran once, in __init__, and swallowed its own failure.

    On 2026-09-19 a 429 during startup produced "assets_loaded=0" and the
    backend carried on for the life of the process. Empty metadata is not
    harmless: every coin falls back to szDecimals=3 and a 1e-3 price tick, so
    sizes and trigger prices get rounded to a precision the venue never asked
    for -- silently malformed orders on the path whose job is placing stops.
    """

    class Meta:
        def __init__(self, fail_times=0):
            self.fail_times, self.calls = fail_times, 0

        def meta(self):
            self.calls += 1
            if self.calls <= self.fail_times:
                raise RuntimeError("429 rate limited")
            return {"universe": [{"name": "UNI", "szDecimals": 1},
                                 {"name": "BTC", "szDecimals": 5}]}

    def backend(self, fail_times=0):
        b = Backend.__new__(Backend)
        b._sz_decimals, b._max_leverage, b.debug = {}, {}, False
        b.info = self.Meta(fail_times)
        return b

    def test_populated_metadata_needs_no_reload(self):
        b = self.backend()
        b._sz_decimals = {"UNI": 1}
        self.assertTrue(b._ensure_meta())
        self.assertEqual(b.info.calls, 0)

    def test_empty_metadata_is_retried_and_recovers(self):
        b = self.backend()
        self.assertTrue(b._ensure_meta())
        self.assertEqual(b._sz_decimals["BTC"], 5)

    def test_still_failing_metadata_reports_false(self):
        b = self.backend(fail_times=99)
        self.assertFalse(b._ensure_meta())

    def test_protection_refuses_without_metadata(self):
        b = StubBackend([good_stop(), good_tp()])
        b._sz_decimals, b._max_leverage = {}, {}
        b.info = self.Meta(fail_times=99)
        res = b.protect()
        self.assertEqual(res["status"], "missing")
        self.assertIn("asset_metadata_unavailable", res["error"])

    def test_stop_only_refuses_without_metadata(self):
        b = StubBackend([good_stop()])
        b._sz_decimals, b._max_leverage = {}, {}
        b.info = self.Meta(fail_times=99)
        res = b.place_stop_only(coin="UNI", side="LONG", size_coin=46.3,
                                stop_price=95.0)
        self.assertEqual(res["status"], "missing")
        self.assertIn("asset_metadata_unavailable", res["error"])


class CancelOrderTest(unittest.TestCase):
    """The SDK is cancel(name, oid: int). We were sending a string.

    The venue answered 422 "Failed to deserialize the JSON body into the
    target type", cancel_order returned False, and every caller only printed
    -- so every cancel this system ever attempted failed silently. Those are
    the cleanup paths: protection orders after an emergency exit, and the
    expiry of a resting bid. Their entire job is to leave nothing behind.
    """

    class Ex:
        def __init__(self):
            self.seen = []

        def cancel(self, name, oid):
            self.seen.append((name, oid))
            if not isinstance(oid, int):
                raise RuntimeError("422 Failed to deserialize the JSON body")
            return {"status": "ok"}

    def backend(self):
        b = Backend.__new__(Backend)
        b.exchange, b.debug = self.Ex(), False
        return b

    def test_the_oid_is_sent_as_an_int(self):
        b = self.backend()
        self.assertTrue(b.cancel_order("UNI", "549652894820"))
        self.assertEqual(b.exchange.seen, [("UNI", 549652894820)])

    def test_an_int_oid_also_works(self):
        b = self.backend()
        self.assertTrue(b.cancel_order("UNI", 549652894820))

    def test_a_non_numeric_oid_is_refused_without_calling_the_venue(self):
        b = self.backend()
        self.assertFalse(b.cancel_order("UNI", "not-an-oid"))
        self.assertEqual(b.exchange.seen, [])

    def test_missing_arguments_are_refused(self):
        b = self.backend()
        self.assertFalse(b.cancel_order("", "1"))
        self.assertFalse(b.cancel_order("UNI", ""))


class EndpointPreferenceTest(unittest.TestCase):
    """The plain openOrders endpoint cannot carry a verifiable answer."""

    def test_frontend_open_orders_is_preferred(self):
        calls = []

        class Info:
            def open_orders(self, addr, dex=""):
                calls.append("open_orders")
                return []

            def frontend_open_orders(self, addr, dex=""):
                calls.append("frontend_open_orders")
                return []

        b = Backend.__new__(Backend)
        b.info, b.debug, b._sz_decimals = Info(), False, {"UNI": 1}
        b._user_state_query_address = lambda: "0xabc"
        b.get_open_orders("UNI")
        self.assertEqual(calls, ["frontend_open_orders"])


if __name__ == "__main__":
    unittest.main()

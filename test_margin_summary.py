"""Tests for unified-account margin reading in live_execution_backend.

Every position this system opens is sized from `available`. If that number
reports collateral as free while it is committed to an open position, the
next trade is sized against money that is already working -- which is how
2026-09-18 produced 48 entry attempts and 30 "Insufficient margin" rejections
inside 50 seconds.

`available` was `USDC.total - USDC.hold`, and `hold` is a SPOT order lock.
Whether perp margin ever appears in it could not be established from a flat
account: "hold is 0 because nothing is held" and "hold never reflects perp
margin" look identical when nothing is open.

Verified against the live venue on 2026-09-18: for a unified account the perp
clearinghouse reports accountValue, totalMarginUsed and withdrawable all
"0.0" even when funded, and spotClearinghouseState carries
tokenToAvailableAfterMaintenance -- the venue's own answer to how much is
actually free, net of maintenance margin. That field is correct whichever way
`hold` behaves, so the ambiguity stops mattering.
"""

import unittest

from live_execution_backend import LiveExecutionBackend as Backend


def spot_state(total="768.58249", hold="0.0", available="768.58249",
               token=0, include_tam=True):
    state = {
        "balances": [
            {"coin": "USDC", "token": token, "total": total, "hold": hold,
             "entryNtl": "0.0"},
            {"coin": "USDT0", "token": 268, "total": "0.0", "hold": "0.0"},
        ],
    }
    if include_tam:
        state["tokenToAvailableAfterMaintenance"] = [[token, available]]
    return state


class StubInfo:
    def __init__(self, state, boom=False):
        self.state, self.boom = state, boom

    def spot_user_state(self, addr):
        if self.boom:
            raise RuntimeError("venue unreachable")
        return self.state


class StubBackend(Backend):
    def __init__(self, state, boom=False):
        self.info = StubInfo(state, boom)
        self.debug = False

    def read(self):
        return self._get_margin_summary_unified("0xabc", "MAINNET")


class UnifiedMarginTest(unittest.TestCase):

    def test_flat_account_reports_everything_free(self):
        equity, used, available = StubBackend(spot_state()).read()
        self.assertAlmostEqual(equity, 768.58249)
        self.assertAlmostEqual(available, 768.58249)
        self.assertAlmostEqual(used, 0.0)

    def test_open_position_reduces_available_even_when_hold_is_zero(self):
        # The regression. hold stays "0.0" while collateral is committed; only
        # tokenToAvailableAfterMaintenance reports the truth. Under the old
        # `total - hold` reading this returned the full balance as free.
        equity, used, available = StubBackend(
            spot_state(total="768.58", hold="0.0", available="500.00")).read()
        self.assertAlmostEqual(equity, 768.58)
        self.assertAlmostEqual(available, 500.00)
        self.assertAlmostEqual(used, 268.58, places=2)

    def test_used_and_available_always_reconcile_to_equity(self):
        equity, used, available = StubBackend(
            spot_state(total="1000.0", available="250.0")).read()
        self.assertAlmostEqual(used + available, equity)

    def test_falls_back_to_total_minus_hold_when_field_absent(self):
        equity, used, available = StubBackend(
            spot_state(total="1000.0", hold="150.0", include_tam=False)).read()
        self.assertAlmostEqual(equity, 1000.0)
        self.assertAlmostEqual(available, 850.0)
        self.assertAlmostEqual(used, 150.0)

    def test_token_id_is_matched_not_assumed_zero(self):
        equity, _used, available = StubBackend(
            spot_state(total="500.0", available="400.0", token=7)).read()
        self.assertAlmostEqual(equity, 500.0)
        self.assertAlmostEqual(available, 400.0)

    def test_available_for_a_different_token_is_ignored(self):
        state = spot_state(total="500.0", hold="0.0", include_tam=False)
        state["tokenToAvailableAfterMaintenance"] = [[268, "9999.0"]]
        equity, _used, available = StubBackend(state).read()
        self.assertAlmostEqual(available, 500.0, msg="fell back, not borrowed")

    def test_available_is_clamped_to_equity(self):
        _equity, used, available = StubBackend(
            spot_state(total="100.0", available="250.0")).read()
        self.assertAlmostEqual(available, 100.0)
        self.assertAlmostEqual(used, 0.0)

    def test_negative_available_floors_at_zero(self):
        _equity, used, available = StubBackend(
            spot_state(total="100.0", available="-5.0")).read()
        self.assertAlmostEqual(available, 0.0)
        self.assertAlmostEqual(used, 100.0)

    def test_missing_usdc_balance_returns_none(self):
        self.assertIsNone(StubBackend({"balances": []}).read())

    def test_query_failure_returns_none(self):
        # None means "no data". A caller must never read it as zero equity.
        self.assertIsNone(StubBackend(spot_state(), boom=True).read())

    def test_unparseable_amount_falls_back_rather_than_inventing(self):
        state = spot_state(total="500.0", hold="0.0", include_tam=False)
        state["tokenToAvailableAfterMaintenance"] = [[0, "not-a-number"]]
        _equity, _used, available = StubBackend(state).read()
        self.assertAlmostEqual(available, 500.0)


if __name__ == "__main__":
    unittest.main()

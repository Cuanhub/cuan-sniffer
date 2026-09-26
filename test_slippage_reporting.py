"""What 'slippage' means in the order logs.

Slippage is how much WORSE than the market we filled. That makes the
reference the price the market was showing -- not the price we asked for.

The two differ on exits by construction. execute_exit sets
ref_price = min(mid, trigger) for a sell, which is how the IOC limit is
guaranteed to sit at or below the book so the close cannot fail to fill. It is
a floor on what we will accept, not a prediction of what we will get. Measuring
against it reports the gap between the trigger and the market, which on a
profitable exit is the P&L wearing slippage's name: PONS closed at the mid
exactly on 2026-09-26 and was logged as 376bps of slippage, which is what sent
somebody looking for an execution problem that was never there.

Nothing gates on this number today. It is logged, and a logged number that
lies costs review time -- which is the only reason it is pinned here.
"""

import unittest

from live_execution_backend import LiveExecutionBackend


class FakeFillParse(unittest.TestCase):
    """_parse_fill is pure arithmetic over a response dict, so it is exercised
    directly rather than through a live backend."""

    def parse(self, fill_px, sz, ref_price, is_exit, slip_ref=None, is_buy=None):
        resp = {"status": "ok", "response": {"data": {"statuses": [
            {"filled": {"avgPx": str(fill_px), "totalSz": str(sz), "oid": 1}}
        ]}}}
        return LiveExecutionBackend._parse_fill(
            _NoNetwork(), resp, coin="PONS",
            requested_notional=sz * fill_px, ref_price=ref_price,
            is_exit=is_exit, slip_ref=slip_ref, is_buy=is_buy)

    # -- the PONS case -----------------------------------------------------

    def test_a_sell_filled_at_the_mid_has_no_slippage(self):
        # The real numbers: trigger 0.6184 (the entry), mid 0.6417, filled at
        # the mid. A perfect fill.
        r = self.parse(0.6417, 1.0, ref_price=0.6184, is_exit=True,
                       slip_ref=0.6417, is_buy=False)
        self.assertTrue(r.filled)
        self.assertAlmostEqual(r.slippage_bps, 0.0, places=6)

    def test_measuring_against_the_trigger_is_what_reported_376bps(self):
        # Kept as the counter-example: this is the old behaviour, and the
        # number it produces is the move from entry to market, not slippage.
        r = self.parse(0.6417, 1.0, ref_price=0.6184, is_exit=True,
                       slip_ref=None, is_buy=None)
        self.assertGreater(r.slippage_bps, 300)

    # -- sign ---------------------------------------------------------------

    def test_a_sell_below_the_mid_is_real_slippage(self):
        r = self.parse(99.0, 1.0, ref_price=100.0, is_exit=True,
                       slip_ref=100.0, is_buy=False)
        self.assertAlmostEqual(r.slippage_bps, 100.0, places=6)

    def test_a_buy_above_the_mid_is_real_slippage(self):
        r = self.parse(101.0, 1.0, ref_price=100.0, is_exit=False,
                       slip_ref=100.0, is_buy=True)
        self.assertAlmostEqual(r.slippage_bps, 100.0, places=6)

    def test_a_buy_below_the_mid_is_a_better_fill_not_slippage(self):
        r = self.parse(99.0, 1.0, ref_price=100.0, is_exit=False,
                       slip_ref=100.0, is_buy=True)
        self.assertAlmostEqual(r.slippage_bps, 0.0, places=6)

    def test_a_short_entry_sells_so_a_low_fill_is_adverse(self):
        # The case is_exit alone gets wrong: an entry that SELLS.
        r = self.parse(99.0, 1.0, ref_price=100.0, is_exit=False,
                       slip_ref=100.0, is_buy=False)
        self.assertAlmostEqual(r.slippage_bps, 100.0, places=6)

    def test_a_short_exit_buys_so_a_high_fill_is_adverse(self):
        r = self.parse(101.0, 1.0, ref_price=100.0, is_exit=True,
                       slip_ref=100.0, is_buy=True)
        self.assertAlmostEqual(r.slippage_bps, 100.0, places=6)


class _NoNetwork:
    """Just enough of the backend for _parse_fill: it cancels stray resting
    IOC orders, and there are none in these responses."""

    def _cancel_resting_oids(self, *a, **k):
        return None

    def estimate_exit_fee(self, notional):
        return notional * 0.00045

    def estimate_entry_fee(self, notional):
        return notional * 0.00045

    def _reject(self, *a, **k):
        raise AssertionError("these responses all contain a fill")


if __name__ == "__main__":
    unittest.main()

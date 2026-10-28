"""The backtest's copies of live constants must equal the live ones.

research/ may not import core/ (it places orders), so research/backtest keeps
its own copy of the few live constants it needs. This file -- which stays in
the live repo and is NOT shipped to the OpenClaw lab -- is where the two are
held equal.
"""

import unittest

from core import protect
from research.backtest import band


class LiveParity(unittest.TestCase):
    def test_bid_ttl_is_production_s(self):
        self.assertEqual(band.BID_TTL_BARS, protect.BID_TTL_HOURS)

    def test_the_outcome_recorder_uses_the_live_bid_life(self):
        from research.levels import zone_log
        self.assertEqual(zone_log.BID_TTL_BARS, protect.BID_TTL_HOURS)


if __name__ == "__main__":
    unittest.main()

"""Writing down the setups we actually traded, so outcomes can be joined to them.

The trader computed a full 55-column episode through find_candidates, acted on
it, and threw it away. Measured 2026-10-07: 0 of 39 traded episode ids existed
in episodes.csv, so not one of the 654 rows of open interest, funding and book
depth could be joined to a real outcome. We were collecting data with no path
to using it.

The two sides also mint incompatible ids -- scanner.scan_coin uses
make_episode_id(coin, ...) while call.find_candidates uses coin + "-" + tf --
so they could never have matched even if both had been written. That is left
alone: they are genuinely different populations (the scanner is 4h and
onset-only; call scans 4h and 1h and includes flipped resistances), and the
join that matters is trade -> its own episode -> its own outcome.
"""

import unittest

import tools.live_levels_trader as T
from research.levels import storage


def episode(**over):
    e = {"episode_id": "AAA-1H_1791385200000_100", "coin": "AAA",
         "timeframe": "1h", "bar_ts_ms": 1791385200000, "level_price": 100.0,
         "sigma_4h": 0.01, "suggested_entry": 100.35, "suggested_stop": 98.0,
         "suggested_tp": 110.0, "suggested_rr": 4.1, "near_sigma": 5.0,
         "_px": 105.0, "_ladder": [(110.0, 4.1, 5, 0.8)], "_tf": "1h"}
    e.update(over)
    return e


class Persisting(unittest.TestCase):
    def setUp(self):
        self.written = []
        self._ae, self._ro = storage.append_episode, T._ro_client
        storage.append_episode = lambda row: (self.written.append(row) or True)

        class RO:
            def get_perp_contexts(self_):
                return {"AAA": {"oi_base": 1000.0, "oi_notional_usd": 100000.0,
                                "funding_rate": 1.25e-05, "premium": 0.0001,
                                "mark_px": 100.0, "oracle_px": 100.1,
                                "impact_bid": 99.9, "impact_ask": 100.1,
                                "impact_spread_bps": 20.0, "day_ntl_vlm": 7.9e6,
                                "prev_day_px": 98.0}}
        T._ro_client = lambda: RO()

    def tearDown(self):
        storage.append_episode, T._ro_client = self._ae, self._ro

    def test_it_writes_the_episode(self):
        self.assertTrue(T.persist_traded_episode(episode()))
        self.assertEqual(len(self.written), 1)
        self.assertEqual(self.written[0]["episode_id"], "AAA-1H_1791385200000_100")

    def test_private_keys_are_stripped(self):
        T.persist_traded_episode(episode())
        self.assertFalse([k for k in self.written[0] if str(k).startswith("_")])

    def test_the_context_is_attached(self):
        T.persist_traded_episode(episode())
        row = self.written[0]
        self.assertEqual(row["oi_notional_usd"], 100000.0)
        self.assertEqual(row["impact_spread_bps"], 20.0)
        self.assertTrue(row["ctx_captured_at_utc"].endswith("Z"))

    def test_everything_the_resolver_needs_survives(self):
        """resolve_outcomes reads exactly these; without them the row is
        written but never acquires an outcome, which is the whole point."""
        T.persist_traded_episode(episode())
        row = self.written[0]
        for k in ("coin", "bar_ts_ms", "suggested_entry", "suggested_stop",
                  "suggested_tp", "sigma_4h"):
            self.assertIn(k, row)
            self.assertNotIn(row[k], ("", None))

    def test_the_coin_field_stays_the_plain_symbol(self):
        """resolve_outcomes groups by coin to fetch candles once. If this
        carried the '-1H' suffix the fetch would fail for every traded row."""
        T.persist_traded_episode(episode())
        self.assertEqual(self.written[0]["coin"], "AAA")


class ItNeverDisturbsAnOpenPosition(unittest.TestCase):
    """A research row is worth nothing next to a live position, so every
    failure here is swallowed."""

    def setUp(self):
        self._ae, self._ro = storage.append_episode, T._ro_client

    def tearDown(self):
        storage.append_episode, T._ro_client = self._ae, self._ro

    def test_a_failed_context_fetch_still_persists_the_row(self):
        written = []
        storage.append_episode = lambda row: (written.append(row) or True)

        class Boom:
            def get_perp_contexts(self):
                raise RuntimeError("429 rate limited")

        T._ro_client = lambda: Boom()
        self.assertTrue(T.persist_traded_episode(episode()))
        self.assertEqual(written[0].get("oi_base", ""), "")

    def test_a_failed_write_returns_false_rather_than_raising(self):
        def boom(row):
            raise OSError("disk full")
        storage.append_episode = boom
        T._ro_client = lambda: type("R", (), {"get_perp_contexts": lambda s: {}})()
        self.assertFalse(T.persist_traded_episode(episode()))

    def test_a_malformed_episode_does_not_raise(self):
        storage.append_episode = lambda row: True
        T._ro_client = lambda: type("R", (), {"get_perp_contexts": lambda s: {}})()
        for bad in ({}, {"coin": None}):
            self.assertIn(T.persist_traded_episode(bad), (True, False))


class WiredAtBothFills(unittest.TestCase):
    def test_the_market_path_persists(self):
        import inspect
        self.assertIn("persist_traded_episode", inspect.getsource(T.execute_one))

    def test_the_resting_bid_path_persists_on_FILL_not_on_placement(self):
        """A bid that is never hit is not a trade. Writing it at placement
        would put setups we did not take into the set used to judge the ones
        we did."""
        import inspect
        place = inspect.getsource(T.rest_one_bid)
        fill = inspect.getsource(T.persist_filled_bids)
        self.assertNotIn("persist_traded_episode", place)
        self.assertIn("persist_traded_episode", fill)
        self.assertIn("pending_episode", place,
                      "the episode must be carried until something observes "
                      "a position")
        # and the pass loop has to actually call it, or the episode is
        # carried for ever and never written
        self.assertIn("persist_filled_bids", inspect.getsource(T.main))


if __name__ == "__main__":
    unittest.main()

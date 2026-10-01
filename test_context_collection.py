"""Positioning and liquidity data: collected, and provably not traded on.

Every one of the 43 features the episode set already carried is a transform of
OHLCV. Screening 19 of them against expectancy over 1,060 resolved episodes
found exactly one that survives controlling for target distance
(level_pivot_count, rho +0.088). The shortage was never price indicators -- it
was that there is no positioning, flow or liquidity data at all.

So these twelve columns exist to be SCREENED, not used. The point of this file
is the inertness: the trading path must not read them, and a missing reading
must not look like a real zero when the screen eventually runs.
"""

import inspect
import unittest

from research.levels import scanner, storage

CTX_FIELDS = ("ctx_captured_at_utc", "oi_base", "oi_notional_usd",
              "funding_rate", "premium", "mark_px", "oracle_px", "impact_bid",
              "impact_ask", "impact_spread_bps", "day_ntl_vlm", "prev_day_px")


def bars(n=200, base=100.0):
    out, t = [], 1_700_000_000_000
    for k in range(n):
        px = base + 6.0 * __import__("math").sin(k / 23.0) + 0.02 * k
        low = px - (4.5 if k % 37 == 0 else 0.6)
        out.append({"t": t + k * 14_400_000, "o": px, "h": px + 0.6,
                    "l": low, "c": px, "v": 1000.0})
    return out


class Schema(unittest.TestCase):
    def test_the_columns_exist_and_are_appended_at_the_end(self):
        """storage widens an existing file positionally, so a new column that
        is not a suffix would corrupt every historical row."""
        self.assertEqual(tuple(storage.EPISODE_FIELDNAMES[-12:]), CTX_FIELDS)

    def test_no_duplicates_crept_in(self):
        f = storage.EPISODE_FIELDNAMES
        self.assertEqual(len(f), len(set(f)))


class Inertness(unittest.TestCase):
    """The guarantee that matters while this is still an experiment."""

    def test_the_live_filter_does_not_read_any_of_them(self):
        import tools.live_levels_trader as T
        src = inspect.getsource(T.passes_quality)
        for f in CTX_FIELDS:
            self.assertNotIn(f, src, f"passes_quality reads {f}")

    def test_the_logging_gate_does_not_read_any_of_them(self):
        from research.levels import gate
        src = inspect.getsource(gate)
        for f in CTX_FIELDS:
            self.assertNotIn(f, src, f"the gate reads {f}")

    def test_the_trading_module_never_mentions_them(self):
        import tools.live_levels_trader as T
        src = inspect.getsource(T)
        for f in CTX_FIELDS:
            self.assertNotIn(f, src, f"the trader references {f}")

    def test_the_trader_does_not_even_fetch_contexts(self):
        """The trading path is unchanged: the collector fetches this, not the
        thing that places orders."""
        import tools.live_levels_trader as T
        self.assertNotIn("get_perp_contexts", inspect.getsource(T))


class Capture(unittest.TestCase):
    """The gate is deliberately stubbed here.

    A synthetic series that satisfies run>=6 sigma, pull>=1 sigma AND a retest
    is fiddly to construct, and the thing under test is the context columns,
    not the gate. Stubbing it means these assertions actually run -- an earlier
    version of this file skipped three of them because the series gated nothing,
    which is a test that reports success for having done nothing.
    """

    CTX = {"AAA": {"oi_base": 1000.0, "oi_notional_usd": 100000.0,
                   "funding_rate": 1.25e-05, "premium": 0.0001,
                   "mark_px": 100.0, "oracle_px": 100.1,
                   "impact_bid": 99.9, "impact_ask": 100.1,
                   "impact_spread_bps": 20.0, "day_ntl_vlm": 7.9e6,
                   "prev_day_px": 98.0}}

    def setUp(self):
        self._evaluate = scanner.evaluate
        self._fired = scanner._fired_previous_bar

        def always_fires(bars_, i, level_price):
            return {"sigma_4h": 0.01, "run_sigma": 9.0, "pull_sigma": 2.0,
                    "run_pct": 9.0, "pull_pct": 2.0, "bars_since_high": 5,
                    "retests": 3, "bars_since_last_retest": 2,
                    "near_sigma": 4.0, "dist_to_level_pct": 4.0,
                    "price": bars_[i]["c"], "cond_run_pullback": True,
                    "cond_retest": True, "cond_near": True, "gate_fires": True}

        scanner.evaluate = always_fires
        scanner._fired_previous_bar = lambda *a, **k: False

    def tearDown(self):
        scanner.evaluate = self._evaluate
        scanner._fired_previous_bar = self._fired

    def rows(self, ctx):
        out = scanner.scan_coin("AAA", bars(), ctx=ctx)
        self.assertTrue(out, "the stubbed gate should have produced rows")
        return out

    def test_the_snapshot_lands_on_every_row(self):
        for r in self.rows(self.CTX):
            self.assertEqual(r["oi_notional_usd"], 100000.0)
            self.assertEqual(r["impact_spread_bps"], 20.0)
            self.assertEqual(r["funding_rate"], 1.25e-05)
            self.assertTrue(r["ctx_captured_at_utc"].endswith("Z"))

    def test_a_missing_coin_leaves_blanks_not_zeros(self):
        """A failed reading must never be mistaken for an open interest of nil
        by whatever screens this later."""
        for r in self.rows({"SOMETHING_ELSE": {}}):
            for f in CTX_FIELDS:
                self.assertEqual(r[f], "", f"{f} should be blank, got {r[f]!r}")

    def test_no_context_at_all_is_handled(self):
        for ctx in (None, {}):
            for r in self.rows(ctx):
                self.assertEqual(r["oi_base"], "")
                self.assertEqual(r["ctx_captured_at_utc"], "")

    def test_an_explicit_none_becomes_blank_too(self):
        for r in self.rows({"AAA": {"oi_base": None, "funding_rate": None}}):
            self.assertEqual(r["oi_base"], "")
            self.assertEqual(r["funding_rate"], "")

    def test_the_lookup_matches_the_key_the_producer_emits(self):
        """scan_coin looks up coin.upper(), which is correct precisely because
        get_perp_contexts uppercases its keys. Pinned as one fact rather than
        softened with a case-insensitive lookup for an input that cannot
        occur -- the producer is the contract.
        """
        import inspect
        from research.levels.venue_client import HyperliquidReadOnlyClient
        src = inspect.getsource(HyperliquidReadOnlyClient.get_perp_contexts)
        self.assertIn('.upper()', src,
                      "get_perp_contexts must emit uppercase coin keys")
        for r in self.rows({"AAA": {"oi_base": 7.0}}):
            self.assertEqual(r["oi_base"], 7.0)

    def test_every_context_column_is_present_on_every_row(self):
        for r in self.rows(self.CTX):
            for f in CTX_FIELDS:
                self.assertIn(f, r)

    def test_scanning_still_works_with_the_argument_omitted(self):
        """Backfill calls scan_series, which does not supply context."""
        self.assertTrue(scanner.scan_coin("AAA", bars()))


class ContextFetchFailureIsSurvivable(unittest.TestCase):
    def test_run_pass_continues_when_the_context_call_raises(self):
        class Client:
            def get_perp_contexts(self):
                raise RuntimeError("429 rate limited")

            def get_candles(self, coin, tf, start, end):
                return [{"t": b["t"], "o": b["o"], "h": b["h"], "l": b["l"],
                         "c": b["c"], "v": b["v"]} for b in bars()]

        found = scanner.run_pass(Client(), [{"symbol": "AAA"}], dry_run=True)
        self.assertIsInstance(found, list)
        for r in found:
            self.assertEqual(r["oi_base"], "")


if __name__ == "__main__":
    unittest.main()

"""Refusing a long when the perp trades rich to spot.

The first feature in this project to reach statistical significance. Screened
over 248 resolved episodes carrying positioning context, scored on expectancy
and stratified by reward/risk exactly as the price features were -- all
nineteen of which failed one check or the other:

    mark-vs-oracle basis        n    mean R   win rate
    Q1   -52.5 .. -2.0 bps      62    -0.052      27%
    Q2    -1.9 .. +1.9          62    +0.006      27%
    Q3    +2.0 .. +7.7          62    -0.485      16%
    Q4    +7.8 .. +29.3         62    -0.694      11%

    perp ABOVE oracle  n=153  mean R -0.524  p = 0.0035
    perp AT or BELOW   n= 95  mean R +0.044  p = 0.0024

Mark above oracle means perp buyers are paying a premium over spot --
leveraged longs crowding in. It is the mechanism funding is meant to measure
and cannot, because Hyperliquid pins funding at base 77% of the time.

0.0 is not tuned. It is the natural boundary and the only cut in the sweep
where the KEPT set has positive expectancy: +0.044 at 0bps against -0.031 at
2, -0.083 at 4, -0.127 at 6.
"""

import unittest

import tools.live_levels_trader as T
from research.levels import config as C


class Backend:
    def __init__(self, mid): self._mid = mid
    def mid(self, coin): return self._mid


def bars(low=99.5, n=40):
    out = [{"t": i*3_600_000, "o": 100.0, "h": 101.0, "l": 99.5, "c": 100.0}
           for i in range(n)]
    out.append({"t": n*3_600_000, "o": 100.0, "h": 100.5, "l": low, "c": 100.0})
    return out


class TheVeto(unittest.TestCase):
    LEVEL, STOP, TP, ENTRY, MID = 100.0, 98.0, 110.0, 100.35, 100.3

    def check(self, mark, oracle):
        return T.entry_is_actionable(
            Backend(self.MID), "TEST", self.ENTRY, self.STOP, self.TP,
            level=self.LEVEL, bars_1h=bars(),
            ctx={"mark_px": mark, "oracle_px": oracle})

    def test_perp_at_parity_passes(self):
        ref, why = self.check(100.0, 100.0)
        self.assertIsNotNone(ref)

    def test_perp_below_oracle_passes(self):
        """The +0.044R bucket -- the only one with positive expectancy."""
        ref, why = self.check(99.95, 100.0)
        self.assertIsNotNone(ref)

    def test_perp_rich_to_oracle_is_refused(self):
        ref, why = self.check(100.10, 100.0)   # +10 bps
        self.assertIsNone(ref)
        self.assertIn("rich to oracle", why)
        self.assertIn("longs are paying up", why)

    def test_the_boundary_is_parity(self):
        just_under, _ = self.check(99.999, 100.0)
        self.assertIsNotNone(just_under)
        just_over, _ = self.check(100.001, 100.0)
        self.assertIsNone(just_over)

    def test_the_reported_basis_is_in_bps(self):
        _, why = self.check(100.25, 100.0)     # +25 bps
        self.assertIn("+25.0bps", why)


class ItAbstainsRatherThanGuesses(unittest.TestCase):
    LEVEL, STOP, TP, ENTRY, MID = 100.0, 98.0, 110.0, 100.35, 100.3

    def call(self, ctx):
        return T.entry_is_actionable(
            Backend(self.MID), "TEST", self.ENTRY, self.STOP, self.TP,
            level=self.LEVEL, bars_1h=bars(), ctx=ctx)

    def test_no_context_means_no_veto(self):
        """A failed contexts fetch must not invent a refusal."""
        for ctx in (None, {}):
            ref, why = self.call(ctx)
            self.assertIsNotNone(ref)

    def test_missing_or_zero_prices_mean_no_veto(self):
        for ctx in ({"mark_px": None, "oracle_px": 100.0},
                    {"mark_px": 100.5, "oracle_px": None},
                    {"mark_px": 100.5, "oracle_px": 0.0},
                    {"mark_px": 0.0, "oracle_px": 100.0}):
            ref, why = self.call(ctx)
            self.assertIsNotNone(ref, f"invented a refusal from {ctx}")

    def test_it_can_be_switched_off(self):
        saved = T.ENTRY_MAX_BASIS_BPS
        try:
            T.ENTRY_MAX_BASIS_BPS = -1.0
            ref, _ = self.call({"mark_px": 105.0, "oracle_px": 100.0})
            self.assertIsNotNone(ref)
        finally:
            T.ENTRY_MAX_BASIS_BPS = saved

    def test_the_older_gates_still_own_their_refusals(self):
        """The veto sits after drift, the stop and the break check, so it
        cannot take credit for a refusal it did not make."""
        rich = {"mark_px": 110.0, "oracle_px": 100.0}
        ref, why = T.entry_is_actionable(Backend(97.0), "TEST", self.ENTRY,
                                         self.STOP, self.TP, level=self.LEVEL,
                                         bars_1h=bars(), ctx=rich)
        self.assertIsNone(ref)
        self.assertIn("from the planned entry", why)

    def test_it_only_removes_trades(self):
        for mark in (99.0, 100.0, 100.5, 105.0):
            with_ctx, _ = self.call({"mark_px": mark, "oracle_px": 100.0})
            without, _ = self.call(None)
            if with_ctx is not None:
                self.assertIsNotNone(without,
                                     "the veto approved something the base path refused")


class WiredAtZeroExtraCost(unittest.TestCase):
    def test_the_pass_fetches_contexts_not_just_mids(self):
        """get_perp_contexts carries mark and oracle beside the mid, so the
        veto costs no additional request -- still one call per pass."""
        import inspect
        src = inspect.getsource(T.build_intents)
        self.assertIn("client.get_perp_contexts()", src)
        # The CALL, not the word: the comment explaining the switch names the
        # old method, and matching bare text flagged that as a regression.
        self.assertNotIn("client.get_all_mids()", src)
        self.assertIn("mid_px", src)

    def test_the_contexts_reach_the_gate(self):
        import inspect
        self.assertIn("ctx_cache", inspect.getsource(T.main))
        self.assertIn("ctx=", inspect.getsource(T.execute_one))

    def test_the_threshold_is_the_natural_boundary(self):
        self.assertEqual(C.ENTRY_MAX_BASIS_BPS, 0.0)


if __name__ == "__main__":
    unittest.main()

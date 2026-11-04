"""The predictability scan finds planted structure, with the right sign, and
nothing in a martingale. Synthetic returns only; no market claim here."""

import math
import unittest

import numpy as np

from research.backtest.experiments import predictability_scan as P


def panel(T=3000, C=30, phi_market=0.0, phi_idio=0.0, shock_revert=0.0, seed=1):
    """Hourly log returns: an AR(1) common factor plus AR(1) idiosyncratic
    parts; optionally, idiosyncratic jumps that partly reverse next hour."""
    rng = np.random.default_rng(seed)
    f = np.zeros(T)
    e = np.zeros((T, C))
    for t in range(1, T):
        f[t] = phi_market * f[t - 1] + rng.normal(0, 0.006)
        e[t] = phi_idio * e[t - 1] + rng.normal(0, 0.008, C)
    if shock_revert:
        jumps = (rng.random((T, C)) < 0.004) * rng.choice([-1, 1], (T, C)) * 0.06
        e += jumps
        e[1:] -= shock_revert * jumps[:-1]
    return f[:, None] + e


class Windows(unittest.TestCase):
    def test_past_and_forward_sums_by_hand(self):
        R = np.arange(1.0, 7.0)[:, None]                  # 1..6
        past = P.window_sum(R, 2)[:, 0]
        fwd = P.window_sum(R, 2, forward=True)[:, 0]
        self.assertTrue(math.isnan(past[0]))
        self.assertEqual(list(past[1:]), [3, 5, 7, 9, 11])
        self.assertEqual(list(fwd[:4]), [5, 7, 9, 11])
        self.assertTrue(np.isnan(fwd[4:]).all())

    def test_a_missing_hour_voids_every_window_through_it(self):
        R = np.array([1.0, 2.0, np.nan, 4.0, 5.0])[:, None]
        self.assertTrue(np.isnan(P.window_sum(R, 2)[[2, 3], 0]).all())
        self.assertEqual(P.window_sum(R, 2)[4, 0], 9.0)

    def test_the_trailing_rms_ignores_signs_and_the_current_hour(self):
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (400, 3))
        flipped = X * rng.choice([-1.0, 1.0], 400)[:, None]
        a, b = P.trailing_rms(X), P.trailing_rms(flipped)
        np.testing.assert_allclose(a[200:], b[200:])
        self.assertAlmostEqual(a[300, 1], math.sqrt(np.mean(X[300 - P.SHOCK_LOOKBACK:300, 1] ** 2)))


class TercileBook(unittest.TestCase):
    def test_long_the_low_short_the_high(self):
        sig = np.array([[1.0, 2, 3, 4, 5, 6, 7, 8, 9]])
        fwd = -sig / 100                                  # the low ones fall least
        spread, w = P.tercile_book(sig, fwd, np.array([0]))
        self.assertAlmostEqual(spread[0], -0.02 - (-0.08))   # low leg mean minus high leg mean
        self.assertEqual(list(np.sign(w[0])), [1, 1, 1, 0, 0, 0, -1, -1, -1])

    def test_too_few_coins_is_no_book(self):
        spread, _ = P.tercile_book(np.array([[1.0, 2, 3, np.nan]]), np.zeros((1, 4)), np.array([0]))
        self.assertTrue(math.isnan(spread[0]))


class FindsWhatIsThere(unittest.TestCase):
    def _scan(self, R):
        return P.scan(R, draws=150, log=lambda *_: None)["cells"]

    def test_a_martingale_has_nothing(self):
        cells = self._scan(panel(seed=4))
        self.assertLess(max(abs(v["z"]) for v in cells.values()), 3.5)
        for v in cells.values():
            self.assertLess(abs(v["null_mean"]), 4 * v["null_sd"] / math.sqrt(150) + 1e-12)

    def test_market_momentum_is_found_and_signed(self):
        cells = self._scan(panel(phi_market=0.3, seed=5))
        self.assertGreater(cells["market_1h"]["z"], 4)

    def test_cross_sectional_reversal_is_found_and_signed(self):
        cells = self._scan(panel(phi_idio=-0.3, seed=6))
        self.assertGreater(cells["relative_1h"]["z"], 4)          # long laggards earned

    def test_cross_sectional_momentum_reads_negative(self):
        cells = self._scan(panel(phi_idio=0.3, seed=7))
        self.assertLess(cells["relative_1h"]["z"], -4)

    def test_shock_reversal_is_found_and_signed(self):
        cells = self._scan(panel(shock_revert=0.5, seed=8))
        self.assertLess(cells["shock_1h"]["z"], -4)               # the move reversed

    def test_the_null_keeps_every_shock_and_the_cross_section(self):
        R = panel(seed=9)
        eps = np.random.default_rng(1).choice([-1.0, 1.0], R.shape[0])
        np.testing.assert_array_equal(P.shock_mask(R), P.shock_mask(R * eps[:, None]))
        c0 = np.corrcoef(R[1:].T)
        c1 = np.corrcoef((R * eps[:, None])[1:].T)
        np.testing.assert_allclose(c0, c1, atol=0.05)


class PremiumAlignment(unittest.TestCase):
    def test_a_premium_known_at_t_sits_on_the_bar_closing_at_t(self):
        """Funding for the hour ENDING at T is known at T: it must land on
        the row of the bar that opened at T - 1h, so the forward window from
        that row starts with the bar that opens at T -- never before."""
        import json
        import tempfile
        from pathlib import Path
        H = 3_600_000
        times = np.asarray([10 * H, 11 * H, 12 * H, 13 * H])        # bar OPEN times
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "AAA.json").write_text(json.dumps(
                [{"coin": "AAA", "time": 12 * H + 37, "premium": "0.0005", "fundingRate": "0.0000125"}]))
            P_, Fr, info = P.load_premium(Path(d), times, ["AAA"])
        self.assertEqual(np.flatnonzero(np.isfinite(P_[:, 0])).tolist(), [1])   # the bar 11h-12h
        self.assertAlmostEqual(P_[1, 0], 0.0005)
        self.assertAlmostEqual(Fr[1, 0], 0.0000125)
        self.assertAlmostEqual(info["annualised_pct"], 0.0000125 * 24 * 365 * 100)
        R = np.arange(4.0)[:, None]
        self.assertEqual(P.window_sum(R, 1, forward=True)[1, 0], R[2, 0])   # next bar only


class RegimePersistence(unittest.TestCase):
    def test_clustered_volatility_is_found_and_a_martingale_s_trendiness_is_not(self):
        rng = np.random.default_rng(12)
        T, C = 168 * 40, 12
        level = np.exp(np.cumsum(rng.normal(0, 0.15, (T // 168, C)), axis=0))   # slow vol regimes
        sig = np.repeat(level, 168, axis=0) * 0.008
        R = rng.normal(0, 1, (T, C)) * sig
        out = P.regime_persistence(R, draws=200)
        self.assertGreater(out["vol"]["value"], 0.5)
        self.assertLess(out["vol"]["p"], 0.01)
        self.assertGreater(out["trendiness"]["p"], 0.01)
        self.assertGreater(out["trend_pays_gap_bps"]["p"], 0.01)

    def test_blocks_are_whole_weeks_and_a_partial_week_is_dropped(self):
        up = np.full((168 * 8 + 100, 1), 0.001)
        out = P.regime_persistence(np.hstack([up] * 10), draws=20)
        self.assertEqual((out["blocks"], out["block_hours"]), (8, 168))


class WeeklyHorizon(unittest.TestCase):
    def _daily(self, persist=0.0, seed=2, T=900, C=30):
        """Daily returns whose per-coin drift is a slow AR(1): with persist > 0
        last month's winners keep drifting up -- momentum by construction."""
        rng = np.random.default_rng(seed)
        mu = np.zeros((T, C))
        for t in range(1, T):
            mu[t] = persist * mu[t - 1] + rng.normal(0, 0.002, C)
        return mu * (persist > 0) + rng.normal(0, 0.03, (T, C))

    def test_a_martingale_has_no_weekly_effect(self):
        cells = P.scan_weekly(self._daily(seed=3), draws=150)["cells"]
        self.assertLess(max(abs(v["z"]) for v in cells.values()), 3.5)

    def test_planted_momentum_is_found_and_signed(self):
        cells = P.scan_weekly(self._daily(persist=0.97, seed=4), draws=150)["cells"]
        self.assertGreater(cells["xs_momentum_28d"]["z"], 3)
        self.assertGreater(cells["own_trend_28d"]["z"], 3)


if __name__ == "__main__":
    unittest.main()

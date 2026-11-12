"""H-TREND's machinery, on synthetic prices only; no market claim here.

The book must add up by hand, look at nothing from the future, find a
planted trend, and find nothing in a drifting series without one -- which is
the whole reason the controls are rotations rather than zero.
"""

import math
import unittest

import numpy as np

from research.backtest.experiments import trend_daily as T


def prices(Tn=1500, C=4, phi=0.0, drift=0.0, seed=1, vol=0.03):
    """Daily closes whose log returns share a slow AR(1) trend component
    (phi) on top of noise, plus a constant drift."""
    rng = np.random.default_rng(seed)
    m = np.zeros((Tn, C))
    for t in range(1, Tn):
        m[t] = phi * m[t - 1] + rng.normal(0, vol * 0.25, C)
    lr = drift + (m if phi else 0.0) + rng.normal(0, vol, (Tn, C))
    return 100.0 * np.exp(np.cumsum(lr, axis=0))


def setup(close):
    s, v, e = T.features(close)
    r = T.next_returns(close)
    days = np.where(e.any(axis=1) & np.isfinite(r).any(axis=1))[0]
    return s, v, e, r, days


class Features(unittest.TestCase):
    def test_trend_is_the_mean_sign_of_three_trailing_returns(self):
        close = np.linspace(100, 300, 200)[:, None].copy()
        close[150:] = close[150] - np.arange(50)[:, None] * 0.2   # then falls
        s, v, e = T.features(close)
        self.assertEqual(s[149, 0], 1.0)
        # 20 days into the fall: 14d is down, 28d and 56d still up.
        self.assertAlmostEqual(s[170, 0], 1 / 3)
        self.assertTrue(np.isnan(s[T.MIN_HISTORY - 2, 0]))
        self.assertTrue(e[T.MIN_HISTORY - 1, 0])

    def test_volatility_is_the_annualised_30_day_stdev_by_hand(self):
        close = prices(200, 1, seed=3)
        _, v, _ = T.features(close)
        lr = np.log(close[1:, 0] / close[:-1, 0])
        t = 150
        self.assertAlmostEqual(v[t, 0], lr[t - 30: t].std(ddof=1) * math.sqrt(365), places=12)

    def test_nothing_reads_the_future(self):
        close = prices(400, 3, seed=5)
        s1, v1, e1 = T.features(close)
        bent = close.copy()
        bent[300:] *= 3.0
        s2, v2, e2 = T.features(bent)
        np.testing.assert_array_equal(s1[:300], s2[:300])
        np.testing.assert_array_equal(v1[:300], v2[:300])


class Book(unittest.TestCase):
    def test_one_coin_adds_up_by_hand(self):
        close = prices(300, 1, seed=7)
        s, v, e, r, days = setup(close)
        out = T.simulate(s, v, e, r, days, detail=True)
        d0 = days[0]
        w = float(np.clip(s[d0, 0] * T.VOL_TARGET / v[d0, 0], -2, 2))
        expect = (w * r[d0, 0] - abs(w) * T.COST_BPS / 1e4
                  - max(w, 0) * T.FUND_LONG_BPS / 1e4)
        self.assertAlmostEqual(out["net"][0, 0], expect, places=12)
        self.assertAlmostEqual(out["turnover"][0], abs(w), places=12)

    def test_the_band_holds_small_changes_and_trades_a_flip(self):
        close = prices(600, 1, seed=11)
        s, v, e, r, days = setup(close)
        out = T.simulate(s, v, e, r, days, detail=True)
        flips = int((np.sign(s[days[1:], 0]) != np.sign(s[days[:-1], 0])).sum())
        self.assertGreaterEqual(int(out["trades"].sum()), flips)
        self.assertLess(int(out["trades"].sum()), len(days) // 2)   # not every day

    def test_costs_only_ever_subtract(self):
        close = prices(500, 3, seed=13)
        s, v, e, r, days = setup(close)
        free = T.simulate(s, v, e, r, days, cost_bps=0, fund_long_bps=0)["net"][0]
        paid = T.simulate(s, v, e, r, days)["net"][0]
        self.assertTrue((paid <= free + 1e-15).all())

    def test_a_rotated_book_keeps_the_real_books_long_short_mix(self):
        close = prices(900, 3, seed=17, drift=0.001)
        s, v, e, r, days = setup(close)
        sd = np.nan_to_num(s[days])
        for off in (120, 400):
            first = e[days].argmax(axis=0)
            rot = np.empty_like(sd)
            for c in range(sd.shape[1]):
                span = len(days) - first[c]
                idx = first[c] + (np.arange(span) + off) % span
                rot[first[c]:, c] = sd[idx, c]
                rot[:first[c], c] = 0
            np.testing.assert_allclose(rot.mean(axis=0), sd.mean(axis=0), atol=1e-12)

    def test_a_late_listing_is_rotated_inside_its_own_history(self):
        close = prices(900, 2, seed=19)
        close[:400, 1] = np.nan
        s, v, e, r, days = setup(close)
        net = T.simulate(s, v, e, r, days, offsets=(0, 200))["net"]
        self.assertTrue(np.isfinite(net).all())


class Timing(unittest.TestCase):
    def test_a_planted_trend_is_found(self):
        close = prices(2000, 6, phi=0.97, seed=23)
        s, v, e, r, days = setup(close)
        t = T.timing_test(s, v, e, r, days, controls=300, seed=1)
        self.assertGreater(t["excess_bps_day"], 0)
        self.assertLess(t["p"], 0.05)

    def test_drift_without_trend_is_not_edge(self):
        fired = 0
        for seed in range(8):
            close = prices(1500, 6, drift=0.002, seed=100 + seed)
            s, v, e, r, days = setup(close)
            t = T.timing_test(s, v, e, r, days, controls=200, seed=seed)
            self.assertGreater(t["real_bps_day"], 0)        # it does earn the drift
            fired += t["p"] < 0.05
        self.assertLessEqual(fired, 1)

    def test_the_null_archive_keeps_drift_and_size_and_loses_order(self):
        close = prices(1200, 3, phi=0.97, drift=0.001, seed=29)
        fake = T.null_archive(close, np.random.default_rng(1))
        a, b = np.log(close[1:] / close[:-1]), np.log(fake[1:] / fake[:-1])
        mean = a.mean(axis=0)
        np.testing.assert_allclose(np.abs(b - mean), np.abs(a), atol=1e-9)
        self.assertAlmostEqual(fake[0, 0], close[0, 0])


class Stats(unittest.TestCase):
    def test_drawdown_and_sharpe_by_hand(self):
        x = np.array([0.10, -0.50, 0.20])
        self.assertAlmostEqual(T.drawdown(x), -0.5)
        self.assertAlmostEqual(T.sharpe(x), x.mean() / x.std(ddof=1) * math.sqrt(365))

    def test_bootstrap_interval_brackets_the_point(self):
        x = np.random.default_rng(3).normal(0.001, 0.01, 800)
        b = T.block_bootstrap_sharpe(x, B=400)
        self.assertLess(b["ci_lo"], b["sharpe"])
        self.assertGreater(b["ci_hi"], b["sharpe"])


if __name__ == "__main__":
    unittest.main()

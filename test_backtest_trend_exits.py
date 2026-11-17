"""H-EXITS' machinery, on synthetic bars only; no market claim here.

Every fill rule is worked by hand: where a stop fills, that a day reaching
both a stop and a take-profit is the stop, that a take-profit needs trade-
through, that a stop-loss or breakeven stop sits the coin out until its
target changes sign while the daily stop does not, and that the average
entry moves on adds only. Every scenario is also run through the plain loop,
and the vectorised engine must match it on random data for every rule.
"""

import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from research.backtest.experiments import trend_daily as T
from research.backtest.experiments import trend_exits as X
from research.trend import paper as P

PASS = X.PASS_COST_BPS / 1e4
FEE = X.FEE_BPS / 1e4
SLIP = X.STOP_SLIP_BPS / 1e4
FUND = X.FUND_LONG_BPS / 1e4


def bars(rows):
    a = np.asarray(rows, float)
    return a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]


def go(rule, rows, w, days=None):
    """One coin: run the engine and the plain loop, which must agree."""
    O, H, L, C = bars(rows)
    W = np.asarray(w, float).reshape(-1, 1)
    days = np.arange(len(rows) - 1) if days is None else np.asarray(days)
    res = X.run(rule, days, O[None], H[None], L[None], C[None], W[None])
    plain = X.run_plain(rule, days, O, H, L, C, W)
    np.testing.assert_allclose(res["net"][0], plain, rtol=0, atol=1e-15)
    return res


def flat(px, n):
    return [(px, px, px, px)] * n


def random_bars(Tn=400, C=4, seed=3, listed=(0, 0, 60, 150)):
    """Random-walk OHLC with staggered listings, and random targets that
    change sign, go to zero and wander inside and outside the band."""
    rng = np.random.default_rng(seed)
    lr = rng.normal(0, 0.04, (Tn, C))
    close = 100 * np.exp(np.cumsum(lr, axis=0))
    prev = np.vstack([close[:1], close[:-1]])
    o = prev * np.exp(rng.normal(0, 0.002, (Tn, C)))
    h = np.maximum(o, close) * np.exp(np.abs(rng.normal(0, 0.03, (Tn, C))))
    lo = np.minimum(o, close) * np.exp(-np.abs(rng.normal(0, 0.03, (Tn, C))))
    w = np.round(rng.normal(0, 0.08, (Tn, C)) * 4) / 4 * 0.3
    w[rng.random((Tn, C)) < 0.7] = np.nan                         # persist most days
    for j in range(C):
        w[0, j] = 0.0
        for t in range(1, Tn):
            if np.isnan(w[t, j]):
                w[t, j] = w[t - 1, j]
    for j, a in enumerate(listed):
        for X_ in (o, h, lo, close):
            X_[:a, j] = np.nan
        w[:a, j] = 0.0
    return o, h, lo, close, w


class Stops(unittest.TestCase):
    def test_a_short_stops_at_its_level_plus_slippage_and_is_rebuilt_next_pass(self):
        rows = flat(100, 1) + [(100, 126, 99, 120), (120, 121, 119, 120)]
        r = go(X.BASE, rows, [-0.1, -0.1, -0.1])
        fill = 125 * (1 + SLIP)
        self.assertAlmostEqual(r["net"][0, 0], -0.001 * (fill - 100) - 0.1 * PASS - 0.001 * fill * FEE, 15)
        self.assertAlmostEqual(r["net"][0, 1], -0.1 * PASS, 15)          # rebuilt: no lock
        self.assertEqual(int(r["events"]["daily_stops"][0]), 1)
        self.assertEqual(int(r["events"]["exit_stops"][0]), 0)

    def test_a_day_opening_beyond_the_stop_fills_at_the_open(self):
        rows = flat(100, 1) + [(130, 131, 129, 130), (130, 130, 130, 130)]
        r = go(X.BASE, rows, [-0.1, -0.1, -0.1], days=[0])
        fill = 130 * (1 + SLIP)
        self.assertAlmostEqual(r["net"][0, 0], -0.001 * (fill - 100) - 0.1 * PASS - 0.001 * fill * FEE, 15)

    def test_the_daily_stop_is_measured_from_the_latest_mark_not_the_entry(self):
        rule = X.HYPOTHESES["DS10"]
        rows = flat(100, 1) + [(100, 109, 99, 108), (108, 118, 107, 117), (117, 117, 117, 117)]
        r = go(rule, rows, [-0.1] * 4, days=[0, 1])
        self.assertEqual(int(r["events"]["daily_stops"][0]), 0)         # 118 < 108 x 1.10
        self.assertAlmostEqual(r["net"][0, 0], -0.001 * 8 - 0.1 * PASS, 15)
        self.assertAlmostEqual(r["net"][0, 1], -0.001 * 9, 15)          # held inside the band

    def test_a_stop_loss_from_entry_fills_and_the_coin_sits_out(self):
        rule = X.HYPOTHESES["SL10"]
        rows = flat(100, 1) + [(100, 111, 99, 108)] + flat(108, 2)
        r = go(rule, rows, [-0.1] * 4, days=[0, 1, 2])
        fill = 110 * (1 + SLIP)
        self.assertAlmostEqual(r["net"][0, 0], -0.001 * (fill - 100) - 0.1 * PASS - 0.001 * fill * FEE, 15)
        np.testing.assert_allclose(r["net"][0, 1:], 0.0, atol=1e-18)     # target unchanged: flat
        self.assertEqual(int(r["events"]["exit_stops"][0]), 1)

    def test_breakeven_arms_at_a_pass_stops_at_entry_and_sits_out_until_the_sign_changes(self):
        rule = X.HYPOTHESES["BE10"]
        rows = (flat(100, 1) + [(100, 101, 88, 89), (89, 101, 88, 95)]
                + flat(95, 3))
        w = [-0.1, -0.1, -0.1, 0.0, -0.1, -0.1]
        r = go(rule, rows, w, days=[0, 1, 2, 3, 4])
        fill = 100 * (1 + SLIP)                                         # entry, not 89 x 1.25
        self.assertAlmostEqual(r["net"][0, 0], 0.011 - 0.1 * PASS, 15)
        self.assertAlmostEqual(r["net"][0, 1], -0.001 * (fill - 89) - 0.001 * fill * FEE, 15)
        self.assertEqual(r["net"][0, 2], 0.0)                           # sits out
        self.assertEqual(r["net"][0, 3], 0.0)                           # target 0: the lock clears
        self.assertAlmostEqual(r["net"][0, 4], -0.1 * PASS, 15)         # and it re-enters
        self.assertEqual(int(r["events"]["exit_stops"][0]), 1)

    def test_breakeven_does_not_arm_short_of_its_trigger(self):
        rule = X.HYPOTHESES["BE10"]
        rows = flat(100, 1) + [(100, 101, 90.5, 91), (91, 101, 90, 95)]
        r = go(rule, rows, [-0.1] * 3)
        self.assertEqual(int(r["events"]["exit_stops"][0]), 0)          # 91 is 9% in profit


class TakeProfit(unittest.TestCase):
    def test_half_is_taken_at_the_level_and_the_rest_cannot_be_added_to(self):
        rule = X.HYPOTHESES["TP10"]
        rows = flat(100, 1) + [(100, 101, 89.9, 92)] + flat(92, 2)
        r = go(rule, rows, [-0.1, -0.1, 0.1, 0.1], days=[0, 1, 2])
        u = -0.001
        self.assertAlmostEqual(r["net"][0, 0], u * 0.5 * (90 - 100) + u * 0.5 * (92 - 100)
                               - 0.1 * PASS - 0.001 * 0.5 * 90 * FEE, 15)
        self.assertEqual(r["net"][0, 1], 0.0)          # band wants -0.1, the lock holds -0.046
        self.assertAlmostEqual(r["net"][0, 2], -(0.1 + 0.046) * PASS - 0.1 * FUND, 15)
        self.assertEqual(int(r["events"]["takes"][0]), 1)

    def test_a_take_profit_needs_the_day_to_trade_through_it(self):
        rule = X.HYPOTHESES["TP10"]
        rows = flat(100, 1) + [(100, 101, 89.96, 92), (92, 92, 92, 92)]
        r = go(rule, rows, [-0.1] * 3, days=[0])
        self.assertEqual(int(r["events"]["takes"][0]), 0)
        self.assertAlmostEqual(r["net"][0, 0], 0.008 - 0.1 * PASS, 15)

    def test_a_day_reaching_both_the_stop_and_the_take_profit_is_the_stop(self):
        rule = X.HYPOTHESES["TP10"]
        rows = flat(100, 1) + [(100, 126, 85, 95), (95, 95, 95, 95)]
        r = go(rule, rows, [-0.1] * 3, days=[0])
        fill = 125 * (1 + SLIP)
        self.assertEqual(int(r["events"]["takes"][0]), 0)
        self.assertAlmostEqual(r["net"][0, 0], -0.001 * (fill - 100) - 0.1 * PASS - 0.001 * fill * FEE, 15)

    def test_a_level_already_reached_at_the_pass_is_taken_at_the_pass(self):
        rule = X.HYPOTHESES["TP10"]
        rows = flat(100, 1) + [(100, 101, 89.97, 89.98), (89.98, 90, 89.9, 89.98)]
        r = go(rule, rows, [-0.1] * 3, days=[0, 1])
        self.assertEqual(int(r["events"]["pass_takes"][0]), 1)
        self.assertAlmostEqual(r["net"][0, 1], -0.0005 * 89.98 * PASS, 15)

    def test_the_average_entry_moves_on_an_add(self):
        rule = X.NEIGHBOURS["TP10"][1]                               # TP20
        rows = flat(100, 1) + [(100, 101, 80.5, 81), (81, 82, 70.9, 72), (72, 72, 72, 72)]
        r = go(rule, rows, [-0.05, -0.1, -0.1, -0.1], days=[0, 1])
        un = -0.1 / 81
        a = (0.0005 * 100 + (abs(un) - 0.0005) * 81) / abs(un)
        tp = a * 0.8
        self.assertEqual(int(r["events"]["takes"][0]), 1)
        self.assertAlmostEqual(r["net"][0, 1], un * 0.5 * (tp - 81) + un * 0.5 * (72 - 81)
                               - (0.1 - 0.0405) * PASS - abs(un) * 0.5 * tp * FEE, 14)


class Book(unittest.TestCase):
    def test_weights_drift_with_price_and_the_band_reads_the_drift(self):
        rows = [(100,) * 4, (110,) * 4, (140,) * 4, (140,) * 4]
        r = go(X.NOSTOP, rows, [-0.1] * 4)
        self.assertAlmostEqual(r["net"][0, 0], -0.001 * 10 - 0.1 * PASS, 15)
        self.assertAlmostEqual(r["net"][0, 1], -0.001 * 30, 15)       # -0.11 is inside the band
        self.assertAlmostEqual(r["net"][0, 2], -0.04 * PASS, 15)       # -0.14 is not: back to -0.1

    def test_longs_pay_funding_and_shorts_receive_none(self):
        r = go(X.NOSTOP, flat(100, 3), [0.1] * 3)
        self.assertAlmostEqual(r["net"][0, 1], -0.1 * FUND, 15)
        r = go(X.NOSTOP, flat(100, 3), [-0.1] * 3)
        self.assertEqual(r["net"][0, 1], 0.0)

    def test_the_engine_is_the_plain_loop_on_random_bars_for_every_rule(self):
        o, h, lo, c, w = random_bars()
        days = np.arange(0, len(c) - 1)
        rules = [X.BASE, X.NOSTOP, *X.HYPOTHESES.values(),
                 *[r for pair in X.NEIGHBOURS.values() for r in pair],
                 X.harsh(X.BASE), *[X.harsh(r) for r in X.HYPOTHESES.values()],
                 replace(X.BASE, name="ALL", entry_stop=0.08, breakeven_at=0.06, take_at=0.07)]
        for rule in rules:
            net = X.run(rule, days, o[None], h[None], lo[None], c[None], w[None])["net"][0]
            plain = X.run_plain(rule, days, o, h, lo, c, w)
            self.assertLess(float(np.abs(net - plain).max()), 1e-13, rule.name)
        ev = X.run(rules[-1], days, o[None], h[None], lo[None], c[None], w[None])["events"]
        for k in ("exit_stops", "takes"):
            self.assertGreater(int(ev[k][0]), 0, k)                    # every path exercised
        ds05 = X.NEIGHBOURS["DS10"][0]
        ev = X.run(ds05, days, o[None], h[None], lo[None], c[None], w[None])["events"]
        self.assertGreater(int(ev["daily_stops"][0]), 0)

    def test_books_run_together_are_the_books_run_alone(self):
        sets = [random_bars(seed=s) for s in (5, 6, 7)]
        days = np.arange(0, 399)
        stacked = [np.stack([s[i] for s in sets]) for i in range(5)]
        rule = X.HYPOTHESES["TP10"]
        together = X.run(rule, days, *stacked)["net"]
        for k, s in enumerate(sets):
            alone = X.run(rule, days, *(a[None] for a in s))["net"][0]
            np.testing.assert_allclose(together[k], alone, rtol=0, atol=1e-15)


class Targets(unittest.TestCase):
    def test_targets_are_the_live_decide_day_by_day(self):
        rng = np.random.default_rng(9)
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.04, (400, 10)), axis=0))
        close[:50, 3] = np.nan
        close[:200, 7] = np.nan
        W = X.targets(close)
        for t in (130, 175, 260, 330, 399):
            np.testing.assert_allclose(W[t], P.decide(close[: t + 1]), rtol=0, atol=1e-15)


class Null(unittest.TestCase):
    def test_mirrored_bars_keep_their_order_range_and_first_bar(self):
        o, h, lo, c, _ = random_bars(Tn=300)
        no, nh, nl, nc = X.null_ohlc(o, h, lo, c, np.random.default_rng(1))
        ok = np.isfinite(c)
        np.testing.assert_array_equal(np.isfinite(nc), ok)
        self.assertTrue((nh[ok] >= np.maximum(no, nc)[ok] * (1 - 1e-12)).all())
        self.assertTrue((nl[ok] <= np.minimum(no, nc)[ok] * (1 + 1e-12)).all())
        np.testing.assert_allclose(np.log(nh / nl)[ok], np.log(h / lo)[ok], atol=1e-9)
        for j in range(c.shape[1]):
            a = int(np.argmax(ok[:, j]))
            self.assertEqual(nc[a, j], c[a, j])

    def test_one_sign_a_day_for_every_coin(self):
        o, h, lo, c, _ = random_bars(Tn=200, listed=(0, 0, 0, 0))
        no, nh, nl, nc = X.null_ohlc(o, h, lo, c, np.random.default_rng(2))
        real = np.log(c[1:] / c[:-1])
        fake = np.log(nc[1:] / nc[:-1]) - np.nanmean(real, axis=0)
        sign = np.sign(fake / real)
        self.assertTrue((sign == sign[:, :1]).all())


class Stats(unittest.TestCase):
    def test_a_constant_gain_is_found_and_noise_is_not(self):
        rng = np.random.default_rng(4)
        b = X.boot_diff(1e-3 + rng.normal(0, 1e-5, 800), B=999)
        self.assertAlmostEqual(b["p"], 1 / 1000)
        self.assertLess(b["ci_lo_bps_day"], b["mean_bps_day"])
        self.assertGreater(b["ci_hi_bps_day"], b["mean_bps_day"])
        hits = sum(X.boot_diff(rng.normal(0, 1e-3, 800), B=499, seed=s)["p"] < 0.05
                   for s in range(40))
        self.assertLessEqual(hits, 6)

    def test_a_non_finite_difference_is_refused_not_read_as_significant(self):
        with self.assertRaises(ValueError):
            X.boot_diff(np.array([1e-3, np.nan, 2e-3]), B=99)

    def test_drawdown_is_measured_on_fixed_capital(self):
        self.assertAlmostEqual(X.fixed_drawdown(np.array([0.1, -0.2, 0.05])), 0.9 / 1.1 - 1, 15)
        self.assertAlmostEqual(X.fixed_drawdown(np.array([-0.1, 0.05])), -0.1, 15)


class Archive(unittest.TestCase):
    def _src(self, root: Path, close_shift: float = 0.0):
        hashes = {}
        for k, coin in enumerate(X.COINS):
            d = root / f"{coin}USDT" / "1d"
            d.mkdir(parents=True)
            rows = [[1_600_000_000_000 + i * T.DAY_MS, str(10 + k + i), str(11 + k + i),
                     str(9 + k + i), str(10.5 + k + i + close_shift), "0", 0]
                    for i in range(5)]
            (d / "2020-01-01.json").write_text(json.dumps(rows))
            closes = json.dumps([[r[0], float(10.5 + k + i)] for i, r in enumerate(rows)],
                                separators=(",", ":"))
            hashes[coin] = X._sha(closes)
        return hashes

    def test_the_archive_is_built_only_from_closes_that_match_h_trend(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            want = self._src(root / "src")
            X.build_archive(root / "src", root / "out", expected=want)
            times, o, h, lo, c = X.load(X.COINS, root / "out")
            self.assertEqual(c.shape, (5, len(X.COINS)))
            self.assertEqual(float(h[0, 0]), 11.0)
            bad = dict(want, BTC="0" * 64)
            with self.assertRaises(ValueError):
                X.build_archive(root / "src", root / "out2", expected=bad)

    def test_a_tampered_archive_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            want = self._src(root / "src")
            X.build_archive(root / "src", root / "out", expected=want)
            f = root / "out" / "ETH.json"
            f.write_text(f.read_text().replace("12.0", "12.5", 1))
            with self.assertRaises(ValueError):
                X.load(X.COINS, root / "out")


if __name__ == "__main__":
    unittest.main()

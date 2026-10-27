"""Statistics pinned to published values; data handling that repairs nothing."""

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from research.backtest import data as D
from research.backtest import stats as S

FIX = Path(__file__).parent / "tests/fixtures/hyperliquid/candle_snapshot_1h_eth.json"


class TDistribution(unittest.TestCase):
    def test_against_published_tables(self):
        for t, df, p in [(2.0, 10, 0.0734), (2.228, 10, 0.05), (1.96, 1e9, 0.05),
                         (2.576, 1e9, 0.01), (12.706, 1, 0.05), (2.045, 29, 0.05),
                         (3.169, 10, 0.01)]:
            self.assertAlmostEqual(S.t_two_sided_p(t, df), p, places=3, msg=f"t={t} df={df}")

    def test_symmetric_and_bounded(self):
        self.assertAlmostEqual(S.t_two_sided_p(-2.0, 10), S.t_two_sided_p(2.0, 10))
        self.assertAlmostEqual(S.t_two_sided_p(0.0, 10), 1.0)


class Welch(unittest.TestCase):
    def test_a_textbook_case(self):
        a = [27.5, 21.0, 19.0, 23.6, 17.0, 17.9, 16.9, 20.1, 21.9, 22.6, 23.1, 19.6, 19.0, 21.7, 21.4]
        b = [27.1, 22.0, 20.8, 23.4, 23.4, 23.5, 25.8, 22.0, 24.8, 20.2, 21.9, 22.1, 22.9, 20.5, 24.4]
        w = S.welch(a, b)
        # Reference: t = -2.46, df = 24.98, p = 0.021 (Welch's 1947 example as
        # reproduced in the scipy docs for ttest_ind(equal_var=False)).
        self.assertAlmostEqual(w["t"], -2.46, places=2)
        self.assertAlmostEqual(w["df"], 24.98, places=1)
        self.assertAlmostEqual(w["p"], 0.021, places=3)


class Holm(unittest.TestCase):
    def test_step_down(self):
        r = S.holm({"a": 0.01, "b": 0.04, "c": 0.03})
        self.assertTrue(r["a"]["reject"])          # 0.01 <= 0.05/3
        self.assertFalse(r["c"]["reject"])         # 0.03 >  0.05/2
        self.assertFalse(r["b"]["reject"])         # stops once one fails
        self.assertAlmostEqual(r["a"]["p_holm"], 0.03)
        self.assertAlmostEqual(r["c"]["p_holm"], 0.06)
        self.assertAlmostEqual(r["b"]["p_holm"], 0.06)   # monotone, not 0.04


class ClusterBootstrap(unittest.TestCase):
    def test_one_observation_per_cluster_matches_the_iid_error(self):
        rng = np.random.default_rng(1)
        x = rng.normal(0.1, 1.0, 400)
        r = S.cluster_bootstrap(x, [0] * 400, list(range(400)), "mean", B=4000)
        self.assertAlmostEqual(r["se"], x.std(ddof=1) / math.sqrt(400), delta=0.006)

    def test_correlated_clusters_widen_the_error(self):
        """Ten copies of each observation look like 4000 trades to an iid
        test; resampling whole clusters sees 400."""
        rng = np.random.default_rng(2)
        base = rng.normal(0.0, 1.0, 400)
        x = np.repeat(base, 10)
        clusters = np.repeat(np.arange(400), 10)
        r = S.cluster_bootstrap(x, [0] * len(x), clusters, "mean", B=3000)
        iid_se = x.std(ddof=1) / math.sqrt(len(x))
        self.assertGreater(r["se"], 2.5 * iid_se)

    def test_the_difference_is_group1_minus_group0(self):
        v = [1.0, 1.0, 0.0, 0.0]
        g = [1, 1, 0, 0]
        r = S.cluster_bootstrap(v, g, [0, 1, 0, 1], "diff", B=200)
        self.assertAlmostEqual(r["estimate"], 1.0)


class Normalise(unittest.TestCase):
    def setUp(self):
        self.rec = json.loads(FIX.read_text())

    def test_the_forming_bar_is_dropped(self):
        raw, fetched = self.rec["response"], self.rec["fetched_ms"]
        self.assertGreater(raw[-1]["t"] + D.HOUR_MS, fetched, "fixture lost its forming bar")
        bars = D.normalise(raw, fetched)
        self.assertEqual(len(bars), len(raw) - 1)
        self.assertTrue(all(b["t"] + D.HOUR_MS <= fetched for b in bars))

    def test_values_are_the_venue_s_verbatim(self):
        bars = D.normalise(self.rec["response"], self.rec["fetched_ms"])
        for b, r in zip(bars, self.rec["response"]):
            self.assertEqual(b["c"], float(r["c"]))
            self.assertEqual(b["h"], float(r["h"]))

    def test_the_recorded_real_bars_validate_clean(self):
        bars = D.normalise(self.rec["response"], self.rec["fetched_ms"])
        v = D.validate(bars)
        self.assertTrue(v["usable"], v)
        self.assertEqual(v["gaps"], 0)


class Validate(unittest.TestCase):
    def bars(self):
        return [{"t": k * D.HOUR_MS, "o": 10, "h": 11, "l": 9, "c": 10, "v": 1}
                for k in range(5)]

    def test_counts_and_does_not_repair(self):
        b = self.bars()
        b[2] = dict(b[2], l=10.5)                    # low above the open
        del b[3]                                     # a missing hour
        before = json.dumps(b)
        v = D.validate(b)
        self.assertEqual((v["bad_ohlc"], v["gaps"], v["missing_bars"]), (1, 1, 1))
        self.assertFalse(v["usable"])
        self.assertEqual(json.dumps(b), before, "validate modified its input")

    def test_duplicates_are_unusable(self):
        b = self.bars()
        b.insert(2, dict(b[1]))
        self.assertFalse(D.validate(b)["usable"])


class SnapshotIntegrity(unittest.TestCase):
    def test_an_altered_bar_is_refused_on_load(self):
        bars = [{"t": k * D.HOUR_MS, "o": 10, "h": 11, "l": 9, "c": 10, "v": 1}
                for k in range(5)]
        snap = D.Snapshot({"X": bars}, {"coins": {"X": {"sha256": D.digest(bars)}}})
        path = Path(tempfile.mkdtemp()) / "s.json"
        snap.save(path)
        D.Snapshot.load(path)                        # clean: loads
        raw = json.loads(path.read_text())
        raw["bars"]["X"][3]["c"] = 10.01
        path.write_text(json.dumps(raw))
        with self.assertRaises(ValueError):
            D.Snapshot.load(path)


class Resample(unittest.TestCase):
    def test_complete_groups_only_and_true_aggregation(self):
        base = 1_700_006_400_000                     # a 4h boundary
        bars = [{"t": base + k * D.HOUR_MS, "o": 10 + k, "h": 20 + k, "l": 5 - k / 10,
                 "c": 11 + k, "v": 1.0, "n": 1} for k in range(10)]
        four = D.resample(bars, 4)
        self.assertEqual(len(four), 2)               # 10 bars: 2 full groups
        g = four[0]
        self.assertEqual((g["o"], g["h"], g["l"], g["c"], g["v"]),
                         (10, 23, 5 - 0.3, 14, 4.0))


if __name__ == "__main__":
    unittest.main()

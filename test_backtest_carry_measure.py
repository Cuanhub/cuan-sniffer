"""The carry measurement's arithmetic, by hand, on synthetic records only; no
market claim here."""

import json
import math
import tempfile
import unittest
from pathlib import Path

from research.backtest.experiments import carry_measure as C

H = C.H
JAN_2025 = 1735689600000          # 2025-01-01T00:00Z
K = C.INTEREST_HOURLY


def hours(start, n, rate):
    """(times, rates) for n funding hours from `start`; `rate` is a number or
    a function of the hour's index."""
    import numpy as np
    t = np.asarray([start + i * H for i in range(n)], dtype=np.int64)
    r = np.asarray([rate(i) if callable(rate) else rate for i in range(n)], dtype=float)
    return t, r


class Funding(unittest.TestCase):
    def test_the_interest_constant_is_about_eleven_percent_a_year(self):
        t, r = hours(JAN_2025, 24 * 30, K)
        self.assertAlmostEqual(C.annualised_pct(r), 10.95, places=9)
        f = C.funding_record(t, r)
        self.assertEqual(f["share_at_interest_constant"], 1.0)
        self.assertEqual(f["share_negative"], 0.0)

    def test_months_years_and_the_worst_complete_month(self):
        # January (744 h) at the constant, February (672 h) at -K: both complete
        t, r = hours(JAN_2025, 744 + 672, lambda i: K if i < 744 else -K)
        f = C.funding_record(t, r)
        self.assertEqual(f["complete_months"], 2)
        self.assertEqual(f["share_negative_months"], 0.5)
        self.assertEqual(f["worst_month"]["month"], "2025-02")
        self.assertAlmostEqual(f["worst_month"]["pct_of_notional"], -K * 672 * 100, places=12)
        self.assertEqual(list(f["by_year"]), ["2025"])

    def test_a_short_month_is_not_a_complete_month(self):
        t, r = hours(JAN_2025, 744 + 100, lambda i: K if i < 744 else -10 * K)
        f = C.funding_record(t, r)
        self.assertEqual(f["complete_months"], 1)
        self.assertEqual(f["worst_month"]["month"], "2025-01")

    def test_longest_run_of_negative_weeks(self):
        signs = [1, -1, -1, -1, 1, -1]
        t, r = hours(JAN_2025, 168 * len(signs), lambda i: signs[i // 168] * K)
        self.assertEqual(C.longest_negative_weeks(t, r), 3)

    def test_loading_sorts_and_keeps_one_row_an_hour(self):
        rows = [{"time": JAN_2025 + 2 * H + 79, "fundingRate": "0.2"},
                {"time": JAN_2025 + 79, "fundingRate": "0.1"},
                {"time": JAN_2025 + 2 * H + 80, "fundingRate": "0.3"}]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "X.json"
            p.write_text(json.dumps(rows))
            t, r = C.load_funding(p)
        self.assertEqual(list(t), [JAN_2025, JAN_2025 + 2 * H])
        self.assertEqual(list(r), [0.1, 0.3])


class Rule(unittest.TestCase):
    def test_a_constant_positive_rate_is_always_on(self):
        t, r = hours(JAN_2025, 24 * 60, K)
        x = C.rule_record(t, r)
        self.assertEqual(x["time_in_market"], 1.0)
        self.assertEqual(x["switches"], 2)                   # one entry, one exit
        self.assertAlmostEqual(x["gross_always_pct_per_year"], 10.95, places=6)
        self.assertAlmostEqual(x["gross_filtered_pct_per_year"], x["gross_always_pct_per_year"], places=12)
        yrs = x["years"]
        self.assertAlmostEqual(x["net_always_taker_pct_per_year"],
                               x["gross_always_pct_per_year"] - 2 * C.SWITCH_PCT["taker"] / yrs, places=9)

    def test_the_filter_steps_aside_once_the_trailing_week_turns_negative(self):
        t, r = hours(JAN_2025, 24 * 60, lambda i: K if i < 24 * 30 else -K)
        x = C.rule_record(t, r)
        self.assertEqual(x["switches"], 2)                   # in at the first decision, out once, never back
        self.assertGreater(x["gross_filtered_pct_per_year"], x["gross_always_pct_per_year"])
        self.assertTrue(0.4 < x["time_in_market"] < 0.6)
        # the first decision has a full week behind it
        self.assertEqual(x["window_from"], "2025-01-08")

    def test_a_decision_uses_no_funding_after_it(self):
        # one huge payment in the first hour after a 00:00, negative funding all around: that
        # day's decision cannot see it, so the filter misses it -- and then follows it in
        t, r = hours(JAN_2025, 24 * 20, lambda i: 1.0 if i == 24 * 12 + 1 else -K)
        x = C.rule_record(t, r)
        self.assertGreater(x["gross_always_pct_per_year"], 0)
        self.assertLess(x["gross_filtered_pct_per_year"], 0)
        self.assertEqual(x["switches"], 2)


class Capital(unittest.TestCase):
    def test_return_on_capital(self):
        self.assertAlmostEqual(C.return_on_capital(10.0, 3.0), 7.5)
        self.assertAlmostEqual(C.return_on_capital(10.0, 1.0), 5.0)

    def test_liquidation_rise(self):
        self.assertAlmostEqual(C.liquidation_rise(1.0, 5.0), 0.9 / 1.1)
        self.assertAlmostEqual(C.liquidation_rise(2.0, 40.0), 0.4875 / 1.0125)

    def test_worst_rises_by_hand(self):
        bars = [{"t": JAN_2025 + i * C.DAY, "c": c, "h": h} for i, (c, h) in
                enumerate([(100, 100), (100, 130), (100, 110), (120, 150)])]
        w = C.worst_rises(bars)
        self.assertAlmostEqual(w["1d"]["rise"], 0.5)          # 100 -> 150 the next day
        self.assertEqual(w["1d"]["from"], "2025-01-03")
        self.assertAlmostEqual(w["7d"]["rise"], 0.5)

    def test_the_table_flags_a_leverage_the_worst_day_would_have_liquidated(self):
        rises = {"1d": {"rise": 0.45}, "7d": {"rise": 0.9}}
        rows = {r["leverage"]: r for r in C.capital_table(20.0, 5.0, rises)}
        self.assertTrue(rows[1.0]["survives_worst_1d"])       # liquidates at +81.8%
        self.assertFalse(rows[2.0]["survives_worst_1d"])      # liquidates at +36.4%
        self.assertFalse(rows[1.0]["survives_worst_7d"])


class Basis(unittest.TestCase):
    def test_spot_over_perp_in_bps(self):
        spot = [{"t": 0, "c": 102.0}, {"t": H, "c": 99.5}, {"t": 2 * H, "c": 100.0}]
        perp = [{"t": 0, "c": 100.0}, {"t": H, "c": 100.0}]       # the third hour has no perp bar
        b = C.basis_record(spot, perp)
        self.assertEqual(b["hours"], 2)
        self.assertAlmostEqual(b["mean_bps"], 75.0, places=9)
        self.assertAlmostEqual(b["worst_bps"], 200.0, places=9)
        self.assertEqual(b["share_beyond_100bps"], 0.5)


class Vault(unittest.TestCase):
    def test_modified_dietz_and_its_range_by_hand(self):
        d = C.DAY
        det = {"portfolio": [
            ["allTime", {"accountValueHistory": [[0, "100"], [d, "210"], [2 * d, "231"]],
                         "pnlHistory": [[0, "0"], [d, "10"], [2 * d, "31"]]}],
            ["month", {"accountValueHistory": [], "pnlHistory": []}],
            ["week", {"accountValueHistory": [], "pnlHistory": []}]]}
        v = C.hlp_record(det)
        # interval 1: P&L 10 on 100 plus a 100 deposit at mid-interval -> 10/150; interval 2: 21/210
        self.assertAlmostEqual(v["twr_pct"], ((1 + 10 / 150) * 1.1 - 1) * 100, places=9)
        self.assertAlmostEqual(v["twr_range_pct"][0], (1.05 * 1.1 - 1) * 100, places=9)
        self.assertAlmostEqual(v["twr_range_pct"][1], (1.10 * 1.1 - 1) * 100, places=9)


class EndToEnd(unittest.TestCase):
    def test_main_writes_the_record(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "f").mkdir()
            (root / "daily").mkdir()
            (root / "basis").mkdir()
            n = 24 * 40
            (root / "f" / "BTC.json").write_text(json.dumps(
                [{"time": JAN_2025 + i * H + 79, "fundingRate": str(K)} for i in range(n)]))
            (root / "daily" / "BTC.json").write_text(json.dumps(
                [{"t": JAN_2025 + i * C.DAY, "o": "100", "h": "110", "l": "95", "c": "100"} for i in range(40)]))
            candles = [{"t": JAN_2025 + i * H, "o": "1", "h": "1", "l": "1", "c": "100"} for i in range(48)]
            (root / "basis" / "BTC_spot.json").write_text(json.dumps(candles))
            (root / "basis" / "BTC_perp.json").write_text(json.dumps(candles))
            (root / "meta.json").write_text(json.dumps([{"universe": [{"name": "BTC", "maxLeverage": 40}]}, []]))
            (root / "spot.json").write_text(json.dumps([
                {"tokens": [{"name": "USDC", "index": 0}, {"name": "UBTC", "index": 1, "fullName": "Unit Bitcoin"}],
                 "universe": [{"tokens": [1, 0], "name": "@1"}]},
                [{"coin": "@1", "dayNtlVlm": "5"}]]))
            out = root / "out"
            C.main(["--funding", str(root / "f"), "--daily", str(root / "daily"), "--basis", str(root / "basis"),
                    "--meta", str(root / "meta.json"), "--spot-meta", str(root / "spot.json"), "--out", str(out)])
            s = json.loads((out / "summary.json").read_text())
            btc = s["coins"]["BTC"]
            self.assertAlmostEqual(btc["funding"]["pct_per_year"], 10.95, places=6)
            self.assertEqual(btc["max_leverage"], 40.0)
            self.assertAlmostEqual(btc["rises"]["1d"]["rise"], 0.10)
            self.assertEqual(btc["basis"]["mean_bps"], 0.0)
            self.assertEqual(s["spot_tokens"]["BTC"], {"token": "UBTC", "full_name": "Unit Bitcoin", "day_volume": 5.0})
            self.assertTrue(math.isfinite(btc["capital"][0]["return_on_capital_pct"]))
            self.assertIn("Cash-and-carry", (out / "report.md").read_text())


if __name__ == "__main__":
    unittest.main()

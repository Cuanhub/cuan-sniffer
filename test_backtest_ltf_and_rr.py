"""The 15m structure and the reward:risk experiment's machinery.

Hand-built bars test the MACHINERY with known answers; causality is proven
on recorded real 15m bars. No claim about the market comes from this file.
"""

import json
import math
import unittest
from pathlib import Path

import numpy as np

from research.backtest import common as C
from research.backtest import data as D
from research.backtest import ltf
from research.backtest.engine import LIMIT, MARKET_CLOSE, Trade
from research.backtest.experiments import ltf_15m as L
from research.backtest.experiments import rr_floor as R

FIX = Path(__file__).parent / "tests/fixtures/hyperliquid/candle_snapshot_15m_eth.json"
Q = 900_000
H = 3_600_000


def real15():
    rec = json.loads(FIX.read_text())
    return D.normalise(rec["response"], rec["fetched_ms"], Q)


def mk(highs, closes=None, lows=None, t0=1_700_006_400_000, step=Q):
    closes = closes or [h - 1 for h in highs]
    lows = lows or [c - 2 for c in closes]
    return [{"t": t0 + k * step, "o": c, "h": h, "l": lo, "c": c}
            for k, (h, c, lo) in enumerate(zip(highs, closes, lows))]


def trade(kind, entry_i, entry_px, stop=90.0, tp=120.0, decided_i=None, r=-1.0, t=0):
    return Trade(coin="X", tag="band", kind=kind, decided_i=entry_i if decided_i is None else decided_i,
                 entry_i=entry_i, entry_t=t, entry_px=entry_px, stop=stop, tp=tp, exit_i=entry_i + 5,
                 exit_px=stop, outcome="stop", r_gross=r, cost_r=0.0, r_net=r,
                 features={"rr": (tp - entry_px) / (entry_px - stop), "stop_atr_x": 1.6})


class BreakOfStructure(unittest.TestCase):
    def test_causal_on_real_15m_bars(self):
        bars = real15()
        self.assertGreater(len(bars), 600)
        full = ltf.bos_breaks(bars)
        self.assertGreater(sum(full), 5, "too few breaks to mean anything")
        for t in range(40, len(bars), 3):
            self.assertEqual(ltf.bos_breaks(bars[:t + 1])[t], full[t], f"t={t}")

    def test_the_first_close_above_the_last_swing_high(self):
        #        0   1   2   3   4    5   6   7   8   9  10  11
        highs = [10, 11, 12, 13, 20, 14, 13, 12, 11, 10, 15, 22]
        closes = [h - 1 for h in highs]
        closes[11] = 21                                   # first close above 20
        br = ltf.bos_breaks(mk(highs, closes))
        self.assertEqual([i for i, x in enumerate(br) if x], [11])

    def test_not_before_the_swing_high_is_confirmed(self):
        highs = [10, 11, 12, 13, 20, 14, 13, 30]          # pivot 4 confirms at 7
        closes = [h - 1 for h in highs]
        closes[6] = 25                                    # above 20, but at bar 6
        br = ltf.bos_breaks(mk(highs, closes))
        self.assertFalse(br[6])

    def test_one_break_per_level(self):
        highs = [10, 11, 12, 13, 20, 14, 13, 12, 25, 26]
        closes = [h - 1 for h in highs]
        closes[8], closes[9] = 22, 24
        br = ltf.bos_breaks(mk(highs, closes))
        self.assertEqual([i for i, x in enumerate(br) if x], [8])

    def test_swing_lows_mirror_swing_highs(self):
        lows = [10, 9, 8, 7, 2, 6, 7, 8, 9]
        bars = mk([l + 5 for l in lows], [l + 2 for l in lows], lows)
        self.assertEqual(ltf.swing_lows(bars), [4])


class LtfStop(unittest.TestCase):
    def test_never_wider_than_the_band_stop(self):
        lows = [10, 9, 8, 7, 2, 6, 7, 8, 9, 9, 9]
        bars = mk([l + 5 for l in lows], [l + 2 for l in lows], lows)
        sw = ltf.swing_lows(bars)
        atr = [0.1] * len(bars)
        self.assertEqual(ltf.ltf_stop(bars, sw, atr, 10, band_stop=5.0), 5.0)
        tight = ltf.ltf_stop(bars, sw, atr, 10, band_stop=1.0)
        self.assertAlmostEqual(tight, 2 * (1 - 0.0025))

    def test_uses_only_confirmed_swing_lows(self):
        lows = [10, 9, 8, 7, 2, 6, 7, 8, 9]
        bars = mk([l + 5 for l in lows], [l + 2 for l in lows], lows)
        sw = ltf.swing_lows(bars)
        self.assertEqual(ltf.ltf_stop(bars, sw, [0.1] * 9, 6, band_stop=0.5), 0.5)


class ArmIndex(unittest.TestCase):
    def setUp(self):
        self.h = [{"t": 1_700_006_400_000 + k * H, "o": 100, "h": 101, "l": 99, "c": 100}
                  for k in range(10)]
        self.q = [{"t": 1_700_006_400_000 + k * Q, "o": 100, "h": 101, "l": 100, "c": 100}
                  for k in range(40)]
        self.t15 = [b["t"] for b in self.q]

    def test_a_bid_arms_at_the_quarter_that_filled_it(self):
        self.q[4 * 3 + 2]["l"] = 98.5                     # hour 3, third quarter
        tr = trade(LIMIT, 3, 99.0)
        self.assertEqual(ltf.arm_index(tr, self.h, self.t15, self.q), 14)

    def test_a_market_entry_arms_after_its_close(self):
        tr = trade(MARKET_CLOSE, 3, 100.0)
        self.assertEqual(ltf.arm_index(tr, self.h, self.t15, self.q), 16)

    def test_outside_the_15m_window_is_none(self):
        tr = trade(MARKET_CLOSE, 9, 100.0)
        self.assertIsNone(ltf.arm_index(tr, self.h, self.t15, self.q[:20]))


class Measurement(unittest.TestCase):
    def test_15m_bars_decide_what_an_ambiguous_1h_bar_could_not(self):
        """The 1h bar touched both; at 15m the target came first."""
        t0 = 1_700_006_400_000
        h = [{"t": t0, "o": 100, "h": 100, "l": 100, "c": 100},
             {"t": t0 + H, "o": 100, "h": 121, "l": 89, "c": 95}]
        q = ([{"t": t0 + k * Q, "o": 100, "h": 100, "l": 100, "c": 100} for k in range(4)] +
             [{"t": t0 + H, "o": 100, "h": 121, "l": 99, "c": 120},          # target first
              {"t": t0 + H + Q, "o": 120, "h": 120, "l": 89, "c": 95}] +
             [{"t": t0 + H + k * Q, "o": 95, "h": 96, "l": 94, "c": 95} for k in range(2, 4)])
        from research.backtest.costs import MEASURED
        from research.backtest.engine import Order, simulate
        base = simulate(h, Order("X", 0, MARKET_CLOSE, 90.0, 120.0), MEASURED)
        self.assertTrue(base.ambiguous and base.outcome == "stop")
        m = L.measure_on_15m(base, h, [b["t"] for b in q], q)
        self.assertEqual(m.outcome, "tp")


class NullData(unittest.TestCase):
    def test_15m_and_1h_come_from_one_walk(self):
        hourly, fifteen = C.random_walks_15m(0, coins=1, n_1h=50)
        c = next(iter(hourly))
        q, h = fifteen[c], hourly[c]
        self.assertEqual(len(h), len(q) // 4)
        for k in (0, 7, 49):
            g = q[4 * k:4 * k + 4]
            self.assertEqual((h[k]["o"], h[k]["h"], h[k]["l"], h[k]["c"]),
                             (g[0]["o"], max(b["h"] for b in g), min(b["l"] for b in g), g[-1]["c"]))

    def test_null_summary_flags_a_bias_and_false_positives(self):
        biased = [{"x": {"estimate": 0.05 + 0.001 * i, "adds_edge": False}} for i in range(20)]
        self.assertTrue(C.null_summary(biased)["x"]["miscalibrated"])
        fires = [{"x": {"estimate": (-1) ** i * 0.01, "adds_edge": i < 3}} for i in range(20)]
        self.assertTrue(C.null_summary(fires)["x"]["miscalibrated"])
        clean = [{"x": {"estimate": float(e), "adds_edge": False}}
                 for e in np.random.default_rng(3).normal(0, 0.05, 20)]
        self.assertFalse(C.null_summary(clean)["x"]["miscalibrated"])


class Controls(unittest.TestCase):
    def test_draws_are_after_the_entry(self):
        d = C.draws_after(5000, 3000, C.rng_for("ETH", 3000, 1))
        self.assertTrue((d > 3000).all() and (d <= 3240).all())

    def test_draws_read_module_values_at_call_time(self):
        old = C.CONTROL_DRAWS
        try:
            C.CONTROL_DRAWS = 7
            self.assertEqual(len(C.draws_after(5000, 3000, C.rng_for("ETH", 1, 1))), 7)
        finally:
            C.CONTROL_DRAWS = old


class RrBuckets(unittest.TestCase):
    def test_each_bucket_takes_its_rr_range_from_its_own_run(self):
        runs = {2.0: {"X": [trade(LIMIT, 1, 100.0, stop=95, tp=115)]},          # rr 3
                1.0: {"X": [trade(LIMIT, 1, 100.0, stop=95, tp=107.5),          # rr 1.5
                            trade(LIMIT, 2, 100.0, stop=95, tp=112.5)]},        # rr 2.5: not mid
                0.3: {"X": [trade(LIMIT, 1, 100.0, stop=95, tp=102.5)]}}        # rr 0.5
        self.assertEqual(len(R.bucket_trades(runs, "E_live")), 1)
        mid = R.bucket_trades(runs, "E_mid")
        self.assertEqual([round(t.features["rr"], 2) for t in mid], [1.5])
        self.assertEqual(len(R.bucket_trades(runs, "E_low")), 1)

    def test_decide_requires_every_criterion(self):
        from research.backtest import stats as S
        fr = {"E_mid": {"estimate": 0.2, "halves": [0.1, 0.3], "p": 0.001}}
        ok = R.decide(fr, S.holm({"E_mid": 0.001}), {"E_mid": False})
        self.assertTrue(ok["E_mid"]["has_edge"] and ok["E_mid"]["material"])
        self.assertFalse(R.decide(fr, S.holm({"E_mid": 0.001}), {"E_mid": True})["E_mid"]["has_edge"])
        fr2 = {"E_mid": {"estimate": 0.2, "halves": [-0.1, 0.5], "p": 0.001}}
        self.assertFalse(R.decide(fr2, S.holm({"E_mid": 0.001}))["E_mid"]["has_edge"])


if __name__ == "__main__":
    unittest.main()

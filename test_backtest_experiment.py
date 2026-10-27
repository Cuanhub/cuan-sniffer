"""The experiment's own decision helpers, pinned to hand-checked cases.

Synthetic bars here test the MACHINERY; no result about the market comes
from them.
"""

import math
import unittest

import numpy as np

from research.backtest import stats as S
from research.backtest.engine import LIMIT, MARKET_CLOSE, Trade
from research.backtest.experiments import trendline_breakout as X


def bar(o, h, l, c, k):
    return {"t": 1_700_006_400_000 + k * 3_600_000, "o": o, "h": h, "l": l, "c": c}


def base_trade(kind=LIMIT, entry_i=1, stop=95.0, tp=110.0, entry_px=100.0,
               r_net=-1.0, resolved=True):
    return Trade(coin="X", tag="band", kind=kind, decided_i=0, entry_i=entry_i,
                 entry_t=1_700_006_400_000 + entry_i * 3_600_000, entry_px=entry_px,
                 stop=stop, tp=tp, exit_i=5 if resolved else None, exit_px=95.0,
                 outcome="stop" if resolved else "open", r_gross=r_net, cost_r=0.0,
                 r_net=r_net)


class Confirm(unittest.TestCase):
    def flat(self, n=60):
        return [bar(100, 101, 99, 100, k) for k in range(n)]

    def test_a_break_enters_at_the_next_open_with_the_same_stop_and_target(self):
        bars = self.flat()
        bars[4] = bar(100, 102, 99.5, 101.5, 4)
        bars[5] = bar(101.7, 111, 101, 110.5, 5)
        br = [False] * len(bars)
        br[4] = True
        status, tr = X.confirm(bars, br, base_trade(entry_i=1), 48, X.MEASURED)
        self.assertEqual(status, "entered")
        self.assertEqual((tr.entry_i, tr.entry_px), (5, 101.7))
        self.assertEqual((tr.stop, tr.tp), (95.0, 110.0))
        self.assertEqual(tr.outcome, "tp")

    def test_the_stop_first_means_no_trade(self):
        bars = self.flat()
        bars[3] = bar(100, 100, 94, 96, 3)
        br = [False] * len(bars)
        br[6] = True
        self.assertEqual(X.confirm(bars, br, base_trade(entry_i=1), 48, X.MEASURED),
                         ("stopped_first", None))

    def test_a_bid_s_fill_bar_low_can_invalidate(self):
        bars = self.flat()
        bars[1] = bar(100, 100, 94, 99, 1)
        br = [False] * len(bars)
        br[1] = True
        self.assertEqual(X.confirm(bars, br, base_trade(LIMIT, entry_i=1), 48, X.MEASURED)[0],
                         "stopped_first")

    def test_a_market_setup_s_own_decision_low_cannot(self):
        """Its low printed before the close the setup was armed at."""
        bars = self.flat()
        bars[1] = bar(100, 100, 94, 99, 1)
        br = [False] * len(bars)
        br[1] = True
        status, _ = X.confirm(bars, br, base_trade(MARKET_CLOSE, entry_i=1), 48, X.MEASURED)
        self.assertEqual(status, "entered")

    def test_no_break_within_the_wait(self):
        bars = self.flat()
        br = [False] * len(bars)
        br[1 + 49] = True
        self.assertEqual(X.confirm(bars, br, base_trade(entry_i=1), 48, X.MEASURED)[0],
                         "no_break")
        br[1 + 48] = True
        self.assertEqual(X.confirm(bars, br, base_trade(entry_i=1), 48, X.MEASURED)[0],
                         "entered")

    def test_the_rr_floor_refuses_a_late_entry(self):
        bars = self.flat()
        bars[5] = bar(108, 109, 107, 108, 5)        # next open far above entry
        br = [False] * len(bars)
        br[4] = True
        self.assertEqual(X.confirm(bars, br, base_trade(entry_i=1), 48, X.MEASURED,
                                   min_rr=2.0)[0], "rr_below_floor")


class Shift(unittest.TestCase):
    def test_circular_shift_keeps_every_event(self):
        s = [False] * 100
        for i in (3, 40, 97):
            s[i] = True
        for off in (1, 25, 60, 99):
            sh = X._shift(s, off)
            self.assertEqual(sum(sh), 3)
            self.assertEqual(sorted(i for i, x in enumerate(sh) if x),
                             sorted((i + off) % 100 for i in (3, 40, 97)))


class Paired(unittest.TestCase):
    def test_the_primary_is_trendline_minus_random_arm(self):
        b1 = base_trade(r_net=-1.0)
        b2 = base_trade(r_net=+2.0)
        win = base_trade(kind="market_next_open", r_net=+0.5)
        rows = [{"base": b1, "status": "stopped_first", "conf": None, "ctrl": -0.2,
                 "ctrl_entry_rate": 0.3},
                {"base": b2, "status": "entered", "conf": win, "ctrl": 0.1,
                 "ctrl_entry_rate": 0.4}]
        p = X.h2_paired(rows)
        self.assertEqual(p["d_primary"], [0.0 - (-0.2), 0.5 - 0.1])
        self.assertEqual(p["d_vs_base"], [0.0 - (-1.0), 0.5 - 2.0])
        self.assertEqual(p["d_random_vs_base"], [-0.2 - (-1.0), 0.1 - 2.0])

    def test_open_trades_and_missing_controls_are_dropped_not_zeroed(self):
        rows = [{"base": base_trade(resolved=False), "status": "entered", "conf": None,
                 "ctrl": 0.0, "ctrl_entry_rate": 0.0},
                {"base": base_trade(), "status": "entered",
                 "conf": base_trade(resolved=False), "ctrl": 0.0, "ctrl_entry_rate": 0.0},
                {"base": base_trade(), "status": "no_break", "conf": None, "ctrl": None,
                 "ctrl_entry_rate": None}]
        p = X.h2_paired(rows)
        self.assertEqual(p["d_primary"], [])
        self.assertEqual(p["dropped_open"], 3)


class NullData(unittest.TestCase):
    def test_random_walks_are_a_martingale_in_price(self):
        bars = X.random_walks(0, coins=3, n=4000)
        ratios = [b2["c"] / b1["c"] for bs in bars.values() for b1, b2 in zip(bs, bs[1:])]
        m = float(np.mean(ratios))
        se = float(np.std(ratios)) / math.sqrt(len(ratios))
        self.assertLess(abs(m - 1.0), 4 * se)

    def test_ohlc_is_consistent(self):
        for bs in X.random_walks(1, coins=2, n=500).values():
            for b in bs:
                self.assertLessEqual(b["l"], min(b["o"], b["c"]))
                self.assertGreaterEqual(b["h"], max(b["o"], b["c"]))


class Decide(unittest.TestCase):
    def summary(self, est, p, halves, ctrl=0.1, strata=(0.1, 0.1)):
        c = {"estimate": est, "p": p}
        return {
            "H1": {"cluster": c, "split_half": [{"diff": h} for h in halves],
                   "matched_control": {"excess_diff": ctrl},
                   "stop_atr_strata": {"below": {"diff": strata[0]},
                                       "at_or_above": {"diff": strata[1]}}},
            "H2": {"cluster": c, "split_half": [{"mean": h} for h in halves]},
            "H3": {"cluster": c, "split_half": [{"mean": h} for h in halves]},
            "holm": S.holm({"H1": p, "H2": p, "H3": p})}

    def test_every_criterion_is_required(self):
        ok = self.summary(0.2, 0.001, (0.1, 0.3))
        self.assertTrue(all(v["adds_edge"] for v in X.decide(ok).values()))
        self.assertFalse(X.decide(self.summary(0.2, 0.2, (0.1, 0.3)))["H2"]["adds_edge"])
        self.assertFalse(X.decide(self.summary(-0.2, 0.001, (-0.1, -0.3)))["H3"]["adds_edge"])
        self.assertFalse(X.decide(self.summary(0.2, 0.001, (-0.1, 0.5)))["H2"]["adds_edge"])
        self.assertFalse(X.decide(self.summary(0.2, 0.001, (0.1, 0.3), ctrl=-0.01))["H1"]["adds_edge"])
        self.assertFalse(X.decide(self.summary(0.2, 0.001, (0.1, 0.3), strata=(0.1, -0.1)))["H1"]["adds_edge"])
        bad = X.decide(ok, miscalibrated={"H2": True})
        self.assertFalse(bad["H2"]["adds_edge"])
        self.assertTrue(bad["H1"]["adds_edge"])

    def test_nan_is_never_positive(self):
        s = self.summary(float("nan"), 0.001, (float("nan"), 0.3))
        self.assertFalse(X.decide(s)["H3"]["adds_edge"])


if __name__ == "__main__":
    unittest.main()

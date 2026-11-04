"""The short side: the engine's mirrored rules, the mirror's construction,
its controls, and the experiment's statistics (PREREG_short_mirror.md).

Hand-built bars test the machinery, as in test_backtest_engine. The mirror's
claim -- that it IS the live strategy upside down -- is tested on the
recorded real bars in tests/fixtures, where it has to hold exactly.
"""

import json
import math
import unittest
from pathlib import Path

import numpy as np

from research.backtest import band
from research.backtest import common as C
from research.backtest.costs import FRICTIONLESS, Costs
from research.backtest.engine import (LIMIT, LONG, MARKET_CLOSE, SHORT, Order,
                                      Trade, simulate)
from research.backtest.experiments import short_mirror as M
from research.levels import zones as Z

REAL = json.loads((Path(__file__).parent / "tests/fixtures/zones/bars_1h.json").read_text())


def bar(o, h, l, c, t=0):
    return {"t": t, "o": o, "h": h, "l": l, "c": c}


def short(i, kind, stop, tp, limit=0.0, ttl=12):
    return Order("X", i, kind, stop, tp, limit=limit, ttl=ttl, side=SHORT)


class ShortEntries(unittest.TestCase):
    def test_the_decision_bar_s_own_high_cannot_stop_a_market_short(self):
        bars = [bar(100, 110, 99, 100),      # spiked through 105 before the close
                bar(100, 101, 89, 90)]       # then the target
        tr = simulate(bars, short(0, MARKET_CLOSE, 105, 90), FRICTIONLESS)
        self.assertEqual((tr.outcome, tr.side), ("tp", SHORT))

    def test_a_resting_sell_fills_when_the_high_reaches_it(self):
        bars = [bar(95, 96, 94, 95), bar(95, 101, 95, 99), bar(99, 99, 89, 90)]
        tr = simulate(bars, short(0, LIMIT, 105, 90, limit=100), FRICTIONLESS)
        self.assertEqual((tr.entry_i, tr.entry_px, tr.outcome), (1, 100, "tp"))

    def test_the_target_on_the_fill_bar_is_not_counted(self):
        """The low may have printed before the rally that filled the sell."""
        bars = [bar(95, 96, 94, 95),
                bar(95, 101, 89, 96),        # reaches 90 AND fills at 100
                bar(96, 106, 95, 105)]       # then stops out
        tr = simulate(bars, short(0, LIMIT, 105, 90, limit=100), FRICTIONLESS)
        self.assertEqual(tr.outcome, "stop")
        self.assertTrue(tr.fill_bar_tp)

    def test_the_stop_on_the_fill_bar_does_count(self):
        bars = [bar(95, 96, 94, 95), bar(95, 107, 95, 104)]
        tr = simulate(bars, short(0, LIMIT, 105, 90, limit=100), FRICTIONLESS)
        self.assertEqual((tr.outcome, tr.exit_i), ("stop", 1))

    def test_a_sell_that_never_fills_is_no_trade_and_expires(self):
        self.assertIsNone(simulate([bar(95, 96, 94, 95)] * 20, short(0, LIMIT, 105, 90, limit=100)))
        late = [bar(95, 96, 94, 95)] * 13 + [bar(95, 101, 95, 99)]
        self.assertIsNone(simulate(late, short(0, LIMIT, 105, 90, limit=100, ttl=12)))

    def test_an_entry_outside_stop_and_target_is_no_trade(self):
        bars = [bar(100, 100, 100, 100), bar(88, 89, 87, 88)]
        self.assertIsNone(simulate(bars, Order("X", 0, "market_next_open", 105, 90, side=SHORT)))

    def test_entry_slippage_sells_lower(self):
        c = Costs(0.0, 0.0, 0.0, 0.0, market_entry_slip_pct=0.1)
        bars = [bar(100, 100, 100, 100), bar(100, 101, 99, 100)]
        self.assertAlmostEqual(simulate(bars, short(0, MARKET_CLOSE, 105, 90), c).entry_px, 99.9)


class ShortExits(unittest.TestCase):
    def test_a_bar_reaching_both_is_a_loss_and_flagged(self):
        bars = [bar(100, 100, 100, 100), bar(100, 106, 89, 100)]
        tr = simulate(bars, short(0, MARKET_CLOSE, 105, 90), FRICTIONLESS)
        self.assertEqual(tr.outcome, "stop")
        self.assertTrue(tr.ambiguous)

    def test_a_gap_through_the_stop_exits_at_the_open(self):
        bars = [bar(100, 100, 100, 100), bar(110, 112, 108, 109)]
        tr = simulate(bars, short(0, MARKET_CLOSE, 105, 90), FRICTIONLESS)
        self.assertEqual(tr.exit_px, 110)
        self.assertAlmostEqual(tr.r_gross, -2.0)      # worse than -1R, as it was

    def test_a_gap_through_the_target_takes_the_better_open(self):
        bars = [bar(100, 100, 100, 100), bar(85, 86, 84, 85)]
        tr = simulate(bars, short(0, MARKET_CLOSE, 105, 90), FRICTIONLESS)
        self.assertEqual((tr.outcome, tr.exit_px), ("tp", 85))

    def test_fees_and_slippage_by_hand(self):
        c = Costs(maker_fee_pct=0.02, taker_fee_pct=0.05, stop_slip_pct=0.1,
                  tp_slip_pct=0.0, market_entry_slip_pct=0.0)
        bars = [bar(95, 96, 94, 95), bar(95, 101, 95, 99), bar(99, 106, 99, 105)]
        tr = simulate(bars, short(0, LIMIT, 105, 90, limit=100), c)
        exit_px = 105 * (1 + 0.001)                    # buying back higher
        self.assertAlmostEqual(tr.exit_px, exit_px)
        self.assertAlmostEqual(tr.r_gross, (100 - exit_px) / 5)
        self.assertAlmostEqual(tr.cost_r, (0.0002 * 100 + 0.0005 * exit_px) / 5)
        self.assertAlmostEqual(tr.risk_pct, 5.0)
        self.assertAlmostEqual(tr.planned_rr, 2.0)

    def test_a_long_is_unchanged(self):
        bars = [bar(105, 106, 104, 105), bar(105, 105, 99, 101), bar(101, 111, 100, 110)]
        tr = simulate(bars, Order("X", 0, LIMIT, 95, 110, limit=100), FRICTIONLESS)
        self.assertEqual((tr.side, tr.outcome, tr.risk_pct, tr.planned_rr), (LONG, "tp", 5.0, 2.0))


class TheMirrorIsTheStrategyUpsideDown(unittest.TestCase):
    def test_inverting_twice_gives_the_bars_back(self):
        for coin in ("ETH", "HYPE", "FARTCOIN"):
            back = band.invert(band.invert(REAL[coin]))
            for a, b in zip(REAL[coin], back):
                for k in "ohlc":
                    self.assertAlmostEqual(a[k], b[k], delta=1e-9 * a[k])

    def test_highs_and_lows_swap_and_stay_consistent(self):
        inv = band.invert(REAL["ETH"])
        for b in inv:
            self.assertLessEqual(b["l"], min(b["o"], b["c"]))
            self.assertGreaterEqual(b["h"], max(b["o"], b["c"]))
        self.assertAlmostEqual(inv[0]["c"], REAL["ETH"][0]["c"])

    def test_supply_bands_are_the_demand_bands_of_the_inverted_bars(self):
        """zones.supply_zones calls itself the mirror of demand; on real bars
        it must be exactly that, or the experiment trades something else."""
        compared = 0
        for coin, bars in REAL.items():
            k = band.invert_k(bars)
            sup = Z.supply_zones(bars, coin=coin, min_bars=200)
            dem = Z.demand_zones(band.invert(bars), coin=coin, min_bars=200)
            mapped = sorted((z.base_i, z.bos_i, k / z.hi, k / z.lo) for z in dem)
            got = sorted((z.base_i, z.bos_i, z.lo, z.hi) for z in sup)
            self.assertEqual([m[:2] for m in mapped], [g[:2] for g in got], coin)
            for m, g in zip(mapped, got):
                self.assertAlmostEqual(m[2], g[2], delta=1e-9 * g[2])
                self.assertAlmostEqual(m[3], g[3], delta=1e-9 * g[3])
            compared += len(got)
        self.assertGreater(compared, 50)

    def test_a_short_resolves_exactly_as_the_long_on_inverted_bars(self):
        """Same fill bar, exit bar and outcome; frictionless R related by the
        exact identity R_long_inverted = R_short x stop / exit."""
        rng = np.random.default_rng(7)
        checked = 0
        for coin in ("ETH", "HYPE", "AAVE", "WLD"):
            bars = REAL[coin]
            inv, k = band.invert(bars), band.invert_k(bars)
            for _ in range(60):
                i = int(rng.integers(0, len(bars) - 50))
                c = bars[i]["c"]
                risk, reward = rng.uniform(0.005, 0.04), rng.uniform(0.005, 0.08)
                kind = LIMIT if rng.random() < 0.5 else MARKET_CLOSE
                lim = c * (1 + rng.uniform(0.0, 0.02)) if kind == LIMIT else 0.0
                base = lim if kind == LIMIT else c
                so = Order(coin, i, kind, base * (1 + risk), base * (1 - reward), limit=lim, side=SHORT)
                lo = Order(coin, i, kind, k / so.stop, k / so.tp, limit=k / lim if lim else 0.0)
                ts, tl = simulate(bars, so, FRICTIONLESS), simulate(inv, lo, FRICTIONLESS)
                self.assertEqual(ts is None, tl is None)
                if ts is None:
                    continue
                for f in ("entry_i", "exit_i", "outcome", "ambiguous", "fill_bar_tp"):
                    self.assertEqual(getattr(ts, f), getattr(tl, f), f)
                if ts.resolved:
                    self.assertAlmostEqual(tl.r_gross, ts.r_gross * so.stop / ts.exit_px, places=9)
                    checked += 1
        self.assertGreater(checked, 100)

    def test_the_replay_shorts_on_the_long_s_decisions_upside_down(self):
        made = 0
        for coin in ("ETH", "HYPE", "AAVE", "WLD", "SUI", "ENA", "NEAR", "BCH"):
            bars = REAL[coin]
            shorts = band.run(coin, bars, FRICTIONLESS, side=SHORT)
            longs_inv = band.run(coin, band.invert(bars), FRICTIONLESS)
            self.assertEqual([(t.decided_i, t.kind, t.entry_i, t.outcome) for t in shorts],
                             [(t.decided_i, t.kind, t.entry_i, t.outcome) for t in longs_inv], coin)
            for t in shorts:
                self.assertEqual(t.side, SHORT)
                self.assertTrue(t.tp < t.entry_px < t.stop)
                self.assertLess(t.features["zone_lo"], t.features["zone_hi"])
            made += len(shorts)
        self.assertGreater(made, 0)

    def test_side_must_be_known(self):
        with self.assertRaises(ValueError):
            band.run("ETH", REAL["ETH"], side="sideways")


class ShortControls(unittest.TestCase):
    def _orders(self, trade, bars):
        seen = []
        real = C.simulate
        try:
            C.simulate = lambda b, o, costs=None: (seen.append(o), real(b, o))[1]   # type: ignore
            C.same_order_control(bars, trade)
        finally:
            C.simulate = real   # type: ignore
        return seen

    def test_a_short_bid_s_controls_are_sells_above_the_close_with_its_geometry(self):
        bars = REAL["ETH"]
        dec = 600                       # controls are drawn after the 799-bar warm-up
        c = bars[dec]["c"]
        ent = c * 1.01
        t = Trade("ETH", "band_short", LIMIT, dec, dec + 1, 0, ent, ent * 1.03, ent * 0.94,
                  dec + 5, ent, "tp", 2.0, 0.0, 2.0, side=SHORT)
        orders = self._orders(t, bars)
        self.assertEqual(len(orders), C.CONTROL_DRAWS)
        for o in orders:
            self.assertEqual((o.side, o.kind), (SHORT, LIMIT))
            close = bars[o.decided_i]["c"]
            self.assertAlmostEqual(o.limit / close, 1.01, places=9)
            self.assertAlmostEqual(o.stop / o.limit, 1.03, places=9)
            self.assertAlmostEqual(o.tp / o.limit, 0.94, places=9)
            self.assertGreater(o.decided_i, t.entry_i)

    def test_a_market_short_s_controls_are_market_shorts(self):
        bars = REAL["ETH"]
        c = bars[600]["c"]
        t = Trade("ETH", "band_short", MARKET_CLOSE, 600, 600, 0, c, c * 1.02, c * 0.96,
                  605, c, "tp", 2.0, 0.0, 2.0, side=SHORT)
        orders = self._orders(t, bars)
        self.assertEqual(len(orders), C.CONTROL_DRAWS)
        for o in orders:
            self.assertEqual((o.side, o.kind), (SHORT, MARKET_CLOSE))
            close = bars[o.decided_i]["c"]
            self.assertAlmostEqual(o.stop / close, 1.02, places=9)
            self.assertAlmostEqual(o.tp / close, 0.96, places=9)

    def test_funding_is_received_short_and_paid_long_in_r(self):
        mk = lambda side, stop: Trade("X", "", MARKET_CLOSE, 0, 0, 0, 100.0, stop, 0.0, 10, 100.0,  # noqa: E731
                                      "tp", 0.0, 0.0, 0.0, side=side)
        self.assertAlmostEqual(C.funding_r(mk(SHORT, 102.0), 0.01), 0.01 * 10 / 2.0)
        self.assertAlmostEqual(C.funding_r(mk(LONG, 98.0), 0.01), -0.01 * 10 / 2.0)
        self.assertEqual(C.funding_r(mk(SHORT, 102.0), 0.0), 0.0)


class TheStatisticsFindAnEdgeWhenThereIsOne(unittest.TestCase):
    def _rows(self, mean, n=600, seed=3):
        rng = np.random.default_rng(seed)
        t0 = 1_775_000_000_000
        out = []
        for k in range(n):
            t = Trade("C%d" % (k % 30), "", LIMIT, 0, 0, t0 + k * 3_600_000 * 8, 100.0, 101.0, 98.0,
                      1, 100.0, "tp", 0.0, 0.0, float(rng.normal(-0.1, 2.0)), side=SHORT)
            out.append((t, float(rng.normal(mean, 2.0))))
        return out

    def test_a_planted_edge_is_found(self):
        fr = M.framing_stats(self._rows(0.4), B=2000)
        self.assertLess(fr["p"], 0.01)
        self.assertGreater(min(fr["halves"]), 0)
        d = M.decide({"S_live": fr}, {"S_live": False})["S_live"]
        self.assertTrue(d["has_edge"] and d["material"])

    def test_noise_is_rarely_an_edge(self):
        fired = sum(M.decide({"S_live": M.framing_stats(self._rows(0.0, seed=s), B=500,
                                                        two_way_se=False)})["S_live"]["has_edge"]
                    for s in range(20))
        self.assertLessEqual(fired, 3)       # ~0.5 expected: p < 0.05 AND positive AND both halves

    def test_every_criterion_is_required_and_nan_is_never_positive(self):
        fr = {"estimate": 0.3, "p": 0.01, "halves": [0.2, float("nan")]}
        self.assertFalse(M.decide({"S_live": fr})["S_live"]["has_edge"])
        fr = {"estimate": 0.3, "p": 0.01, "halves": [0.2, 0.4]}
        self.assertFalse(M.decide({"S_live": fr}, {"S_live": True})["S_live"]["has_edge"])
        self.assertTrue(M.decide({"S_live": fr}, {"S_live": False})["S_live"]["has_edge"])

    def test_the_two_way_se_is_never_below_the_week_se(self):
        fr = M.framing_stats(self._rows(0.1), B=1000)
        self.assertGreaterEqual(fr["se_2way"], fr["se"] - 1e-12)


class TheOracleIsSeen(unittest.TestCase):
    """The pipeline's controls must be able to tell a strategy that knows
    the future from timing-free shorts on the same real bars."""

    def test_an_oracle_short_beats_its_controls(self):
        excess = []
        for coin, bars in C.random_walks(5, coins=2, n=3000).items():
            oracle = lambda o, b=bars: (lambda tr: tr is not None and tr.outcome == "tp")(simulate(b, o))  # noqa: E731
            for t in band.run(coin, bars, side=SHORT, gate=oracle):
                self.assertEqual(t.outcome, "tp")
                m = C.same_order_control(bars, t)
                if m is not None:
                    excess.append(t.r_net - m)
        self.assertGreater(len(excess), 5)
        self.assertGreater(float(np.mean(excess)), 1.0)


class TheExperimentRuns(unittest.TestCase):
    def test_end_to_end_on_a_small_random_walk(self):
        bars = C.random_walks(0, coins=2, n=1400)
        with C.pool(bars, workers=2) as pool:
            s = M.analyse(pool, bars, B=200, log=lambda *_: None)
        self.assertIn("S_live", s["framings"])
        for key in ("absolute", "by_entry_kind", "stop_atr_strata", "by_market_week",
                    "long_side_same_code", "combined_book", "funding_sensitivity", "trades"):
            self.assertIn(key, s)
        self.assertTrue(all(isinstance(r["excess"], float) for r in s["trades"]))
        self.assertTrue(math.isfinite(s["combined_book"]["long_weekly_mean"]))


if __name__ == "__main__":
    unittest.main()

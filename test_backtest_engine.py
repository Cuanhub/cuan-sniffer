"""The backtest engine resolves every ambiguity against the trade.

Each test here pins one rule from engine.py's docstring. The synthetic bars
in this file are hand-built so the right answer is known in advance -- they
test the MACHINERY, and no result about the market comes from them. The
properties that matter on real data (ATR agreement, selection agreement,
no lookahead) are tested on the recorded real bars in tests/fixtures.
"""

import json
import unittest
from pathlib import Path

from research.backtest import band
from research.backtest.costs import FRICTIONLESS, MEASURED, Costs
from research.backtest.engine import (LIMIT, MARKET_CLOSE, MARKET_NEXT_OPEN,
                                      Order, atr_pct_series, simulate)
from research.levels import zone_pass
from research.levels import zones as Z

REAL = json.loads((Path(__file__).parent / "tests/fixtures/zones/bars_1h.json").read_text())


def bar(o, h, l, c, t=0):
    return {"t": t, "o": o, "h": h, "l": l, "c": c}


class MarketCloseEntriesStartAfterTheClose(unittest.TestCase):
    """The decision bar's high and low happened BEFORE its close. The old
    scratch harness let them resolve a trade entered at that close."""

    def test_the_decision_bar_s_own_low_cannot_stop_the_trade(self):
        bars = [bar(100, 101, 90, 100),      # dipped through 95 before the close
                bar(100, 111, 99, 110)]      # then the target
        tr = simulate(bars, Order("X", 0, MARKET_CLOSE, stop=95, tp=110), FRICTIONLESS)
        self.assertEqual(tr.outcome, "tp")
        self.assertEqual(tr.entry_i, 0)

    def test_the_decision_bar_s_own_high_cannot_win_the_trade(self):
        bars = [bar(100, 120, 99, 100),      # spiked through 110 before the close
                bar(100, 101, 94, 95)]
        tr = simulate(bars, Order("X", 0, MARKET_CLOSE, stop=95, tp=110), FRICTIONLESS)
        self.assertEqual(tr.outcome, "stop")


class LimitFills(unittest.TestCase):
    def test_fills_at_the_limit_when_the_low_reaches_it(self):
        bars = [bar(105, 106, 104, 105), bar(105, 105, 99, 101), bar(101, 111, 100, 110)]
        tr = simulate(bars, Order("X", 0, LIMIT, stop=95, tp=110, limit=100), FRICTIONLESS)
        self.assertEqual((tr.entry_i, tr.entry_px), (1, 100))
        self.assertEqual(tr.outcome, "tp")

    def test_the_target_on_the_fill_bar_is_not_counted(self):
        """The high may have printed before the dip that filled the bid."""
        bars = [bar(105, 106, 104, 105),
                bar(105, 111, 99, 104),      # reaches 110 AND fills at 100
                bar(104, 105, 94, 95)]       # then stops out
        tr = simulate(bars, Order("X", 0, LIMIT, stop=95, tp=110, limit=100), FRICTIONLESS)
        self.assertEqual(tr.outcome, "stop")
        self.assertTrue(tr.fill_bar_tp)

    def test_the_stop_on_the_fill_bar_does_count(self):
        """Price reached the bid on its way down: a low under the stop came
        after the fill."""
        bars = [bar(105, 106, 104, 105), bar(105, 105, 93, 96)]
        tr = simulate(bars, Order("X", 0, LIMIT, stop=95, tp=110, limit=100), FRICTIONLESS)
        self.assertEqual((tr.outcome, tr.exit_i), ("stop", 1))

    def test_a_bid_that_never_fills_is_no_trade(self):
        bars = [bar(105, 106, 104, 105)] * 20
        self.assertIsNone(simulate(bars, Order("X", 0, LIMIT, 95, 110, limit=100, ttl=12)))

    def test_the_bid_expires_after_its_ttl(self):
        bars = [bar(105, 106, 104, 105)] * 13 + [bar(105, 105, 99, 101)]
        self.assertIsNone(simulate(bars, Order("X", 0, LIMIT, 95, 110, limit=100, ttl=12)))
        bars = [bar(105, 106, 104, 105)] * 12 + [bar(105, 105, 99, 101)]
        self.assertIsNotNone(simulate(bars, Order("X", 0, LIMIT, 95, 110, limit=100, ttl=12)))


class Exits(unittest.TestCase):
    def test_a_bar_reaching_both_is_a_loss_and_flagged(self):
        bars = [bar(100, 100, 100, 100), bar(100, 111, 94, 100)]
        tr = simulate(bars, Order("X", 0, MARKET_CLOSE, 95, 110), FRICTIONLESS)
        self.assertEqual(tr.outcome, "stop")
        self.assertTrue(tr.ambiguous)

    def test_a_gap_through_the_stop_exits_at_the_open(self):
        bars = [bar(100, 100, 100, 100), bar(90, 92, 88, 91)]
        tr = simulate(bars, Order("X", 0, MARKET_CLOSE, 95, 110), FRICTIONLESS)
        self.assertEqual(tr.exit_px, 90)
        self.assertAlmostEqual(tr.r_gross, -2.0)      # worse than -1R, as it was

    def test_a_trade_still_open_at_the_end_is_marked_open(self):
        bars = [bar(100, 100, 100, 100), bar(100, 105, 98, 103)]
        tr = simulate(bars, Order("X", 0, MARKET_CLOSE, 95, 110), FRICTIONLESS)
        self.assertEqual(tr.outcome, "open")
        self.assertIsNone(tr.exit_i)
        self.assertFalse(tr.resolved)

    def test_an_entry_outside_stop_and_target_is_no_trade(self):
        bars = [bar(100, 100, 100, 100), bar(112, 113, 111, 112)]
        self.assertIsNone(simulate(bars, Order("X", 0, MARKET_NEXT_OPEN, 95, 110)))


class Costs_(unittest.TestCase):
    def test_fees_and_slippage_by_hand(self):
        c = Costs(maker_fee_pct=0.02, taker_fee_pct=0.05, stop_slip_pct=0.1,
                  tp_slip_pct=0.0, market_entry_slip_pct=0.0)
        bars = [bar(105, 106, 104, 105), bar(105, 105, 99, 101), bar(101, 101, 94, 95)]
        tr = simulate(bars, Order("X", 0, LIMIT, 95, 110, limit=100), c)
        exit_px = 95 * (1 - 0.001)
        self.assertAlmostEqual(tr.exit_px, exit_px)
        self.assertAlmostEqual(tr.r_gross, (exit_px - 100) / 5)
        self.assertAlmostEqual(tr.cost_r, (0.0002 * 100 + 0.0005 * exit_px) / 5)
        self.assertAlmostEqual(tr.r_net, tr.r_gross - tr.cost_r)

    def test_the_measured_defaults_are_the_measured_numbers(self):
        """Pinned so a casual edit cannot turn a measurement into a guess."""
        self.assertEqual(MEASURED.maker_fee_pct, 0.0144)
        self.assertEqual(MEASURED.taker_fee_pct, 0.0432)
        self.assertEqual(MEASURED.stop_slip_pct, 0.1193)
        self.assertEqual(MEASURED.tp_slip_pct, 0.0)

    def test_a_control_can_be_charged_the_entry_fee_of_its_trade(self):
        bars = [bar(100, 100, 100, 100), bar(100, 111, 99, 110)]
        maker = simulate(bars, Order("X", 0, MARKET_CLOSE, 95, 110, entry_fee="maker"))
        taker = simulate(bars, Order("X", 0, MARKET_CLOSE, 95, 110))
        self.assertLess(maker.cost_r, taker.cost_r)


class AgreesWithProduction(unittest.TestCase):
    def test_atr_series_is_zones_atr_on_real_bars(self):
        for coin in ("ETH", "FARTCOIN", "BNB"):
            bars = REAL[coin]
            series = atr_pct_series(bars)
            for t in (1, 2, 13, 14, 15, 200, 551, len(bars) - 1):
                self.assertAlmostEqual(series[t], Z.atr_pct(bars[:t + 1]), places=10,
                                       msg=f"{coin} t={t}")

    def test_bid_ttl_is_production_s(self):
        from core import protect
        self.assertEqual(band.BID_TTL_BARS, protect.BID_TTL_HOURS)

    def test_the_window_is_production_s(self):
        self.assertEqual(band.WINDOW, zone_pass.ZONE_PASS_1H_HOURS)

    def test_order_at_makes_the_live_pass_s_choice(self):
        """Feed the same real bars to the live pass's own function through a
        client that serves them, and require the identical order -- at bars
        where the strategy actually decides something, or the comparison is
        None == None and proves nothing (which is what the first version of
        this test did: no coin had a tradeable band at the final bar)."""
        compared = nones = 0
        for coin in ("ETH", "BNB", "HYPE", "NEAR", "AAVE", "WLD", "SUI"):
            stamped = [dict(b, t=1_700_000_000_000 + k * 3_600_000)
                       for k, b in enumerate(REAL[coin])]
            for i in range(800, len(stamped), 9):
                bars = stamped[:i + 1]

                class Recorded:
                    def get_candles(self, c, interval, start, end, _b=bars):
                        return [{k: str(v) for k, v in b.items()} for b in _b]

                intents, _, _ = zone_pass.plans_by_coin(
                    Recorded(), [{"symbol": coin}], min_rr=2.0,
                    mids={coin: bars[-1]["c"]})
                o = band.order_at(coin, bars, i, min_rr=2.0)
                if not intents:
                    self.assertIsNone(o, f"{coin} i={i}")
                    nones += 1
                    continue
                ep = intents[0]
                self.assertIsNotNone(o, f"{coin} i={i}")
                self.assertAlmostEqual(o.stop, ep["suggested_stop"], places=9)
                self.assertAlmostEqual(o.tp, ep["suggested_tp"], places=9)
                self.assertEqual(o.kind == MARKET_CLOSE, bool(ep["zone_immediate"]))
                if o.kind == LIMIT:
                    self.assertAlmostEqual(o.limit, ep["suggested_entry"], places=9)
                compared += 1
        self.assertGreater(compared, 10, "too few real decisions compared")
        self.assertGreater(nones, 10, "never compared a refusal")

    def test_first_tradeable_matches_the_old_loop(self):
        """plans_by_coin's selection was refactored into first_tradeable;
        the refactor must choose exactly what the loop chose."""
        for coin, bars in REAL.items():
            plans = Z.plans_for(bars, bars[-1]["c"], coin=coin)
            taken = None
            for p in plans:
                if zone_pass.rejection(p, 2.0) is None:
                    taken = p
                    break
            self.assertIs(zone_pass.first_tradeable(plans, 2.0), taken, coin)


class NoLookahead(unittest.TestCase):
    def test_a_decision_is_the_same_with_the_future_removed(self):
        """order_at(bars, i) must not change when bars after i are deleted."""
        checked = 0
        for coin in ("ETH", "HYPE", "WLD", "SUI"):
            bars = REAL[coin]
            for i in range(800, len(bars) - 1, 7):
                full = band.order_at(coin, bars, i)
                cut = band.order_at(coin, bars[:i + 1], i)
                self.assertEqual(full, cut, f"{coin} i={i}")
                checked += full is not None
        self.assertGreater(checked, 0, "no decision was ever made -- vacuous")

    def test_the_replay_takes_one_position_per_coin_at_a_time(self):
        trades = band.run("ETH", REAL["ETH"])
        for a, b in zip(trades, trades[1:]):
            self.assertIsNotNone(a.exit_i)
            self.assertGreaterEqual(b.decided_i, a.exit_i)


if __name__ == "__main__":
    unittest.main()

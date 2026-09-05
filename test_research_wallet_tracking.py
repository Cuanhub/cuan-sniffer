"""
Tests for research/wallet_tracking's trade reconstruction -- fully
synthetic, no network calls, since getting the FIFO matching and
buy/sell classification right is the part that must not be wrong before
trusting it against real on-chain data.
"""

import unittest

from research.wallet_tracking.reconstruction import (
    build_trade_events, fifo_match_trades, aggregate_wallet_stats, WETH_ADDRESS,
)

WALLET = "0xWALLET000000000000000000000000000000001"
TOKEN_A = "0xTOKENA00000000000000000000000000000001"


def _transfer(tx_hash, token_addr, symbol, decimals, amount, from_addr, to_addr, ts):
    return {
        "hash": tx_hash, "contractAddress": token_addr, "tokenSymbol": symbol,
        "tokenDecimal": str(decimals), "value": str(int(amount * (10 ** decimals))),
        "from": from_addr, "to": to_addr, "timeStamp": str(ts),
    }


def _normal_tx(tx_hash, from_addr, to_addr, eth_amount, ts):
    return {"hash": tx_hash, "from": from_addr, "to": to_addr,
            "value": str(int(eth_amount * 1e18)), "timeStamp": str(ts)}


ROUTER = "0xROUTER0000000000000000000000000000001"


class TestBuildTradeEvents(unittest.TestCase):

    def test_eth_buy_classified_correctly(self):
        # wallet sends 1 ETH, receives 1000 TOKEN_A
        transfers = [_transfer("0xtx1", TOKEN_A, "TKA", 18, 1000, ROUTER, WALLET, 100)]
        normal = [_normal_tx("0xtx1", WALLET, ROUTER, 1.0, 100)]
        events, skipped = build_trade_events(transfers, normal, WALLET)
        self.assertEqual(len(skipped), 0)
        self.assertEqual(len(events), 1)
        ev = events[0]
        self.assertEqual(ev.side, "buy")
        self.assertAlmostEqual(ev.token_amount, 1000)
        self.assertAlmostEqual(ev.quote_amount, 1.0)
        self.assertEqual(ev.quote_asset, "ETH")

    def test_eth_sell_classified_correctly(self):
        transfers = [_transfer("0xtx2", TOKEN_A, "TKA", 18, 1000, WALLET, ROUTER, 200)]
        normal = [_normal_tx("0xtx2", ROUTER, WALLET, 1.5, 200)]
        events, skipped = build_trade_events(transfers, normal, WALLET)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].side, "sell")
        self.assertAlmostEqual(events[0].quote_amount, 1.5)

    def test_weth_buy_classified_correctly(self):
        # wallet sends WETH, receives TOKEN_A -- both as token transfers, no native ETH
        transfers = [
            _transfer("0xtx3", WETH_ADDRESS, "WETH", 18, 2.0, WALLET, ROUTER, 300),
            _transfer("0xtx3", TOKEN_A, "TKA", 18, 500, ROUTER, WALLET, 300),
        ]
        events, skipped = build_trade_events(transfers, [], WALLET)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].side, "buy")
        self.assertEqual(events[0].quote_asset, "WETH")
        self.assertAlmostEqual(events[0].quote_amount, 2.0)

    def test_token_to_token_swap_is_skipped(self):
        TOKEN_B = "0xTOKENB00000000000000000000000000000002"
        transfers = [
            _transfer("0xtx4", TOKEN_A, "TKA", 18, 100, WALLET, ROUTER, 400),
            _transfer("0xtx4", TOKEN_B, "TKB", 18, 50, ROUTER, WALLET, 400),
        ]
        events, skipped = build_trade_events(transfers, [], WALLET)
        self.assertEqual(len(events), 0)
        self.assertEqual(len(skipped), 1)
        self.assertIn("non-quote tokens", skipped[0].reason)

    def test_plain_transfer_no_quote_counterpart_is_skipped(self):
        transfers = [_transfer("0xtx5", TOKEN_A, "TKA", 18, 100, ROUTER, WALLET, 500)]
        events, skipped = build_trade_events(transfers, [], WALLET)
        self.assertEqual(len(events), 0)
        self.assertEqual(len(skipped), 1)

    def test_sell_via_internal_eth_transfer_classified_correctly(self):
        # Real pattern verified 2026-09-04: router unwraps WETH and
        # forwards native ETH to the wallet via an INTERNAL call, not the
        # top-level tx value. Without internal_txs, this tx has zero
        # quote-asset delta and gets skipped -- exactly the bug that
        # produced "331 buys, 0 sells" against real data before this fix.
        transfers = [_transfer("0xtx7", TOKEN_A, "TKA", 18, 1000, WALLET, ROUTER, 700)]
        normal = [_normal_tx("0xtx7", WALLET, ROUTER, 0.0, 700)]  # no top-level value
        internal = [{"hash": "0xtx7", "from": ROUTER, "to": WALLET,
                     "value": str(int(0.52 * 1e18)), "isError": "0"}]
        events, skipped = build_trade_events(transfers, normal, WALLET, internal_txs=internal)
        self.assertEqual(len(skipped), 0)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].side, "sell")
        self.assertAlmostEqual(events[0].quote_amount, 0.52)

    def test_failed_internal_tx_ignored(self):
        transfers = [_transfer("0xtx8", TOKEN_A, "TKA", 18, 1000, WALLET, ROUTER, 800)]
        internal = [{"hash": "0xtx8", "from": ROUTER, "to": WALLET,
                     "value": str(int(1.0 * 1e18)), "isError": "1"}]
        events, skipped = build_trade_events(transfers, [], WALLET, internal_txs=internal)
        self.assertEqual(len(events), 0)  # errored internal call must not count as real value

    def test_unrelated_transfer_in_same_tx_ignored(self):
        # a transfer not touching WALLET at all should not pollute the tx grouping
        transfers = [
            _transfer("0xtx6", TOKEN_A, "TKA", 18, 1000, ROUTER, WALLET, 600),
            _transfer("0xtx6", TOKEN_A, "TKA", 18, 5, "0xSOMEONE", "0xELSE", 600),
        ]
        normal = [_normal_tx("0xtx6", WALLET, ROUTER, 1.0, 600)]
        events, skipped = build_trade_events(transfers, normal, WALLET)
        self.assertEqual(len(events), 1)
        self.assertAlmostEqual(events[0].token_amount, 1000)


class TestFifoMatching(unittest.TestCase):

    def test_simple_round_trip_pnl(self):
        transfers = [
            _transfer("0xb1", TOKEN_A, "TKA", 18, 1000, ROUTER, WALLET, 100),
            _transfer("0xs1", TOKEN_A, "TKA", 18, 1000, WALLET, ROUTER, 200),
        ]
        normal = [
            _normal_tx("0xb1", WALLET, ROUTER, 1.0, 100),
            _normal_tx("0xs1", ROUTER, WALLET, 2.0, 200),
        ]
        events, _ = build_trade_events(transfers, normal, WALLET)
        closed = fifo_match_trades(events)
        self.assertEqual(len(closed), 1)
        t = closed[0]
        self.assertAlmostEqual(t.cost_basis, 1.0)
        self.assertAlmostEqual(t.proceeds, 2.0)
        self.assertAlmostEqual(t.pnl, 1.0)
        self.assertAlmostEqual(t.pnl_pct, 100.0)
        self.assertEqual(t.hold_seconds, 100)

    def test_partial_sell_leaves_remaining_lot_open(self):
        transfers = [
            _transfer("0xb1", TOKEN_A, "TKA", 18, 1000, ROUTER, WALLET, 100),
            _transfer("0xs1", TOKEN_A, "TKA", 18, 400, WALLET, ROUTER, 200),
        ]
        normal = [
            _normal_tx("0xb1", WALLET, ROUTER, 1.0, 100),
            _normal_tx("0xs1", ROUTER, WALLET, 0.6, 200),
        ]
        events, _ = build_trade_events(transfers, normal, WALLET)
        closed = fifo_match_trades(events)
        self.assertEqual(len(closed), 1)
        self.assertAlmostEqual(closed[0].amount, 400)
        self.assertAlmostEqual(closed[0].cost_basis, 0.4)  # 400/1000 of the 1.0 ETH cost

    def test_sell_spans_multiple_buy_lots_fifo_order(self):
        transfers = [
            _transfer("0xb1", TOKEN_A, "TKA", 18, 500, ROUTER, WALLET, 100),
            _transfer("0xb2", TOKEN_A, "TKA", 18, 500, ROUTER, WALLET, 150),
            _transfer("0xs1", TOKEN_A, "TKA", 18, 700, WALLET, ROUTER, 200),
        ]
        normal = [
            _normal_tx("0xb1", WALLET, ROUTER, 1.0, 100),   # 0.002 ETH/token
            _normal_tx("0xb2", WALLET, ROUTER, 2.0, 150),   # 0.004 ETH/token
            _normal_tx("0xs1", ROUTER, WALLET, 2.1, 200),   # 0.003 ETH/token proceeds
        ]
        events, _ = build_trade_events(transfers, normal, WALLET)
        closed = fifo_match_trades(events)
        # 700 sold: first 500 from lot1 (cost 1.0), then 200 from lot2 (cost 2.0*200/500=0.8)
        self.assertEqual(len(closed), 2)
        self.assertAlmostEqual(closed[0].amount, 500)
        self.assertAlmostEqual(closed[0].cost_basis, 1.0)
        self.assertAlmostEqual(closed[0].buy_timestamp, 100)
        self.assertAlmostEqual(closed[1].amount, 200)
        self.assertAlmostEqual(closed[1].cost_basis, 0.8)
        self.assertAlmostEqual(closed[1].buy_timestamp, 150)

    def test_sell_with_no_prior_buy_produces_no_closed_trade(self):
        # e.g. an airdropped token sold later -- no real cost basis on record
        transfers = [_transfer("0xs1", TOKEN_A, "TKA", 18, 100, WALLET, ROUTER, 200)]
        normal = [_normal_tx("0xs1", ROUTER, WALLET, 0.5, 200)]
        events, _ = build_trade_events(transfers, normal, WALLET)
        closed = fifo_match_trades(events)
        self.assertEqual(len(closed), 0)


class TestAggregateStats(unittest.TestCase):

    def test_empty_input(self):
        self.assertEqual(aggregate_wallet_stats([]), {"n_closed_trades": 0})

    def test_win_rate_and_profit_factor(self):
        transfers = [
            _transfer("0xb1", TOKEN_A, "TKA", 18, 1000, ROUTER, WALLET, 100),
            _transfer("0xs1", TOKEN_A, "TKA", 18, 1000, WALLET, ROUTER, 150),
            _transfer("0xb2", TOKEN_A, "TKA", 18, 1000, ROUTER, WALLET, 200),
            _transfer("0xs2", TOKEN_A, "TKA", 18, 1000, WALLET, ROUTER, 250),
        ]
        normal = [
            _normal_tx("0xb1", WALLET, ROUTER, 1.0, 100),
            _normal_tx("0xs1", ROUTER, WALLET, 2.0, 150),  # win: +1.0
            _normal_tx("0xb2", WALLET, ROUTER, 1.0, 200),
            _normal_tx("0xs2", ROUTER, WALLET, 0.5, 250),  # loss: -0.5
        ]
        events, _ = build_trade_events(transfers, normal, WALLET)
        closed = fifo_match_trades(events)
        stats = aggregate_wallet_stats(closed)
        self.assertEqual(stats["n_closed_trades"], 2)
        self.assertAlmostEqual(stats["win_rate_pct"], 50.0)
        self.assertAlmostEqual(stats["total_pnl_quote"], 0.5)
        self.assertAlmostEqual(stats["profit_factor"], 2.0)  # 1.0 gross win / 0.5 gross loss


if __name__ == "__main__":
    unittest.main()

"""
CLI/entry point: pull a wallet's real on-chain history and report
reconstructed trade stats. Collection + analysis only -- this produces a
descriptive report, not a recommendation or a signal.

Usage:
    python -m research.wallet_tracking.analyze 0xADDRESS...
    python -m research.wallet_tracking.analyze 0xADDR1 0xADDR2 --max-rows 5000
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Only agent.py's own startup loads .env automatically (via config.py) --
# an ad-hoc/research entry point run standalone must load it explicitly
# or ETHERSCAN_API_KEY silently isn't there. Same gotcha hit earlier this
# session with other research scripts.
load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)

from research.wallet_tracking.etherscan_client import EtherscanReadOnlyClient
from research.wallet_tracking.reconstruction import build_trade_events, fifo_match_trades, aggregate_wallet_stats


def analyze_wallet(client: EtherscanReadOnlyClient, address: str, max_rows: Optional[int] = None) -> dict:
    print(f"[{address}] fetching token transfers...", flush=True)
    token_transfers = client.fetch_all_token_transfers(address, max_rows=max_rows)
    print(f"[{address}] {len(token_transfers)} token transfers fetched", flush=True)

    print(f"[{address}] fetching normal transactions...", flush=True)
    normal_txs = client.fetch_all_normal_txs(address, max_rows=max_rows)
    print(f"[{address}] {len(normal_txs)} normal txs fetched", flush=True)

    print(f"[{address}] fetching internal transactions...", flush=True)
    internal_txs = client.fetch_all_internal_txs(address, max_rows=max_rows)
    print(f"[{address}] {len(internal_txs)} internal txs fetched", flush=True)

    events, skipped = build_trade_events(token_transfers, normal_txs, address, internal_txs=internal_txs)
    closed = fifo_match_trades(events)
    stats = aggregate_wallet_stats(closed)

    return {
        "address": address, "n_token_transfers": len(token_transfers),
        "n_normal_txs": len(normal_txs), "n_trade_events": len(events),
        "n_skipped_txs": len(skipped), "skip_reasons_sample": [s.reason for s in skipped[:5]],
        "stats": stats,
    }


def print_report(result: dict) -> None:
    print("\n" + "=" * 78)
    print(f"WALLET TRADE RECONSTRUCTION — {result['address']}")
    print("=" * 78)
    print(f"  token transfers fetched: {result['n_token_transfers']}")
    print(f"  normal txs fetched:      {result['n_normal_txs']}")
    print(f"  classified trade events: {result['n_trade_events']}")
    print(f"  skipped (ambiguous) txs: {result['n_skipped_txs']}")
    if result["skip_reasons_sample"]:
        print(f"    sample skip reasons: {result['skip_reasons_sample']}")

    stats = result["stats"]
    if stats.get("n_closed_trades", 0) == 0:
        print("\n  no closed (FIFO-matched) trades found")
        return

    print(f"\n  closed trades:     {stats['n_closed_trades']}")
    print(f"  win rate:          {stats['win_rate_pct']:.1f}%")
    print(f"  total PnL (quote): {stats['total_pnl_quote']:+.4f}")
    print(f"  total cost basis:  {stats['total_cost_basis_quote']:.4f}")
    print(f"  ROI:               {stats['roi_pct']:+.1f}%")
    print(f"  profit factor:     {stats['profit_factor']:.3f}")
    print(f"  avg hold time:     {stats['avg_hold_hours']:.1f}h")
    print(f"\n  PnL by token (top 10):")
    for i, (sym, pnl) in enumerate(stats["pnl_by_token"].items()):
        if i >= 10:
            break
        print(f"    {sym:12s} {pnl:+.4f}")


def main():
    parser = argparse.ArgumentParser(description="Reconstruct trades for one or more wallets from on-chain history")
    parser.add_argument("addresses", nargs="+")
    parser.add_argument("--max-rows", type=int, default=None, help="cap rows fetched per endpoint (for a quick first look)")
    args = parser.parse_args()

    client = EtherscanReadOnlyClient()
    for addr in args.addresses:
        result = analyze_wallet(client, addr, max_rows=args.max_rows)
        print_report(result)


if __name__ == "__main__":
    sys.exit(main())

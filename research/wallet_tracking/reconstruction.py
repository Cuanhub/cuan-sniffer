"""
Trade reconstruction from raw on-chain transfers.

Core idea: group all ERC20 transfers + native ETH movements by transaction
hash. A DEX swap is one transaction where a wallet's balance of some
"quote" asset (ETH, WETH, or a major stablecoin) moves one way and a
non-quote token moves the other way -- that's a BUY (quote out, token in)
or SELL (token out, quote in). Ambiguous transactions (multiple non-quote
tokens touched, or no quote counterpart at all -- e.g. a plain airdrop or
a token-to-token swap not routed through a quote asset) are explicitly
flagged as skipped, never guessed at.

Realized PnL is computed FIFO per token: each BUY adds a lot (amount,
quote cost, timestamp); each SELL consumes lots oldest-first and realizes
PnL on the matched portion. All PnL is denominated in the quote asset
(ETH-equivalent, treating WETH 1:1 with ETH and stablecoins as their own
quote unit) -- USD conversion is a separate, later step, not done here.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

WETH_ADDRESS = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"
QUOTE_TOKEN_ADDRESSES = {
    WETH_ADDRESS: "WETH",
    "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48": "USDC",
    "0xdac17f958d2ee523a2206206994597c13d831ec7": "USDT",
    "0x6b175474e89094c44da98b954eedeac495271d0f": "DAI",
}


@dataclass
class TradeEvent:
    tx_hash: str
    timestamp: int
    side: str          # "buy" or "sell" (from the wallet's perspective)
    token_address: str
    token_symbol: str
    token_amount: float
    quote_amount: float   # ETH-equivalent value paid (buy) or received (sell)
    quote_asset: str      # "ETH", "WETH", "USDC", etc.


@dataclass
class SkippedTx:
    tx_hash: str
    reason: str


@dataclass
class Lot:
    amount: float
    quote_cost: float  # total quote paid for this lot
    timestamp: int


@dataclass
class ClosedTrade:
    token_address: str
    token_symbol: str
    buy_timestamp: int
    sell_timestamp: int
    amount: float
    cost_basis: float
    proceeds: float
    pnl: float
    pnl_pct: float
    hold_seconds: int


def _norm(addr: str) -> str:
    return (addr or "").lower()


def build_trade_events(
    token_transfers: List[Dict[str, Any]],
    normal_txs: List[Dict[str, Any]],
    wallet: str,
    internal_txs: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[TradeEvent], List[SkippedTx]]:
    """internal_txs matters, not optional in practice: many DEX routers
    unwrap WETH and forward native ETH to the trader via an internal call
    rather than the top-level tx value -- verified directly on a real
    sell (WETH contract -> router -> wallet, both internal calls, zero
    top-level value). Omitting it silently misclassifies real sells as
    having no quote-asset counterpart."""
    wallet = _norm(wallet)

    eth_value_by_tx: Dict[str, float] = {}
    for tx in normal_txs:
        if _norm(tx.get("from")) == wallet or _norm(tx.get("to")) == wallet:
            val = int(tx.get("value", 0) or 0) / 1e18
            if val > 0:
                sign = -1 if _norm(tx.get("from")) == wallet else 1
                eth_value_by_tx[tx["hash"]] = eth_value_by_tx.get(tx["hash"], 0.0) + sign * val
    for tx in (internal_txs or []):
        if _norm(tx.get("isError", "0")) != "0":
            continue  # failed internal call moved no real value
        if _norm(tx.get("from")) == wallet or _norm(tx.get("to")) == wallet:
            val = int(tx.get("value", 0) or 0) / 1e18
            if val > 0:
                sign = -1 if _norm(tx.get("from")) == wallet else 1
                eth_value_by_tx[tx["hash"]] = eth_value_by_tx.get(tx["hash"], 0.0) + sign * val

    by_tx: Dict[str, List[Dict[str, Any]]] = {}
    for t in token_transfers:
        by_tx.setdefault(t["hash"], []).append(t)

    events: List[TradeEvent] = []
    skipped: List[SkippedTx] = []

    for tx_hash, transfers in by_tx.items():
        quote_delta: Dict[str, float] = {}   # quote_symbol -> net amount (signed, wallet perspective)
        token_delta: Dict[Tuple[str, str], float] = {}  # (address, symbol) -> net amount

        for t in transfers:
            addr = _norm(t.get("contractAddress", ""))
            decimals = int(t.get("tokenDecimal", 18) or 18)
            amount = int(t.get("value", 0) or 0) / (10 ** decimals)
            is_in = _norm(t.get("to")) == wallet
            is_out = _norm(t.get("from")) == wallet
            if not (is_in or is_out):
                continue
            signed = amount if is_in else -amount

            if addr in QUOTE_TOKEN_ADDRESSES:
                sym = QUOTE_TOKEN_ADDRESSES[addr]
                quote_delta[sym] = quote_delta.get(sym, 0.0) + signed
            else:
                key = (addr, t.get("tokenSymbol", "UNKNOWN"))
                token_delta[key] = token_delta.get(key, 0.0) + signed

        eth_delta = eth_value_by_tx.get(tx_hash, 0.0)
        if eth_delta != 0:
            quote_delta["ETH"] = quote_delta.get("ETH", 0.0) + eth_delta

        # Drop near-zero noise (dust / rounding)
        token_delta = {k: v for k, v in token_delta.items() if abs(v) > 1e-12}
        quote_delta = {k: v for k, v in quote_delta.items() if abs(v) > 1e-12}

        if len(token_delta) != 1:
            if token_delta:
                skipped.append(SkippedTx(tx_hash, f"{len(token_delta)} distinct non-quote tokens touched"))
            continue
        if len(quote_delta) != 1:
            skipped.append(SkippedTx(tx_hash, f"{len(quote_delta)} distinct quote assets touched (expected exactly 1)"))
            continue

        (token_addr, token_sym), token_amt = next(iter(token_delta.items()))
        quote_sym, quote_amt = next(iter(quote_delta.items()))

        if token_amt > 0 and quote_amt < 0:
            side = "buy"
        elif token_amt < 0 and quote_amt > 0:
            side = "sell"
        else:
            skipped.append(SkippedTx(tx_hash, "token/quote deltas same sign -- not a simple swap"))
            continue

        timestamp = int(transfers[0].get("timeStamp", 0))
        events.append(TradeEvent(
            tx_hash=tx_hash, timestamp=timestamp, side=side,
            token_address=token_addr, token_symbol=token_sym,
            token_amount=abs(token_amt), quote_amount=abs(quote_amt), quote_asset=quote_sym,
        ))

    events.sort(key=lambda e: e.timestamp)
    return events, skipped


def fifo_match_trades(events: List[TradeEvent]) -> List[ClosedTrade]:
    open_lots: Dict[str, List[Lot]] = {}
    closed: List[ClosedTrade] = []

    for ev in events:
        lots = open_lots.setdefault(ev.token_address, [])
        if ev.side == "buy":
            lots.append(Lot(amount=ev.token_amount, quote_cost=ev.quote_amount, timestamp=ev.timestamp))
            continue

        # sell: consume oldest lots first
        remaining_to_sell = ev.token_amount
        proceeds_per_unit = ev.quote_amount / ev.token_amount if ev.token_amount > 0 else 0.0
        while remaining_to_sell > 1e-12 and lots:
            lot = lots[0]
            matched = min(lot.amount, remaining_to_sell)
            cost_per_unit = lot.quote_cost / lot.amount if lot.amount > 0 else 0.0
            cost_basis = matched * cost_per_unit
            proceeds = matched * proceeds_per_unit
            pnl = proceeds - cost_basis
            closed.append(ClosedTrade(
                token_address=ev.token_address, token_symbol=ev.token_symbol,
                buy_timestamp=lot.timestamp, sell_timestamp=ev.timestamp,
                amount=matched, cost_basis=cost_basis, proceeds=proceeds, pnl=pnl,
                pnl_pct=(pnl / cost_basis * 100.0) if cost_basis > 0 else 0.0,
                hold_seconds=ev.timestamp - lot.timestamp,
            ))
            lot.amount -= matched
            lot.quote_cost -= cost_basis
            remaining_to_sell -= matched
            if lot.amount <= 1e-12:
                lots.pop(0)
        # remaining_to_sell > 0 here means a sell with no matching buy on
        # record (e.g. an airdropped/transferred-in token sold later) --
        # not counted as a closed trade since there's no real cost basis.

    return closed


def aggregate_wallet_stats(closed_trades: List[ClosedTrade]) -> Dict[str, Any]:
    if not closed_trades:
        return {"n_closed_trades": 0}
    wins = [t for t in closed_trades if t.pnl > 0]
    total_pnl = sum(t.pnl for t in closed_trades)
    total_cost = sum(t.cost_basis for t in closed_trades)
    gross_win = sum(t.pnl for t in wins)
    gross_loss = abs(sum(t.pnl for t in closed_trades if t.pnl <= 0))
    by_token: Dict[str, float] = {}
    for t in closed_trades:
        by_token[t.token_symbol] = by_token.get(t.token_symbol, 0.0) + t.pnl

    return {
        "n_closed_trades": len(closed_trades),
        "win_rate_pct": len(wins) / len(closed_trades) * 100.0,
        "total_pnl_quote": total_pnl,
        "total_cost_basis_quote": total_cost,
        "roi_pct": (total_pnl / total_cost * 100.0) if total_cost > 0 else 0.0,
        "profit_factor": (gross_win / gross_loss) if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0),
        "avg_hold_hours": sum(t.hold_seconds for t in closed_trades) / len(closed_trades) / 3600.0,
        "pnl_by_token": dict(sorted(by_token.items(), key=lambda kv: -kv[1])),
    }

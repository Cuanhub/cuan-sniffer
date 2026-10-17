# Fixture provenance

Recorded by `tests/record_fixtures.py` (read-only `/info` POSTs) from MAINNET on 2026-10-17.
Account address replaced with a placeholder; `cloid` dropped. Sizes, prices, statuses, field names and nesting are verbatim.

| file | source |
|---|---|
| `order_status_by_inner_status.json` | `historicalOrders`, one record per distinct inner status |
| `order_status_envelopes.json` | `orderStatus` by oid, one per inner status |
| `clearinghouse_state_flat.json` | `clearinghouseState` (account was flat) |
| `clearinghouse_state_with_position.json` | **Official documented example**, not a recording -- this account held no position at record time. Source: Hyperliquid docs, info endpoint / perpetuals. |
| `spot_user_state.json` | `spotClearinghouseState` |
| `frontend_open_orders.json` | `frontendOpenOrders` |
| `user_fills_30d.json` | `userFillsByTime`, last 30 days |
| `meta.json` | `meta` (perp universe) |
| `abstraction_state.json` | `userAbstraction` |
| `frontend_open_orders_protected.json` | Derived from the REAL trigger order records above (`Stop Market` + `Take Profit Market`), re-pointed to one coin and size. `frontendOpenOrders` returns the same frontend-order object; the live call returned `[]` because the account was flat. |
| `equity_hold_series.txt` | 265 `(spot USDC total, hold)` readings scraped from the live trader's own log (`~/Library/Logs/cuan-sniffer/trader.log`), oldest first, one per `get_margin_summary` call. Used to establish that `total` marks to market: across consecutive readings with an open position, the median of Δtotal/Δhold is exactly 5.0000, the flat leverage this system trades at, so Δtotal equals the change in position notional. |

## Identifier scrubbing

Beyond the address and `cloid`, these are replaced because the fixtures live
in a PUBLIC repository:

- **`hash`** (fills) — zeroed. A Hyperliquid transaction hash is public
  on-chain data, and anyone holding one can look up the wallet that made it.
  Leaving them would have defeated the address scrubbing entirely. No test
  reads this field.
- **`oid`, `tid`** — renumbered sequentially. `tid` is renumbered in
  timestamp order, which is the only property anything depends on (it breaks
  ties in `core.streak.trades_from_fills`).

Sizes, prices, `closedPnl`, `fee`, `startPosition`, statuses, field names and
nesting are still verbatim.

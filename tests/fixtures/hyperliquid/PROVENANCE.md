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

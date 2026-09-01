# Funding Collector — Data Schema

One CSV per symbol per UTC day, under `research/funding/data/`, named
`{SYMBOL}_{YYYY-MM-DD}.csv`. Append-only; a restart re-running the
collector will not duplicate a row for a timestamp already on disk.

| Column | Type | Source | Notes |
|---|---|---|---|
| `timestamp_utc` | ISO 8601 string | collector clock | `YYYY-MM-DDTHH:MM:SSZ`, one row per symbol per collection cycle (~5min) |
| `symbol` | string | venue universe | e.g. `SOL` |
| `funding_rate` | float | `metaAndAssetCtxs` ctx `funding` | Live, continuously-updated rate — **not** the settled value; sampled every cycle |
| `realized_funding_rate` | float or empty | `fundingHistory` | Only populated on the cycle nearest each hourly settlement boundary; empty on other rows. This is ground truth for what actually paid out, distinct from `funding_rate` above |
| `funding_interval_hours` | float, constant `1.0` | documented assumption | Hyperliquid settles hourly for all perps checked; **not exposed as a field on any endpoint** — see collector.py's constant comment |
| `mark_price` | float | ctx `markPx` | |
| `oracle_price` | float | ctx `oraclePx` | Hyperliquid does not distinguish index vs. oracle price — oracle **is** the reference price, so `index_price` from the original spec is omitted as a separate column rather than duplicating this one |
| `open_interest` | float | ctx `openInterest` | Base units (coin), not USD |
| `open_interest_usd` | float | computed: `open_interest * mark_price` | Not directly exposed by the venue |
| `best_bid` / `best_ask` | float or empty | `l2Book` levels[0][0] / levels[1][0] | Empty if the book fetch failed that cycle — never interpolated |
| `mid` | float or empty | computed: `(best_bid + best_ask) / 2` | |
| `spread_bps` | float or empty | computed: `(best_ask - best_bid) / mid * 10000` | |
| `volume_24h_usd` | float | ctx `dayNtlVlm` | Already USD notional, no conversion needed |
| `raw_response_json` | JSON string | ctx + top-of-book | Full fidelity fallback if a derived column above is ever found to be wrong |

## Explicitly not collected

- **`predicted_funding_rate`** — Hyperliquid's `metaAndAssetCtxs` exposes
  exactly one `funding` value per asset; there is no separate
  predicted-vs-realized split on any checked endpoint. Verified directly
  against a real response before deciding to omit this column rather than
  ship a permanently-empty one.

## Universe

Top `FUNDING_UNIVERSE_TOP_N` (env var, default 30) perps by 24h notional
volume (`dayNtlVlm`), refreshed once per UTC day. See `universe.py` and
`DECISIONS.md` (2026-09-01 entry) for why this was chosen over the full
~233-perp venue listing or the 8 currently-traded coins.

## Fetch failures

Every failed fetch (after exhausting retries with exponential backoff) is
logged to both stdout (`[FUNDING_COLLECTOR][FETCH_FAILURE]`) and
`research/funding/data/_fetch_failures.log`, with the endpoint, error, and
timestamp. A gap in a symbol's file is real and visible in that log —
nothing is silently interpolated to fill it.

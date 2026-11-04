# Predictability scan -- EXPLORATORY, decides nothing

42 coins, 5002 hours of 1h bars, data fetched 2026-10-28T00:46:11.551000+00:00. Null: 1000 sign-flip draws (seed 20261104). Positive = the rule named earned money: market = follow the universe's own trend; relative = long laggards / short leaders (reversal); shock = continue the move; premium = long the cheap / short the rich; xs_momentum = long past winners / short losers; own_trend = each coin follows its own sign.

## Hours to a week, on the 1h archive

| cell | n | gross bps/period | null sd | z | p | Holm p | traded/period | break-even one-way fee (bps) |
|---|---|---|---|---|---|---|---|---|
| market_1h | 4998 | -1.50 | 0.99 | -1.52 | 0.124 | 1.000 | 1.04 | 1.45 |
| relative_1h | 5000 | +2.65 | 0.62 | +4.26 | 0.001 | 0.013 | 2.66 | 1.00 |
| relative_skip_1h (diagnostic) | 4999 | +1.03 | 0.62 | +1.66 | 0.103 | n/a | 2.66 | 0.39 |
| market_4h | 1248 | -5.70 | 3.76 | -1.52 | 0.122 | 1.000 | 1.05 | 5.44 |
| relative_4h | 1248 | +6.40 | 2.40 | +2.67 | 0.013 | 0.156 | 2.67 | 2.40 |
| relative_skip_4h (diagnostic) | 1248 | +2.97 | 2.32 | +1.28 | 0.192 | n/a | 2.66 | 1.12 |
| market_24h | 206 | -8.90 | 22.92 | -0.39 | 0.707 | 1.000 | 1.00 | 8.85 |
| relative_24h | 206 | -14.38 | 14.48 | -0.99 | 0.334 | 1.000 | 2.57 | 5.59 |
| market_72h | 67 | +24.42 | 69.36 | +0.35 | 0.745 | 1.000 | 1.06 | 23.02 |
| relative_72h | 67 | -31.98 | 41.98 | -0.76 | 0.463 | 1.000 | 2.61 | 12.26 |
| market_168h | 27 | +137.41 | 174.27 | +0.79 | 0.451 | 1.000 | 0.85 | 162.39 |
| relative_168h | 27 | +41.57 | 99.80 | +0.42 | 0.636 | 1.000 | 2.66 | 15.60 |
| shock_1h | 3389 | -2.87 | 3.39 | -0.85 | 0.393 | 1.000 | 2.00 | 1.44 |
| shock_4h | 3389 | +6.08 | 6.11 | +0.99 | 0.316 | 1.000 | 2.00 | 3.04 |
| shock_24h | 3387 | +14.05 | 12.58 | +1.12 | 0.260 | 1.000 | 2.00 | 7.02 |

This account pays 1.44 bps one way as maker and 4.32 as taker.

## Premium, on the 60 days the funding history covers

| cell | n | gross bps/period | null sd | z | p | Holm p | traded/period | break-even one-way fee (bps) |
|---|---|---|---|---|---|---|---|---|
| premium_1h [carry +0.38 bps; rank corr with past return +0.05] | 1439 | -0.62 | 1.22 | -0.51 | 0.626 | 1.000 | 0.62 | 1.00 |
| premium_4h [carry +1.54 bps; rank corr with past return +0.11] | 359 | -2.54 | 4.75 | -0.53 | 0.601 | 1.000 | 1.04 | 2.43 |
| premium_24h [carry +9.81 bps; rank corr with past return +0.18] | 59 | -22.85 | 26.89 | -0.85 | 0.423 | 1.000 | 1.54 | 14.82 |
| premium_72h [carry +42.33 bps; rank corr with past return +0.18] | 19 | -12.60 | 84.38 | -0.15 | 0.888 | 1.000 | 1.83 | 6.90 |
| premium_168h [carry +34.24 bps; rank corr with past return +0.11] | 7 | -134.80 | 215.33 | -0.63 | 0.521 | 1.000 | 1.76 | 76.51 |

## Funding

- 42 coins, 60480 coin-hours: mean +0.076 bps/hour (+6.7% a year, paid by longs to shorts); exactly the formula's fixed interest component in 72.2% of coin-hours; negative in 9.6%
- percentiles 0.1/1/50/99/99.9: -18.701 / -0.574 / +0.125 / +1.018 / +2.250 bps/hour
- a perp-only short held 60 days, mean over 42 coins: funding collected +1.10%, price move +22.49% (median +7.29%, sd across coins 45.0%), net to the short -21.40%; the short made money on 38% of coins

## Weeks, on daily candles (42 coins, 1403 days, 2023-01-01 to 2026-11-03)

| cell | n | gross bps/week | null sd | z | p | Holm p | traded/period | break-even one-way fee (bps) |
|---|---|---|---|---|---|---|---|---|
| xs_momentum_7d | 198 | +37.73 | 40.02 | +0.94 | 0.344 | 1.000 | 2.64 | 14.30 |
| own_trend_7d | 6188 | -10.02 | 49.25 | -0.20 | 0.830 | 1.000 | n/a | n/a |
| market_trend_7d | 183 | +54.51 | 71.98 | +0.76 | 0.455 | 1.000 | n/a | n/a |
| xs_momentum_28d | 195 | +64.25 | 38.84 | +1.65 | 0.099 | 0.593 | 1.24 | 51.82 |
| own_trend_28d | 6062 | +56.41 | 47.39 | +1.19 | 0.256 | 1.000 | n/a | n/a |
| market_trend_28d | 180 | +49.21 | 68.56 | +0.72 | 0.464 | 1.000 | n/a | n/a |

## Do regimes persist?

- 1h archive, 29 blocks of 168: volatility rank-corr +0.74 (shuffled +0.50, p 0.002); trendiness -0.02 (shuffled -0.01, p 0.737); trend-following after the trendiest third minus after the least trendy third +226.7 bps (shuffled -15.5, p 0.116)
- daily candles, 200 blocks of 7: volatility rank-corr +0.52 (shuffled +0.30, p 0.002); trendiness -0.02 (shuffled -0.00, p 0.607); trend-following after the trendiest third minus after the least trendy third -25.7 bps (shuffled +6.9, p 0.687)

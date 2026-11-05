# Cash-and-carry on this venue -- MEASUREMENT, in-sample, no p-values

Hold the coin as spot, short the same amount of its perp; the short collects funding. Costs: an entry or an exit is one spot and one perp fill (0.1132% taker, 0.0544% maker). Funding in % of notional a year.

## Funding to a short

| coin | since | % a year | last 12m | by year | at constant | negative hours | negative months | worst month | longest negative weeks |
|---|---|---|---|---|---|---|---|---|---|
| BTC | 2023-05-12 | +14.1 | +5.8 | 2023 +15.1, 2024 +24.1, 2025 +10.6, 2026 +5.5 | 52% | 13% | 8% of 40 | 2023-08 -0.70% | 6 |
| ETH | 2023-05-12 | +14.2 | +6.4 | 2023 +22.4, 2024 +22.0, 2025 +8.5, 2026 +5.9 | 49% | 13% | 8% of 40 | 2025-04 -0.24% | 3 |
| FARTCOIN | 2024-12-20 | +23.0 | +14.7 | 2024 +89.6, 2025 +27.7, 2026 +14.8 | 66% | 2% | 0% of 22 | 2025-10 +0.61% | 1 |
| HYPE | 2024-12-05 | +20.2 | +9.9 | 2024 +118.8, 2025 +22.1, 2026 +9.5 | 67% | 7% | 0% of 23 | 2026-04 +0.42% | 1 |
| NEAR | 2023-11-01 | +20.6 | +10.9 | 2023 +73.1, 2024 +30.4, 2025 +9.9, 2026 +11.4 | 65% | 8% | 0% of 36 | 2025-10 +0.26% | 1 |
| PUMP | 2025-07-10 | +14.3 | +9.8 | 2025 +22.2, 2026 +9.9 | 75% | 7% | 0% of 15 | 2026-03 +0.33% | 1 |
| SOL | 2023-05-12 | +12.2 | +0.9 | 2023 +13.0, 2024 +28.1, 2025 +5.3, 2026 +0.8 | 42% | 24% | 28% of 40 | 2026-02 -1.35% | 11 |
| XMR | 2026-01-15 | +28.0 | +28.0 | 2026 +28.0 | 61% | 4% | 0% of 9 | 2026-02 +1.06% | 1 |
| ZEC | 2025-10-02 | +4.7 | +4.3 | 2025 -0.6, 2026 +6.2 | 67% | 17% | 23% of 13 | 2025-11 -2.42% | 4 |

## Always on, or only while the trailing week paid (same window, net)

| coin | from | years | always, taker | always, maker | filtered, taker | filtered, maker | in market | switches |
|---|---|---|---|---|---|---|---|---|
| BTC | 2023-05-19 | 3.42 | +14.1 | +14.2 | +13.4 | +13.9 | 92% | 30 |
| ETH | 2023-05-19 | 3.42 | +14.2 | +14.2 | +13.1 | +13.8 | 93% | 36 |
| FARTCOIN | 2024-12-28 | 1.85 | +22.0 | +22.0 | +21.7 | +21.9 | 99% | 4 |
| HYPE | 2024-12-13 | 1.90 | +18.8 | +18.9 | +18.4 | +18.6 | 99% | 8 |
| NEAR | 2023-11-09 | 2.99 | +20.2 | +20.2 | +19.4 | +19.8 | 96% | 20 |
| PUMP | 2025-07-18 | 1.30 | +12.0 | +12.1 | +11.2 | +11.6 | 97% | 8 |
| SOL | 2023-05-19 | 3.42 | +12.2 | +12.2 | +13.1 | +13.9 | 76% | 50 |
| XMR | 2026-01-23 | 0.78 | +27.5 | +27.7 | +27.2 | +27.5 | 98% | 4 |
| ZEC | 2025-10-10 | 1.07 | +3.5 | +3.6 | +4.4 | +5.6 | 79% | 22 |

## On capital, and what liquidates the short leg

Return on capital at the last 12 months' funding, perp leg at L x; the rise that liquidates that short; the worst rises in the daily candles.

| coin | max lev | worst 1d rise | worst 7d rise | L 1 | L 1.5 | L 2 | L 3 |
|---|---|---|---|---|---|---|---|
| BTC | 40.0 | +22% (2023-10-22) | +38% (2023-03-10) | +2.9%, liq +98% ok | +3.5%, liq +65% ok | +3.9%, liq +48% ok | +4.3%, liq +32% ok |
| ETH | 25.0 | +23% (2025-05-07) | +51% (2025-05-07) | +3.2%, liq +96% ok | +3.8%, liq +63% ok | +4.3%, liq +47% ok | +4.8%, liq +31% ok |
| FARTCOIN | 10.0 | +64% (2025-01-18) | +271% (2025-01-12) | +7.3%, liq +90% ok | +8.8%, liq +59% LIQ 1d | +9.8%, liq +43% LIQ 1d | +11.0%, liq +27% LIQ 1d |
| HYPE | 10.0 | +29% (2025-04-08) | +114% (2024-12-10) | +4.9%, liq +90% ok | +5.9%, liq +59% ok | +6.6%, liq +43% ok | +7.4%, liq +27% LIQ 1d |
| NEAR | 10.0 | +42% (2025-11-06) | +100% (2026-09-15) | +5.5%, liq +90% ok | +6.6%, liq +59% ok | +7.3%, liq +43% ok | +8.2%, liq +27% LIQ 1d |
| PUMP | 10.0 | +31% (2026-08-19) | +105% (2026-08-16) | +4.9%, liq +90% ok | +5.9%, liq +59% ok | +6.6%, liq +43% ok | +7.4%, liq +27% LIQ 1d |
| SOL | 20.0 | +36% (2023-01-13) | +90% (2023-01-07) | +0.5%, liq +95% ok | +0.6%, liq +63% ok | +0.6%, liq +46% ok | +0.7%, liq +30% LIQ 1d |
| XMR | 5.0 | +40% (2025-04-27) | +84% (2026-01-07) | +14.0%, liq +82% ok | +16.8%, liq +52% ok | +18.7%, liq +36% LIQ 1d | +21.0%, liq +21% LIQ 1d |
| ZEC | 10.0 | +75% (2025-09-30) | +218% (2025-09-27) | +2.2%, liq +90% ok | +2.6%, liq +59% LIQ 1d | +2.9%, liq +43% LIQ 1d | +3.2%, liq +27% LIQ 1d |

## Spot against perp (hourly closes)

| coin | spot token | hours | mean bps | sd | 1st pct | 99th pct | worst | beyond 1% |
|---|---|---|---|---|---|---|---|---|
| BTC | UBTC | 5000 | -2.0 | 6.1 | -13.4 | +13.1 | +101.6 (2026-07-08 11:00) | 0.02% |
| ETH | UETH | 5000 | -2.0 | 6.3 | -15.3 | +13.4 | +42.2 (2026-10-29 22:00) | 0.00% |
| FARTCOIN | UFART | 5000 | -10.2 | 30.2 | -105.6 | +69.8 | -252.4 (2026-08-26 20:00) | 1.48% |
| HYPE | HYPE | 5000 | +1.0 | 4.9 | -13.1 | +10.1 | -37.7 (2026-10-30 01:00) | 0.00% |
| NEAR | ONEAR | 1073 | -2.3 | 37.1 | -66.3 | +112.3 | -430.1 (2026-09-22 02:00) | 1.96% |
| PUMP | UPUMP | 5000 | -5.8 | 14.7 | -43.7 | +34.6 | +279.3 (2026-09-06 18:00) | 0.04% |
| SOL | USOL | 5000 | -1.5 | 6.9 | -17.2 | +16.3 | +42.2 (2026-04-14 14:00) | 0.00% |
| XMR | XMR1 | 5000 | -13.6 | 36.7 | -98.1 | +84.6 | -806.4 (2026-09-04 03:00) | 1.66% |
| ZEC | UZEC | 5000 | +0.9 | 34.4 | -42.2 | +97.5 | +726.4 (2026-08-21 08:00) | 1.02% |

## Spot tokens (the issuer is the operator's claim, as the venue lists it)

| coin | token | listed as | 24h spot volume |
|---|---|---|---|
| BTC | UBTC | Unit Bitcoin | $50,946,013 |
| ETH | UETH | Unit Ethereum | $15,900,403 |
| FARTCOIN | UFART | Unit Fartcoin | $116,053 |
| HYPE | HYPE | Hyperliquid | $104,022,384 |
| NEAR | ONEAR | NEAR Protocol | $4,050,450 |
| PUMP | UPUMP | Unit Pump Fun | $1,998,233 |
| SOL | USOL | Unit Solana | $4,326,068 |
| XMR | XMR1 | XMR - Wagyu.xyz | $20,771,069 |
| ZEC | UZEC | Unit Zcash | $22,796,706 |

## HLP, the venue's market-making vault, for comparison

- 2023-05-10 to 2026-11-05: time-weighted +264.8% (+44.9% a year); +170.3% to +395.9% depending on when deposits landed within each sampling interval
- by year: 2023 +77.3%, 2024 +62.9%, 2025 +17.3%, 2026 +7.7%
- trailing: 365d +7.0%, 182d +1.0%, 91d +0.9%

# H-TREND: daily trend-following, long and short (generated)

Data: github.com/finom/static-klines @ 2a160c07c975d4e569bd5a264a9351073e6778db (Binance spot 1d klines). Costs 9.32 bps per unit traded; longs pay 3.0 bps a day funding; shorts receive 0.0.

Null calibration: 1 of 20 fired; mean excess +0.80 +/- 0.72 bps a day; PASSED.

## The pre-registered test (timing: real book minus rotated controls)

| | P&L days | real bps/day | controls bps/day | excess | z | p | Holm p | halves' excess | edge |
|---|---|---|---|---|---|---|---|---|---|
| T_all | 2507 | +10.48 | -2.13 | +12.60 | +3.41 | 0.0010 | 0.0020 | +13.26, +12.16 | YES |
| T_majors | 3237 | +13.04 | -2.11 | +15.15 | +3.73 | 0.0010 | 0.0020 | +19.12, +12.66 | YES |

## Forward-trial checks (T_all)

- edge_on_T_all: pass
- sharpe_at_least_0.7: pass
- sharpe_ci_above_0: pass
- both_halves_net_positive: pass
- each_lookback_net_positive: pass
- survives_one_day_delay: pass
- survives_20bps_costs: pass

**Worth a forward trial: YES**

## T_all as a strategy, net (2020-01-01 to 2026-11-11)

- +38.2% a year at 32.8% volatility; Sharpe +1.17 (95% CI +0.40 to +1.90); worst drawdown -35.5%; gross Sharpe +1.38.
- Halves (split 2023-06-07): Sharpe +1.32 then +1.01; total +263.9% then +162.8%.
- Activity: 836 trades and 46.7x turnover a year; costs 4.36% and funding 2.87% a year.
- Legs, gross % a year: long +44.4, short +1.1.
- By coin, gross % a year: BTC +7.5, ETH +6.0, BNB +4.8, SOL +5.0, XRP +3.6, ADA +5.3, DOGE +6.0, AVAX +3.5, LINK +1.7, DOT +1.9.
- Lookbacks alone, net Sharpe: 7d +0.60, 112d +0.54, 14d +1.12, 28d +1.09, 56d +0.66.
- Sensitivity, net Sharpe: one_day_delay +1.14, cost_20bps +1.01, fee_only_4.32bps +1.24, shorts_credited_funding +1.25, no_funding +1.25.
- Capital check: $+274 a year on $716.

| year | net % | Sharpe | worst drawdown % |
|---|---|---|---|
| 2020 | +105.5 | +2.14 | -13.0 |
| 2021 | +109.0 | +2.75 | -12.1 |
| 2022 | +11.1 | +0.48 | -21.2 |
| 2023 | +31.7 | +1.03 | -27.6 |
| 2024 | +30.7 | +1.01 | -34.4 |
| 2025 | -1.4 | +0.12 | -24.9 |
| 2026 | +18.0 | +0.72 | -25.8 |

## T_majors as a strategy, net (2018-01-01 to 2026-11-11)

- +47.6% a year at 41.7% volatility; Sharpe +1.14 (95% CI +0.46 to +1.80); worst drawdown -47.9%; gross Sharpe +1.35.
- Halves (split 2022-06-07): Sharpe +1.41 then +0.86; total +869.1% then +227.8%.
- Activity: 167 trades and 53.7x turnover a year; costs 5.00% and funding 3.74% a year.
- Legs, gross % a year: long +47.9, short +8.5.
- By coin, gross % a year: BTC +30.0, ETH +26.3.
- Lookbacks alone, net Sharpe: 7d +0.57, 112d +0.48, 14d +0.92, 28d +0.90, 56d +0.96.
- Sensitivity, net Sharpe: one_day_delay +1.06, cost_20bps +1.00, fee_only_4.32bps +1.20, shorts_credited_funding +1.21, no_funding +1.23.
- Capital check: $+341 a year on $716.

| year | net % | Sharpe | worst drawdown % |
|---|---|---|---|
| 2018 | +4.2 | +0.30 | -39.7 |
| 2019 | +63.5 | +1.26 | -33.3 |
| 2020 | +196.6 | +2.62 | -17.4 |
| 2021 | +85.9 | +1.87 | -13.4 |
| 2022 | +8.8 | +0.41 | -32.1 |
| 2023 | +50.7 | +1.14 | -34.3 |
| 2024 | +15.3 | +0.55 | -47.9 |
| 2025 | +9.8 | +0.44 | -23.4 |
| 2026 | +63.0 | +1.52 | -23.5 |

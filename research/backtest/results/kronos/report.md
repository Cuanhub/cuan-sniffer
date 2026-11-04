# Kronos as a reader of the band strategy's trades -- results

Pre-registration: `research/backtest/PREREG_kronos.md`. Data fetched 2026-10-28T00:46:11.551000+00:00; 44 coins; 876 trades.

## Null calibration (20 random-walk datasets, real size)

| statistic | fired | mean D | SE | verdict |
|---|---|---|---|---|
| K1_direction:raw | 1/20 | -0.0033 | 0.0195 | ok |
| K1_direction:excess | 1/20 | -0.0012 | 0.0266 | ok |
| K2_expected_r:raw | 0/20 | -0.0073 | 0.0218 | ok |
| K2_expected_r:excess | 1/20 | -0.0218 | 0.0308 | ok |

## Null spread (Amendment 1)

How far D spreads across the null datasets, against the SE the test reports. A ratio inside the chance band means the SE is the right size.

| statistic | datasets | spread of D | week SE | ratio | two-way SE | ratio | chance band | fired (week) | fired (two-way) |
|---|---|---|---|---|---|---|---|---|---|
| K1_direction:raw | 20 | 0.087 | 0.081 | 1.07 | 0.093 | 0.94 | 0.68-1.31 | 1 | 1 |
| K1_direction:excess | 20 | 0.119 | 0.102 | 1.17 | 0.110 | 1.08 | 0.68-1.31 | 1 | 1 |
| K2_expected_r:raw | 20 | 0.097 | 0.091 | 1.07 | 0.097 | 1.00 | 0.68-1.31 | 0 | 0 |
| K2_expected_r:excess | 20 | 0.138 | 0.113 | 1.22 | 0.120 | 1.14 | 0.68-1.31 | 1 | 0 |

## Approved minus rejected trades (net R, real outcomes)

| hypothesis | statistic | split | scored | approved | D | 95% CI | p | Holm p | two-way SE x widened | Holm p, two-way | halves | approved mean R | rejected mean R | as first registered | verdict (amended) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| K1_direction | raw | sign | 876 | 422 | +0.193 | [-0.076, +0.525] | 0.2056 | 0.4112 | 0.162 x 1.00 | 0.4681 | +0.314, +0.129 | +0.010 | -0.183 | no | **no** |
| K2_expected_r | raw | sign | 876 | 306 | +0.080 | [-0.223, +0.431] | 0.6309 | 0.6309 | 0.166 x 1.00 | 0.6313 | +0.121, +0.066 | -0.038 | -0.118 | no | **no** |

## Descriptive (no decisions)

- unscored decisions: 0; mean valid paths 16.0
- 24h close, mean absolute error: Kronos 7.66% vs no-change 4.13% (n 875)
- K1_direction: approval rate 0.48, Spearman(score, net R) -0.145; excess-based D +0.207 (p 0.2169, n 555)
  - limit: n 719 (sign), approved +0.046, rejected -0.213
  - market_close: n 157 (sign), approved -0.115, rejected +0.009
  - by score quintile (mean score -> mean net R): -0.1309 -> -0.070 (n 176); -0.0422 -> -0.189 (n 175); -0.0032 -> -0.254 (n 175); +0.0348 -> -0.018 (n 175); +0.1275 -> +0.080 (n 175)
- K2_expected_r: approval rate 0.35, Spearman(score, net R) -0.110; excess-based D +0.001 (p 0.9962, n 555)
  - limit: n 719 (sign), approved -0.004, rejected -0.142
  - market_close: n 157 (sign), approved -0.169, rejected +0.001
  - by score quintile (mean score -> mean net R): -1.0888 -> -0.002 (n 176); -0.6153 -> -0.282 (n 175); -0.0327 -> -0.081 (n 219); +0.1583 -> +0.034 (n 131); +0.9587 -> -0.093 (n 175)

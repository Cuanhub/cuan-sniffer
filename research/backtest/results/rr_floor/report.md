# Reward:risk floor -- results

Pre-registration: `research/backtest/PREREG_rr_floor.md`. Data fetched 2026-10-28T00:46:11.551000+00:00; 44 coins.

Null calibration: E_live fired 0/20, mean -0.0106 +/- 0.0198, E_mid fired 0/20, mean -0.0199 +/- 0.0094, E_low fired 0/20, mean +0.0070 +/- 0.0072

## Edge by reward:risk bucket (excess over random entries with the same geometry)

| bucket | n | excess R | 95% CI | p (week-cluster) | Holm p | halves | absolute net R | control R | edge |
|---|---|---|---|---|---|---|---|---|---|
| E_live | 555 | +0.055 | [-0.145, +0.273] | 0.6009 | 1.0000 | +0.128, -0.017 | -0.058 | -0.113 | **no** |
| E_mid | 1283 | -0.060 | [-0.201, +0.084] | 0.4132 | 1.0000 | -0.101, -0.019 | -0.098 | -0.039 | **no** |
| E_low | 1556 | -0.024 | [-0.095, +0.050] | 0.5211 | 1.0000 | +0.002, -0.050 | -0.054 | -0.031 | **no** |

## Each floor as a strategy (descriptive; portfolio rails not modelled)

| MIN_RR | trades | win | mean net R | total net R | weekly vs live (CI) |
|---|---|---|---|---|---|
| 0.3 | 3195 | 49.2% | -0.061 | -193.6 | -4.40 [-12.61, +3.64] |
| 0.5 | 2860 | 45.9% | -0.057 | -163.1 | -3.23 [-11.00, +4.40] |
| 1.0 | 2086 | 38.2% | -0.080 | -166.9 | -3.38 [-9.71, +2.56] |
| 1.5 | 1373 | 31.5% | -0.110 | -151.5 | -2.78 [-7.56, +1.66] |
| 2.0 | 876 | 27.6% | -0.090 | -79.1 | +0.00 [+0.00, +0.00] |
| 2.5 | 551 | 24.7% | -0.085 | -47.1 | +1.23 [-1.79, +4.20] |
| 3.0 | 356 | 23.6% | -0.016 | -5.8 | +2.82 [-1.09, +6.81] |

rr >= 3.0 within the live trades: n=120, excess +0.002 (p=0.9908)

## Stop/ATR strata (excess R)

- E_live: below -0.006 (n 277), at-or-above +0.116 (n 278), median stop 1.97x ATR
- E_mid: below -0.036 (n 641), at-or-above -0.084 (n 642), median stop 1.96x ATR
- E_low: below -0.010 (n 778), at-or-above -0.037 (n 778), median stop 1.97x ATR

## By entry kind (excess R)

- E_live:limit: n 398, +0.061 (iid p 0.4748)
- E_live:market_close: n 157, +0.041 (iid p 0.7516)
- E_mid:limit: n 959, -0.095 (iid p 0.0142)
- E_mid:market_close: n 324, +0.046 (iid p 0.4880)
- E_low:limit: n 851, -0.060 (iid p 0.0499)
- E_low:market_close: n 705, +0.020 (iid p 0.4971)

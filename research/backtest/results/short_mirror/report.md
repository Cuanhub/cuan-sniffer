# Short-side mirror -- results

Pre-registration: `research/backtest/PREREG_short_mirror.md`. Data fetched 2026-10-28T00:46:11.551000+00:00; 44 coins.

Null calibration: S_live fired 0/20, mean -0.0325 +/- 0.0315

## The test (excess over random shorts with the same geometry)

| | n | excess R | 95% CI | p (week-cluster) | p (two-way) | halves | absolute net R | controls' net R | edge |
|---|---|---|---|---|---|---|---|---|---|
| S_live | 676 | -0.146 | [-0.329, +0.070] | 0.1490 | 0.1490 | -0.106, -0.186 | -0.263 | -0.118 | **no** |

Criteria: p_below_alpha NO, positive NO, both_halves_positive NO, null_calibration_ok yes

## The mirror as a strategy (descriptive; portfolio rails not modelled)

- 1048 resolved shorts, win rate 27.6%, mean net R -0.132, total -138.1R; 289 targets, 759 stops; median hold 13 bars, median stop 2.67%
- planner rr median 2.40 (inverted prices) vs 2.19 as a real short
- long side through the same code: 876 trades, mean net R -0.090; excess +0.055 (p 0.6009, n 555)
- combined book over 26 weeks: long-only -3.04R/week (sd 17.30), long + short -8.35R/week (sd 14.49); shorts add -5.31 [-13.11, +1.88]; weekly correlation -0.71

## Secondary splits of the excess (descriptive)

- entry limit: n 512, -0.152 (iid p 0.0223)
- entry market_close: n 164, -0.127 (iid p 0.3451)
- stop below 1.99x ATR: -0.109 (n 338); at or above: -0.183 (n 338)
- weeks the universe fell: n 308, +0.196
- weeks the universe rose: n 368, -0.431
- with 0.00125%/h funding to shorts: absolute -0.118, excess -0.146 (p 0.1490)

# Trendline breakout -- results

Pre-registration: `research/backtest/PREREG_trendline_breakout.md`.  Data fetched 2026-10-27T14:18:34.685000+00:00 from https://api.hyperliquid.xyz/info candleSnapshot (mainnet, public, read-only).

45 coins, 218,542 closed 1h bars. Excluded: 0.

## Null calibration

20 random-walk datasets (12 coins x 5000 bars). Decision fired (false positives): H1 0/20, H2 0/20, H3 0/20. Holm rejections: H1 1, H2 0, H3 0. Mean null estimate (must be ~0): H1 -0.0248 +/- 0.0376, H2 -0.0224 +/- 0.0289, H3 -0.0065 +/- 0.0075.

## Baseline (the live strategy, replayed)

907 resolved trades (748 bids, 159 market), mean net R -0.051 (gross -0.023, costs +0.028), win 28.8%, total -46.6R, median rr 2.36, median risk 2.60%. 8 still open at the data end.

## Decisions

| framing | adds edge | criteria |
|---|---|---|
| H1 | **no** | holm_rejected=NO, positive=NO, same_sign_both_halves=NO, matched_control_agrees=NO, both_stop_atr_strata_agree=NO, null_calibration_ok=yes |
| H2 | **no** | holm_rejected=NO, positive=yes, same_sign_both_halves=yes, null_calibration_ok=yes |
| H3 | **no** | holm_rejected=NO, positive=yes, same_sign_both_halves=yes, null_calibration_ok=yes |

## H1 -- filter

break in the prior 24 bars: n=488, mean -0.165;  none: n=419, mean +0.081

difference -0.246, week-cluster 95% CI [-0.473, -0.022], p=0.0324 (iid Welch p=0.0263)

matched-random control: excess difference -0.166 (p=0.1374); stop/ATR strata: -0.193 below / -0.299 at-or-above the median

halves: -0.074 (n 253/200)  -0.407 (n 235/219)

filter applied as a strategy: n=565, mean -0.127, total -71.7R

## H2 -- confirmation entry

915 setups: entered 307, stopped_first 503, no_break 61, invalid_at_entry 44

**trendline arm minus random-event arm** per setup +0.035, CI [+0.001, +0.067], p=0.0368

trendline arm minus baseline +0.050 (p=0.5449); random-event arm minus baseline +0.015 (p=0.8354) -- the second number is what abstention alone is worth

entry rate: trendline 0.336, random 0.399; confirmation trades mean -0.005 (n 302), median rr 0.89

breakdown: base_win_conf_skipped 90, base_win_conf_win 170, base_win_conf_loss 1, base_loss_conf_skipped 515, base_loss_conf_win 0, base_loss_conf_loss 131

with rr >= 2 re-applied: +0.028 (p=0.1121);  entry slippage 0.0564%: +0.037

## H3 -- standalone breakout

3942 resolved trades: absolute mean -0.095 (win 33.5%), random-entry control -0.141, excess +0.048, CI [-0.091, +0.170], p=0.4740

halves: +0.031  +0.065

## Holm family

| | p | Holm p | rejected |
|---|---|---|---|
| H1 | 0.0324 | 0.0973 | False |
| H2 | 0.0368 | 0.0973 | False |
| H3 | 0.4740 | 0.4740 | False |

## Robustness (never used to choose)

- H1: 6/30 variants positive; 0 positive with uncorrected p<0.05, 7 with p<0.05 either way (chance alone: ~1.5)
- H2: 29/29 variants positive; 14 positive with uncorrected p<0.05, 14 with p<0.05 either way (chance alone: ~1.5)
- H3: 9/12 variants positive; 4 positive with uncorrected p<0.05, 4 with p<0.05 either way (chance alone: ~0.6)

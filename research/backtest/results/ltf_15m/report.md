# 15-minute read -- results

Pre-registration: `research/backtest/PREREG_ltf_15m.md`. 1h data fetched 2026-10-28T00:46:11.551000+00:00, 15m 2026-10-28T00:54:02.982000+00:00 (52.1 days of 15m bars, 44 coins).

Null calibration: B1_band_stop fired 0/20, mean -0.0196 +/- 0.0385, B2_ltf_stop fired 0/20, mean -0.0188 +/- 0.0398

## B -- 15m confirmation: each confirmed trade vs random entries after it, same geometry

| framing | confirmed trades | excess R over later random entries | 95% CI | p (week-cluster) | Holm p | halves | adds edge |
|---|---|---|---|---|---|---|---|
| B1_band_stop | 163 | -0.067 | [-0.289, +0.120] | 0.5258 | 1.0000 | -0.058, -0.076 | **no** |
| B2_ltf_stop | 163 | -0.006 | [-0.202, +0.204] | 0.9533 | 1.0000 | -0.085, +0.072 | **no** |

B1_band_stop: entered 163, stopped_first 75, invalid_at_entry 1; entry rate 0.682 (random arm 0.732); confirmed trades -0.047 net R (n 163), median rr at entry 1.40; vs baseline -0.002; per setup vs random (secondary) -0.026

B2_ltf_stop: entered 163, stopped_first 75, invalid_at_entry 1; entry rate 0.682 (random arm 0.656); confirmed trades -0.057 net R (n 163), median rr at entry 2.14; vs baseline -0.009; per setup vs random (secondary) +0.312

B2 stop distance: median 1.80x 1h ATR (baseline band stop 2.00x)

## A -- measurement: the same 1h-engine trades replayed on 15m bars

281 trades: mean net R -0.110 on 1h vs -0.110 on 15m (difference +0.000); outcome changed for 0; 1h-ambiguous bars 0 (0 were wins at 15m); fill-hour targets 1 (0 wins at 15m).

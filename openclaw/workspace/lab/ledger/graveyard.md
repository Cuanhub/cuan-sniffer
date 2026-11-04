# Graveyard: ideas that did not survive

Before testing anything near one of these, say in the PREREG what is
materially different. "Maybe it works now" is not different.

| idea | what it looked like | what killed it | when |
|---|---|---|---|
| **Killzone timing** (London/NY session entries) | better win rate in session hours, pooled | reversed sign under control for time/regime | 2026-10 |
| **HTF bias check** (4h trend agrees) | better outcomes with HTF agreement | the effect vanished under control; the proxy tracked the period, not the setup | 2026-10 |
| **Sweep confirmation** (liquidity sweep -> reclaim before entry) | +0.080R | n = 15: noise, not evidence | 2026-10 |
| **`pull_over_run`** (pullback depth relative to the run) | looked predictive; a revival was recommended | inverts under control -- the recommendation was wrong | 2026-10-26 |
| **Constant-score split** | a 0.715R spread between halves | the column was constant; the median split divided by list position, i.e. time | 2026-10-26 |
| **`outcome_r` as corroboration** | "agrees with the result" | it is +rr or -1 by construction -- an identity | 2026-10-26 |
| **Per-coin edge** | some coins looked reliably profitable | did not replicate across halves of the period | 2026-10 |
| **The old level gate** | +203R over the backfill | -0.0413R vs random entry (z = -0.93, n = 688): all 28 coins rose, median +38.5%, and random entries with the same geometry earned more | 2026-09 |
| **Tight fixed stops (0.25%)** | high rr, many trades | 7 straight live stop-outs, all sub-1.5% stops; stop/ATR is the dominant effect (t = +9.75) | 2026-10-26 |
| **Trendline break as a filter** | "trade the band after the downtrend line breaks" | -0.246R vs no break (CI -0.473 to -0.022), wrong sign in both halves and both stop/ATR strata; pre-registered H-TRENDLINE H1 | 2026-10-27 |
| **Trendline breakout, standalone** | +0.048R over random entry, iid p 0.04 | week-clustered p 0.47 -- breakouts cluster across coins in time; absolute -0.095R/trade over 3,942 trades; H-TRENDLINE H3 | 2026-10-27 |
| **Lowering the reward:risk floor** (take the bands the gate refuses) | refused bands won 33 of 37 on the recorder's first day | one rally; over 30 weeks rr 1-2 -0.060R and rr 0.3-1 -0.024R vs random, and MIN_RR 1.0 loses -166.9R vs -79.1R at 2.0; H-RRFLOOR | 2026-10-27 |
| **15m break-of-structure confirmation** | the classic LTF entry inside an HTF zone | -0.067R (band stop) and -0.006R (15m stop) vs later random entries on 52 days; NOT ruled out at a material size -- re-test with ~6 months of archived 15m bars, same design; H-LTF15M | 2026-10-27 |
| **Kronos (pre-trained candlestick model) as a trade filter** | "a model that reads candles like text" | approved minus rejected over 876 trades: K1 24h direction +0.193R (Holm p 0.41; 0.47 two-way), K2 expected R +0.080R (p 0.63); its 24h forecasts are 1.85x no-change's error on real bars and on random walks alike. K1 NOT ruled out at a material size -- forward re-test queued; H-KRONOS | 2026-11-03 |
| **Short-side mirror of the band strategy** (supply bands as entries, stops above, demand targets below) | "work both directions"; the profitable vaults trade both ways | -0.146R vs random shorts with the same geometry (CI -0.329 to +0.070, p 0.149), negative in both halves; 1,048 shorts at -0.132R; long + short loses faster than long alone (weekly correlation -0.71: steadier, not better); pre-registered H-SHORT | 2026-11-04 |

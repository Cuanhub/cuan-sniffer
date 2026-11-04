# The band strategy's short-side mirror -- interpretation

`report.md` and `summary.json` are generated and are the record. `posthoc.json`
(from `research/backtest/experiments/short_mirror_posthoc.py`) holds the post-
hoc splits below, which decide nothing. This page is the reading of them,
written afterwards and labelled as such.

**Verdict: no edge.** The live strategy run upside down -- supply bands as
entries, stops above, demand bands below as targets -- times its shorts
slightly WORSE than random shorts with the same geometry, in both halves of
the period. Nothing goes live; the system stays long-only, and the standing
decision of 2026-09-18 is untouched.

## The pre-registered test

| | n | excess R | 95% CI | p | halves | absolute net R | controls' net R |
|---|---|---|---|---|---|---|---|
| S_live | 676 | -0.146 | -0.329 to +0.070 | 0.149 (two-way 0.149) | -0.106, -0.186 | -0.263 | -0.118 |

Edge needed p < 0.05, a positive estimate and both halves positive; it has
none of the three. The null calibration passed first (0 of 20 random-walk
datasets fired; mean -0.033 +/- 0.032).

- As a strategy: 1,048 resolved shorts, 27.6% wins, -0.132R a trade, -138R
  in total (portfolio rails not modelled). 676 of them have controls; the
  rest entered too near the end of the data for 50 random shorts to resolve.
- The long side through the same code reproduces H-RRFLOOR exactly (876
  trades, -0.090R; excess +0.055, p 0.60, n 555): the engine change did not
  touch longs.
- Both sides together: -8.35R a week against -3.04R long-only over 26 weeks.
  The two sides' weekly results correlate -0.71, so the book is steadier
  (weekly sd 14.5 vs 17.3) and loses faster. Hedging does not create edge.
- The archive rose (median coin +36%), which is why random shorts lost
  -0.118R; the mirror lost more than that.

## Post hoc (decides nothing)

| split | n | excess R | 95% CI | p |
|---|---|---|---|---|
| native short rr >= 2.0 | 452 | -0.172 | -0.390 to +0.072 | 0.149 |
| native rr < 2.0 | 224 | -0.094 | -0.302 to +0.136 | 0.401 |
| resting sells | 512 | -0.152 | -0.334 to +0.063 | 0.138 |
| in-band market shorts | 164 | -0.127 | -0.360 to +0.146 | 0.329 |

- The registration called the inverted-price thresholds second-order. For
  the reward:risk gate they are not: a target below reads ~9% further on
  inverted prices (median rr 2.40 there, 2.19 as a real short), so the
  mirror took some trades a native gate would refuse. Those 224 did no worse
  than the rest; the strict subset is no better. The verdict does not turn
  on it.
- Resting sells looked significant on the iid p in the report (0.022); on
  the week-clustered SE it is 0.138. Bids on the long side showed the same
  pattern (H-RRFLOOR: filled bids in rr 1-2 -0.095R, iid p 0.014): an order
  resting at a band fills when price runs through it.
- The split by market week (+0.196 in weeks the universe fell, -0.431 in
  weeks it rose) is mostly the drift difference between a trade's week and
  its controls' later bars, not timing skill, and a week's direction is not
  known when the trade is placed.

## What it means

Long (+0.055, CI -0.145 to +0.273) and short (-0.146, CI -0.329 to +0.070)
together say the band rules carry no detectable information about direction
on these bars, either way up. That agrees with the predictability scan
(`results/predictability_scan/`): at the strategy's 13-hour median hold,
nothing in these 44 coins' own prices predicts the next hours beyond the
last hour's bid-ask bounce.

# Kronos as a reader of the band strategy's trades -- interpretation

`report.md` and `summary.json` are generated and are the record. `posthoc.json`
(from `research/backtest/experiments/kronos_posthoc.py`) holds the post-hoc
numbers below, which decide nothing. This page is the reading of them,
written afterwards and labelled as such.

**Verdict: Kronos-small does not detectably separate the band strategy's
winning trades from its losing ones**, under the rules as first registered
and as amended. On every measure here, its forecasts on real candles are
indistinguishable from its forecasts on random walks. It is not wired into
live.

## The pre-registered test

876 trades of the live strategy as replayed (MIN_RR 2.0, measured costs),
which lose -0.090R per trade. Each was read at its decision bar: 16 sampled
48h futures from the 464 closed 1h candles up to it.

| reading | approved | D: approved - rejected, net R | 95% CI | p | Holm p | Holm p, two-way SE | halves | approved trades' mean |
|---|---|---|---|---|---|---|---|---|
| K1: 24h direction | 422 of 876 | +0.193 | -0.076 to +0.525 | 0.21 | 0.41 | 0.47 | +0.31, +0.13 | +0.010R |
| K2: expected R on the paths | 306 of 876 | +0.080 | -0.223 to +0.431 | 0.63 | 0.63 | 0.63 | +0.12, +0.07 | -0.038R |

Information needed Holm p < 0.05; neither comes close. Both statistics
passed the null calibration first (20 random-walk datasets: fired 1 and 0
times; means -0.003 and -0.007), and the same analysis scores +1.5R when
handed an oracle that knows the future.

- **K1** is above the 0.10R materiality bar at face value, but its interval
  runs from -0.08R to +0.52R. Against same-geometry controls it is the same
  (+0.207R, p 0.22), so it is not the stop-width effect either. And even at
  face value the trades it approves average +0.010R: it would turn a losing
  strategy into a break-even one, not a profitable one.
- **K2** shows the weakness the pre-registration anticipated. Against random
  orders with each trade's own geometry its D is +0.001R (p 0.996): what
  little it has is the trade's shape, not the market.
- By entry kind, K1's gap is +0.26R on the 719 bids and -0.12R on the 157
  market entries. A subgroup picked after seeing the data proves nothing;
  this is not a lead.

## Does Kronos read these candles at all? (post hoc)

| | real, 875 decisions | 20 random-walk datasets |
|---|---|---|
| 24h close error, Kronos / no change | 1.85 (7.66% vs 4.13%) | 1.85 +/- 0.04 |
| Kronos closer than no change | 32.2% | 30.4% +/- 1.2% |
| P(up 24h \| K1 up) - P(up 24h \| K1 down) | +0.087 (two-way SE 0.032) | +0.051 +/- 0.018, positive in 20 of 20 |

- Kronos's 24h forecast misses the real close by 85% more than "no change"
  does -- the same as on pure noise, where nothing can be forecast.
- The direction gap (price is up 24h later after 28.5% of the decisions K1
  calls up, 19.8% of those it calls down) looks significant on its own: z =
  2.7 on the two-way SE. But random walks produce +0.051 of it, in every one
  of the 20 datasets, and the largest of them (+0.088) exceeds the real gap.
  A bid becomes a trade only when price comes down to it,
  so most decisions are followed by a fall (24% up, on real bars and on noise
  alike), and that selection ties the future to the past Kronos reads. What
  is left, +0.036, is one to one and a half standard errors. Without the null
  this would have read "Kronos calls direction", and it would have been
  false.
- Spearman(K1, net R) is -0.145 although D is positive. It comes from inside
  the 634 stop-outs, whose net R differs only in how far past -1R the exit
  lands (fees, slippage and gaps, against the stop's width): K1 runs against
  that (-0.353), a geometry effect. On wins against losses the sign is positive:
  29.9% of the trades K1 approves reach their target, against 25.6% of those
  it rejects.

## Amendment 1, in hindsight

Committed before any real forecast, when 8 null datasets suggested the
week-clustered SE was too small. On all 20 the week SE was inside the chance
band (ratios 1.07 for K1 and K2 on raw R; band 0.68-1.31): the first 8
overstated the problem, and the two-way SE came out slightly conservative
(0.94 and 1.00). It changed no verdict.

## What would settle it

K1 is not ruled out at a material size. Confirming an effect as large as
this estimate, at a two-way SE of 0.162, with 80% power needs about 4,800
trades -- two and a half years of the replay's ~35 a week. Six months of new
trades would detect it about one time in four. The pipeline exists and runs
in ten minutes, so the same K1 test on trades from bars after 2026-10-27 is
queued at the bottom, as a long shot.

Scope: Kronos-small as frozen in the pre-registration, reading this
strategy's trades. Nothing here tests a larger Kronos, or a use other than a
trade filter.

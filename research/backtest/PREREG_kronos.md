# Pre-registration: Kronos as a reader of the band strategy's trades

Committed before any Kronos forecast has been computed on real data.

What had been looked at before this was written: the hold-time distribution
of the live strategy's trades on the snapshot below (a property of the
strategy, not of Kronos; it set the horizon); Kronos's speed and arithmetic
on random walks; and a smoke run of the pipeline on tiny random walks.

## Question

Does Kronos -- a foundation model pre-trained on candlesticks -- reading the
closed 1h candles up to a trade's decision bar, separate the live band
strategy's winning trades from its losing ones well enough to matter?

## The model, frozen

- Kronos-small (24.7M parameters) with Kronos-Tokenizer-base, from
  safetensors at Hugging Face revisions `901c26c1` and `0e011738`
  (research/kronos/vendor/PROVENANCE.md has full revisions and SHA-256s).
- Model code: upstream commit `67b630e`, vendored unmodified.
- Pre-training data ends June 2024 (the paper); the weights were last
  modified 2025-09-09. Every trade below enters between 2026-05-06 and
  2026-10-27, so none of the test period can be in the weights.
- Eval mode (dropout off). Sampling T=1.0, top_p=0.9, top_k=0 -- upstream's
  predict() defaults.
- Lookback L = 464 closed bars ending AT the decision bar; horizon H = 48
  bars; L + H = 512, the model's context, so the window never slides.
- S = 16 sampled paths per decision. Upstream's predict() averages its paths;
  here each path is kept (forecast.sample_paths).
- Sampling through research/kronos/fast.py, a key/value-cached copy of
  upstream's loop: same logits (|difference| < 1e-3 at every step,
  teacher-forced, on CPU and MPS) and identical paths for the same seed
  (research/kronos/test_fast.py).
- Decisions are sorted by (coin, decision bar) and forecast in batches of 8;
  batch b is seeded SEED + b (real) or SEED + 1000(k+1) + b (null dataset k),
  SEED = 20261103.

Why H = 48: the live strategy's median trade lasts 13 bars, 87% resolve
within 48, and a bid fills within 11 bars 90% of the time.

## Data

`research/backtest/cache/snapshot_1h_fresh.json`, the snapshot rr_floor used:
44 coins, fetched 2026-10-27, digests verified on load.

## Trades

The live strategy as `band.run` replays it at MIN_RR 2.0 with MEASURED costs
-- the run rr_floor's E_live bucket came from: 876 resolved trades (719 bids,
157 market entries). Each is read at its decision bar; nothing after that bar
reaches the model.

## The two readings (research/kronos/score.py)

Every sampled candle is made self-consistent (high = max of open, high,
close; low = min of open, low, close). A path with a non-finite or
non-positive price is dropped; a decision with fewer than 12 valid paths is
unscored.

- **K1 direction**: mean over paths of (close 24 bars after the decision) /
  (decision close) - 1. Approve iff > 0.
- **K2 expected R**: the trade's own order -- kind, stop, target, bid price,
  12-bar bid life -- run through the conservative engine on [decision bar] +
  each path. Net R if it resolves on the path; marked to the path's last
  close, paying the taker exit fee, if it filled and is still open; 0 if a
  bid never fills on the path. Mean over paths. Approve iff > 0.

If either side of the approve/reject split holds fewer than 50 trades, the
split is at the median score instead, which tests information only, and is
reported as such.

## Primary statistic, per hypothesis

D = mean net R of approved trades - mean net R of rejected trades, using the
trades' REAL outcomes. Week-clustered bootstrap by entry week (B = 10,000),
two-sided p. Holm across K1 and K2.

Anticipated weakness, which the null measures: K2 runs each trade's own
geometry through the engine, so it can score trades higher for their SHAPE
-- stop width against fees, distance to the bid -- rather than for anything
Kronos reads. On real outcomes that shape effect would look like edge.

## Null calibration, before real data

20 driftless random-walk datasets the size of the real one: 44 coins x 5,000
bars (common.random_walks, seeds 0-19, salt 31), each given volume that
follows its own bar's range and carries no direction. The identical pipeline
runs on each: band.run -> Kronos paths -> scores -> D.

A statistic is miscalibrated for a hypothesis if it fires on more than 2 of
20 datasets (fires: Holm-rejected across K1 and K2, D > 0, D > 0 in both
chronological halves) or if its mean across datasets is more than 2.5 SE
from zero.

Fallback, fixed now: D is also computed on EXCESS R -- each trade's net R
minus its same-order random controls (common.same_order_control), which
share its geometry and so remove a shape effect. Per hypothesis, the
statistic used is D on raw net R if that calibrated, else D on excess if
that calibrated, else the hypothesis is not testable by this design and is
reported so.

## Decision rules

- **Information**: calibrated statistic; Holm p < 0.05; D > 0; D > 0 in both
  chronological halves of the trades.
- **Material**: information and D >= +0.10R.
- **Worth wiring into live**: material, AND the approved trades' mean net R
  is positive (the filtered strategy makes money, not merely loses less).
  Even then only after the lab replicates it on bars collected after this
  run.

## Secondary (descriptive, no decisions)

Spearman correlation of each score with net R; approval rates; D by entry
kind; the 24h close's mean absolute error against a no-change forecast; net R
by score quintile; unscored count.

## Amendment 1 (2026-11-03, before any Kronos forecast on real data)

What prompted it: an interim read of the first 8 null datasets, from the
run's own cached scores (no real data has been read by Kronos). Every mean
sat near zero and K1 fired on 1 of 8, both inside the rules above. But D
swung between datasets more than its week-clustered SE allowed: the spread
of D across the 8 was 1.39x the root-mean-square SE for K1 on raw R, 1.53x
for K1 on excess, 1.21x and 1.27x for K2 (95% chance band for 8 datasets:
0.49-1.51). The cause, on the same 8: on a random walk a coin's drift lasts
for weeks, Kronos reads it, and the band's trades win or lose with it, so a
coin's trades are not independent from week to week. Clustering by coin
widens the SE (raw R: K1 0.077 -> 0.092, K2 0.084 -> 0.090). With an SE
that small, a p-value overstates the evidence and the test can pass a
fake edge. The registered rule (more than 2 of 20 fires) is too blunt to
catch a miss this size.

The change, which can only remove a verdict:

- For each D, the SE is also computed clustered by coin and by coin-week.
  These are combined with the week SE as Cameron, Gelbach & Miller (2011)
  combine one-way variances, and the result is never below either one-way
  SE. This is the two-way SE.
- `--spread` re-analyses the 20 null datasets from their cached scores.
  For each statistic, r = (spread of D across the 20) / (root-mean-square
  two-way SE).
- **Information** additionally requires Holm p < 0.05 across K1 and K2,
  computed on D / (two-way SE x max(1, r)). Material and worth-wiring
  follow from information as before.
- The real run refuses to start until `--spread` covers all 20 null
  datasets, built with the registered configuration.
- The report gives both verdicts: as first registered, and amended. The
  amended verdict decides.

Nothing else changes: the model, data, trades, readings, D, the null and its
miscalibration rule, the fallback to excess R, and the 0.10R materiality bar.

## Not modelled

- Portfolio rails (4 slots, 60 min spacing), as in band.py.
- The filter's knock-on effect on later trades: a refused trade frees its
  coin, and the strategy could take a different trade soon after. This test
  partitions the trades the strategy takes; it does not replay a gated
  strategy, which would need a forecast at every candidate bar.

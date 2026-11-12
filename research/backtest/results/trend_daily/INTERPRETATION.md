# Daily trend-following -- interpretation

`report.md` and `summary.json` are generated and are the record. This page
reads them, written afterwards and labelled as such.

**Verdict: edge, on both hypotheses, and every pre-registered forward-trial
check passes.** This is the first test in this repository to do so. It earns
a forward run at small size; it does not earn live capital at scale.

## The pre-registered test

| | P&L days | excess over rotated controls | z | p | Holm p | halves |
|---|---|---|---|---|---|---|
| T_all (10 coins, 2020-) | 2,507 | +12.6 bps a day | +3.4 | 0.001 | 0.002 | +13.3, +12.2 |
| T_majors (BTC, ETH, 2018-) | 3,237 | +15.2 bps a day | +3.7 | 0.001 | 0.002 | +19.1, +12.7 |

The real book beat every one of its 2,000 controls in both. The null
calibration passed first (1 of 20 fired; +0.80 +/- 0.72 bps). The engine was
re-implemented independently as a plain loop for T_majors and agrees to four
decimals (13.0384 bps a day).

## As a strategy, net of 9.32 bps a trade and funding on longs

- T_all: +38% a year at 33% volatility, Sharpe 1.17 (95% CI 0.40 to 1.90),
  worst drawdown -35.5%. About 840 small trades a year.
- It survives a full day's delay on every trade (Sharpe 1.14), 20 bps costs
  (1.01), and each lookback alone (14d 1.12, 28d 1.09, 56d 0.66). The
  neighbours 7d and 112d are weaker (0.60, 0.54): a plateau with edges.
- No losing year but 2025 (-1.4%). Every year since 2022 had a drawdown of
  21% to 34%.

## What to hold against it

- **It has decayed, as expected in advance.** Sharpe 2.1 and 2.8 in 2020-21;
  0.5, 1.0, 1.0, 0.1, 0.7 since. The second half alone is +12.2 bps of
  excess at p 0.10 -- positive, not separately established. Plan on the
  2022-2026 figures (Sharpe roughly 0.5 to 1.0), not the whole-period ones.
- **The money is the long leg.** Gross +44% a year long, +1% short on T_all
  (+48 / +8.5 on the majors). Shorts have roughly paid for themselves; what
  the rule earns is being long in rises and out of the way in falls.
- **Spot bars, perp venue.** Funding is a flat charge here; the real one
  spikes in exactly the rallies the book is long. Basis, liquidations and
  Hyperliquid's own fills are not in this test. That is what the forward run
  is for.
- **Ten survivors.** T_majors has no survivor selection and passes on its
  own, but the eight others were chosen with hindsight.
- **Published before.** Time-series momentum in crypto is in the literature;
  this confirms it on this cost model, it does not discover it.
- **Capital.** At this volatility the measured return is about $270 a year
  on $716. The edge is real in percent and small in dollars at this size.

## State at the last close (2026-11-11)

Short all ten (AVAX at one-third weight), about 0.97x gross. Nine of ten
are down over 56 days. The band strategy is long-only into this.

## What it would take

A daily pass at 00:00 UTC that computes the ten targets and trades the
differences through core/venue.py, with core/protect.py's stops replaced by
the book's own exits (there is no stop; the position is the signal) and
core/safety.py's drawdown rail kept. Paper first, beside the band strategy,
then small. Pre-register the forward run's own kill criteria before it
starts. The build is the user's call.

# Pre-registration: exits for the trend book

Written BEFORE any return, P&L or exit of any rule below is computed on the
archive. Only the integrity pass has touched it (row counts, gaps, bar
ordering, opens against the previous close), and its closes have been
matched byte for byte to H-TREND's archive. Nothing below changes after
results exist; a change made after the null calibration is a dated
amendment, with its reason, made before the real run. The code refuses the
real run unless the calibration passed against this file's exact SHA-256.

## Grounding

- 2026-11-15: the live trend book (research/trend/PREREG_live.md) is short
  all ten coins and about 11% up on its $350 after a sharp fall. The operator
  asked whether the 25% stop should be tighter, whether the stop should move
  to entry once a trade is well in profit ("ex. trade sits 50% profitable"),
  or whether to take partial profit -- and asked for a backtest on real,
  uninflated numbers.
- The live 25% stop is a disaster stop, reset at every daily pass to 25%
  beyond that day's mark (PREREG_live.md, deviation 1). It fires only on a
  25% move within one day; it is not a stop from entry. The tested strategy
  (H-TREND) has no stop and no profit target: a trade ends when the trend
  rule turns.
- "50% profitable" is read as the venue's return on margin. At the book's 5x
  margin setting that is a 10% price move. All distances below are price
  moves.
- Disclosed priors. (1) The question was asked during a winning trade.
  (2) Published trend-following work mostly finds that profit targets and
  tight stops lower returns, because the returns come from a few large
  moves. (3) This uses H-TREND's archive and window, so it is not a fresh
  test of the trend edge; it compares exits on that same history.
  (4) H-TREND's book had a -35.5% worst drawdown, so K1 would have fired in
  it. Expectation, stated in advance: none of the four beats the live rule;
  the tight stop and the profit target do worse.

## Data

H-TREND's archive: Binance spot daily klines (UTC days) for BTC, ETH, BNB,
SOL, XRP, ADA, DOGE, AVAX, LINK and DOT, from github.com/finom/static-klines
at commit 2a160c07c975d4e569bd5a264a9351073e6778db, now with each day's open,
high and low beside its close. Every coin's closes, serialised as H-TREND
serialised them, reproduce H-TREND's SHA-256 (results/trend_daily/
summary.json); the build refuses otherwise. Integrity pass: no gaps, no bar
whose high is below its open or close or whose low is above them, nothing
non-positive; 13 opens more than 1% from the previous close (largest: DOGE
641 bps, BNB 448 bps), which the stop rule fills at the open.

P&L days 2020-01-01 to 2026-11-11, as H-TREND's T_all. Halves split at the
middle day.

Spot, not perp. Hyperliquid triggers a stop on its mark price and fills a
limit order on its own trades; neither is Binance's traded high or low. The
fill rules below lean against every exit for that reason, and the harsh
check leans harder.

## The book, unchanged

H-TREND's T_all rule -- trend, volatility, target weight, the 25% band --
computed by the same functions the live book runs. What differs from
H-TREND's own simulation, for every rule alike:

- Positions are held in units between passes, so a weight drifts with price
  and the band reads the drifted weight, as the live book reads it from the
  venue. Capital is fixed and not compounded.
- Each daily pass trades at that day's close and pays 9.32 bps per unit of
  notional traded, as in H-TREND. Long notional pays 3 bps a day of funding;
  shorts receive nothing.

## Fills inside a day

- A stop fills at its level, or at the day's open if the open is already
  beyond it, then 25 bps worse, plus the 4.32 bps taker fee.
- A take-profit fills at its level only if the day traded at least 5 bps
  through it, never better than its level, plus 4.32 bps.
- A day whose range reaches both a stop and a take-profit is taken as the
  stop, for the whole position.
- A take-profit level already reached at a pass is taken at the pass price,
  as a pass trade.

## The rules

"Entry" is the average entry price, as the venue shows it: it moves when the
position is added to, not when it is reduced. A trade starts when a coin's
position opens or flips sign and ends when it is flat.

**BASE**, the live rule: a stop 25% beyond the pass's mark, reset every pass;
after it fills, the next pass rebuilds the position if the rule still holds.

The four hypotheses, each one change to BASE:

- **DS10**: the daily stop 10% beyond the mark instead of 25%. Everything
  else as BASE, including the rebuild at the next pass.
- **SL10**: a stop-loss 10% beyond the average entry, beside the 25% daily
  stop (the nearer of the two rests). After it fills, the coin stays flat
  until its target changes sign.
- **BE10**: once a pass finds the trade 10% in profit, its stop moves to the
  average entry for the rest of the trade (the 25% daily stop still rests if
  it is nearer). After that stop fills, the coin stays flat until its target
  changes sign.
- **TP10**: a resting order takes half the position off at 10% profit, once
  a trade. After it fills, the position may be reduced by the rule but not
  added to until its target changes sign.

Neighbours, used only by the plateau check: DS05 and DS15; SL05 and SL20;
BE05 and BE20; TP05 and TP20. NOSTOP (H-TREND's rule with no stop at all) is
reported beside BASE and decides nothing.

## The statistic and the test

For each hypothesis, d = its daily net return minus BASE's, on the same days.
Statistic: the mean of d. p: two-sided, from a circular block bootstrap of d
centred on zero (28-day blocks, B = 10,000, seed 20261115), with the +1
correction. Holm across the four, alpha 0.05.

## Decision rule

A hypothesis is **better** only with all five:
1. Holm-adjusted p < 0.05 and a positive mean d;
2. a positive mean d in both halves;
3. a worst drawdown (capital fixed: the book's value against its peak) no
   deeper than BASE's;
4. both neighbours with a positive mean d -- a plateau, not a peak;
5. a positive mean d when every stop fills 50 bps beyond its level and a
   take-profit needs 20 bps of trade-through, BASE recomputed the same way.

Holm-adjusted p < 0.05 with a negative mean d is **worse**. Anything else is
**not established**, and BASE stands.

Nothing goes live from this test. A "better" earns a dated amendment to
PREREG_live.md, which starts a new live run.

## Null calibration, before the real run

Twenty synthetic archives from the real one, as H-TREND's calibration but for
whole bars: each day, all coins together, is kept or mirrored about the
previous close by one random sign (the high becomes the mirrored low and the
low the mirrored high), then each coin's own mean daily log return is added
back to the whole bar. Trend is gone; drift, fat tails, volatility clustering,
cross-coin correlation and each day's range survive. Targets come from each
archive's own closes. The four tests run on each, seeds 20261116 (archives)
and 20261215 + k (bootstrap). Pass: at most 2 of the 20 archives with any
hypothesis Holm-adjusted p < 0.05 and a positive mean d. Each hypothesis's
mean d across the nulls is reported as the mechanical effect of the rule --
costs, fill assumptions, exposure to drift -- and decides nothing. A failure
is fixed by amendment before any real result is read.

## Checks on the machinery

The engine is vectorised; a plain loop written separately must give the same
daily net on the real days for BASE and the four hypotheses (to 1e-12), or
the run stops. Any non-finite daily net or difference stops the run.

## Reported beside the verdict (descriptive, deciding nothing)

Each rule's annual return on capital, volatility, Sharpe, worst drawdown and
whether K1 (30% below peak) would have fired; stops and take-profits a year;
cost and funding; the difference in dollars a year at the live book's $350;
P&L by year; NOSTOP beside BASE; and H-TREND's own simulation of the same days.

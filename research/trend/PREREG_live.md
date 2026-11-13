# Pre-registration: the trend book, live at small size

Written and committed BEFORE the first live pass. Nothing below changes once
the run has started; a change is a dated amendment, with its reason, and it
starts a new run.

## Why live, and what it can and cannot show

On 2026-11-13 the operator chose to run the book live at small size rather
than 90 days on paper first: fills, delays and funding on a real account are
what the paper run could only model. The paper run (PREREG_forward.md) keeps
running beside it, as the model this run is measured against.

Ninety days cannot establish the edge either way. In the backtest
(research/backtest/results/trend_daily), 35% of all 90-day windows lost money,
the 2.5th percentile was -18.2% and the worst -28.1%. A losing quarter is
ordinary, and a winning one proves nothing. What this run measures is what the
backtest assumed: that the book can be traded once a day on this venue, what a
trade really costs, and what funding really charges.

## The book, unchanged

H-TREND's T_all rule (PREREG_trend_daily.md, "The strategy, defined"),
computed by `research.trend.paper.decide` and `rebalance` -- the functions the
paper run uses, and the same `features` the backtest ran -- on Hyperliquid's
own daily perp candles for BTC, ETH, BNB, SOL, XRP, ADA, DOGE, AVAX, LINK and
DOT. No parameter is touched.

## How it trades (tools/trend_live.py)

- **Capital** is `TREND_CAPITAL_USD`, fixed for the run and not compounded.
  Weights are fractions of it. It may not exceed account equity at the first
  pass.
- **Once per UTC day**, on the first hourly run after 00:00 UTC that can read
  the venue. Held weights are read from the venue (signed notional over the
  capital), the band rule decides which coins trade, and each trade is one IOC
  through `core.execute`: every close and reduction first, then opens and adds.
  Cross margin at a 5x setting, which sets margin, not exposure.
- **The venue minimum.** A trade under $11 of notional is skipped and recorded
  (the venue refuses orders under $10). A close is never skipped.
- **Shared account.** While the book is armed the band trader does not trade
  its ten coins (core/books.py). A coin that still carries a band plan or a
  band bid is left to the band book until that trade is over.

## Deviations from the backtest, fixed in advance

1. **A disaster stop.** A reduce-only stop 25% beyond each position's mark,
   reset at every pass. The tested strategy has no stop. This one is there
   because the host has been down for days at a time, and it should fire only
   on a move that size between two passes. If it fires, the next pass rebuilds
   the position if the rule still holds it.
2. **The account's rails stay in force, for the whole account:** the 5% daily
   loss stop (an automatic halt until the next UTC day) and the 10% drawdown
   over 24 hours (a halt until the operator resumes). Both close every position,
   this book's included. The book does not trade while a halt is in force and
   rebuilds on its first pass after one clears.

## Kill, automatic

**K1: the book's value falls 30% below its peak.** Value is the capital, plus
closed P&L less fees on the ten coins, plus their funding, plus their open P&L,
all read from the venue's own fills, funding and positions. The pass closes the
ten coins one at a time and stops. It stays stopped until a new run is
pre-registered.

## Stop for review, by alert

- **S3**, as in PREREG_forward.md: the venue stops serving a coin's daily
  candles for 3 days.
- The watchdog reports the pass silent for three hours of the Mac being awake.
- A disaster stop that cannot be confirmed resting.

## Review on day 90

All four together earn a decision to change the size. None of them is evidence
of edge.

| # | Criterion | Why this number |
|---|---|---|
| R1 | At least 80 of the 90 days have a ledger row | A daily book that cannot be run daily is not this strategy |
| R2 | Mean cost per trade <= 20 bps: the fill against the decision close, plus fees, weighted by notional | The cost level the backtest was shown to survive |
| R3 | Funding paid, annualised, <= 9% of the capital | Three times the backtest's 2.9% a year |
| R4 | Live minus model >= -5 bps a day | The model is paper.replay on the venue's own daily closes for the same days |

`python -m tools.trend_live --status` reports all of these.

## Fixed

- No parameter of the book changes during the run: the lookbacks, the
  volatility target, the band, the 25% stop, the $11 minimum and the capital.
- The ledger (research/trend/data/live/ledger.jsonl) is appended, never edited.
- This run changes nothing about the band strategy's own rules or EDGE_PLAN.md's
  kill criteria, which stand on their own dates.

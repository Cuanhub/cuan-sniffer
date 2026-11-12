# Pre-registration: the trend book's paper forward run on Hyperliquid perps

Written and committed BEFORE the first paper pass. Nothing below changes
once the run has started; a change is an amendment, dated, with its reason.

## What this run is for, and what it cannot do

H-TREND (research/backtest/results/trend_daily) found a timing edge on
Binance SPOT daily bars. This run does not re-test that edge: ninety days
cannot. On the backtest itself, 35% of all 90-day windows lost money, the
2.5th percentile was -18.2% and the worst -28.1%. A losing ninety days here
would be ordinary; a winning one would prove nothing.

What ninety days CAN measure is everything the backtest assumed about this
venue: that the book can be read and rebalanced once a day, what the delay
between the daily close and the fill costs, and what funding really charges
compared with the flat 3 bps a day on longs that the backtest used.

## The book, unchanged

Exactly H-TREND's T_all rule (PREREG_trend_daily.md, "The strategy,
defined"), computed by the same `features` function the backtest used, on
Hyperliquid's own daily perp candles: BTC, ETH, BNB, SOL, XRP, ADA, DOGE,
AVAX, LINK, DOT. No parameter is touched. No order is placed: the package
is import-isolated from every order-capable module.

## The paper pass (research/trend/paper.py)

Runs hourly; acts once per UTC day, on the first run after 00:00 UTC that
can read all ten coins' closed daily bars, all ten mids and the funding
history. If anything is unreadable it changes nothing and tries again next
hour -- an unreadable venue is not a flat one.

- Held weights are marked from the last pass's mids to this pass's mids.
- Funding is the venue's own hourly rates since the last pass, on the held
  weights: longs pay a positive rate, shorts receive it.
- Targets come from the closed daily bars; the band rule decides which
  coins trade; a trade fills at the current mid and pays 9.32 bps of the
  notional traded (4.32 measured taker fee + 5 assumed), as in the backtest.
- Recorded per trade, not charged: the adverse move from the decision close
  to the fill mid. Marking from mids already puts it in the P&L; recording
  it is what lets the backtest's cost assumption be checked.
- Paper equity starts at 1.0. Every pass appends one ledger row.

## Decision day

The 90th UTC day after the first pass. On that day `--status` is run and
read against the rules below, once.

## Go: all four, or it does not proceed to live size

| # | Criterion | Why this number |
|---|---|---|
| G1 | At least 80 of the 90 days have a ledger row | A daily book that cannot be run daily is not this strategy |
| G2 | Mean recorded adverse move per trade <= 10.68 bps | With the 9.32 charged that is 20 bps a trade, the level the backtest was shown to survive |
| G3 | Net funding paid, annualised, <= 9% of equity | Three times the backtest's 2.9% a year; beyond it the cost model is wrong |
| G4 | Paper minus model: mean daily difference >= -5 bps | The model is the backtest engine re-run on the venue's own daily closes for the same days. -5 bps a day is ~18% a year, about half the measured return. Daily tracking noise is large; this catches a gross mismatch, not a fine one |

A pass earns a separately designed live run at small size, with its own
kill criteria, as the operator's decision. It is not evidence of edge.

## Stop: any one ends the run early, for review

| # | Trigger | Why |
|---|---|---|
| S1 | Paper drawdown from its peak reaches 30% | Beyond the worst 90-day window in seven years of backtest (-28.1%) |
| S2 | Ten consecutive UTC days with no ledger row | The pass is not running |
| S3 | The venue stops serving daily candles for any of the ten for 3 days | The universe is no longer the one tested |

## Fixed

- No parameter of the book changes during the run. A change restarts it.
- The paper ledger is the record; rows are appended, never edited.
- This run changes nothing about the live band strategy, its rails, or
  EDGE_PLAN.md's kill criteria, which stand on their own dates.

# Pre-registration: daily trend-following, long and short, on a long history

Written and committed BEFORE any return, signal or P&L is computed on the
archive named below. Only an integrity pass has touched it (row counts, date
ranges, gaps, OHLC ordering). Nothing below changes after results exist; a
change made after the null calibration is an amendment, dated, with its
reason, and made before the real run.

## Grounding

- The operator asked (2026-11-12) for the system to be rebuilt around
  something with an edge, after comparing it with a four-strategy gold EA
  whose core is trend-following validated on 2020-2026.
- Six pre-registered tests say the band entry carries no information on 1h
  bars (graveyard.md). The predictability scan's only open price lead was at
  the weekly horizon, and it lacked power: on 200 weeks of Hyperliquid daily
  candles, each coin following its own 28-day sign was +56 bps a week
  (z <= 1.2) and the universe's own trend +49 to +55 (z <= 0.8). Queue item
  14 says a longer history than this venue's is needed. This is that test,
  for the time-series (own-trend) form.
- What is materially different from everything in the graveyard: the horizon
  (weeks, not hours), the instrument set (majors with 6-9 years of history),
  the form (always-in, volatility-scaled, both directions -- not an entry
  pattern with a stop and a target), and the null (below).
- Disclosed priors. (1) Published work finds time-series momentum in crypto
  through about 2021 and weaker after. (2) I know the broad price history of
  these coins (2020-21 bull, 2022 bear, 2023-25 recovery). (3) 2023-2026
  overlaps the scan's exploratory window, so that stretch is not untouched.
  (4) The ten coins are today's survivors. Expectation, stated in advance:
  positive over the whole period, carried by 2020-2022; the second half is
  where it is most likely to fail.
- Volatility persists on these coins (scan: rank correlation 0.72 day to
  day, p 0.001), which is the stated basis for volatility-scaled sizing.

## Data

Binance spot daily klines (UTC days) for BTC, ETH, BNB, SOL, XRP, ADA, DOGE,
AVAX, LINK, DOT against USDT, from the public mirror
github.com/finom/static-klines at commit 2a160c07c975d4e569bd5a264a9351073e6778db,
through 2026-11-11. Zero gaps and zero mis-ordered bars in the integrity
pass. The experiment records a SHA-256 of every coin's series. Spot, not
perp: the perp's funding is modelled as a cost (below), its basis is not.

## The strategy, defined

For coin i on day t (all from closes up to and including t):

1. trend s = the mean of sign(close_t / close_{t-L} - 1) over L = 14, 28, 56
   days; so s is one of -1, -1/3, +1/3, +1. Three lookbacks, equally
   weighted, so that no single lookback is a fitted number.
2. volatility v = the standard deviation of the last 30 daily log returns,
   annualised by sqrt(365).
3. target weight w* = s x (0.50 / v) / N, capped at +/- 2/N, where N is the
   number of coins eligible that day. A coin is eligible once it has 120
   days of history.
4. the held weight moves to w* only when w* has a different sign, is zero,
   or differs from the held weight by more than 25% of |w*|. Otherwise it
   stays (held weights are treated as constant between trades).
5. the day's P&L is the held weight after the day-t decision times the
   simple return from close t to close t+1, summed over coins, less costs.

Costs: every unit of traded notional pays 9.32 bps (this account's measured
4.32 bps taker fee plus 5 bps assumed slippage -- market-entry slippage has
never been measured here). Long notional pays 0.03% a day of funding (the
formula's constant, ~11% a year; the scan measured a mean nearer 6.7%);
short notional receives nothing.

## Hypotheses -- two, Holm across them, alpha 0.05, two-sided

- **T_all**: the ten coins, P&L days 2020-01-01 to 2026-11-11.
- **T_majors**: BTC and ETH only, P&L days 2018-01-01 to 2026-11-11. These
  two were the two largest throughout, so they carry no survivor selection.

## Edge, defined: timing, not drift

These coins rose. A book that is long more often than short earns that
drift with no skill. So the statistic is the strategy's mean daily net
return MINUS the mean over 2,000 control books, each built by rotating the
whole trend matrix s (all coins together) in time by one random offset of
at least 90 days, wrapping around, and running steps 2-5 unchanged. A
control has the same mix of long and short, the same persistence of
position, the same volatility scaling and costs, and the same market -- only
its alignment with the returns that follow is broken. Seed 20261112.

p = the two-sided rank of the real book among its controls,
2 x min(share >=, share <=), with the +1 correction.

## Decision rule

**Edge** needs all three, per hypothesis: Holm-adjusted p < 0.05; a positive
excess; and a positive excess in BOTH halves of its P&L days (split at the
middle day, controls recomputed within each half).

**Worth a forward trial** needs edge on T_all AND all of:
- net annualised Sharpe >= 0.7 on T_all, with the lower bound of its 95% CI
  above 0 (block bootstrap, 28-day blocks, B = 10,000, seed 20261112);
- net P&L positive in both halves in absolute terms;
- each lookback alone (14, 28, 56) net positive -- a plateau, not a peak;
- net mean still positive when every trade is delayed a full day, and when
  costs are 20 bps per unit traded.

Anything less: no edge, a graveyard row, nothing changes. Either way nothing
goes live from this test; a pass earns a separately designed forward run at
small size on Hyperliquid perps.

## Null calibration, before the real run

Twenty synthetic archives from the real one: every day's returns, all coins
together, get one random sign (fat tails, volatility clustering and
cross-coin correlation survive; trend does not), then each coin's own mean
daily return is added back, so the drift is there and the trend is not. The
T_all test runs on each. Pass: at most 2 of 20 with raw p < 0.05, and the
mean excess within 2 standard errors of zero. A failure is fixed by
amendment before any real result is read.

## Reported beside the verdict (descriptive, deciding nothing)

Net and gross annual return, volatility, Sharpe, worst drawdown, turnover
and trades a year; a year-by-year table; long and short legs apart; each
coin; lookbacks 7 and 112 as neighbours; shorts credited the funding longs
pay; and the capital check -- what the measured return is in dollars a year
on this account's equity.

## Amendment 1 (2026-11-12, before the null calibration and before any real result)

The registration rotates "the whole trend matrix, all coins together, by one
random offset". Written as code that cannot be done literally: SOL, DOT and
AVAX become eligible about a year into T_all, so a whole-matrix rotation
would move their signals onto days they did not trade and leave them flat on
days they did -- the controls would hold those coins less than the real book
does and earn less of their drift, biasing the excess upward.

So each coin's trend is rotated by the SAME offset inside its OWN eligible
stretch of the window (wrapping there). The seven coins eligible from the
first day stay exactly aligned with one another; the three later ones stay
aligned to within their listing dates. Every control then holds each coin on
exactly the days the real book does, with the same long/short mix. Nothing
else changes. The null calibration below is run on this design.

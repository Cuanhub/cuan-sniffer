# Pre-registration: does a trendline breakout add edge to the band strategy?

Written and committed **before** the data for this test was fetched and before
any result was computed. The git commit that adds this file is the proof; the
results commit must come after it. Nothing below may be changed after results
exist. A changed definition is a new experiment with a new file.

This project has already been fooled four times by features that looked good
until controlled: killzone timing, the HTF bias proxy, sweep confirmation and
`pull_over_run` all reversed or vanished under control. This test is designed
around those failures.

## Question

Would adding a trendline-breakout feature to the live band strategy improve
its net expectancy per trade? It is tested three ways, because "adding" a
breakout can mean three different things.

## Data

- Hyperliquid mainnet 1h candles from the public `candleSnapshot` info
  endpoint, fetched once by `research/backtest/data.py`.
- Universe: the live universe at fetch time, from
  `research.levels.universe.select_universe` (top 45 perps by 24h notional
  volume, after the standing exclusions).
- Up to 5000 **closed** bars per coin (the venue's ceiling, about 208 days).
  The forming bar is dropped.
- A coin with fewer than 1000 closed bars, or with any duplicate timestamp or
  OHLC-inconsistent bar, is excluded. **Nothing is repaired or filled.**
- Every coin's bars are SHA-256 digested into the manifest saved with the
  results.
- Known bias, accepted: the universe is today's top 45, so coins that fell
  out of the top 45 during the period are missing (survivorship). It affects
  every arm of every comparison alike.

## Costs: measured from this account, not assumed (`costs.py`)

| | value | source |
|---|---|---|
| maker fee | 0.0144% | 28 real maker fills |
| taker fee | 0.0432% | 333 real taker fills |
| stop slippage | 0.1193% | mean adverse, 171 real Stop Market exits |
| take-profit slippage | 0 | measured −0.041% (favourable), counted as zero |
| market-entry slippage | 0 | **unmeasurable** (no band market entry has ever filled); sensitivity at 0.0564% |

## Baseline: the production strategy, replayed (`band.py`)

At each 1h bar close, `zones.plans_for` on the trailing 1200 closed bars, and
`zone_pass.first_tradeable(plans, MIN_RR=2.0)` -- the function the live pass
calls. In the band: market entry at the close. Above it: a bid at the band
entry resting 12 bars. One position or bid per coin. The 6h post-loss
cooldown on market entries, as live applies it today. Fills and exits by the
conservative rules in `engine.py` (a bar touching stop and target is a loss;
no target on a bid's fill bar; gaps exit at the open).

Not modelled: the portfolio rails (4 concurrent, 20/day, 60 min spacing) and
the 3-loss stand-down. They select by timing, not by any feature here.

## The trendline (`trendline.py`), primary parameters

| parameter | value | why this value |
|---|---|---|
| swing-high strength `k` | 3 | `zones.ZONE_PIVOT_WINDOW`; not a new knob |
| `lookback` | 120 bars | five days of 1h bars |
| `min_span` A to B | 6 bars | 2k |
| `min_touches` | 2 | the two anchors; a third touch is a robustness variant |
| touch tolerance | 0.25 ATR | |
| break buffer | 0 | a close above the line; 0.25 ATR is a variant |

At the close of bar t, using only swing highs **confirmed** by t (index + k <= t)
within the lookback: A is the highest; B is the swing high after A giving the
least negative slope (the upper envelope). Rejected if not descending, if B is
under 6 bars after A, or if any close between A and B is above the line. A
**break** is the first close above the line after B.

ATR is `zones.atr_pct`'s definition exactly (mean true range, 14 bars).

## Hypotheses

Each has ONE primary statistic. Everything else is secondary.

### H1 -- filter

Band trades whose decision bar has a trendline break within the previous 24
bars (inclusive) are compared with band trades without one.

- **Primary statistic:** mean net R (with break) minus mean net R (without).
- **Must also hold, or the result is not edge:**
  1. the same sign after a matched-random control: each trade's net R minus
     the mean of 200 random market entries on the same coin within +/-240
     bars, with the same risk %, reward % and entry fee type (removes drift
     and regime);
  2. the same sign in both stop/ATR strata (below and above the median
     `stop_atr_x`), because stop/ATR is the strongest effect yet measured in
     this strategy (t = +9.75).
- **Secondary:** the strategy fully re-simulated with the filter applied.

### H2 -- confirmation entry

For every baseline setup that reached its entry (a bid that filled, or price
in the band at the decision), do not enter. Instead, from that bar, wait up
to 48 bars for a trendline break. Enter at the next bar's open, with the SAME
stop and SAME target. No trade if a bar's low reaches the stop first, if no
break comes within 48 bars, or if the open is not strictly between stop and
target.

- **Primary statistic:** the paired mean, over setups, of
  (trendline-arm net R, or 0 if no trade) minus (random-event-arm net R, or
  0 if no trade). The random-event arm is the identical procedure with the
  coin's break series circularly shifted by a random offset -- same number
  of events, same clustering, no relation to price -- averaged over 100
  offsets per coin. Only what the trendline knows differs between the arms.
- **Secondary:** (trendline arm) minus (baseline), and (random-event arm)
  minus (baseline), which together show how much of any improvement is
  abstention rather than information; the rr >= 2.0 floor re-applied at the
  confirmation entry; per-trade means; the outcome breakdown; market-entry
  slippage at 0.0564%.

**Why the primary is not "(confirmation) minus (baseline)".** That was the
first draft. A pilot on synthetic random walks -- bars with no market
information in them, run before any real data was fetched -- declared it
"adds edge" at p < 0.0001. A losing baseline trade is one where price fell
to the stop, and that path almost never breaks a descending trendline on the
way, so waiting for a break skips almost every loser and scores it 0R.
Whenever the baseline's expectancy is negative, from costs alone even on
noise, any rule that trades less looks like an improvement. That statistic
measured abstention, not trendline information.

### H3 -- standalone

Every trendline break is a long entry at the next open, stop 1.5 ATR(14) below
the signal close, target 2R from it. One position per coin. Same period as
the baseline (from bar 800 of each coin).

- **Primary statistic:** mean of (trade net R minus the mean net R of 200
  random entries on the same coin within +/-240 bars, same rules).
- **Secondary:** the absolute mean net R of the breakout trades.

## Null calibration (run before the real data)

The whole experiment -- all three framings, the decision rule exactly as
below -- is first run on 20 synthetic datasets of driftless random walks
(12 coins x 5000 bars each, hourly volatility drawn per coin from 0.6% to
1.6%). They contain no market information, so every "adds edge" verdict on
them is a false positive. The observed false-positive rate per framing is
reported with the real results. A framing whose decision rule fires on more
than 2 of the 20 null datasets is reported as miscalibrated, and its real
result is not trusted as edge.

The engine itself was calibrated the same way: on 40 random walks the band
strategy's 1,515 trades averaged -0.020R +/- 0.042 gross, win rate 27.7%
against a martingale expectation of 28.4%; fixed 3%/6% random entries
averaged +0.013R +/- 0.011, win rate 33.8% against 33.3%.

## Inference

- **Decision statistic:** a cluster bootstrap resampling whole ISO weeks
  (B = 10,000, seed 20261027), two-sided p from the bootstrap SE. Crypto
  coins move together, so trades in the same week are not independent. The
  iid Welch / one-sample t is reported beside it, never instead of it.
- **Multiple testing:** H1, H2, H3 are one family; Holm-Bonferroni at
  alpha = 0.05.
- **A framing "adds edge" only if ALL of:**
  1. its primary statistic is rejected by Holm;
  2. the effect is positive;
  3. it has the same sign in both halves of the period (split at the median
     entry time);
  4. for H1, its two must-hold controls above agree in sign;
  5. the null calibration did not mark that framing miscalibrated.
- **Smallest effect worth acting on:** +0.10R per trade. A significant effect
  below that is reported as real but not worth the complexity.
- An effect that passes but leaves the arm with negative absolute expectancy
  is reported as "improves a losing strategy", not as edge.

## Robustness (reported in full, never used to choose)

- H1: k in {2, 3, 5} x lookback in {72, 120, 240} x window in {12, 24, 48}
- H2: k x lookback x max wait in {24, 48, 96}
- H3: k x lookback
- plus, for each: break buffer 0.25 ATR; min_touches 3; and (H1, H3) a 4h
  trendline built from complete groups of real 1h bars.

Reported as the share of variants with the primary's sign and the count with
uncorrected p < 0.05 -- out of N variants, about 0.05 x N would reach that by
chance alone.

## Amendment 1 -- 2026-10-27, after the null calibration, before any real data

**Change:** every random control (H1's matched control, H3's control) draws
its entry bars from the 240 bars AFTER the trade's signal or entry bar,
not from +/-240 bars around it.

**Why, from the pre-registered null calibration:** on random walks the H3
excess averaged -0.027R +/- 0.008 across 20 datasets, where a martingale
says zero. Decomposed on 16 more null datasets (gross, no costs):

| | mean R |
|---|---|
| every bar as an entry | +0.026 +/- 0.002 |
| every breakout bar | +0.017 +/- 0.010 |
| H3 trades, one at a time | +0.015 +/- 0.011 |
| their +/-240 controls | **+0.047 +/- 0.002** |

The control draws represent their windows faithfully (difference -0.003 +/-
0.002), so the window was the problem: a breakout at t exists because price
rose INTO t, and control entries placed before t ride that same rise. The
control was being handed the move that defines the signal. (The +0.026 that
every entry earns on a martingale is a separate, uniform effect: exits are
booked exactly at the stop or target while a discrete path overshoots
between samples; theory predicts +0.024 at these step sizes. It applies to
both arms alike and cancels in every comparison here.)

The same flaw would have biased H1's "matched control must agree" criterion
against the filter. Draws after the signal cannot overlap what defines it:
unbiased under the null, and conservative (attenuated) if an effect is real.
The null calibration is re-run with this design and must pass again before
real data is fetched.

**Added calibration check:** the false-positive count alone let a -0.027R
bias through (it fired 0/20 because the bias pointed away from "adds edge").
So each framing's mean estimate across the 20 null datasets must also sit
within 2.5 standard errors of zero; a framing that fails this is reported as
miscalibrated, exactly as for too many false positives. Nothing else changes.

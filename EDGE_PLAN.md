# Edge Plan — what the entry gate claims, and what it would take to know

**Date:** 2026-10-17 · **Status:** live trading DISARMED · equity $720.17
**Sample so far:** 109 closed trades, 2026-09-20 → 2026-10-17

This document exists because the audit's top finding was not a bug. It was
that the system has no demonstrated edge and no agreed way to find out. It
contains no code and proposes none.

---

## 1. The hypothesis

**What the gate asserts.** A horizontal price level that (a) formed from a run
of ≥6σ followed by a pullback of ≥1σ, (b) has been retested at least once,
(c) is approached from above with price within 12σ, and (d) has resistance
mapped far enough above to give reward/risk ≥2.0, will hold more often than
chance when price returns to it — enough that buying it with a structural stop
beneath the structure and a target at a mapped resistance rung produces
positive expectancy in R, net of fees and slippage.

Stated as a testable null:

> **H₀:** expectancy per trade = 0R
> **H₁:** expectancy per trade > 0R
> Unit: R = realised PnL ÷ the risk recorded at entry.

**What has been added on top**, each of which is a sub-hypothesis with its own
evidence and its own doubt:

| Rail | Claim | Evidence | Status |
|---|---|---|---|
| Perp basis ≤ 0 | buying support while longs pay a premium is buying a crowded trade | p=0.0002 at n=439, monotone quartiles, both halves agree | strongest |
| pull/run ≤ 0.60 | a level that has given back most of its run is distribution, not a dip | p=0.0000 on P(hit target) at n=492, day- and rr-controlled | simulated only |
| 3 losses → 24h stand-down | losses arrive in day-sized clumps | p=0.0002, both halves agree | fitted (best of 9) |
| 60-min entry spacing | clustered entries are correlated bets | 45% win alone vs 12% clustered | shipped, held |

**What the system already says about itself.** The trader prints this on every
start:

> `NOTE: this gate measured -0.0413R vs random entry (z=-0.93, n=688 backfill trades).`

That is the gate failing to beat a coin flip on its own backfill. The rails
above were all added after that measurement; none of them has been shown to
turn it positive.

---

## 2. How many trades it would take to know

Measured from the 109 real closes, in R:

```
n = 109      mean = -0.0891R      stdev = 2.0608R
median -1.077R      min -2.32R      max +8.49R
standard error = 0.1974R
95% CI on current expectancy = [-0.476R, +0.298R]
```

**The confidence interval is the headline.** After 109 live trades the data
cannot distinguish "loses half a unit of risk per trade" from "makes a third
of one". Nothing about the current sample constrains the answer.

σ = 2.06R is very large relative to the effect being sought, and that is
structural, not bad luck: losses cluster at −1R while wins run to +8.5R. A
2.46:1 payoff with a 25% hit rate *is* a high-variance distribution.

Sample size for a one-sided test at α=0.05:

| Effect to detect | 80% power | 90% power | At 4 trades/day | At 1.6/day* |
|---|---|---|---|---|
| +0.05R | 13,334 | 17,851 | 9.1 years | 23 years |
| **+0.10R** | **3,333** | **4,463** | **2.3 years** | **5.7 years** |
| +0.15R | 1,482 | 1,983 | 1.0 year | 2.5 years |
| +0.20R | 833 | 1,116 | 7 months | 1.4 years |
| +0.30R | 370 | 496 | 3 months | 7.7 months |

\* 1.6/day is the rate the 3-loss/24h stand-down implies (it kept 41 of 98 in replay).

**The answer to the question asked: detecting +0.10R expectancy at 95%
confidence requires about 3,333 trades — roughly two and a half years at the
current rate, and over five with the stand-down active.**

### What follows from that

The experiment as currently designed **cannot answer its own question**. Three
honest responses, and only the third is a strategy:

1. **Accept a longer horizon.** Costs ~$3 of risk per trade × 3,333 = the
   whole account several times over if expectancy is even slightly negative.
   Not viable at this size.
2. **Only test for a large effect.** +0.30R is detectable in ~370 trades, about
   three months. But a +0.30R gate would be an exceptional edge; if it were
   there, 109 trades would likely have hinted at it. The CI's upper bound is
   +0.298R, so this is right at the edge of what is already nearly excluded.
3. **Attack the variance, not the sample.** σ=2.06R is the binding constraint.
   Anything that cuts σ cuts the required n *quadratically*. Capping winners at
   a fixed R, or sizing to equalise R across trades, would shrink σ far faster
   than waiting shrinks the standard error. **This is the only route that makes
   the question answerable at this account size, and it has not been tried.**

A fourth response deserves naming because it is what usually happens: keep
trading, keep tuning, never decide. That is what the kill criteria below are
for.

---

## 3. Pre-registered kill criteria

Fixed now, before the next trade. Any single trigger ends the experiment at
its stated size. These are deliberately about **ruling out**, not confirming,
because section 2 shows confirming a small edge is out of reach.

### Hard stops — end the experiment

| # | Trigger | Rationale |
|---|---|---|
| **K1** | Cumulative realised PnL ≤ **−$150** from 2026-10-17 (equity ≤ $570) | ~21% of the account. Beyond this the remaining capital cannot fund a meaningful sample at any size |
| **K2** | **300 further closed trades** (n≥409) with the 95% CI upper bound still **below +0.10R** | Futility: at that point +0.10R is excluded, and anything smaller is undetectable within this account's lifetime |
| **K3** | Any **10-trading-day window** with profit factor < 0.60 **and** ≥15 closes | Not noise. PF 0.60 over 15+ trades is a structurally losing configuration |
| **K4** | Two **account-drawdown halts** (10%/24h) within any 14-day window | The rails firing repeatedly means the configuration, not the market, is the problem |
| **K5** | **2026-12-31** reached without K6 being met | A date, so "collect more data" cannot run forever |

### Continuation criterion — the only thing that justifies scaling

| # | Trigger | Then |
|---|---|---|
| **K6** | ≥200 further closes (n≥309) **and** 95% CI lower bound **above 0R** **and** profit factor ≥1.20 | Edge is established at this size. Only then consider increasing `MARGIN_PCT_PER_TRADE` |

### Variance-reduction gate

| # | Trigger | Then |
|---|---|---|
| **K7** | By **n=209** (100 further closes), σ has not fallen below **1.5R** | Adopt response 3 from section 2 — fixed-R exits or R-equalised sizing — or accept that the question is unanswerable at this size and stop |

### Conditions of the experiment, fixed now

- **Size:** `MARGIN_PCT_PER_TRADE = 2.5%` (halved from 5% while unproven).
  Not to be raised except by K6.
- **No new admission rails** until n≥209. Every rail added mid-experiment
  resets the sample, and four have been added in the last five days.
- **`pull_over_run` is provisional.** Built on 492 *simulated* episodes in one
  falling regime; the 33 matched real trades were non-monotone. Re-screen
  against real fills at ≥100 closes carrying the field (K7's checkpoint).
- **Measurement is fixed:** R = realised PnL ÷ `risk_usd` at entry, from
  `live_trades.csv`, fees included. No re-definition mid-experiment.
- **Reporting:** the CI and profit factor get recomputed every 25 closes and
  checked against K1–K7. If that is not done, K5 applies.

---

## 4. Stated plainly

There is **no evidence of an edge**. There is evidence of competent risk
control around a strategy whose entry signal has never beaten random on its
own backfill, and which has lost money over 109 live trades at a profit factor
of 0.81.

The gap to breakeven is genuinely small — 28.9% win rate needed against 24.8%
achieved, about six more winners in 109 — and that is why this is worth another
bounded run rather than abandonment. But "small gap" and "edge" are different
claims, and only the first is supported.

The most valuable thing in this document is not a kill criterion. It is the
number **3,333**: the recognition that the current design cannot answer its own
question within the account's lifetime, and that reducing σ is the only lever
that changes it. Everything else is bookkeeping around that fact.

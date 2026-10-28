# Pre-registration: the reward:risk floor -- do refused bands carry edge?

Written and committed BEFORE the data for this test is fetched and before any
result is computed on it. Nothing below changes after results exist.

## Grounding

- The live recorder's first day, re-resolved conservatively (2026-10-27):
  bands the gate REFUSES (rr < 2.0) filled 37 times and won 33, +0.465R net
  per trade; bands it accepted filled 3 times and won 1. All 40 were detected
  within four hours of one market move: one observation, not evidence. This
  test asks the same question over ~30 weeks.
- Disclosed prior peeking: a scratch harness on 2026-10-26 reported MIN_RR
  1.0 +17.1R/491 trades, 1.5 +8.6R/265, 2.0 +25.2R/173. That harness
  resolved market entries with their own decision bar and selected
  plans[0]; its numbers are not relied on, and the hypotheses below are
  two-sided.
- The live configuration on the corrected engine (H-TRENDLINE baseline):
  -0.051R +/- 0.055 per trade over 907 trades.

## Data

A fresh snapshot fetched AFTER this commit, by the trendline test's rules:
the live universe (top 45 by 24h volume), up to 5000 closed 1h bars per coin,
forming bar dropped, nothing repaired, SHA-256 per coin. Fetched one coin per
10 seconds, so the live bot keeps its share of the venue's per-IP budget.

## Costs

`costs.MEASURED`, unchanged: maker 0.0144%, taker 0.0432%, stop slippage
0.1193%, take-profit slippage 0.

## Strategy runs

`band.run` with MIN_RR in {0.3, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0}; everything
else exactly as live (1.5 ATR stop, 12h bids, one per coin, the market-entry
cooldown).

## Edge, defined

A band trade's EXCESS = its net R minus the mean net R of 200 random market
entries on the same coin, drawn from the 240 bars AFTER its entry, with the
same risk %, reward % and entry fee type. The controls cannot see the signal
(trendline Amendment 1), and geometry, costs, drift and every rr-dependent
property of the engine are shared by trade and control, so they cancel.

## Hypotheses -- one Holm family, alpha 0.05, two-sided

| framing | trades | question |
|---|---|---|
| **E_live** | the live config (MIN_RR 2.0); every trade has rr >= 2.0 | does the live band selection beat random timing at its own geometry? |
| **E_mid** | the MIN_RR 1.0 run, trades with rr in [1.0, 2.0) | do the bands a floor of 1.0 would ADD carry edge? |
| **E_low** | the MIN_RR 0.3 run, trades with rr in [0.3, 1.0) | do the close-target bands (most of the first day's "wins") carry edge? |

Statistic for each: the mean excess, by a week-clustered bootstrap
(B = 10,000, seed 20261027).

## Decision rule

A bucket **has edge** only if ALL of: Holm-rejected; estimate positive;
positive in both halves (split at the median entry time); null calibration
fires on at most 2 of 20 datasets AND its null mean is within 2.5 SE of zero.
Material only at >= +0.10R per trade.

**A change to the live floor is recommended only if** the bucket it adds has
edge AND is material AND that bucket's absolute mean net R is positive AND
the configuration's total net R beats the live one (secondary below) -- and
even then it goes live only after the same test passes on the first 45 days
of bars that did not exist at fetch time.

## Secondary (descriptive, never decided on)

- Per configuration: trades, win rate, mean and total net R, mean weekly net
  R, and the paired weekly difference against live with a week bootstrap CI.
- Per bucket: absolute mean net R and the control mean (what drift paid).
- Each bucket by entry kind (bid vs in-band market).
- The rr >= 3.0 bucket's excess (would RAISING the floor help?).
- Each primary in stop/ATR strata (below / at-or-above the median), because
  stop distance in ATR multiples is the strongest effect known here
  (t = +9.75) and rr correlates with it.

## Null calibration

As the trendline test, amended design: 20 random-walk datasets, 12 coins x
5000 bars. A framing that fires on more than 2, or whose mean is more than
2.5 SE from zero, is reported as miscalibrated and not trusted.

## What would make this wrong

- Stop/ATR confounding (the strata above).
- Survivorship: today's top 45.
- Portfolio rails not modelled: low floors take more trades than the account
  could hold at once (max 4), so their totals overstate what is achievable.
- One regime: the halves check.

## Amendment 1 -- 2026-10-28, after the null calibration, before any real data

**Change:** each trade's control uses the trade's OWN order type. A resting
bid gets random bids -- placed at random decision bars after the trade's
fill, at the same distance below that bar's close as the trade's bid sat
below its decision close, resting 12 bars, with the same stop and target
distances from the bid -- and only filled controls count. An in-band market
entry keeps random market entries.

**Why, from the pre-registered null calibration:** E_mid's null mean was
-0.042R +/- 0.008 and E_low's -0.021R +/- 0.008 (E_live -0.031 +/- 0.016,
inside the limit). The engine judges a bid's fill hour conservatively --
the stop counts there, the target does not, because the high may have come
before the dip that filled it -- while a market-entry control has no fill
hour at all. The closer the target, the more often the fill hour reaches it:
on 6 null datasets the rr 0.3-1 bucket lost 218 fill-hour targets this way.
With same-order-type controls the null excess went from -0.027 +/- 0.016 to
+0.004 +/- 0.016 (E_low) and, on 10 further datasets, -0.017 +/- 0.017 for
bids and +0.018 +/- 0.041 for market entries (E_mid). The calibration is
re-run in full with this design and must pass before the real data is
analysed. Nothing else changes.

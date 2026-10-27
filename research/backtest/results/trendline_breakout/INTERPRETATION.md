# Trendline breakout -- interpretation

`report.md` and `summary.json` are generated and are the record. This page
is the reading of them, written after the fact, and labelled as such.

**Verdict: a trendline breakout does not add edge to the band strategy in
any of the three pre-registered framings.** No primary statistic survives
Holm (family p-values 0.097, 0.097, 0.474), so by the rule fixed before the
data was fetched, none adds edge.

## As a filter (H1): it points the wrong way

Band trades whose decision came within 24 bars after a trendline break did
**worse**: -0.165R against +0.081R without one, a difference of -0.246R
(95% CI -0.473 to -0.022). The direction holds in both halves of the period,
in both stop/ATR strata, and after the matched-random control. Applied as a
filter, it would have turned -46.6R into -71.7R. This is not established as
a negative edge either (Holm p 0.097, and it was not the pre-registered
direction) -- "avoid band trades right after a breakout" would need its own
pre-registration and a holdout.

## As a confirmation entry (H2): small, consistent, below every bar

Waiting in the band for a break and entering on the next open beat waiting
for an event at a random time by +0.035R per setup (95% CI +0.001 to
+0.067; p 0.037 uncorrected, 0.097 after Holm). Positive in both halves and
in all 29 robustness variants (correlated variants -- they share the same 907
setups -- so not 29 independent confirmations). But it fails Holm, it is a
third of the +0.10R materiality bar, the confirmed trades themselves average
-0.005R, and against the CURRENT strategy the gain is +0.050R with a CI of
-0.118 to +0.204. Worth re-testing as the lab's archive grows; not worth
building now.

## Standalone (H3): loses money, no edge over random entry

3,942 breakout trades averaged -0.095R; random entries with the same stop
and target averaged -0.141R; the excess of +0.048R has a week-clustered p of
0.47. An iid test on the same numbers gives p = 0.04 -- breakouts fire across
many coins in the same weeks, and treating those trades as independent would
have reported an edge that is not there.

## The context that matters more than the trendline

The live strategy itself, replayed bar by bar with the live selection rule,
measured costs and conservative fills: **907 trades, -0.051R per trade
(+/- 0.055 SE; gross -0.023R), -46.6R total, over 2026-04-02 to 2026-10-27
on today's 45-coin universe.** That is indistinguishable from zero edge, with
costs tipping it negative. The portfolio rails (4 concurrent, spacing, the
stand-down) are not modelled, so this is not the account's P&L -- but it says
the band entry itself has no demonstrated edge on this window. The previous
session's +25.2R for the 1.5x ATR configuration came from a scratch harness
later found to resolve market entries on their own decision bar and to select
plans[0]; it should not be relied on. Re-measuring the stop configuration on
this engine is first in the lab's queue.

## What made the test trustworthy

- Pre-registered and committed (f81ed0b) before the data was fetched.
- A synthetic pilot caught a statistic that rewarded abstention (it "found
  edge" in pure noise at p < 0.0001); fixed before any real data.
- The null calibration caught a control window that let the control see the
  signal (-0.027R bias); amended (489da27) and re-calibrated (f8a891c) before
  any real data. Final calibration: 0/20 false positives, all null means ~0.
- Real Hyperliquid candles only (218,542 closed 1h bars, forming bars dropped,
  zero gaps), SHA-256 per coin in summary.json; costs measured from this
  account's own fills.

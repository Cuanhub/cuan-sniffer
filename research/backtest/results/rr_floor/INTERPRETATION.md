# Reward:risk floor -- interpretation

`report.md` and `summary.json` are generated and are the record. This page is
the reading of them, written afterwards and labelled as such.

**Verdict: no reward:risk bucket shows edge, and no floor makes the strategy
profitable. Lowering the floor would lose more; raising it loses less only
because it trades less.** By the pre-registered rule, nothing changes live.

## The bands the gate refuses do not carry edge

Edge was measured per trade against random orders of the trade's own type,
placed after it, with its geometry (Amendment 1, which the null calibration
forced: market-entry controls biased the test by -0.02 to -0.04R).

| bucket | trades | excess over random | 95% CI | p | absolute net R |
|---|---|---|---|---|---|
| live (rr >= 2) | 555 | +0.055 | -0.145 to +0.273 | 0.60 | -0.058 |
| refused, rr 1-2 | 1,283 | -0.060 | -0.201 to +0.084 | 0.41 | -0.098 |
| refused, rr 0.3-1 | 1,556 | -0.024 | -0.095 to +0.050 | 0.52 | -0.054 |

The recorder's first day -- refused bands winning 33 of 37 -- was one rally
across four hours. Over 30 weeks the same bands earn nothing beyond random
timing, and lose after costs.

## As strategies (descriptive; portfolio rails not modelled)

Every floor from 0.3 to 3.0 loses money on this window. MIN_RR 1.0 loses
-166.9R over 2,086 trades against -79.1R over 876 at the live 2.0. Higher
floors lose less (3.0: -5.8R over 356) because they take fewer trades of the
same quality: the rr >= 3 trades inside the live set show +0.002R of edge,
which is none. Win rate rises as the floor falls (49% at 0.3) while
expectancy does not: close targets win often and pay little.

## Caveats

- Coverage: 555 of the live floor's 876 trades had enough same-type controls
  to be scored (a deep bid's random twins often do not fill). The unscored
  trades average about -0.145R, so they are not hiding a profitable group.
- Composition: the universe rotated between this snapshot and this morning's
  (APT, ETHFI, MET, WLFI out; BCH, INIT, TRX in), and the live floor scored
  -0.090R here against -0.051R there. Both negative; neither should be read
  to two decimals.
- Descriptive only, not tested: bids in the lower buckets did worse than
  random bids (rr 1-2: -0.095, iid p 0.014); market entries did not. An
  idea for the lab's queue, not a finding.

## What it means

The question was whether a different reward:risk floor would find edge the
live gate is leaving on the table. It would not. Together with the trendline
test's baseline, the evidence now says the band entry itself has no
demonstrated edge on this window -- the floor is not what is wrong.

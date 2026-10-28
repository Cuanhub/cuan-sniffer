# 15-minute read -- interpretation

`report.md` and `summary.json` are generated and are the record. This page is
the reading of them, written afterwards and labelled as such.

**Verdict: a 15m read does not help the band strategy -- not as an entry
confirmation, and not as a measurement tool.** 52 days of 15m history (the
venue's limit) cannot rule out a material effect, so this is "no evidence",
not "proven useless".

## As an entry rule

Waiting inside the band for a 15m break of structure (the first 15m close
above the last confirmed 15m swing high) and entering on the next open:

| | confirmed trades | excess over later random entries, same geometry | 95% CI | p | per setup vs live |
|---|---|---|---|---|---|
| B1: band stop | 163 | -0.067R | -0.289 to +0.120 | 0.53 | -0.002R |
| B2: tighter 15m stop | 163 | -0.006R | -0.202 to +0.204 | 0.95 | -0.009R |

Neither beats random timing at the same geometry, both lose in absolute terms
(-0.047R and -0.057R per trade), and against the live strategy on the same
setups they change nothing.

## As a measurement tool

281 live-strategy trades replayed on 15m bars: mean net R -0.110 at 1h and
-0.110 at 15m, and not one outcome changed. With stops about 2.5% away and
targets about 5-6%, a single hour almost never reaches both, and only one
fill-hour target occurred. The conservative 1h engine already resolves this
strategy's trades correctly.

## Why the null calibration mattered here more than anywhere

It rejected two designs before any real data was analysed:

- per-setup (the trendline H2 design): abstention bias, because a real 15m
  break comes later than a random event after price drops into a band;
- unpaired per-trade means (Amendment 1): fired on 14 of 20 random-noise
  datasets for B1 and 11 of 20 for B2.

On this real data those rejected designs say B2 is +0.613R per trade at
p = 0.003 and B1 +0.121R at p = 0.029. Reported without the calibration, that
would have been the headline "the 15m read works" -- and it would have been
false: the same statistic finds it in pure noise.

## What would settle it

More 15m history. The lab now archives 15m bars daily; at six months the same
pre-registered test has roughly three times the trades.

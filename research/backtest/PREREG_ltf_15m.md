# Pre-registration: does a 15-minute read help the band strategy?

Written and committed BEFORE the 15m data is fetched and before any result
exists. Two questions, answered separately:

- **A (measurement):** how much do the 1h engine's conservative intrabar
  rules misstate outcomes, judged against 15m bars?
- **B (entry rule):** does waiting inside the band for a 15m break of
  structure -- the SMC "lower-timeframe confirmation" -- improve entries?

## Data

The fresh 1h snapshot fetched for PREREG_rr_floor, and a 15m snapshot of the
same coins fetched with it (same validation; one request per 10 seconds).
The venue serves at most 5000 15m bars: about 52 days, 2026-09-05 onward on
the day this was written. Setups are the live configuration's band trades on
the 1h snapshot whose arming moment lies inside the 15m window with at least
192 15m bars (48h) after it.

**Power, stated now:** 52 days gives roughly a quarter of the 1h test's
setups. At the trendline H2's per-setup SD (about 0.5R) the standard error is
about 0.033R, so a material +0.10R effect is detectable (~3 SE) and an effect
the size of H2's (+0.035R) is not. A null result here does not rule out small
effects.

## The 15m break of structure (`research/backtest/ltf.py`)

Swing highs and lows with k = 3 (the band pivot's own window), on 15m bars.
At 15m bar t the LEVEL is the high of the most recent swing high confirmed by
t (index + 3 <= t). A BREAK is the first 15m close above that level after it
was confirmed -- in a decline into a band, the first close above the last
lower high (a change of character). Causal by construction, and tested by
recomputing on a series cut at t.

## Arming: placing the 1h setup on 15m bars

- a bid that filled in 1h bar j: armed at the first 15m bar of hour j whose
  low reaches the bid; that bar's low after the fill can invalidate;
- an in-band market entry at the close of 1h bar i: armed at the first 15m
  bar after that close.

## B1 -- 15m confirmation, band stop (primary)

From the arm, for up to 192 15m bars: a low at or under the band stop first
means no trade; a break means entry at the next 15m open with the SAME band
stop and target, exits by the engine's conservative rules on 15m bars.

## B2 -- 15m confirmation, 15m stop (primary)

As B1, but the stop is the TIGHTER of the band stop and the last confirmed
15m swing low minus max(0.25%, 1.5 x ATR(14) on 15m) -- the lower-timeframe
refinement that tightens risk. Never wider than the band stop.

## Statistic for B1 and B2

**Primary -- entry quality, which abstention cannot score:** over setups
where the arm entered and its trade resolved, the mean of (that trade's net
R) minus (the mean net R of the random arm's ENTERED, resolved trades on the
same setup; at least 10 of its 100 replicates). The random arm is the
identical procedure, including B2's stop rule, with the coin's 15m break
series circularly shifted. Week-clustered bootstrap, B = 10,000, seed
20261027. Holm family {B1, B2}, alpha 0.05, two-sided.

**Why not the per-setup comparison used for the trendline's H2.** It was the
first draft here too. A pilot on synthetic random walks -- no market
information, run before this file was committed -- declared B1 "adds edge" at
p < 0.0001 (+0.156R per setup). After price drops into a band, a real 15m
break needs an up-move first, so it comes LATER than a random event: the
random arm entered on 72% of setups, the 15m arm on 57%. Whenever trades
lose on average -- from costs alone, even on noise -- the arm that enters
less looks better. At 1h the two rates were close enough that H2 calibrated
(null mean -0.022 +/- 0.029); at 15m they are not. The per-setup comparison
is kept, as a secondary, because it is the one that maps to profit -- but
only entry quality is decided on.

**Adds edge** only if: Holm-rejected; positive; positive in both halves;
null calibration fires on at most 2 of 20 and its mean is within 2.5 SE of
zero. Material at >= +0.10R per trade.

## A -- measurement (descriptive; not a hypothesis test)

For live-configuration trades whose whole life falls inside the 15m window:
the same order simulated on 15m bars against the 1h engine. Reported: the
mean R difference (15m minus 1h), the share of trades whose outcome changes,
and how the 1h engine's ambiguous bars (both stop and target touched) and
fill-hour targets resolve when seen at 15m.

## Secondary

The per-setup comparison (arm net R or 0, minus random arm net R or 0);
B1/B2 against the plain baseline (how much is abstention); entry rates;
median rr at entry; the confirmed trades' absolute mean net R; B2's stop in
1h-ATR multiples (tighter stops have lost before: the stop/ATR effect,
t = +9.75); market-entry slippage sensitivity at 0.0564%.

## Null calibration

20 datasets, each ONE random walk at 15m resolution -- 12 coins x 9,600 15m
bars with 24 sub-steps each, and 2,400 1h bars built by exact aggregation --
the 15m window being the last 5,000 15m bars. Same thresholds as above. (24
sub-steps rather than the 1h calibration's 12: a coarse synthetic path
overshoots a stop or target between samples while the engine books the exit
at the level, an artifact real continuous trading does not have. Finer paths
keep the calibration measuring the DESIGN.)

## What would make this wrong

- 52 days is one regime; each half is 26 days.
- Entering at the next 15m open pays taker fees and unmeasured entry
  slippage (the sensitivity above).
- B2 tightens stops, which this strategy has paid for before; its stop/ATR
  is reported so a "win" from tighter stops in one regime is visible.

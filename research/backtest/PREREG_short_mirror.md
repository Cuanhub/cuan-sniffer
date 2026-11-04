# Pre-registration: the band strategy's short-side mirror

Written and committed BEFORE any short trade is simulated on real bars.
Nothing below changes after results exist; any change made after the null
calibration is an amendment, dated, with its reason, and made before the real
run.

## Grounding

- The operator asked (2026-11-04) whether working both directions would be an
  upgrade. The standing decision of 2026-09-18 defers shorts until the long
  side clears 50% wins over 20+ trades; this test is offline, changes nothing
  live, and does not revisit that decision on its own.
- The long side on the same archive (H-RRFLOOR, E_live): +0.055R per trade
  over random entries with the same geometry (95% CI -0.145 to +0.273, p 0.60,
  n 555); absolute -0.090R per trade over 876 trades.
- Disclosed prior knowledge of these bars: the archive rose (BTC +20.6%,
  median coin +36% from 2026-04-02 to 2026-10-27), so short trades are
  expected to lose in absolute terms whatever their timing. No short trade,
  supply-band entry or short outcome has been computed on these bars.
- Disclosed prior from the vault audit (2026-11-04): the profitable vaults'
  shorts were hedges with a signal of their own, held for days to weeks -- not
  a mirror of a long pattern. Expectation, stated in advance: no edge, because
  the long side shows none.

## The mirror, defined

The live strategy, run unchanged on the price series turned upside down, and
traded on the real one:

1. each coin's bars are inverted: every price p becomes K/p with K = the first
   close squared (highs and lows swap; a fall becomes a rise);
2. `band.order_at` runs on the inverted bars exactly as live -- the same zone
   detector, recency, reach, 1.5x ATR stop, nearest-band target, MIN_RR 2.0,
   in-band market entry or a 12-bar resting order, one per coin, the 6h
   post-loss cooldown on market entries;
3. each order is mapped back (entry, stop, target -> K/x) and simulated as a
   SHORT on the real bars.

So the entries are `zones.supply_zones` bands (the base of an impulse that
closed below a prior swing low; the band runs from that candle's close to its
high), the stop sits above the band, and the target is the top edge of the
nearest demand band below. Percent thresholds act on the inverted prices --
"within 5% below" becomes "band low within 5% above price" -- which differs
from a hand-written mirror only in second-order terms. That is the definition.

## Engine

Shorts are simulated natively, every rule in engine.py mirrored: a resting
sell fills at its limit the first time a bar's HIGH reaches it (maker); a bar
reaching both stop and target is a loss; on the fill bar the stop counts and
the target does not (the low may have printed before the rally that filled
it); a bar that opens through the stop exits at that open; stop exits pay the
measured slippage on top, as a higher buy-back price. R is measured against
the planned stop distance from the actual fill, net of fees.

## Data and costs

`cache/snapshot_1h_fresh.json` -- the archive H-RRFLOOR and H-KRONOS ran on,
verified against its SHA-256 manifest. `costs.MEASURED` unchanged. Funding is
not modelled, as it is not for the long side.

## Edge, defined

A short trade's EXCESS = its net R minus the mean net R of random shorts of
its own order type with its own geometry (rr_floor Amendment 1, mirrored): at
200 random decision bars in the 240 after its entry, a resting sell the same
distance ABOVE that bar's close as the trade's sat above its own decision
close, resting 12 bars, with the trade's stop and target distances -- only
the ones that fill count; an in-band market entry gets random market shorts.
At least 50 must resolve or the trade has no excess. Controls share the
trade's coin, geometry, costs and the market's drift, so those cancel.

## Hypothesis -- one, alpha 0.05, two-sided

**S_live**: the mean excess of the mirror's resolved trades, by a
week-clustered bootstrap (B = 10,000, seed 20261027).

## Decision rule

The mirror **has edge** only if ALL of: p < 0.05; estimate positive; positive
in both halves (split at the median entry time); null calibration fires on at
most 2 of 20 datasets AND its null mean is within 2.5 SE of zero. Material
only at >= +0.10R per trade.

**A short side is recommended for live only if** it has edge AND is material
AND its absolute mean net R is positive AND the same test passes on the first
45 days of bars that did not exist when the snapshot was fetched. Otherwise
the live system stays long-only.

## Secondary (descriptive, never decided on)

- Absolute: trades, win rate, mean and total net R; the controls' mean (what
  timing-free shorts earned -- the drift).
- The excess on the two-way (week x coin) SE of Kronos Amendment 1.
- By entry kind (resting sell vs in-band market); in stop/ATR strata (below /
  at-or-above the median); by half.
- The long side through the same code on the same bars, and the combined
  book: weekly net R of long + short against long alone, paired by week.
- By market direction: excess in weeks when the equal-weighted universe fell
  against weeks it rose.
- Funding sensitivity: the absolute and excess means with the formula's
  zero-premium funding added to every short (0.00125% per hour held, paid by
  longs to shorts), as an upper bound on what the formula alone pays.

## Null calibration

20 random-walk datasets (`common.random_walks`, 12 coins x 5000 bars), the
same pipeline end to end. If S_live fires on more than 2, or its mean is more
than 2.5 SE from zero, it is reported as miscalibrated and not trusted.

## What would make this wrong

- Drift: the archive rose. The controls share it; the absolute numbers do not.
- Supply bands were never fitted to drawn bands (zones.py says so); the
  mirror tests the rules, not a trader's eye for shorts.
- One regime, 30 weeks: the halves check, and the forward test before live.
- Survivorship: today's top 45 coins.
- Portfolio rails not modelled; totals overstate what one account could hold.

# Predictability scan -- interpretation (EXPLORATORY: decides nothing)

`report.md` and `summary.json` are generated and are the record. This page
reads them. Nothing here was pre-registered; every lead below needs its own
pre-registration on data that did not exist when this ran.

## Why this, instead of another strategy

If E[return over the next H hours | anything known now] = 0, every rule built
on that information has zero expected gross P&L and loses its costs. Five
pre-registered rules on these bars have now come out at "random timing minus
costs" (the level gate, trendline breakouts, 15m confirmation, Kronos, and
the band strategy both ways up). So this measures the conditional mean
directly, family by family, against a null that keeps everything about the
returns except their order in sign: fat tails, volatility clustering and the
cross-coin correlation all survive a sign flip of each whole hour.

## What the 1h archive shows (42 coins, 2026-04-02 to 10-27)

- **One effect is real, and it is the spread.** Long the coins that lagged
  over the last hour, short those that led: +2.65 bps an hour gross (z +4.3,
  Holm p 0.013), +6.4 bps per 4h (z +2.7). Skip the latest hour and it falls
  to +1.03 (z +1.7) and +2.97 (z +1.3): most of it is the bid-ask bounce of
  hourly closes. The book turns over ~2.7x a period, so it breaks even at a
  1.0 bps one-way fee at 1h and 2.4 bps at 4h. This account pays 1.44 bps as
  maker and 4.32 as taker. It is a market maker's income -- the largest grid
  vault audited (L/S Grids) fills 100% as maker at 0.8 bps -- not something
  this strategy can take.
- **Nothing else.** The universe's own trend (1h to a week), cross-sectional
  momentum or reversal beyond 4h, and what follows a 3-sigma coin-specific
  shock (1h, 4h, 24h) are all inside the null (|z| < 1.6). Beyond a day the
  power is low -- 206 daily, 67 three-day and 27 weekly periods -- so "not
  found" there is not "not there".

## Premium and funding (42 coins, 60 days to 2026-10-27)

- **Premium predicts nothing at 1h to a week** (|z| <= 0.85; every point
  estimate has long-cheap / short-rich LOSING on price). The long-side basis
  veto (H-BASIS) is untouched by this -- it conditions band longs, not the
  cross-section -- but premium is not a standalone cross-sectional signal
  in this window.
- **The carry is mechanical and small next to the price noise.** A
  long-cheap / short-rich book collects ~+10 bps a day of funding at the 24h
  horizon; its price spread moves ~+/-200 bps a day.
- **Funding is mostly the formula's constant.** Mean +0.076 bps an hour, +6.7%
  a year, paid by longs to shorts; exactly the fixed interest component (0.01%
  per 8h) in 72% of coin-hours; negative in 9.6%; a long left tail (0.1th
  percentile -18.7 bps an hour). Majors: BTC +8.8%, ETH +9.1%, SOL +7.1%,
  HYPE +8.7% a year, with no week negative in this window.
- **Carry without a hedge is a directional bet.** A perp-only short held the
  60 days collected +1.1% in funding while prices moved +22.5% on average (sd
  45% across coins): net -21.4%, profitable on 38% of coins.

## Do regimes persist? (joint-shuffle null)

- **Volatility does**: rank correlation 0.72 day to day and 0.74 week to week
  against 0.40 and 0.50 shuffled (p 0.001). Volatility-scaled sizing has a
  basis.
- **Trendiness does not** (week to week -0.02, p 0.72; day to day +0.04, p
  0.03 and economically nil), and following the trend after the trendiest
  third of weeks is +227 bps a week better than after the least trendy third
  -- p 0.12. A first version of this null shuffled coins independently and
  gave p 0.012; that null was too narrow (it made the coins independent) and
  was replaced before this was read as anything.

## Fat tails (1h archive, each coin on its own volatility)

Median excess kurtosis 9.7 (range 3.6 to 60). The 1-in-1,000 hourly move is
2.1x and the 1-in-10,000 move 2.9x what a normal model gives; moves beyond 5
sd happen ~5,200x more often than a normal model says. A normal-VaR limit is
wrong exactly where it is supposed to bind.

## Weekly horizon (daily candles, 42 coins, 2023-01-01 to 2026-11-03)

- **Momentum is the direction, and not established.** Five of six cells are
  positive, as Liu, Tsyvinski & Wu (2022, JF) found on 2014-2018 data. The
  strongest: long the top third by 28-day return, short the bottom third, hold
  a week -- +64 bps a week gross over 195 weeks (z +1.65, p 0.10, Holm 0.59).
- **Costs are not its problem; evidence is.** It turns over 1.24x a week, so
  it breaks even at a 52 bps one-way fee against this account's 4.32. But the
  weekly spread swings ~540 bps, so confirming +64 at 80% power needs ~560
  weeks (~11 years). Hyperliquid has 195. Settling it needs a longer, broader
  history than this venue has -- and today's top 45 are survivors.
- **Each coin following its own sign**: -10 bps (7-day), +56 (28-day), z <=
  1.2. **The universe's own trend**: +49 to +55 bps, z <= 0.8.
- **Regimes on 200 weeks**: volatility persists (0.52 vs 0.30 shuffled, p
  0.002); trendiness does not (p 0.61); trend-following after the trendiest
  weeks did 26 bps WORSE than after the least trendy (p 0.69).

## Leads, and what they would need

1. Weekly cross-sectional momentum (28d formation, 7d hold): a broader and
   longer daily history (another venue's, or a cross-venue panel) to reach
   power, pre-registered, then a forward test on this venue.
2. Delta-neutral funding carry on the majors: not an alpha hypothesis but a
   yield to measure -- the full funding history since listing (worst months,
   negative spells), spot availability and spot fees (0.070% / 0.040% base),
   and the spot-perp basis at entry and exit.
Nothing else here is worth a pre-registration on this data.

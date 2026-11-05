# Cash-and-carry -- interpretation (MEASUREMENT, in-sample)

`report.md` and `summary.json` are generated and are the record; the metrics
were fixed in commit 9633ce4 before the full histories were read. This page
reads them. It measures a yield; it tests no edge, so there are no p-values.

## Why this one

The predictability scan found nothing in these bars' conditional mean beyond
the bid-ask bounce. The one structural payment it did find is funding: the
formula pays shorts its fixed interest component (0.01% per 8h, ~11% a year)
whenever the premium sits inside the clamp. Holding the coin as spot and
shorting the same amount of perp collects it without a price bet. This reads
what that actually paid, over each perp's whole funding history.

## What it paid

- **Every coin measured paid shorts on average** since its funding history
  begins: from +4.7% a year of notional (ZEC) to +28.0% (XMR). BTC and ETH
  +14.1% / +14.2% over 3.4 years.
- **It has shrunk.** BTC +24.1% (2024), +10.6% (2025), +5.5% (2026 to date);
  ETH +22.0, +8.5, +5.9; SOL +28.1, +5.3, +0.8. HLP, the venue's
  market-making vault, shrank the same way: +62.9% (2024), +17.3% (2025),
  +7.7% (2026 to date), +1.0% over the last six months.
- **Last 12 months:** XMR +28.0, FARTCOIN +14.7, NEAR +10.9, HYPE +9.9,
  PUMP +9.8, ETH +6.4, BTC +5.8, ZEC +4.3, SOL +0.9.
- **The tails of funding itself are mild.** XMR, NEAR, HYPE, PUMP and FARTCOIN
  had no losing month; BTC and ETH 3 of 40, the worst -0.70% of notional; SOL
  11 of 40 (worst -1.35%); ZEC 3 of 13 (worst -2.42%).
- **Stepping aside after a negative week adds nothing** where funding rarely
  turns (every coin but SOL and ZEC came out slightly behind always-on). In-sample.

## What breaks it

- **The squeeze on the short leg, not funding.** Worst 7-day rises: XMR +84%,
  NEAR +100%, PUMP +105%, HYPE +114%, ZEC +218%, FARTCOIN +271% (BTC +38%, ETH
  +51%). A short at 1x is liquidated at roughly +82% (XMR, max leverage 5) to
  +90% (10x coins). Spot gains what the short loses, but on this account the
  spot does not margin the short (portfolio margin needs > $10k), so the
  short's margin has to be topped up from spot within days of a big rise.
  Without that, one squeeze ends the trade at a loss.
- **The spot token.** Every spot leg but HYPE is a wrapped token (Unit: UBTC,
  UETH, USOL, UZEC, UFART, UPUMP; Wagyu: XMR1; ONEAR). Its issuer failing
  breaks the hedge outright; nothing in this record prices that.
- **The spot-perp gap.** Tight on BTC, ETH, SOL and HYPE (sd 5-7 bps, at most
  0.02% of hours beyond 1%) and PUMP (15 bps). Loose on XMR, NEAR, FARTCOIN and
  ZEC (sd 30-37 bps, 1-2% of hours beyond 1%), with one-hour dislocations of
  2.5% to 8%: XMR1 traded 8.1% under the perp at 2026-09-04 03:00 and was back
  within the hour, and wicked again at 07:00 as the perp rose -- the worst
  combination, because the short needs margin exactly when spot is cheapest to
  sell. Rebalance with limit orders and a buffer, never at market into a gap.

## On capital

Spot plus the short's margin at 1x halves the notional yield; 2-3x is safe
only where the worst day is far from liquidation (BTC, ETH). At the last 12
months' rates: XMR ~14%, FARTCOIN ~7% (spot trades ~$116k a day), NEAR ~5.5%,
HYPE and PUMP ~5% at 1x (~6.6% at 2x), BTC and ETH ~3% at 1x, ~4-5% at 2-3x.
On $716 spread across the tight-hedge coins plus a small XMR sleeve that is
roughly $35-55 a year. Real, positive, small at this size.

## What it would take

Spot orders in the bot (core/venue.py loads an empty spot meta today), an
hourly rebalancer that keeps each short's margin above a floor by selling spot
with limit orders, exits on a funding regime change or a spot-perp gap beyond
a bound, and a forward run at small size before any of it matters. Queue item
15 is measured; the build is the user's call.

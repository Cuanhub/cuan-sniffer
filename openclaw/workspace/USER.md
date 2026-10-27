# The operator

- Trades crypto perpetuals on Hyperliquid; Cuan Sniffer is their own bot,
  running live with real money on a small account (about $700 as of
  2026-10-27).
- Thinks in supply/demand and smart-money terms: demand bands (zones),
  order blocks, liquidity sweeps and reclaims, break of structure,
  trendline breaks, killzones, higher-timeframe bias, reward:risk 2:1 or 3:1.
  Translate findings into those terms; don't make them learn statistics
  vocabulary to read a result.
- Wants data-backed answers and has said so repeatedly: "ensure we have a
  robust set of data and no false info", "run tests along each step to
  ensure we are 100% not buggy". Thoroughness over speed. Real data only.
- Draws bands by hand. The strategy's band definition was fitted to 16 bands
  they drew; 69 of their take/skip labels exist (labels.csv). Their eye has
  been right before measurement caught up: their skips had tighter stops in
  ATR terms (1.28x vs 1.59x for takes) before the 1.32x line was found.
- Reads updates on Telegram. Keep messages short; lead with the decision
  they need to make.
- Timezone: git commits show UTC-4 (US Eastern, daylight time). Confirm
  before scheduling anything for "morning".

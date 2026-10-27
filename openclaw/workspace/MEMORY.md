# Durable memory

What is known about the strategy and the evidence behind it. Update when a
result changes what is known; cite the ledger id.

## The strategy (live as of 2026-10-27)

- **Band (demand zone):** the base candle of an impulse that CLOSED above a
  prior swing high (pivot window 3). The band runs from that candle's low to
  its close. Fitted to 16 bands the operator drew: 16/16 found, 0.97% mean
  edge error, width agreeing to 0.01pp. Bands do not expire; a band price
  closed under is still a band (the operator keeps them).
- **Selection:** bands formed within the last 240 bars, reachable (price
  inside, or no more than 5% above), with a supply band overhead as target.
  The most recent qualifying band per coin, and the first that passes
  rr >= 2.0 (`zone_pass.first_tradeable`).
- **Entry:** price inside the band -> market at that price; above it -> a
  resting bid at the band midpoint for 12h. ~90% of entries are bids.
- **Stop:** max(0.25%, 1.5 x ATR(14)) under the band low.
- **Target:** the near edge of the first supply band above.
- **Rails:** 5% margin per trade at 5x, max 4 concurrent, 20/day, 60 min
  between entries, 6h cooldown after a losing close on a coin (applied on
  market entries only -- the bid path skips it; see the 2026-10-27 audit),
  daily stop 5%, drawdown halt 10%, stand-down after 3 losses in a row.
- **Universe:** top 45 perps by 24h volume, refreshed daily.

## Established (survived controls)

- **Stop distance in ATR multiples is the dominant effect.** t = +9.75; win
  rate 18.6% below 1.32x ATR vs 60.8% above; 10/10 coins agree; survives
  holding the target fixed. This is why the stop is 1.5x ATR. Any feature
  that correlates with stop distance will look predictive -- control for it.
- **Perp above oracle (basis) is a veto.** -0.524R vs +0.044R at or below,
  p = 0.0035 over 248 resolved episodes; the first feature in the project to
  reach significance where 19 price features failed. Live as a veto only.
- **The operator's eye reads the right axis.** Their skips had median stop
  1.28x ATR vs 1.59x for takes, before the 1.32x line was measured.

## Execution costs (measured from this account's fills, 2026-10-27)

maker 0.0144%, taker 0.0432% (referral discount applied); stop slippage
mean 0.1193% (median 0.056%, worst 1.79%) over 171 real stop exits; take
profits fill slightly better than trigger (counted as zero); market-entry
slippage unmeasured (no band market entry had filled).

## Data facts

- The venue serves the last ~5000 1h bars (~208 days). The lab's archive
  keeps everything it has fetched, so history grows from 2026-10-27 on.
- The universe is today's top 45: coins that left it are missing from older
  history (survivorship). Equal across arms of a comparison; not across time.
- On a random walk every entry earns about +0.025R gross in this engine
  (exits booked at the barrier while discrete paths overshoot). It cancels
  in comparisons; never read a small positive absolute R as edge.

## Dead (see ledger/graveyard.md for the numbers)

Killzone timing, HTF-bias proxy, sweep confirmation, `pull_over_run`,
per-coin edge, the old level gate (-0.0413R vs random, z = -0.93, n = 688).

## The live strategy has no demonstrated edge on the conservative engine

Replayed bar by bar (live selection, measured costs, conservative fills),
2026-04-02 to 2026-10-27, today's 45 coins: 907 trades, -0.051R +/- 0.055
per trade (gross -0.023R), -46.6R. Indistinguishable from zero; costs tip
it negative. Portfolio rails not modelled. The earlier +25.2R claim came
from a flawed scratch harness. (H-TRENDLINE baseline, 2026-10-27.)

## Trendline breakout (H-TRENDLINE, pre-registered, 2026-10-27): no edge

- Filter (trade only after a recent break): WORSE, -0.246R (CI -0.473 to
  -0.022), same sign in both halves and both stop/ATR strata. Dead.
- Confirmation entry (wait in the band for a break): +0.035R per setup over
  random-event waiting, p 0.037 uncorrected, Holm 0.097; below the +0.10R
  bar; confirmed trades average -0.005R. Not established -- queued for a
  forward re-test once the archive has new data.
- Standalone: -0.095R per trade; +0.048R over random entry, week-clustered
  p 0.47 (iid said 0.04 -- the clustering is not optional). Dead.
Report: lab/research/backtest/results/trendline_breakout/.

## How this lab came to be

Built 2026-10-27 after a week in which five analytical errors were caught
only by luck or by re-checking: a constant-column split matching a
timestamp, an identity read as corroboration, two features that reversed
under control, and a sweep that tested one config nine times. Every rule in
AGENTS.md traces to one of them.

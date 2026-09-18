# Pre-Registered Hypotheses — Cascade-Fade Edge Research

Written 2026-09-06, ~1.5 days after `research/cascade/collector.py` began
running (launched 2026-09-04 23:41, at the user's explicit request, before
this document existed). **Honesty note, and a real difference from
`HYPOTHESES.md`'s own note:** unlike the funding doc, this is NOT written
blind. Aggregate stats have already been computed and shown to the user —
n=503 filled trades, WR 55.5%, PF 1.287, AvgR +0.123R (gross of fees) —
before this file existed. That impurity needs to be named plainly, not
hidden. What this document can still do is lock in the evaluation bar and
minimum sample *before* more data comes in, so the goalposts can't drift
toward whatever the growing dataset happens to show. Treat this as
"pre-registration relative to the eventual conclusion," a weaker guarantee
than funding's "pre-registration relative to the first look" — worth
remembering when weighing any result this produces.

## Question

Does fading large, high-volume 1-minute price cascades on Hyperliquid
perps produce a real, cost-covering edge, or is the early positive read
(n=503, PF 1.287) noise from a short, single-regime sample — the same
failure mode that already produced four false positives elsewhere in this
research pivot?

## Strategy mechanics (fixed, not re-tuned — see `research/cascade/detector.py`)

- **Detect**: `|1m move| >= 0.3%` AND volume >= 3x the trailing 60-bar
  median.
- **Fade**: enter counter to the move, 0.3% beyond the triggering close,
  filled only if price returns to entry within 3 bars (~54% of detected
  events never fill).
- **Exit**: stop at 0.5%, TP1 at 1R (60% partial), TP2 at 2R, time-stop 10
  bars after fill.
- **Fees**: modeled at 0.045%/side (0.09% round-trip).

## Minimum sample before any conclusion

- **At least 2 full weeks** of continuous forward collection — same bar as
  funding, for consistency. No conclusion from less.
- **At least 300 FILLED trades within each evaluated window** — detected-event
  count overstates real sample size given the ~46% fill rate. 503 filled
  trades already exist, but over 1.5 days of one regime; the 300 floor
  applies per non-overlapping evaluation window below, not as a one-time
  cumulative total.
- **At least 3 non-overlapping weekly windows**, each checked
  independently — mirrors funding's week-over-week unit of independence,
  so one hot day or one regime can't carry the whole result.
- **At least 15 symbols with n>=15 filled trades each.** In the first 1.5
  days, CASHCAT + PONS alone accounted for ~36% of all filled trades — a
  result driven by 2 symbols is a symbol-specific story, not a venue-level
  cascade-fade edge.

## Proceed if

- **Net-of-fees** average R stays positive across at least 3 consecutive
  non-overlapping weekly windows.
- **No single symbol accounts for more than 30%** of total realized R
  (stricter than funding's 40% — cascade events are inherently
  symbol-idiosyncratic, so concentration risk is more likely here, not
  less).
- The edge survives being recomputed strictly net-of-fees. **Correction
  to name now, before it gets glossed over**: `detector.py`'s `pnl_r =
  gross / STOP_PCT` is GROSS of fees — `net_pnl_pct` already deducts fees
  but the +0.123R AvgR quoted to the user so far was the gross figure.
  Any real evaluation must use the net figure, not carry the gross one
  forward as if it were final.

## Stop if

- Net-of-fees AvgR is flat or negative once corrected and evaluated over
  the full window.
- The edge is concentrated in fewer than 5 symbols.
- It disappears or inverts in any one of the 3+ weekly windows once they
  exist.
- Minimum sample is reached with an ambiguous result. Same as funding:
  that is a valid, successful stopping point, not a failure to find
  something.

## Confounds to test explicitly

- **Extremity dependence**: regress realized net R against
  `cascade_vol_ratio` and `abs(cascade_move_pct)`. If the edge is
  concentrated in the most extreme cascades, that's a real, distinct
  phenomenon; if it's flat across the range, that's more consistent with
  noise or a fee-timing artifact than a genuine cascade-specific signal.
- **Generic mean reversion, not cascade-specific**: `time_stop` exits are
  currently the best-looking bucket (AvgR +0.631, gross) — before
  crediting the cascade trigger itself, this needs a random-entry control
  (same symbols, same exit rules, entries at random 1m bars instead of
  cascade-triggered ones) to check this isn't just ordinary short-horizon
  mean reversion that would show up starting from almost any entry.
- **Market-beta confound**: same check funding's doc requires — regress
  realized R against BTC returns over the same window, so a directional
  bias isn't mistaken for a cascade-fade-specific edge.

## Capital reality check

Same as funding: before treating any statistically valid result as worth
pursuing, compute the actual dollar edge at real position sizing and
current capital, and weigh it against the operational cost of running a
live fade strategy — slippage beyond the fixed 0.3% entry-offset
assumption, and execution latency on a signal that is inherently about
reacting fast to a already-fast move.

## Protocol note

Editing the thresholds above after more data is examined against them is
a protocol violation, mirroring `HYPOTHESES.md`'s own rule — the fact that
this document wasn't written fully blind makes holding this line *more*
important here, not less, since there's no "purity" left to lose by
comparison.

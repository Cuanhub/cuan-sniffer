# Pre-Registered Hypotheses — Funding/Basis Edge Research

Written 2026-09-01, ~15 minutes after `research/funding/collector.py` began
running. **Note on sequencing:** the collector was already live when this
was written (launched at the user's explicit request in the prior turn,
before this document existed) — flagging that honestly rather than
implying perfect sequencing. In substance, blindness is preserved: at
this point only a handful of 5-minute snapshots exist, no analysis of any
kind has been run against them, and nothing below was chosen with
knowledge of what the data looks like. This file is not to be edited once
that stops being true — see the protocol note at the bottom.

## Question

Does a persistent, harvestable funding/basis edge exist on Hyperliquid,
at capital sizes this account could actually deploy, that survives real
transaction costs and isn't just disguised market-beta exposure?

## Minimum sample before any conclusion

- **At least 2 full weeks** of continuous collection (the collector's
  stated goal) — no conclusion drawn from less, full stop.
- **At least 20 of the 30 tracked symbols** with usable (non-gappy) data
  over that window. Universe is top-30-by-volume and refreshes daily, so
  some attrition/rotation is expected; 20 is chosen so a handful of
  gappy or delisted symbols don't invalidate the whole window.
- **At least 3 distinct weekly periods** showing consistent direction,
  once enough weeks exist to check this. Week-over-week (not day-over-day
  or 5-minute-snapshot-over-snapshot) is the unit of independence — funding
  rates are autocorrelated intraday and this session already learned the
  hard way (four separate false positives) that oversampling a single
  short-lived regime looks like edge until it isn't. A "regime" here means
  a calendar week; if a cleaner regime definition (volatility-based,
  funding-level-based) becomes obviously better once real data exists,
  that's a Phase 2 modeling choice, not a retroactive edit to this file.

## Proceed if

- Net-of-cost carry exceeds **8% annualized (≈800bps/yr)** on at least
  **5 symbols independently**.
  - *Why 8%, not just >0%:* the operational complexity and tail risk of
    running a delta-neutral carry strategy (basis risk, liquidation risk
    on the hedge leg, execution risk on two legs instead of one) isn't
    worth taking on for an edge that only marginally beats doing nothing.
    8% is a deliberately conservative bar — clearly above "technically
    positive," not a number chosen to make the answer come out either way.
- Persists across **3 consecutive weeks** (once that much data exists).
- **No single symbol accounts for more than 40%** of the aggregate
  measured edge. Below that, "the edge" is actually one name's story, not
  a venue-level phenomenon — exactly the kind of concentration risk this
  pivot exists to stop treating as a finding.

## Stop if

- Net-of-cost carry is below the 8%/yr bar, on average or after removing
  outlier symbols.
- The edge is concentrated in **fewer than 3 symbols** at all.
- It **disappears in any single week** once 3+ weeks of data exist — one
  bad week doesn't necessarily kill it, but this is the tripwire for
  "look harder before proceeding," not a number to explain away.
- Reaching minimum sample without a clear signal in either direction.
  Ambiguous-after-enough-data is still an answer: stop and write up what
  was measured, per the brief's own framing — that's a successful outcome
  of this project, not a failure to find something.

## Confound to test explicitly: is this just market beta?

Regress the candidate return series (whatever the eventual carry/basis
metric turns out to be) against BTC returns over the same window. If beta
explains a meaningful share of the apparent edge, it is not alpha — it's
directional exposure wearing a funding-rate costume. This is the specific
check the original score_v3/v3c work never ran on its short-only edge, and
it must run here before anything gets called a finding.

## Capital reality check

At the capital actually available (~$437 as of 2026-09-01), compute what
daily dollar return the measured edge implies **before** treating any
statistically significant result as worth pursuing. A real, well-measured,
5%-beating-cost edge that pays $0.30/day is a true finding and also not
worth the operational risk of running it — in which case the honest
answer is still stop, regardless of the statistics. This check happens
even if every criterion above is met.

## Protocol note

Editing the thresholds above after data has been examined against them is
a protocol violation, not a refinement — the same discipline that caught
four false-positive backtests earlier in this pivot by insisting on
out-of-sample checks applies here for the same reason. If a number above
turns out to be poorly chosen once real data exists, that's a note for
`DECISIONS.md` on the *next* research question, not a retroactive edit to
this one.

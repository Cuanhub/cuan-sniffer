# Decisions Log

Dated record of non-obvious decisions made during the 2026-09-01 data-only
pivot, with the evidence behind each one. See `PIVOT_STATUS.md` for the
overall state of the pivot.

---

## 2026-09-01 — Zero PREFERRED_SYMBOLS/NEGATIVE_SYMBOLS in score_v3c.py

**Decision:** `PREFERRED_SYMBOLS` and `NEGATIVE_SYMBOLS` are set to empty
sets for v3c's scoring path. v3c now scores on structure, context, and
session only — no symbol term.

**Why:** Both sets were inherited unchanged from `score_v3.py` (measured
Pearson r=-0.008 against real outcomes — no real predictive edge) and
were never re-validated against the current 8-coin traded universe
(SOL, JTO, WIF, HYPE, PENGU, TAO, NEAR, SUI).

Concretely, at the time of this decision:
- `PREFERRED_SYMBOLS = {"FARTCOIN", "JTO", "SOL", "WIF", "SUI"}` overlaps
  only 4 of the 8 currently-traded coins. HYPE, PENGU, TAO, and NEAR
  received no symbol adjustment at all — by omission from an old list
  that predates the current universe, not because they were evaluated
  and found neutral.
- `NEGATIVE_SYMBOLS = {"ETH", "ZEC", "BNB"}` was entirely inert: none of
  its members are in `TRACKED_COINS`, so the penalty branch could never
  fire in practice.
- Live replay data cuts against the inherited bonus directly: WIF, a
  member of the preferred set, sat at -1.74R / PF 0.675 across 8 replay
  trades, while TAO and HYPE — both unadjusted — were among the better
  performers.

An unvalidated bonus firing on half the universe (and a penalty that
never fires at all) is a systematic confound baked into every signal
record before the funding-rate research collection even starts. That's
exactly the kind of thing this pivot exists to stop doing — the goal is
clean data, and zeroing an inherited, unvalidated dummy is the same goal
as collecting more data, not a tradeoff against it.

**How to apply going forward:** if a real, symbol-specific effect exists,
it should surface as a measurable residual in the research data collected
under this pivot — a genuine finding, re-derived from resolved outcomes
on the current universe, the same way every other v3c weight was derived.
An inherited classification is not that, however plausible it looks.
The empty sets are kept in place (not deleted) in `score_v3c.py` so a
real, re-derived set can be reinstated there directly.

**Not touched:** `score_v3.py`'s own copies of these constants are
untouched. It's being archived as evidence of the model that was
decommissioned, not edited.

---

## 2026-09-01 — Funding collector universe: volume-filtered top-30, not 8 or 233

**Decision:** `research/funding/`'s collector tracks the top `N` (default
30, `FUNDING_UNIVERSE_TOP_N`) perps by 24h notional volume (`dayNtlVlm`),
refreshed daily — not the 8 currently-traded coins, and not the full
~233-perp Hyperliquid listing.

**Why:** The directional-signal side of this pivot already has more
throughput than it needs (measured ~1,141 eligible signals/day on the
8-coin universe, 2026-09-01 restart) — sample size was never the
constraint there, and this collector doesn't compete with it for anything
(it's a passive read-only feed, gated by nothing).

The funding-carry question is a different one entirely: it needs
cross-sectional breadth to make `HYPOTHESES.md`'s eventual "does not
depend on any single symbol for more than X%" check meaningful, which 8
symbols can't really support. But the full venue listing is mostly
illiquid tail — thin OI, wide spreads, no realistic path to a deployable
carry position — and would dominate storage/API load with assets that
were never real candidates. Top-30-by-volume is the middle ground: real
breadth, filtered to symbols that could actually support a position if
Phase 2 ever finds something.

**How to apply going forward:** if 30 turns out too narrow or too wide
once real data accumulates, `FUNDING_UNIVERSE_TOP_N` is an env-tunable
constant, not a hardcoded assumption — change it and note why here.

---

## 2026-09-16 — KPEPE excluded from the research universe as unservable

**Decision:** `select_universe()` now drops symbols in `UNIVERSE_EXCLUDE`
(env `FUNDING_UNIVERSE_EXCLUDE`, default `KPEPE`) *before* truncating to
`top_n`, so the universe stays 30 wide and the freed slot goes to the
next genuinely-liquid symbol. On the 2026-09-16 refresh that was XLM,
verified serving `candleSnapshot`, `fundingHistory` and `l2Book`.

**Why:** Hyperliquid ranks KPEPE inside the top-30 by 24h notional volume,
so it kept entering the universe, but the venue 500s on every per-symbol
endpoint either collector needs for it — `candleSnapshot` (cascade),
`fundingHistory` (realized funding, the field `SCHEMA.md` calls ground
truth) and the L2 book. Only the bulk `metaAndAssetCtxs` snapshot works,
which is why partly-filled rows existed at all and why this went
unnoticed: across 2,088 collected rows KPEPE carried **zero** realized
funding values and **zero** spread values, versus 208 and 2,368 for BTC
over the same span. It produced no cascade events, ever.

The cost was not just empty columns. Each failed fetch retries 5 times
with backoff: ~16s per 300s funding cycle and ~32s of every 60s cascade
cycle, one symbol consuming over half the cascade poll interval. It also
generated 33,165 of the 34,700 lines (96%) in
`research/cascade/data/_fetch_failures.log`, which made that log useless
for spotting real, transient venue failures on the other 29 symbols.

**Honesty note:** this changes a pre-registered universe mid-study, which
`CASCADE_HYPOTHESES.md` otherwise guards against. It is defensible only
because KPEPE contributed no observations to drop — no cascade events, no
realized funding, no spreads — so no result can shift as a consequence.
Recorded here rather than left silent precisely because "it couldn't have
mattered" is the kind of claim that should be checkable, not asserted.
The exclusion is a data-availability fact about the venue, not a view on
the asset; if those endpoints start working, remove it and note that here.

**How to apply going forward:** exclusions belong in
`FUNDING_UNIVERSE_EXCLUDE`, not hardcoded at a call site, and each one
needs an entry here saying which endpoint failed and what was verified
before the symbol was dropped.

---

## 2026-09-16 — Cascade collector: order-type-aware fees + multi-arm stop widths

**Decision:** two changes to `research/cascade/`, both landed before the
2026-09-16 forward run accumulated meaningful data:

1. **Fees are now attributed by how the order actually executed.** Entry is
   a resting limit 0.3% beyond the cascade close that only fills if price
   returns to it, and TP1/TP2 are limits — all maker. Only stops and
   time-stops cross the spread. Previously `FEE_PER_SIDE * 2` charged taker
   on both sides unconditionally.
2. **Every detected event is resolved at each of `STOP_ARMS`**
   (`CASCADE_STOP_ARMS`, default 0.005/0.0075/0.01) from the same candle
   fetch. Arm 0 is `STOP_PCT`, the pre-registered rule, unchanged.

**Why:** measured on the 1,771 filled trades collected 2026-09-04..10, the
flat-taker model charged 0.0900%/trade against a gross edge of 0.0751%,
giving −0.0298R. Order-type-aware fees cost 0.0497%, giving **+0.0507R** —
the fee model, not the strategy, carried the sign of the result. Venue
rates confirmed from `userFees` for this account: taker 0.045%, maker
0.015%. The corrected figure survives dropping LIT (+0.0292R, n=1,696),
the symbol that carried most of the gross edge.

Stop width is the only lever that attacks the underlying arithmetic rather
than nibbling at it: fees are a fixed fraction of notional while R is
defined by the stop, so drag in R terms is `2*fee/stop` — 0.18R at the 0.5%
baseline under the old model. It also cannot be tested retroactively:
resolutions store outcomes, not the post-entry path, and this venue drops
1m candles after ~3–4 days (verified — T-3d returns candles, T-4d returns
none). Collecting the arms in parallel measures the trade-off prospectively
at zero extra API cost, instead of fitting a stop width to a sample after
seeing it.

**Honesty note:** the +0.0507R above is an upper bound, and the reason is
recorded here so it is not quoted later without it. The fill model counts a
trade as filled when price *touches* the entry, but a resting limit needs
price to trade *through* it to clear the queue — and the trades most often
missed are the good ones, where price tags the level and reverses. The fee
correction and this bias push in opposite directions. Rather than invent a
haircut, resolutions now record `fill_penetration_pct` (how far past entry
the filling bar actually traded), so the forward sample can be re-read
under a minimum-penetration filter as a measurement. Stop fills are
likewise still modelled at exactly `stop_price` with no slippage, on
symbols selected for violent moves — untouched, and still optimistic.

**Also fixed:** the `tp1+time` exit label was unreachable. `exit_reason` is
only non-empty when a `break` set `remaining = 0`, so the old
`if not exit_reason` test could never select it and a TP1 that later timed
out was recorded as `time_stop`. Now keyed off `tp1_hit`. This is not
cosmetic: that exit leaves 60% as a resting limit and 40% as a market
order, so the mislabel also mis-priced its fee.

**Schema:** `resolutions.csv` gains `resolution_id`, `arm_stop_pct`,
`net_pnl_r`, `entry_fee_pct`, `exit_fee_pct`, `fill_penetration_pct`.
Original columns keep their order and meaning, so older files stay readable.
**Row counts changed** — one row per (event, arm), so any analysis counting
rows must filter `arm_stop_pct == 0.005` to reproduce pre-change semantics.
Dedupe is keyed on `resolution_id` when a row carries one and on `event_id`
when it does not, so pre-change files remain idempotent on restart.

**Verification:** arm 0 was diffed against the committed detector over
4,000 randomised synthetic resolutions — zero mismatches on `exit_reason`,
`avg_exit_price`, `gross_pnl_pct`, `pnl_r`, `bars_held`, `fill_ts_ms`. The
forward sample's baseline arm therefore stays directly comparable to
everything collected before this change.

**How to apply going forward:** arms are for *measuring* the stop-width
trade-off, not for picking a winner post hoc. Choosing an arm because it
scored best on this sample is the same error as re-tuning thresholds on it;
`CASCADE_HYPOTHESES.md`'s minimum-sample and weekly-window bars apply per
arm, independently.

---

## 2026-09-18 — Shorts deferred until the long side clears 50% wins

**Decision:** the level system stays long-only. Short setups are not built
until the long side is "dialled in", defined by the user as a win rate at or
above 50%.

**Why it came up:** the user identified a short on UNI -- price double-tapped
resistance at 8.8889 twice on 2026-09-18 (03:00 high 8.8889, 07:00 high
8.8353, 0.35 sigma apart, both confirmed on the 1h), with the stop just above
the double top and support 6.96-8.99R below. The detector already holds that
level: 2 tests, 100% held, 1.0 sigma above price. The geometry is there; the
system simply has no path that produces a short.

**What long-only actually means in the code:** every candidate is
`kind="support"`, every entry `side="LONG"`, the trigger asks whether price
held ABOVE a level, stops go below and targets above, and
`flipped_supports()` converts resistance into support but never the reverse.
Most of this mirrors cleanly -- `detect_levels(kind="resistance")` is already
computed on every scan and used for targets -- so the code is not the hard
part.

**The hard part is that no short has ever been labelled.** All 18 labels are
longs, and every threshold in the trade filter was fitted to them:
TRADE_MIN_NEAR_SIGMA=5.0, the 9-bar rest, the 5 sigma headroom. Applying
those to shorts would repeat exactly the error this filter was built to
correct -- carrying numbers across to a population they were never measured
on. Shorts need their own labels first.

**A second gap the same setup exposed:** nothing in the system models trend.
The user's own note was "since the trend is still up this could be more risky
to trade the correction", and the system has no way to express that. It would
short a double top in a rally as readily as in a downtrend. This matters
because the September book that motivated the whole strategy was 23 fills,
all long, in a window where all 28 universe coins rose a median 38.5%.

**Two things to be honest about when the criterion is checked:**

At the ~3.4 reward/risk this filter selects for, break-even is about a 23%
win rate, so 50% is a demanding bar rather than a natural one -- it implies
roughly +1.0R per trade. The long side may be strongly profitable well below
it. The criterion is a deliberate scope gate, not a profitability threshold,
and it should not be read as "below 50% means it does not work".

"50%" also needs a sample size, or two lucky trades satisfy it. The
pre-registration discipline used elsewhere in this project suggests at least
20-30 closed trades before the number means anything.

**How to apply going forward:** when the long side has 20+ closed trades at
50%+, collect short labels through `research.levels.call` the same way the
long labels were collected, fit the short thresholds to those, and only then
mirror the code. Do not start by mirroring the code.

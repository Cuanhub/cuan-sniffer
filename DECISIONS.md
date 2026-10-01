# Decisions Log

Dated record of non-obvious decisions made during the 2026-09-01 data-only
pivot, with the evidence behind each one. `README.md` has the current state
of the system; `research/levels/SCHEMA.md` has the data model.

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

---

## 2026-09-18 — Volume is not a discriminator; captured, never gated

**Decision:** no volume reading enters the gate or the trade filter. All three
continue to be recorded on every episode.

**Evidence.** SCHEMA.md set this test in advance, and deliberately: three
competing readings of "volume contracted into the level" are captured on every
episode so the question gets settled from data rather than argued, with the
criterion stated up front — near-zero gaps on all three settle it in the
negative. At 25 labels (14 take / 5 skip / 6 maybe), `review --stats` gives:

| feature | take avg | skip avg | gap |
|---|---|---|---|
| `vol_ratio_recent_ref` | 0.848 | 0.780 | +0.068 |
| `vol_trend_slope` | 0.014 | 0.028 | −0.014 |
| `vol_down_up_ratio` | 0.495 | 0.432 | +0.063 |

Against the features the filter actually uses, those are noise: `near_sigma`
separates takes from skips by **+4.121** and `suggested_rr` by **+1.313** —
sixty times the largest volume gap. The pre-registered criterion is met.

This also confirms the original note behind the whole question: the remembered
feature was not the one first computed. At the fill bar every window measured
0.90–1.25x its reference, normal to elevated.

**What this does not establish.** Five skips. The criterion was written
against exactly this comparison, so meeting it is the outcome it was designed
to produce, and a gap resting on n=5 would move on a handful of new labels.
This closes the question of gating on volume *today*. It does not close the
question.

**How to apply going forward:** do not add a volume threshold. Re-read
`review --stats` when the skip count reaches roughly 15, and treat a gap that
has grown rather than shrunk as the signal to look again. Capture costs
nothing, so nothing is removed in the meantime.

---

## 2026-09-20 — The stop goes below the structure, because a break is information

**Decision:** stops are placed just under the lows that formed the level
(`structural_stop`), never at a noise-width distance chosen to flatter the
reward/risk. Stated by the user: *"I would lean more towards the wider
structural stop to be less likely to get shaken out. If it broke structure
high chance it will continue down further."*

**Why it matters.** A stop can be wrong in two ways, and they are not
symmetrical. A stop too tight gets taken by ordinary movement and tells you
nothing -- the level may still be perfectly good, and you have paid to find
out that ETH wiggles. A stop below the structure only triggers when the thing
you were betting on has actually failed, and that failure is itself evidence:
the level broke, so the next level down is now in play.

This is the same reasoning already recorded against `STOP_SIGMA_BELOW` -- *"A
level breaking and price reaching the next level are different events and only
the first invalidates the trade"* -- arrived at independently from the chart
rather than from the code.

**What it costs.** Very little, and that is the point. On the ETH 2562.9 setup
priced at 5x on 3% margin:

| stop | below the level | risk | rr |
|---|---|---|---|
| structural 2550.67 | 12.2 | $1.08 | 3.20 |
| just under the level, 2559 | 3.9 | $0.71 | 4.89 |

The tighter stop prices better by every number the filter can see, and it sits
3.9 points under a level ETH crosses routinely. The 37 cents of extra risk buys
a stop that means something.

**How to apply going forward:** do not tune `STOP_MIN_SIGMA` (0.5) upward to
improve reward/risk on a rejected setup. If a setup only clears the rr floor
with a stop inside the noise, it has not cleared the floor. The structural stop
may be widened by evidence that levels are being broken and recovered; it must
not be narrowed by a setup wanting to qualify.

## 2026-10-01 -- universe 30 -> 45, to be reviewed after three days

The top 30 perps by 24h volume is a list that barely moves, so the same names
recurred on every board while 204 of the venue's 234 perps were never scanned.
The cut also fell straight through setups called by hand: CASHCAT, FARTCOIN,
USELESS and BNB all sit at ranks 31-45.

Affordable only because of the request budget. A pass cost 90 venue calls at 30
coins; removing the duplicate 1h fetch (7c745f6) and caching the 4h series to
its bar boundary (ca60aa3) brought that to 31, so 45 coins runs at about 46 --
roughly half the load that was producing rate limits at half the universe size.

The 15 added, with 24h notional at the time of the change:

  PENGU 9.1M, GRAM 9.1M, XMR 9.0M, FARTCOIN 8.6M, CASHCAT 8.1M, HBAR 8.0M,
  TRUMP 7.5M, XLM 7.4M, ETHFI 7.2M, LDO 6.7M, USELESS 6.1M, BNB 5.9M,
  KBONK 5.4M, JUP 5.3M, VIRTUAL 4.9M

The volume floor moves from $10.3M to $4.9M. At a $119 position that is not a
liquidity constraint; what it buys is thinner, more erratic names, which is the
thing the review is for.

BASELINE to judge it against -- the book immediately before the change:

  26 closed since 2026-09-20, 8W/18L (31%), net +$28.94
  avg win +$8.58, avg loss -$2.20, profit factor 1.73
  equity $768.58 -> $797.01, peak $817.51
  coins traded: LIT x4, DOGE x3, PUMP x3, HYPE x3, PONS x2, XPL x2, WLD x2,
                ZEC, NEAR, LTC, ETH, VVV, BTC, SUI

The question the review answers is NOT whether the three days are profitable --
n will be far too small for that. It is whether the added names behave
differently from the existing 30: their share of trades taken, their win rate
and average loss against the incumbents', and whether any of them produce the
LIT pattern of repeated entries into a falling market. The per-coin cooldown
(18a688d) now bounds that, so this is also the first real test of it.

Rails unchanged and now the binding constraint: 4 concurrent, 20/day, 5% daily
loss stop, 10% account drawdown halt from the high-water mark.

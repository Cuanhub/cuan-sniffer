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

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

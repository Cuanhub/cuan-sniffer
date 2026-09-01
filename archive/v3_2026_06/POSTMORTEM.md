# Postmortem — score_v3

Archived 2026-09-01. Decommissioned, no longer selectable as
`LIVE_ELIGIBILITY_MODEL` (see the parent repo's `DECISIONS.md`). This
directory preserves the model as evidence, not as maintained code.

## What it was

An "expectancy-grounded factor model" — flat, side-agnostic weights per
regime/structure tag, intended to replace the earlier v1/v2 scoring.

**Fit window:** 249 replayed signals, 2026-06-16 to 2026-06-20, all
continuation setups. Verified directly against this file's own header
comment at the time of archiving.

## Why it was killed

**Primary finding, established independently this pivot** (full-pipeline
replay against real resolved Hyperliquid outcomes, 503 candidates, 45
days): **Pearson r = -0.008** between `score_v3` and realized R — no
measurable predictive edge across the full score range. The live-eligible
slice (`score_v3 >= 0.80`) was net-negative: AvgR = -0.080, PF = 0.875.

**Prior audit findings** (from the pre-pivot audit that first flagged
this model for review — cited here for completeness, not independently
re-run this pivot beyond the two items verified below):
- Out-of-sample: n=149, PF 1.007, +0.65R, MaxDD 43R.
- Live result: -6.88R / -$42.64 over 25 trades.

**Root cause, per that audit and consistent with the independent finding
above:** the model applied ONE flat weight per regime tag regardless of
trade side or the current traded universe, masking real but conditional
effects instead of capturing them. Concretely — and verified directly
against this file's code at archive time — `macro_regime=="chop"` scored
**+0.10** and `market_regime=="chop"` scored **+0.05**, applied
identically to LONG and SHORT. This session's own broader audit found the
real chop effect is sharply side-conditional: SHORT+macro_chop PF=2.14
(n=40) vs LONG+macro_chop PF=0.82 (n=23) — a flat bonus for both sides
cancels out exactly the split that mattered. score_v3c (the live
successor) re-derives every weight side-conditioned instead.

The same inherited-classification problem applied to `PREFERRED_SYMBOLS`/
`NEGATIVE_SYMBOLS` (see `score_shared.py`'s docstring and the parent
`DECISIONS.md` 2026-09-01 entry) — both sets predate the current 8-coin
universe and were carried into score_v3c unexamined until this pivot.

## What replaced it

`score_v3c.py` — side-conditioned rescore, live since 2026-08-27,
in-sample Pearson r improved to +0.130. Not proven to have durable edge
either (see the parent repo's own audit trail: roughly breakeven, PF
1.00-1.08, across two non-overlapping 120-day windows) — which is the
immediate reason for this pivot to a data-only posture rather than
continuing to iterate on scoring weights against thin samples.

## What was preserved vs. changed in this move

- `compute_shadow_score_v3()`'s actual scoring logic: **unchanged**,
  byte-for-byte.
- Its own tests (`test_score_v3.py`, moved alongside it): unchanged,
  still pass against the archived module.
- Only change: `_safe_str`/`_safe_bool`/`_has_trigger` and
  `PREFERRED_SYMBOLS`/`NEGATIVE_SYMBOLS`/`NEGATIVE_SESSIONS` are now
  imported from `score_shared.py` (repo root) instead of being defined
  locally — required so this file could move without breaking
  `score_v3b.py`, which still uses the original, unzeroed symbol sets.

# {{ID}}: <fill: one-line title>

Pre-registration. Written and registered (`lab register {{ID}}`) BEFORE the
experiment touches real data. After registration nothing here may change; a
changed design is a new experiment. Delete every `<fill...>` marker --
registration refuses while any remain.

## Question

<fill: one question, answerable yes or no. "Does X improve the band
strategy's net R per trade?" -- not "explore X".>

## Grounding

<fill: what observation or earlier result motivates this. Cite ledger ids
(`lab ledger`) or live evidence (`lab why`). An idea with no grounding is
fishing.>

## Data

<fill: the lab archive (real Hyperliquid 1h bars), development window =
everything before the seal. Any coins excluded and why. Known biases, e.g.
the universe is today's top 45 (survivorship).>

## Feature definition

<fill: exact, causal definition with every parameter and the reason for its
value. Prefer values that already exist in the strategy (k=3, 1.5 ATR, 240
bars) over new knobs. If there is no per-bar feature, write "no per-bar
feature" and why.>

## Framings

<fill: one subsection per name in FRAMINGS, each saying: role (primary /
must_agree / secondary), statistic (diff or mean), what the rows are, and
what POSITIVE means. Positive must always mean "better for the strategy".>

## Controls

<fill: how the information is removed while every mechanic stays identical.
Random entries drawn AFTER the signal (labcore.controls.draws_after) or the
same events at random times (circular_shift). Never a window around the
signal: the signal's own definition leaks into it (trendline Amendment 1).>

## Decision rule

The lab's standard rule (labcore/evaluate.py): Holm-rejected among the
primary framings at alpha 0.05; estimate > 0; positive in both halves of the
period; every must_agree framing positive; >= 30 units per group; null
calibration centred and firing <= 2/20. Then the sealed holdout, once, at
alpha 0.05 / openings. Material only if >= +0.10R.
<fill: anything STRICTER, or "none".>

## What would make this wrong

<fill: the ways this result could mislead -- regime dependence, a
confounder (stop/ATR is the strongest effect known in this strategy, t =
+9.75), selection, costs. Say which secondary analysis checks each.>

## Graveyard check

<fill: related ideas in ledger/graveyard.md and why this one differs
materially. "Nothing related" is an acceptable answer if true.>

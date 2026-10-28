---
name: lab-experiment
description: How to design, pre-register and implement a backtest experiment that the lab will accept -- the PREREG sections, the experiment.py contract, controls that cannot see the signal, and the tests.
metadata: {"openclaw": {"requires": {"bins": ["python3", "git"]}, "emoji": "🧪", "os": ["darwin", "linux"]}}
---

# Designing an experiment

```bash
cd {baseDir}/../../lab && .venv/bin/python lab.py new <slug>     # -> experiments/eNNN_<slug>/
```

## PREREG.md -- fill every section, delete every `<fill...>`

- **Question:** yes/no. "Does X improve net R per trade of the live band
  strategy?"
- **Grounding:** a ledger id, a `lab.py why` finding, or the operator's
  request. No grounding means fishing.
- **Feature definition:** exact and causal. Prefer existing parameters
  (pivot k = 3, 1.5 ATR, 240 bars). Every new knob must say why its value.
- **Framings:** each name in FRAMINGS, its role, statistic, rows, and what
  POSITIVE means (always "better for the strategy").
- **Controls:** remove the information, keep every mechanic identical.
  - filter on baseline trades -> `stat: diff` between kept (group 1) and
    refused (group 0), plus a `must_agree` framing on excess R over
    `research.backtest.common.same_order_control` -- random orders of the
    trade's OWN type (a bid's control is a bid) drawn after its entry;
  - a new entry rule or a wait-for-confirmation rule -> compare against the
    SAME rule with its events moved to random times
    (`controls.circular_shift`), never against the baseline (that rewards
    trading less whenever the baseline loses) -- and compare UNPAIRED
    per-trade means, never only the setups where the signal fired (that
    selects the setups where price moved, and the control rides the move);
  - a standalone signal -> excess over `controls.draws_after` random entries
    with the same stop/target rule.
- **What would make this wrong:** at least the stop/ATR confound (stratify
  or show the feature is unrelated to stop distance), regime/time, costs.
- **Graveyard check.**

## experiment.py -- the contract (see experiments/TEMPLATE)

- `TITLE`, `FRAMINGS`, `FEATURES` (fn(bars) -> one value per bar, causal),
  `records(ctx)` -> rows `{"framing", "t" (ms), "coin", "value", "group"}`.
- `ctx.baseline()` is the live strategy's trades (research.backtest.band).
  `ctx.map(fn, args)` runs module-level functions in workers that read bars
  with `labcore.pool.bars(coin)`.
- Simulate trades only with `research.backtest.engine.simulate`; costs from
  `research.backtest.costs.MEASURED`. Do not write fill or exit logic.
- Read a feature at the DECISION bar (`trade.decided_i`), never later.
- Values are net R. No statistics in this file.

## Tests

Keep the template's three tests; add one per non-trivial feature with a
hand-built case where you know the answer. Then:

```bash
.venv/bin/python -m pytest -q experiments/eNNN_<slug>
.venv/bin/python lab.py register ENNN
```

---
name: lab-propose
description: Turn an experiment that passed the sealed holdout with a material effect into a proposal for the operator. Never applies anything.
metadata: {"openclaw": {"requires": {"bins": ["python3", "git"]}, "emoji": "🧪", "os": ["darwin", "linux"]}}
---

# Proposing a change

Only after `lab.py holdout <ID>` passed and the effect is material:

```bash
cd {baseDir}/../../lab && .venv/bin/python lab.py propose <ID>
```

Then complete `proposals/<ID>.md` -- leave the lab-filled evidence block
exactly as written:

- **The change:** the smallest exact edit to the LIVE repo (file, function,
  constant or a unified diff against research/levels/ or the trader config).
- **Risks and what to watch:** what could go wrong live that the backtest
  cannot see (fills, slippage, the portfolio rails, regime), and the metric
  that would show it within two weeks.

Message the operator: one line verdict, the dev and holdout effects with
intervals, the denominator, and the file path. Then stop. They decide; you
do not chase.

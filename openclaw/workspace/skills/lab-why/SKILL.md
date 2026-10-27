---
name: lab-why
description: Why the live strategy is losing -- the band strategy's own resolved record, read through guards that handle ordered wins, break-even, constant columns, time confounds and persistence.
metadata: {"openclaw": {"requires": {"bins": ["python3", "git"]}, "emoji": "🧪", "os": ["darwin", "linux"]}}
---

# Why are we losing

```bash
cd {baseDir}/../../lab && .venv/bin/python lab.py why
```

This reads the live records synced read-only into the lab. Report what the
tool says; do not recompute its numbers. Every split it prints carries its
own t and a persistence check -- a split that fails persistence is history,
not a reason to change anything. Turn anything that looks real into a
grounded queue item, not into a conclusion.

If it says the records are missing, the sync is not configured
(CUAN_LIVE_DATA) -- tell the operator rather than working around it.

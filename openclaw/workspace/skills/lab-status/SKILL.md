---
name: lab-status
description: Where the research lab stands -- data freshness, the seal, the ledger, the queue and proposals awaiting the operator.
metadata: {"openclaw": {"requires": {"bins": ["python3", "git"]}, "emoji": "🧪", "os": ["darwin", "linux"]}}
---

# Lab status

Run from `{baseDir}/../../lab`:

```bash
cd {baseDir}/../../lab && .venv/bin/python lab.py status
```

Report in at most six lines: data age, experiments by state, the hypotheses
denominator, proposals waiting, and anything flagged with `!!`. If the ledger
chain is not INTACT, lead with that -- it means the record was edited.

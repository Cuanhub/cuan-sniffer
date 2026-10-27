---
name: lab-weekly
description: The weekly research digest for the operator -- what was tested, what was learned, what is next.
metadata: {"openclaw": {"requires": {"bins": ["python3", "git"]}, "emoji": "🧪", "os": ["darwin", "linux"]}}
---

# Weekly digest

```bash
cd {baseDir}/../../lab && .venv/bin/python lab.py status && .venv/bin/python lab.py ledger && .venv/bin/python lab.py why
```

Write at most 12 lines:

1. Experiments this week and their verdicts (ids, effects, intervals).
2. Anything that changed what is known (MEMORY.md edits).
3. What `lab.py why` shows about the live record this week, with its own
   guards' caveats.
4. The next two ideas in the queue and their grounding.
5. "Hypotheses tested to date: N." Proposals awaiting a decision.

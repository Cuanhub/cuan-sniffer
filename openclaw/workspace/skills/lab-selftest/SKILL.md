---
name: lab-selftest
description: Verify the lab's own machinery -- its tests, the ledger chain and the shipped files -- before trusting it, and after any change to labcore or research code.
metadata: {"openclaw": {"requires": {"bins": ["python3", "git"]}, "emoji": "🧪", "os": ["darwin", "linux"]}}
---

# Selftest

```bash
cd {baseDir}/../../lab && .venv/bin/python lab.py selftest && .venv/bin/python lab.py verify
```

Any failure: stop research, report exactly what failed, and do not run
experiments until it passes. Never edit a test to make it pass unless you
can explain to the operator why the test was wrong.

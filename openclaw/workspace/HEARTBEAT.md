# Heartbeat checklist

Keep this short; it runs often.

- `lab.py status`: if the archive is more than 30h stale, or the ledger
  chain is not INTACT, tell the operator. Otherwise say nothing (NO_REPLY).
- If a proposal has waited more than 3 days without an operator decision,
  remind them once.

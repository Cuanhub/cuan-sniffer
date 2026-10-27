---
name: lab-cycle
description: One autonomous research cycle -- pick the best-grounded idea, pre-register it, calibrate, run, record, and report. Respects the daily budget.
metadata: {"openclaw": {"requires": {"bins": ["python3", "git"]}, "emoji": "🧪", "os": ["darwin", "linux"]}}
---

# One research cycle

Follow AGENTS.md. This is the procedure; the rules there override anything
here.

1. `cd {baseDir}/../../lab`, then `.venv/bin/python lab.py status`. Stop and
   report if the archive is stale (> 30h) or the ledger chain is broken.
2. **Budget.** If an experiment already ran on real data today (check
   `lab.py ledger` and today's `memory/` log), do not start another.
   Instead advance one already in flight (null -> run -> holdout), or
   report "no new experiment today: budget".
3. **In flight first.** Any experiment registered but not calibrated, or
   calibrated but not run, or passed development but not held out? Advance
   exactly that one step.
4. **Otherwise pick an idea:** the top of `ledger/queue.md`, unless
   `lab.py why` or a recent result grounds something better. Check
   `ledger/graveyard.md`. Write the grounding down.
5. **Design it** with the lab-experiment skill. Register it. If registration
   fails, fix what it says -- never weaken a check.
6. `lab.py null <ID>`. If MISCALIBRATED, stop: the test is broken. Record a
   note, put the idea back in the queue with what was wrong, report.
7. `lab.py run <ID>`. Report the result, positive or negative.
8. Update `ledger/queue.md` (remove the item, note the id), and on a clear
   failure add a graveyard row with the numbers. Update `MEMORY.md` only if
   something is now established or dead.
9. Write today's `memory/YYYY-MM-DD.md`: what you ran and why, in a few
   lines.

**Report** (3-6 lines): what ran, the verdict, the effect with its interval
and p, what happens next, and "hypotheses tested to date: N".

"""The static-analysis gates, as tests, so CI is not a separate thing to run.

Audit finding F-1: a 2,065-line module that places every order had no type
checking and the project had no dependency scanning.

These are RATCHETS, not clean bills of health. The backend had 15 mypy
findings when checking was introduced; four were defensive isinstance guards
that mypy called unreachable because venue responses were annotated
Dict[str, Any] when the venue can in fact return anything -- the annotation
was the lie, not the guard, and widening it to Any fixed them honestly.
The rest are missing annotations in older code, left as a declining budget
rather than fixed in a batch that was supposed to be about tooling.

The point of a ratchet is that it cannot get worse while someone works it
down. Lower the number when you fix some; never raise it.
"""

import subprocess
import sys
import unittest

# Zero, and it has to stay zero. The 10 this replaces was a budget for
# live_execution_backend.py, which no longer exists; core/ was written typed
# from the start and is checked strictly, so there is no debt to budget for.
MYPY_BUDGET = 0


class Mypy(unittest.TestCase):
    def test_the_order_placing_modules_do_not_get_worse(self):
        """The modules that place every order, checked strictly.

        This used to point at live_execution_backend.py and
        execution_backend.py. Once those were deleted it kept PASSING while
        checking nothing -- mypy on a missing file emits no "error:" line, so
        the count came back 0 and the budget was satisfied vacuously. A green
        test that tests nothing is worse than a red one, so it now names a
        directory and asserts the run actually happened.
        """
        out = subprocess.run(
            [sys.executable, "-m", "mypy", "core/"],
            capture_output=True, text=True).stdout
        self.assertIn("source file", out.lower() + "checked",
                      f"mypy did not check anything:\n{out}")
        self.assertNotIn("can't read file", out, f"missing target:\n{out}")
        n = sum(1 for line in out.splitlines() if "error:" in line)
        self.assertLessEqual(
            n, MYPY_BUDGET,
            f"mypy findings rose from {MYPY_BUDGET} to {n}:\n{out}")
        if n < MYPY_BUDGET:
            print(f"\n  mypy is down to {n} (budget {MYPY_BUDGET}) -- "
                  f"lower MYPY_BUDGET in this file")

    def test_the_config_scopes_the_strict_rules_to_the_order_path(self):
        cfg = open("mypy.ini").read()
        self.assertIn("[mypy-core.*]", cfg)
        self.assertIn("disallow_untyped_defs = True", cfg)
        self.assertNotIn("[mypy-live_execution_backend]", cfg,
                         "mypy.ini still scopes rules to a deleted module")


class Pinning(unittest.TestCase):
    def test_a_lockfile_exists_and_pins_the_venue_sdk(self):
        lock = open("requirements.lock").read()
        for pkg in ("hyperliquid-python-sdk==", "eth-account==", "requests=="):
            self.assertIn(pkg, lock, f"{pkg} is not pinned")

    def test_requirements_txt_is_still_the_human_list(self):
        """requirements.txt stays readable ranges; the lock is what installs."""
        self.assertIn("hyperliquid-python-sdk", open("requirements.txt").read())

    def test_the_retention_archive_is_not_tracked(self):
        out = subprocess.run(["git", "ls-files",
                              "research/levels/data/archive"],
                             capture_output=True, text=True).stdout.strip()
        self.assertEqual(out, "",
                         "the archive is tracked again; a live process will "
                         "dirty the working tree (audit F-4)")


if __name__ == "__main__":
    unittest.main()

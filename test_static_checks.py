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

import pathlib
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


class EnvExampleMatchesTheCode(unittest.TestCase):
    """.env.example must offer only knobs that exist.

    It declared 121 variables and the code read 4 of them. The other 117 came
    from an earlier system -- regime scoring, factor buckets, partial
    take-profits, an SMC engine -- so setting one configured nothing. That is
    worse than not offering it: it reads as a supported control.
    """

    @staticmethod
    def declared():
        import re
        return [l.split("=")[0]
                for l in open(".env.example").read().splitlines()
                if re.match(r"^[A-Z][A-Z0-9_]*=", l)]

    @staticmethod
    def source():
        src = ""
        for f in pathlib.Path(".").rglob("*.py"):
            if ".venv" in str(f) or "scratchpad" in str(f):
                continue
            src += f.read_text()
        return src

    def test_every_declared_variable_is_read_somewhere(self):
        src = self.source()
        dead = [d for d in self.declared() if d not in src]
        self.assertEqual(dead, [], f"declared but never read: {dead}")

    def test_the_credentials_are_offered(self):
        """A setup that cannot find these does not trade at all."""
        d = self.declared()
        for k in ("HL_ACCOUNT_ADDRESS", "HL_SECRET_KEY", "HL_TESTNET",
                  "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID",
                  "LEVELS_LIVE_ARMED"):
            self.assertIn(k, d, f"{k} is not in .env.example")

    def test_the_network_is_stated_never_defaulted(self):
        """The example is for a real account, so it says false explicitly.
        The CODE used to default to true, so a missing .env could not put real
        money on mainnet by omission -- but it could, and would silently, run
        the live bot on testnet. It now has no default at all: Venue() refuses
        unless HL_TESTNET is exactly true or false (test_venue_is_real)."""
        self.assertIn("HL_TESTNET=false", open(".env.example").read())
        from core import venue
        import inspect
        src = inspect.getsource(venue)
        self.assertNotIn('"HL_TESTNET", "true"', src)
        self.assertNotIn('"HL_TESTNET", "false"', src)

    def test_the_band_knobs_are_offered_since_they_are_the_strategy(self):
        d = self.declared()
        for k in ("LEVELS_ZONE_RECENT_BARS", "LEVELS_ZONE_MIN_WIDTH_PCT",
                  "LEVELS_ZONE_REACH_PCT", "LEVELS_ZONE_STOP_BUFFER_PCT",
                  "LEVELS_MIN_RR"):
            self.assertIn(k, d, f"{k} is the strategy and is not offered")


if __name__ == "__main__":
    unittest.main()


class TheRailsLineDescribesTheRailsThatRun(unittest.TestCase):
    """The line the operator reads to know what the bot is doing.

    It has gone stale twice. First it advertised `depth>=5.0s rested>=9b
    headroom>=3.0s` after the pass moved to bands -- thresholds nothing
    applied. Then it said `stop 0.25% under its low` after the stop became
    max(0.25%, ZONE_STOP_ATR_MULT x ATR). Both times it was describing a
    system that no longer existed, on the one line that is supposed to say
    what the system is.
    """

    def summary(self):
        from research.levels.config import summary
        return summary()

    def test_it_names_the_volatility_scaled_stop(self):
        from research.levels import zones
        s = self.summary()
        assert "ATR" in s, s
        assert f"{zones.ZONE_STOP_ATR_MULT:g}x" in s, s

    def test_it_does_not_claim_a_bare_fixed_buffer(self):
        """"stop 0.25% under its low" was true and became false."""
        from research.levels import zones
        s = self.summary()
        bare = f"stop {zones.ZONE_STOP_BUFFER_PCT}% under"
        self.assertNotIn(bare, s, f"the fixed buffer is described as the stop: {s}")

    def test_it_reads_every_band_knob_from_the_module(self):
        """So a knob change cannot leave the description behind."""
        from research.levels import zones
        s = self.summary()
        for v in (zones.ZONE_MIN_WIDTH_PCT, zones.ZONE_MAX_WIDTH_PCT,
                  zones.ZONE_RECENT_BARS, zones.ZONE_REACH_PCT):
            self.assertIn(f"{v:g}", s, f"{v} missing from: {s}")

    def test_the_per_trade_line_does_not_hardcode_a_median_stop(self):
        """It read "risk at a median 1.76% stop $3.10" with both the 1.76%
        and the 0.088 that derived it hardcoded, and kept reporting them
        after the real median moved to about 3.37%."""
        import inspect
        import tools.live_levels_trader as T
        # Comments stripped: this file's own comment RECORDS the old numbers
        # as history, and matching prose rather than code is how a guard like
        # this gets deleted for crying wolf.
        code = "\n".join(
            line for line in inspect.getsource(T.main).splitlines()
            if not line.strip().startswith("#"))
        self.assertNotIn("median 1.76%", code)
        self.assertNotIn("0.088", code)
        self.assertIn("ZONE_STOP_ATR_MULT", code,
                      "the line must read the multiplier from the module")

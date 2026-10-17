"""Retention for the trade log and the launchd-redirected logs.

Two failures here would do real damage, and they are the first two classes
below: pruning a row from the real-money audit trail, and losing rows the
trader appended while retention was running. Everything else is bookkeeping.

Context, from the 2026-10-16 audit: live_trades.csv was 55MB over 429,311
rows, 99.3% of them skips with result=not_triggered -- the same watched level
restated on every pass. trader.log was 80MB because launchd redirects stdout
and never rotates it.

Nothing is deleted; pruned rows go to a monthly gzip under data/archive/.
"""

import csv
import gzip
import pathlib
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

import tools.retention as R
from research.levels.config import WATCH_TTL_HOURS

NOW = datetime(2026, 10, 16, 12, 0, tzinfo=timezone.utc)
COLS = ["ts_utc", "action", "coin", "result", "detail", "pull_over_run"]


def ts(days_ago):
    return (NOW - timedelta(days=days_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")


def row(action, result, days_ago=0, coin="HYPE"):
    return {"ts_utc": ts(days_ago), "action": action, "coin": coin,
            "result": result, "detail": "x", "pull_over_run": "0.3"}


class Tiering(unittest.TestCase):
    def test_money_rows_are_forever(self):
        for a in ("entry", "closed", "protection", "emergency_exit"):
            self.assertEqual(R.tier(row(a, "filled")), "forever", a)

    def test_a_waiting_skip_is_the_prunable_class(self):
        self.assertEqual(R.tier(row("skip", "not_triggered")), "waiting")

    def test_every_other_skip_is_a_decision(self):
        """not_actionable carries the basis, break and pull/run vetoes -- the
        substrate every feature screen is built from."""
        for res in ("not_actionable", "coin_cooldown", "fetch_failed",
                    "bid_cancelled", ""):
            self.assertEqual(R.tier(row("skip", res)), "decision", res)

    def test_reviews_and_dry_runs_are_decisions(self):
        self.assertEqual(R.tier(row("review", "flagged")), "decision")
        self.assertEqual(R.tier(row("dry_run", "planned")), "decision")

    def test_an_unknown_action_is_kept_as_a_decision(self):
        """A new action added later must not silently fall into the prunable
        class just because nobody updated this file."""
        self.assertEqual(R.tier(row("some_new_action", "whatever")),
                         "decision")

    def test_tiering_does_not_read_the_detail_prose(self):
        """Keyed on the result enum. Two earlier tests in this project passed
        against the wording of comments; detail strings get reworded."""
        a = row("skip", "not_triggered")
        a["detail"] = "completely different wording"
        self.assertEqual(R.tier(a), "waiting")


class KeepDecision(unittest.TestCase):
    def setUp(self):
        self.cut = R.cutoffs(NOW)

    def test_money_rows_survive_any_age(self):
        self.assertTrue(R.keep(row("entry", "filled", days_ago=4000),
                               self.cut))

    def test_a_recent_waiting_skip_is_kept(self):
        self.assertTrue(R.keep(row("skip", "not_triggered", days_ago=1),
                               self.cut))

    def test_an_old_waiting_skip_is_pruned(self):
        self.assertFalse(R.keep(row("skip", "not_triggered", days_ago=90),
                                self.cut))

    def test_a_decision_survives_well_past_the_waiting_window(self):
        self.assertTrue(R.keep(row("skip", "not_actionable", days_ago=90),
                               self.cut))

    def test_a_row_with_no_timestamp_is_kept(self):
        """It cannot be aged out safely. Keeping it costs a line; discarding
        one loses evidence."""
        r = row("skip", "not_triggered")
        r["ts_utc"] = ""
        self.assertTrue(R.keep(r, self.cut))

    def test_the_waiting_window_always_covers_a_full_watch_lifecycle(self):
        """seed_watchlist_from_log replays rows inside WATCH_TTL_HOURS, so the
        window is floored above it however low the config is set."""
        saved = R.RETAIN_WAITING_DAYS
        try:
            R.RETAIN_WAITING_DAYS = 0
            cut = R.cutoffs(NOW)
            inside = row("skip", "not_triggered",
                         days_ago=WATCH_TTL_HOURS / 24.0)
            self.assertTrue(R.keep(inside, cut),
                            "a watch still inside its TTL was prunable")
        finally:
            R.RETAIN_WAITING_DAYS = saved


class Pruning(unittest.TestCase):
    def setUp(self):
        self.tmp = R.DATA = R.ARCHIVE.parent  # placeholder, replaced below

    def build(self, tmp_path, rows):
        R.DATA = tmp_path
        R.ARCHIVE = tmp_path / "archive"
        R.TRADE_LOG = tmp_path / "live_trades.csv"
        with R.TRADE_LOG.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLS)
            w.writeheader()
            w.writerows(rows)

    def read(self):
        with R.TRADE_LOG.open(newline="") as f:
            return list(csv.DictReader(f))

    def test_a_dry_run_changes_nothing(self):
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as d:
            self.build(pathlib.Path(d),
                       [row("skip", "not_triggered", 90)] * 5)
            before = R.TRADE_LOG.read_bytes()
            counts, _ = R.prune_trade_log(apply=False, now=NOW)
            self.assertEqual(counts["pruned"], 5)
            self.assertEqual(R.TRADE_LOG.read_bytes(), before)
            self.assertFalse(R.ARCHIVE.exists())

    def test_apply_prunes_and_archives(self):
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as d:
            old = [row("skip", "not_triggered", 90) for _ in range(5)]
            new = [row("skip", "not_triggered", 1) for _ in range(3)]
            money = [row("entry", "filled", 400)]
            self.build(pathlib.Path(d), old + new + money)
            counts, months = R.prune_trade_log(apply=True, now=NOW)
            self.assertEqual(counts["pruned"], 5)
            left = self.read()
            self.assertEqual(len(left), 4)
            self.assertEqual(sum(1 for r in left if r["action"] == "entry"), 1)
            arch = list(R.ARCHIVE.glob("*.csv.gz"))
            self.assertTrue(arch)
            with gzip.open(arch[0], "rt", newline="") as f:
                self.assertEqual(len(list(csv.DictReader(f))), 5)
            self.assertEqual(sum(months.values()), 5)

    def test_it_REFUSES_to_run_if_a_money_row_would_be_pruned(self):
        """The guard runs before a byte moves. A money row in the prune list
        means tier() is wrong, and the right response is to do nothing."""
        import tempfile, pathlib
        real_keep = R.keep
        with tempfile.TemporaryDirectory() as d:
            self.build(pathlib.Path(d),
                       [row("entry", "filled", 1),
                        row("skip", "not_triggered", 90)])
            before = R.TRADE_LOG.read_bytes()
            # Break the age decision while leaving tier() correct, which is
            # the real failure mode: a retention rule that sweeps up a row
            # the tier table says is permanent.
            R.keep = lambda r, cut: False
            try:
                with self.assertRaises(AssertionError) as cm:
                    R.prune_trade_log(apply=True, now=NOW)
            finally:
                R.keep = real_keep
            self.assertIn("audit-trail", str(cm.exception))
            self.assertEqual(R.TRADE_LOG.read_bytes(), before,
                             "the file was touched before the guard fired")
            self.assertFalse(R.ARCHIVE.exists(),
                             "rows were archived before the guard fired")

    def test_rows_appended_mid_run_are_not_lost(self):
        """The trader appends while this runs. A read-filter-rename loses
        whatever arrived in between, because the writer's fd points at an
        inode rename() has orphaned."""
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as d:
            self.build(pathlib.Path(pathlib.Path(d)),
                       [row("skip", "not_triggered", 90) for _ in range(4)])
            real_archive = R.archive_rows

            def archive_then_append(rows, cols, apply):
                out = real_archive(rows, cols, apply)
                if apply:                       # simulate a concurrent write
                    with R.TRADE_LOG.open("a", newline="") as f:
                        csv.DictWriter(f, fieldnames=COLS).writerow(
                            row("entry", "filled", 0, coin="LATE"))
                return out

            R.archive_rows = archive_then_append
            try:
                counts, _ = R.prune_trade_log(apply=True, now=NOW)
            finally:
                R.archive_rows = real_archive
            left = self.read()
            self.assertIn("LATE", [r["coin"] for r in left],
                          "a row appended mid-run was lost")
            self.assertGreater(counts.get("caught_up_bytes", 0), 0)

    def test_a_missing_trade_log_is_fine(self):
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as d:
            R.TRADE_LOG = pathlib.Path(d) / "nope.csv"
            self.assertEqual(R.prune_trade_log(apply=True, now=NOW), ({}, {}))

    def test_running_twice_prunes_nothing_the_second_time(self):
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as d:
            self.build(pathlib.Path(d),
                       [row("skip", "not_triggered", 90) for _ in range(5)]
                       + [row("entry", "filled", 1)])
            R.prune_trade_log(apply=True, now=NOW)
            counts, _ = R.prune_trade_log(apply=True, now=NOW)
            self.assertEqual(counts["pruned"], 0)


class LogRotation(unittest.TestCase):
    def test_a_small_log_is_left_alone(self):
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "trader.log"
            p.write_text("short\n")
            self.assertIsNone(R.rotate_log(p, apply=True))
            self.assertEqual(p.read_text(), "short\n")

    def test_a_big_log_is_copy_truncated_not_renamed(self):
        """launchd holds these fds open in append mode, so renaming would
        leave the writer appending to an inode nothing can read."""
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "trader.log"
            p.write_text("x" * int(R.LOG_ROTATE_MB * 1024 * 1024 + 10))
            inode = p.stat().st_ino
            res = R.rotate_log(p, apply=True)
            self.assertEqual(res["rotated"], 1)
            self.assertTrue(p.exists(), "the live path must survive")
            self.assertEqual(p.stat().st_ino, inode,
                             "the inode changed -- a writer would be orphaned")
            self.assertEqual(p.stat().st_size, 0)
            gz = list((p.parent / "rotated").glob("trader.log.*.gz"))
            self.assertEqual(len(gz), 1)
            with gzip.open(gz[0], "rt") as f:
                self.assertEqual(len(f.read()),
                                 int(R.LOG_ROTATE_MB * 1024 * 1024 + 10))

    def test_a_dry_run_does_not_rotate(self):
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "trader.log"
            p.write_text("x" * int(R.LOG_ROTATE_MB * 1024 * 1024 + 10))
            res = R.rotate_log(p, apply=False)
            self.assertEqual(res["rotated"], 0)
            self.assertGreater(p.stat().st_size, 0)

    def test_generations_are_bounded(self):
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "trader.log"
            dest = p.parent / "rotated"
            dest.mkdir()
            for i in range(R.LOG_KEEP_GENERATIONS + 4):
                (dest / f"trader.log.2026010{i:02d}T000000Z.gz").write_bytes(b"x")
            p.write_text("x" * int(R.LOG_ROTATE_MB * 1024 * 1024 + 10))
            R.rotate_log(p, apply=True)
            self.assertEqual(len(list(dest.glob("trader.log.*.gz"))),
                             R.LOG_KEEP_GENERATIONS)

    def test_a_missing_log_is_fine(self):
        import pathlib
        self.assertIsNone(R.rotate_log(pathlib.Path("/nope/nothing.log"),
                                       apply=True))


if __name__ == "__main__":
    unittest.main()


class StderrLogsRotateOnASmallerCap(unittest.TestCase):
    """A stderr stream is normally empty, so anything in one is a traceback.

    trader.err.log held 183KB of NameErrors from a half-edited working tree,
    referencing lines that no longer existed -- the first thing anyone
    tailing that file would have seen and believed. The stdout cap of 25MB
    would never have touched it.
    """

    def test_an_err_log_over_the_kb_cap_rotates(self):
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "trader.err.log"
            p.write_bytes(b"Traceback...\n" * 8000)      # ~104KB
            res = R.rotate_log(p, apply=True,
                               min_bytes=R.ERR_ROTATE_KB * 1024)
            self.assertEqual(res["rotated"], 1)
            self.assertEqual(p.stat().st_size, 0, "not truncated in place")
            gz = list((p.parent / "rotated").glob("trader.err.log.*.gz"))
            self.assertEqual(len(gz), 1, "the content was not preserved")

    def test_an_err_log_under_the_cap_is_left_alone(self):
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "trader.err.log"
            p.write_text("one warning\n")
            self.assertIsNone(R.rotate_log(p, apply=True,
                                           min_bytes=R.ERR_ROTATE_KB * 1024))

    def test_the_stdout_cap_would_never_have_caught_it(self):
        """Why this needed its own cap rather than reusing the existing one."""
        self.assertLess(R.ERR_ROTATE_KB * 1024,
                        R.LOG_ROTATE_MB * 1024 * 1024)
        self.assertLess(183 * 1024, R.LOG_ROTATE_MB * 1024 * 1024,
                        "183KB is far under the stdout cap, as it was")
        self.assertGreater(183 * 1024, R.ERR_ROTATE_KB * 1024)

    def test_every_running_job_has_both_streams_covered(self):
        for job in ("trader", "scanner", "protect", "telegram"):
            self.assertIn(f"{job}.err.log", R.ERR_LOGS, f"{job} stderr")
        for job in ("trader", "scanner", "protect", "telegram"):
            self.assertIn(f"{job}.log", R.ROTATABLE, f"{job} stdout")

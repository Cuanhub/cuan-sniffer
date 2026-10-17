"""Error-rate alerting. Audit section D, the second MISSING rail.

Every halt already alerts. Nothing watched the rate of ordinary errors, so a
slow degradation -- rising rejects, a coin whose candles stopped serving,
creeping rate limits -- was silent until it showed up in PnL. The scanner
logged 3,630 failed fetches for each of four non-existent tickers over
several days and nobody was told.

The rail is a sliding one-hour window of recorded errors. Over the threshold
it alerts; it does NOT halt, on purpose: errors are usually the venue having
a bad minute, and a rail that stops trading on transient noise is a worse
failure than the noise. Halting stays with the conditions that mean the
account is wrong about itself.
"""

import unittest
from unittest import mock

import tools.live_levels_trader as T


class Counting(unittest.TestCase):
    def setUp(self):
        T.reset_error_window()

    def test_errors_accumulate(self):
        for _ in range(3):
            T.record_error("fetch", "boom")
        self.assertEqual(T.errors_in_window(), 3)

    def test_old_errors_fall_out_of_the_window(self):
        now = 1_000_000.0
        with mock.patch.object(T.time, "time", return_value=now):
            T.record_error("fetch", "old")
        later = now + T.ERROR_WINDOW_SEC + 1
        with mock.patch.object(T.time, "time", return_value=later):
            self.assertEqual(T.errors_in_window(), 0)

    def test_the_window_is_an_hour(self):
        self.assertEqual(T.ERROR_WINDOW_SEC, 3600)


class Alerting(unittest.TestCase):
    def setUp(self):
        T.reset_error_window()

    def test_below_the_threshold_says_nothing(self):
        with mock.patch.object(T.notify, "send") as sent:
            for _ in range(T.ERROR_RATE_ALERT - 1):
                T.record_error("fetch", "x")
        self.assertFalse(sent.called)

    def test_crossing_the_threshold_alerts_once(self):
        with mock.patch.object(T.notify, "send") as sent:
            for _ in range(T.ERROR_RATE_ALERT + 5):
                T.record_error("fetch", "x")
        self.assertEqual(sent.call_count, 1, "alerted more than once")
        self.assertIn("error", sent.call_args[0][0].lower())

    def test_the_alert_names_the_commonest_source(self):
        with mock.patch.object(T.notify, "send") as sent:
            for _ in range(T.ERROR_RATE_ALERT):
                T.record_error("candle_fetch", "x")
        self.assertIn("candle_fetch", sent.call_args[0][0])

    def test_it_does_NOT_halt(self):
        """Errors are usually the venue having a bad minute. A rail that
        stops trading on transient noise is worse than the noise."""
        st = {"owned": {}, "halted": False}
        with mock.patch.object(T.notify, "send"):
            for _ in range(T.ERROR_RATE_ALERT * 3):
                T.record_error("fetch", "x")
        self.assertFalse(st["halted"])
        src = open("tools/live_levels_trader.py").read()
        body = src[src.index("def record_error"):src.index("def errors_in_window")]
        self.assertNotIn("halt(", body)

    def test_recording_never_raises_into_the_caller(self):
        """It is called from error paths. It must not become the error."""
        with mock.patch.object(T.notify, "send",
                               side_effect=RuntimeError("telegram down")):
            T.record_error("fetch", "x" * 10_000)   # must not raise


class WiredIn(unittest.TestCase):
    def test_the_venue_fetch_path_records(self):
        src = open("tools/live_levels_trader.py").read()
        self.assertGreaterEqual(src.count("record_error("), 3,
                                "error recording is not wired into the "
                                "paths that actually fail")


if __name__ == "__main__":
    unittest.main()

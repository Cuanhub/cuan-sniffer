"""Reusing the 1h bars the scan already fetched, instead of fetching them twice.

call.find_candidates pulls a 1h series per coin, extracts candidates, and used
to discard it. execute_one then asked the venue for the same series again to
run the trigger: 30 duplicate calls out of about 93 a pass, against an endpoint
already returning rate limits.

What makes the reuse safe rather than merely cheap is the WINDOW.
level_has_held derives sigma with log_sigma(closed, len(closed) - 1), which
slices backwards by VOL_LOOKBACK (120). Hand it the 60-day series the scan
fetches and it computes sigma over 121 bars instead of ~70 -- a different
number, which then gates both the touch test and the max-above ceiling. So the
cache must hold exactly the window fetch_1h would have returned, and these
tests pin that equivalence rather than trusting it.
"""

import time
import unittest

from research.levels import config as C
from research.levels.call import find_candidates
from research.levels.trigger import (TRIGGER_1H_HOURS, fetch_1h, level_has_held,
                                     trigger_window)

HOUR = 3_600_000


def series(n_hours, end_ms=None, base=100.0):
    """A 1h series ending at the current hour, oldest first."""
    end = end_ms or int(time.time() * 1000)
    out = []
    for k in range(n_hours):
        t = end - (n_hours - 1 - k) * HOUR
        px = base + (k % 7) * 0.1
        out.append({"t": t, "o": px, "h": px + 0.2, "l": px - 0.2, "c": px,
                    "v": 1000.0})
    return out


class WindowEquivalence(unittest.TestCase):
    def test_the_trim_keeps_only_the_trigger_window(self):
        now = int(time.time() * 1000)
        long_series = series(60 * 24, end_ms=now)       # 60 days, as the scan pulls
        got = trigger_window(long_series, now_ms=now)
        self.assertLessEqual(len(got), TRIGGER_1H_HOURS + 1)
        self.assertGreaterEqual(len(got), TRIGGER_1H_HOURS - 1)
        self.assertGreaterEqual(got[0]["t"], now - TRIGGER_1H_HOURS * HOUR)
        self.assertEqual(got[-1]["t"], long_series[-1]["t"])

    def test_the_trim_is_a_no_op_on_a_series_already_that_long(self):
        now = int(time.time() * 1000)
        short = series(TRIGGER_1H_HOURS, end_ms=now)
        got = trigger_window(short, now_ms=now)
        self.assertEqual([b["t"] for b in got], [b["t"] for b in short])

    def test_it_stays_sorted_oldest_first(self):
        now = int(time.time() * 1000)
        shuffled = list(reversed(series(200, end_ms=now)))
        got = trigger_window(shuffled, now_ms=now)
        self.assertEqual([b["t"] for b in got], sorted(b["t"] for b in got))

    def test_the_bar_dicts_carry_what_the_trigger_reads(self):
        now = int(time.time() * 1000)
        got = trigger_window(series(200, end_ms=now), now_ms=now)
        for key in ("t", "o", "h", "l", "c"):
            self.assertIn(key, got[0])

    def test_malformed_rows_are_dropped_not_raised(self):
        now = int(time.time() * 1000)
        bad = series(80, end_ms=now) + [{"t": now, "o": "x"}, {}, {"t": None}]
        got = trigger_window(bad, now_ms=now)
        self.assertTrue(all(isinstance(b["c"], float) for b in got))

    # -- the property that actually matters ---------------------------------

    def test_the_trigger_answers_identically_from_trimmed_and_fetched_bars(self):
        """A trimmed long series and a real fetch_1h must agree. If they ever
        diverge, the sigma the trigger gates on has moved.

        Both sides take their cutoff from the wall clock, as they do in
        production, where the trim and the fetch it replaces would have run
        seconds apart inside the same pass.
        """
        now = (int(time.time() * 1000) // HOUR) * HOUR
        long_series = series(60 * 24, end_ms=now, base=100.0)
        trimmed = trigger_window(long_series)

        class Client:
            def get_candles(self, coin, tf, start, end):
                return [b for b in long_series if b["t"] >= start]

        fetched = fetch_1h(Client(), "TEST")
        self.assertEqual([b["t"] for b in trimmed], [b["t"] for b in fetched])
        self.assertEqual(trimmed, fetched, "the bar dicts must match exactly")
        for level in (99.0, 99.8, 100.0, 100.3, 101.0):
            self.assertEqual(level_has_held(trimmed, level),
                             level_has_held(fetched, level),
                             f"trigger disagrees at level {level}")

    def test_the_untrimmed_series_really_would_have_changed_the_answer(self):
        """The counter-example justifying the trim. If this ever stops being
        true the trim is harmless, but the reasoning above is then stale."""
        now = (int(time.time() * 1000) // HOUR) * HOUR
        long_series = series(60 * 24, end_ms=now, base=100.0)
        trimmed = trigger_window(long_series, now_ms=now)
        self.assertGreater(len(long_series), C.VOL_LOOKBACK + 2)
        self.assertLess(len(trimmed), C.VOL_LOOKBACK,
                        "the trigger window is shorter than VOL_LOOKBACK, which "
                        "is why series length changes sigma")


class FindCandidatesHandsOverBars(unittest.TestCase):
    class Client:
        def __init__(self):
            self.calls = []

        def get_candles(self, coin, tf, start, end):
            self.calls.append((coin, tf))
            hours = 60 * 24 if tf == "1h" else 125 * 6
            step = HOUR if tf == "1h" else 4 * HOUR
            end_ms = int(time.time() * 1000)
            return [{"t": end_ms - (hours - 1 - k) * step, "o": 100.0, "h": 100.2,
                     "l": 99.8, "c": 100.0, "v": 1.0} for k in range(hours)]

    def test_the_1h_series_is_handed_over(self):
        c = self.Client()
        out = {}
        find_candidates(c, [{"symbol": "AAA"}, {"symbol": "BBB"}], bars_out=out)
        self.assertEqual(set(out), {"AAA", "BBB"})
        self.assertTrue(all(len(v) > 100 for v in out.values()))

    def test_only_the_1h_leg_is_handed_over(self):
        """The 4h series is not what the trigger reads, and storing it under
        the same key would silently feed 4h bars to an hourly rule."""
        c = self.Client()
        out = {}
        find_candidates(c, [{"symbol": "AAA"}], bars_out=out)
        got = out["AAA"]
        spacing = {got[i + 1]["t"] - got[i]["t"] for i in range(min(5, len(got) - 1))}
        self.assertEqual(spacing, {HOUR})

    def test_omitting_bars_out_changes_nothing(self):
        c1, c2 = self.Client(), self.Client()
        a = find_candidates(c1, [{"symbol": "AAA"}])
        b = find_candidates(c2, [{"symbol": "AAA"}], bars_out={})
        self.assertEqual(len(a), len(b))
        self.assertEqual(c1.calls, c2.calls)

    def test_a_populated_cache_means_the_trigger_makes_no_call(self):
        import tools.live_levels_trader as T
        now = (int(time.time() * 1000) // HOUR) * HOUR
        cache = {"AAA": trigger_window(series(500, end_ms=now), now_ms=now)}

        def explode():
            raise AssertionError("cached_1h fetched when the cache was populated")

        saved = T._ro_client
        T._ro_client = explode
        try:
            got = T.cached_1h("AAA", cache)
        finally:
            T._ro_client = saved
        self.assertEqual(got, cache["AAA"])

    def test_a_missing_coin_still_falls_back_to_a_fetch(self):
        """A coin whose 1h leg failed during the scan must still get its own
        chance, rather than inheriting another request's error."""
        import tools.live_levels_trader as T
        calls = []

        class RO:
            def get_candles(self, coin, tf, start, end):
                calls.append(coin)
                return series(80)

        saved = T._ro_client
        T._ro_client = lambda: RO()
        try:
            got = T.cached_1h("ZZZ", {})
        finally:
            T._ro_client = saved
        self.assertEqual(calls, ["ZZZ"])
        self.assertTrue(got)


if __name__ == "__main__":
    unittest.main()

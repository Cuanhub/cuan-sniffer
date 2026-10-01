"""Caching the 4h series between passes without letting the price go stale.

A 4h series was refetched every 450s pass -- about 32 times per bar, 30 coins
deep -- for data that by definition cannot change until the bar closes. That
was 30 of the pass's remaining ~63 venue calls.

The trap is that the last bar is NOT closed. Its close is what everything
downstream treats as "now": _candidates_for sets px = bars[-1]["c"], and px
decides which levels are candidates at all (L.price >= px), the depth the
admission gate measures (near_sigma against TRADE_MIN_NEAR_SIGMA = 5.0) and
which resistance rungs are above price. Caching the series wholesale would
quietly shift which setups qualify.

So the closed history is reused and the forming bar is refreshed from one
allMids request covering the whole universe. High and low only widen within a
bar, so ratcheting them against each observed mid moves toward the true
extremes and can never overstate them.

1h is deliberately NOT cached: the trigger reads its last couple of hours to
decide whether a level has held, which is exactly the data that must be fresh.
"""

import time
import unittest

from research.levels.call import (_apply_live_price, _bar_open_ms, _CACHEABLE_TF,
                                  _fetch_tf, find_candidates)

HOUR = 3_600_000


class Client:
    def __init__(self):
        self.calls = []

    def get_candles(self, coin, tf, start, end):
        self.calls.append((coin, tf))
        n, step = (200, 4 * HOUR) if tf == "4h" else (60 * 24, HOUR)
        now = int(time.time() * 1000)
        return [{"t": now - (n - 1 - k) * step, "o": 100.0, "h": 100.5,
                 "l": 99.5, "c": 100.0, "v": 1.0} for k in range(n)]

    def get_all_mids(self):
        return {"AAA": 101.25, "BBB": 99.0}


class BarBoundary(unittest.TestCase):
    def test_the_key_is_the_bar_open_not_a_ttl(self):
        """Expiry by boundary, not by age: that is what makes reuse correct
        rather than merely time-boxed."""
        t = 1790812800000                      # a 4h boundary
        self.assertEqual(_bar_open_ms(t, "4h"), t)
        self.assertEqual(_bar_open_ms(t + 1, "4h"), t)
        self.assertEqual(_bar_open_ms(t + 4 * HOUR - 1, "4h"), t)
        self.assertEqual(_bar_open_ms(t + 4 * HOUR, "4h"), t + 4 * HOUR)

    def test_an_unknown_timeframe_never_pretends_to_be_cacheable(self):
        self.assertEqual(_bar_open_ms(12345, "7m"), 12345)

    def test_only_4h_is_cacheable(self):
        self.assertEqual(_CACHEABLE_TF, ("4h",))
        self.assertNotIn("1h", _CACHEABLE_TF)


class FormingBar(unittest.TestCase):
    def bars(self):
        return [{"t": 0, "o": 100.0, "h": 100.5, "l": 99.5, "c": 100.0},
                {"t": HOUR, "o": 100.0, "h": 100.5, "l": 99.5, "c": 100.0}]

    def test_the_close_is_replaced_by_the_mid(self):
        got = _apply_live_price(self.bars(), 101.25)
        self.assertEqual(got[-1]["c"], 101.25)

    def test_the_high_ratchets_up_but_never_down(self):
        got = _apply_live_price(self.bars(), 101.25)
        self.assertEqual(got[-1]["h"], 101.25)
        got = _apply_live_price(got, 100.0)
        self.assertEqual(got[-1]["h"], 101.25, "a high must not shrink")

    def test_the_low_ratchets_down_but_never_up(self):
        got = _apply_live_price(self.bars(), 98.0)
        self.assertEqual(got[-1]["l"], 98.0)
        got = _apply_live_price(got, 100.0)
        self.assertEqual(got[-1]["l"], 98.0, "a low must not rise")

    def test_a_mid_inside_the_range_leaves_the_extremes_alone(self):
        got = _apply_live_price(self.bars(), 100.1)
        self.assertEqual((got[-1]["h"], got[-1]["l"]), (100.5, 99.5))
        self.assertEqual(got[-1]["c"], 100.1)

    def test_closed_bars_are_untouched(self):
        got = _apply_live_price(self.bars(), 101.25)
        self.assertEqual(got[0], {"t": 0, "o": 100.0, "h": 100.5, "l": 99.5,
                                  "c": 100.0})

    def test_a_missing_or_bad_mid_changes_nothing(self):
        for mid in (0.0, -1.0):
            got = _apply_live_price(self.bars(), mid)
            self.assertEqual(got[-1]["c"], 100.0)

    def test_it_survives_an_empty_series(self):
        self.assertEqual(_apply_live_price([], 101.0), [])


class CacheBehaviour(unittest.TestCase):
    def test_a_second_4h_call_in_the_same_bar_makes_no_request(self):
        c, cache = Client(), {}
        _fetch_tf(c, "AAA", "4h", cache=cache, mids={"AAA": 101.0})
        n = len(c.calls)
        _fetch_tf(c, "AAA", "4h", cache=cache, mids={"AAA": 101.0})
        self.assertEqual(len(c.calls), n, "the cached bar was refetched")

    def test_the_cached_series_still_carries_the_live_price(self):
        c, cache = Client(), {}
        _fetch_tf(c, "AAA", "4h", cache=cache, mids={"AAA": 100.0})
        got = _fetch_tf(c, "AAA", "4h", cache=cache, mids={"AAA": 101.25})
        self.assertEqual(got[-1]["c"], 101.25,
                         "px must come from the mid, not the cached bar")

    def test_a_1h_call_is_never_served_from_cache(self):
        c, cache = Client(), {}
        _fetch_tf(c, "AAA", "1h", cache=cache, mids={"AAA": 101.0})
        n = len(c.calls)
        _fetch_tf(c, "AAA", "1h", cache=cache, mids={"AAA": 101.0})
        self.assertEqual(len(c.calls), n + 1, "1h must stay fresh for the trigger")

    def test_a_stale_bar_key_forces_a_refetch(self):
        c, cache = Client(), {}
        _fetch_tf(c, "AAA", "4h", cache=cache, mids={})
        (coin, tf), (_, bars) = next(iter(cache.items())), next(iter(cache.values()))
        cache[("AAA", "4h")] = (0, bars)        # pretend it came from an old bar
        n = len(c.calls)
        _fetch_tf(c, "AAA", "4h", cache=cache, mids={})
        self.assertEqual(len(c.calls), n + 1)

    def test_without_a_cache_nothing_changes(self):
        c1, c2 = Client(), Client()
        find_candidates(c1, [{"symbol": "AAA"}])
        find_candidates(c2, [{"symbol": "AAA"}], tf_cache={}, mids={"AAA": 100.0})
        self.assertEqual(c1.calls, c2.calls, "the first pass must cost the same")

    def test_the_saving_across_passes(self):
        c, cache = Client(), {}
        uni = [{"symbol": "AAA"}, {"symbol": "BBB"}]
        find_candidates(c, uni, tf_cache=cache, mids=c.get_all_mids())
        first = len(c.calls)
        find_candidates(c, uni, tf_cache=cache, mids=c.get_all_mids())
        second = len(c.calls) - first
        self.assertEqual(first, 4, "first pass: 2 coins x 2 timeframes")
        self.assertEqual(second, 2, "second pass: only the 1h legs")

    def test_the_cache_holds_one_entry_per_coin_and_timeframe(self):
        c, cache = Client(), {}
        uni = [{"symbol": "AAA"}, {"symbol": "BBB"}]
        for _ in range(5):
            find_candidates(c, uni, tf_cache=cache, mids=c.get_all_mids())
        self.assertEqual(set(cache), {("AAA", "4h"), ("BBB", "4h")})


if __name__ == "__main__":
    unittest.main()

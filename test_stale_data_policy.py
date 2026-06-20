"""
test_stale_data_policy.py
-------------------------
Focused tests for live data freshness, stale-cache blocking, and API backoff.

Run:
    python3 -m unittest test_stale_data_policy.py -v

No network access is required. Tests that need optional live-trading
dependencies skip cleanly when those dependencies are not installed.
"""

import time
import types
import os
import tempfile
import unittest

from live_data_guard import ApiFailureBackoff, cache_is_fresh


class TestLiveDataGuard(unittest.TestCase):
    def test_fresh_cache_allowed(self):
        now = time.time()
        self.assertTrue(cache_is_fresh(now - 30, 90, now=now))

    def test_stale_cache_blocked(self):
        now = time.time()
        self.assertFalse(cache_is_fresh(now - 120, 90, now=now))

    def test_api_burst_triggers_backoff(self):
        guard = ApiFailureBackoff(burst_limit=3, backoff_seconds=30)
        self.assertFalse(guard.status()[0])
        guard.record_failure("unit", "timeout-1")
        guard.record_failure("unit", "timeout-2")
        active, remaining = guard.record_failure("unit", "timeout-3")
        self.assertTrue(active)
        self.assertGreater(remaining, 0.0)
        self.assertTrue(guard.status()[0])


class TestPerpDataFreshness(unittest.TestCase):
    def _feed(self):
        try:
            from perp_data import PerpDataFeed
        except Exception as exc:
            self.skipTest(f"perp_data import unavailable: {exc}")
        return PerpDataFeed(coin="SOL", interval="1h", max_candles=10, debug=False)

    def test_stale_candle_cache_blocks_live_signal_generation(self):
        feed = self._feed()
        feed._cached_snapshot = [{"close": 100.0}]
        feed._cached_snapshot_ts = time.time() - 120
        status = feed.get_market_data_status(max_age_seconds=90)
        self.assertFalse(status["fresh"])
        self.assertEqual(status["data_source"], "stale_cache")

    def test_fresh_candle_cache_allowed(self):
        feed = self._feed()
        feed._cached_snapshot = [{"close": 100.0}]
        feed._cached_snapshot_ts = time.time() - 30
        status = feed.get_market_data_status(max_age_seconds=90)
        self.assertTrue(status["fresh"])
        self.assertEqual(status["data_source"], "cache")


class TestSentimentFreshness(unittest.TestCase):
    def test_stale_sentiment_becomes_neutral(self):
        try:
            from perp_sentiment import PerpSentimentFeed, PerpSentimentSnapshot
        except Exception as exc:
            self.skipTest(f"perp_sentiment import unavailable: {exc}")

        feed = PerpSentimentFeed(coin="SOL", debug=False)
        feed._snapshot = PerpSentimentSnapshot(
            coin="SOL",
            funding_rate=0.04,
            open_interest=123.0,
            bias=1.2,
            premium=0.2,
            fetched_at=time.time() - 1000,
            data_source="fresh",
        )
        snap = feed.get_snapshot(max_age_sec=900)
        self.assertTrue(snap.stale_neutralized)
        self.assertEqual(snap.funding_rate, 0.0)
        self.assertEqual(snap.open_interest, 0.0)
        self.assertEqual(snap.bias, 0.0)


class TestExecutorFreshness(unittest.TestCase):
    def test_stale_margin_cache_blocks_new_entries(self):
        try:
            from executor import Executor
        except Exception as exc:
            self.skipTest(f"executor import unavailable: {exc}")

        executor = Executor.__new__(Executor)
        executor._live_mode = True
        executor._venue_margin_cache = (100.0, 0.0, 100.0)
        executor._venue_margin_cache_ts = time.time() - 120
        executor._last_margin_cache_age_sec = None
        executor._last_margin_block_reason = ""

        snapshot = Executor._venue_margin_snapshot_with_warning(
            executor,
            reason="unit_test",
        )
        self.assertIsNone(snapshot)
        self.assertEqual(executor._last_margin_block_reason, "stale_margin_cache")

    def test_stale_midprice_blocks_execution(self):
        try:
            from live_execution_backend import LiveExecutionBackend
            from live_data_guard import GLOBAL_API_BACKOFF
        except Exception as exc:
            self.skipTest(f"live backend import unavailable: {exc}")

        class FailingInfo:
            def all_mids(self):
                raise RuntimeError("timeout")

        GLOBAL_API_BACKOFF.reset()
        backend = LiveExecutionBackend.__new__(LiveExecutionBackend)
        backend.debug = False
        backend.info = FailingInfo()
        backend._mid_cache = {"SOL": 100.0}
        backend._mid_cache_ts = time.time() - 20

        self.assertIsNone(backend.get_mid_price("SOL", max_age_sec=10))


class TestAgentStaleSkip(unittest.TestCase):
    def test_skipped_symbols_count_increments_for_stale_data(self):
        try:
            import agent
            from live_data_guard import GLOBAL_API_BACKOFF
        except Exception as exc:
            self.skipTest(f"agent import unavailable: {exc}")

        class StaleFeed:
            def __init__(self):
                self.df_requested = False

            def get_market_data_status(self, max_age_seconds):
                return {
                    "fresh": False,
                    "has_cache": True,
                    "cache_age": 120.0,
                    "data_source": "cache",
                    "fetch_source": "cache_fallback",
                    "fetch_duration_sec": 0.0,
                    "last_error": "timeout",
                    "max_age_seconds": max_age_seconds,
                }

            def get_ohlcv_df(self):
                self.df_requested = True
                raise AssertionError("stale feed should not reach signal generation")

        GLOBAL_API_BACKOFF.reset()
        original_log = agent.log_gate_reject
        agent.log_gate_reject = lambda **kwargs: None
        try:
            state = types.SimpleNamespace(
                coin="SOL",
                perp_feed=StaleFeed(),
            )
            traded, skipped = agent.process_coin(state, executor=object())
        finally:
            agent.log_gate_reject = original_log

        self.assertFalse(traded)
        self.assertTrue(skipped)
        self.assertFalse(state.perp_feed.df_requested)


class TestAgentDataHealthBreaker(unittest.TestCase):
    def test_data_health_breaker_blocks_when_stale_ratio_high(self):
        try:
            import agent
        except Exception as exc:
            self.skipTest(f"agent import unavailable: {exc}")

        class Feed:
            def __init__(self, fresh):
                self.fresh = fresh

            def get_market_data_status(self, max_age_seconds):
                return {
                    "fresh": self.fresh,
                    "has_cache": True,
                    "cache_age": 10.0 if self.fresh else 120.0,
                    "data_source": "fresh" if self.fresh else "stale_cache",
                    "fetch_source": "network",
                    "fetch_duration_sec": 0.0,
                    "last_error": "",
                    "max_age_seconds": max_age_seconds,
                    "poll_thread_alive": True,
                }

        states = {
            f"C{i}": types.SimpleNamespace(perp_feed=Feed(fresh=(i >= 4)))
            for i in range(10)
        }
        summary = agent.summarize_data_health(states)
        self.assertEqual(summary["stale_count"], 4)
        self.assertTrue(agent.data_health_breaker_active(summary))

    def test_operator_pause_takes_precedence_over_data_health(self):
        try:
            import agent
        except Exception as exc:
            self.skipTest(f"agent import unavailable: {exc}")

        original = agent.PAUSE_NEW_SIGNALS
        try:
            agent.PAUSE_NEW_SIGNALS = True
            self.assertEqual(
                agent.signal_generation_pause_reason(health_block=True),
                "operator_pause",
            )
        finally:
            agent.PAUSE_NEW_SIGNALS = original

    def test_data_health_pause_reason_when_operator_pause_disabled(self):
        try:
            import agent
        except Exception as exc:
            self.skipTest(f"agent import unavailable: {exc}")

        original = agent.PAUSE_NEW_SIGNALS
        try:
            agent.PAUSE_NEW_SIGNALS = False
            self.assertEqual(
                agent.signal_generation_pause_reason(health_block=True),
                "data_health_circuit_breaker",
            )
            self.assertEqual(agent.signal_generation_pause_reason(health_block=False), "")
        finally:
            agent.PAUSE_NEW_SIGNALS = original


class TestAgentProcessLock(unittest.TestCase):
    def test_agent_lock_blocks_second_instance(self):
        try:
            import agent
        except Exception as exc:
            self.skipTest(f"agent import unavailable: {exc}")

        with tempfile.TemporaryDirectory() as tmp:
            lock_path = os.path.join(tmp, "agent.lock")
            with agent.agent_process_lock(lock_path):
                with self.assertRaises(agent.AgentLockUnavailable):
                    with agent.agent_process_lock(lock_path):
                        pass


if __name__ == "__main__":
    unittest.main(verbosity=2)

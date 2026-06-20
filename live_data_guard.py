"""
Shared live-data freshness and API backoff guard.

This module is deliberately dependency-free so it can be used by feeds, the
agent, executor, and tests without importing trading components.
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from typing import Deque, Optional, Tuple


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return float(default)


def _int_env(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return int(default)


LIVE_MAX_CANDLE_CACHE_AGE_SECONDS = _float_env("LIVE_MAX_CANDLE_CACHE_AGE_SECONDS", 90.0)
LIVE_MAX_SENTIMENT_CACHE_AGE_SECONDS = _float_env("LIVE_MAX_SENTIMENT_CACHE_AGE_SECONDS", 900.0)
LIVE_MAX_MARGIN_CACHE_AGE_SECONDS = _float_env("LIVE_MAX_MARGIN_CACHE_AGE_SECONDS", 60.0)
LIVE_MAX_MIDPRICE_AGE_SECONDS = _float_env("LIVE_MAX_MIDPRICE_AGE_SECONDS", 10.0)
API_GLOBAL_BACKOFF_SECONDS = _float_env("API_GLOBAL_BACKOFF_SECONDS", 30.0)
API_FAILURE_BURST_LIMIT = _int_env("API_FAILURE_BURST_LIMIT", 5)


def cache_age_seconds(timestamp: float, now: Optional[float] = None) -> Optional[float]:
    if not timestamp or timestamp <= 0:
        return None
    ref = time.time() if now is None else float(now)
    return max(0.0, ref - float(timestamp))


def cache_is_fresh(timestamp: float, max_age_seconds: float, now: Optional[float] = None) -> bool:
    age = cache_age_seconds(timestamp, now=now)
    return age is not None and age <= float(max_age_seconds)


class ApiFailureBackoff:
    def __init__(self, burst_limit: int, backoff_seconds: float):
        self.burst_limit = max(1, int(burst_limit))
        self.backoff_seconds = max(0.0, float(backoff_seconds))
        self._lock = threading.Lock()
        self._failure_ts: Deque[float] = deque(maxlen=max(1, self.burst_limit * 4))
        self._backoff_until = 0.0
        self._last_reason = ""

    def record_success(self) -> None:
        with self._lock:
            self._failure_ts.clear()

    def record_failure(self, source: str, error: object = "") -> Tuple[bool, float]:
        now = time.time()
        with self._lock:
            window = max(1.0, self.backoff_seconds)
            cutoff = now - window
            while self._failure_ts and self._failure_ts[0] < cutoff:
                self._failure_ts.popleft()

            self._failure_ts.append(now)
            active = False
            remaining = max(0.0, self._backoff_until - now)
            if len(self._failure_ts) >= self.burst_limit:
                self._backoff_until = max(self._backoff_until, now + self.backoff_seconds)
                remaining = max(0.0, self._backoff_until - now)
                self._last_reason = f"{source}:{str(error)[:160]}"
                active = True
                print(
                    f"[API_BACKOFF] active source={source} failures={len(self._failure_ts)} "
                    f"limit={self.burst_limit} backoff={remaining:.1f}s err={str(error)[:160]}"
                )
        return active, remaining

    def status(self) -> Tuple[bool, float, str]:
        now = time.time()
        with self._lock:
            remaining = max(0.0, self._backoff_until - now)
            if remaining <= 0:
                return False, 0.0, self._last_reason
            return True, remaining, self._last_reason

    def reset(self) -> None:
        with self._lock:
            self._failure_ts.clear()
            self._backoff_until = 0.0
            self._last_reason = ""


GLOBAL_API_BACKOFF = ApiFailureBackoff(
    burst_limit=API_FAILURE_BURST_LIMIT,
    backoff_seconds=API_GLOBAL_BACKOFF_SECONDS,
)


def record_api_success() -> None:
    GLOBAL_API_BACKOFF.record_success()


def record_api_failure(source: str, error: object = "") -> Tuple[bool, float]:
    return GLOBAL_API_BACKOFF.record_failure(source=source, error=error)


def api_backoff_status() -> Tuple[bool, float, str]:
    return GLOBAL_API_BACKOFF.status()


def api_backoff_active() -> bool:
    active, _, _ = api_backoff_status()
    return active

import time
import threading
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Tuple

import requests
import pandas as pd
import os

# Hyperliquid info endpoint (public, no auth needed)
HYPERLIQUID_INFO_URL = "https://api.hyperliquid.xyz/info"
PERP_FETCH_MAX_RETRIES = int(os.getenv("PERP_FETCH_MAX_RETRIES", "1"))
PERP_FETCH_MAX_TOTAL_SEC = float(os.getenv("PERP_FETCH_MAX_TOTAL_SEC", "1.5"))
PERP_FETCH_TIMEOUT_SEC = float(os.getenv("PERP_FETCH_TIMEOUT_SEC", "0.9"))
PERP_FETCH_RETRY_SLEEP_SEC = float(os.getenv("PERP_FETCH_RETRY_SLEEP_SEC", "0.1"))


class PerpDataFeed:
    """
    REST-based OHLCV feed for Hyperliquid perpetuals.

    Improvements:
    - Shared HTTP session per feed
    - In-memory candle cache with TTL
    - Global request spacing per process
    - Exponential backoff on 429 / transient failures
    - Poll-loop aware refresh skipping
    - Cleaner debug output
    """

    # ------------------------------------------------------------------
    # Class-level throttling shared across all feed instances
    # ------------------------------------------------------------------
    _global_request_lock = threading.Lock()
    _last_global_request_ts = 0.0
    _min_global_request_gap_sec = 0.35

    def __init__(
        self,
        coin: str = "SOL",
        interval: str = "1m",
        max_candles: int = 500,
        debug: bool = True,
    ):
        self.coin = coin
        self.interval = interval
        self.max_candles = max_candles
        self.debug = debug

        # internal buffer: list of dicts {time, open, high, low, close, volume}
        self._candles = deque(maxlen=max_candles)
        self._lock = threading.Lock()

        # shared requests session per feed
        self._session = requests.Session()

        # local cache for latest snapshot
        self._cached_snapshot: List[Dict] = []
        self._cached_snapshot_ts: float = 0.0

        # controls how often refresh is allowed to hit network
        self._last_refresh_ts: float = 0.0
        self._last_fetch_source: str = "init"
        self._last_fetch_duration_sec: float = 0.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _interval_minutes(self) -> int:
        """
        Convert interval string like "1m", "5m", "15m", "1h" into minutes.
        Defaults to 1 if parsing fails.
        """
        s = str(self.interval).lower().strip()

        try:
            if s.endswith("m"):
                return int(s[:-1])
            if s.endswith("h"):
                return 60 * int(s[:-1])
        except Exception:
            pass

        return 1

    def _recommended_poll_interval_sec(self) -> int:
        """
        Poll less aggressively relative to candle size.
        """
        bar_minutes = self._interval_minutes()

        if bar_minutes <= 1:
            return 15
        if bar_minutes <= 5:
            return 30
        if bar_minutes <= 15:
            return 60
        if bar_minutes <= 60:
            return 180

        return 300

    def _cache_ttl_sec(self) -> int:
        """
        Cache snapshots long enough to avoid hammering the endpoint,
        but short enough to keep the latest candle reasonably fresh.
        """
        bar_minutes = self._interval_minutes()

        if bar_minutes <= 1:
            return 10
        if bar_minutes <= 5:
            return 20
        if bar_minutes <= 15:
            return 45
        if bar_minutes <= 60:
            return 90

        return 180

    def _respect_global_rate_limit(self, deadline: Optional[float] = None) -> bool:
        """
        Enforce a minimum gap between outbound requests across all
        PerpDataFeed instances in the current process.
        """
        with PerpDataFeed._global_request_lock:
            now = time.time()
            elapsed = now - PerpDataFeed._last_global_request_ts
            wait_needed = PerpDataFeed._min_global_request_gap_sec - elapsed

            if wait_needed > 0:
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return False
                    wait_needed = min(wait_needed, max(0.0, remaining - 0.02))
                if wait_needed > 0:
                    time.sleep(wait_needed)

            PerpDataFeed._last_global_request_ts = time.time()
        return True

    def _post_with_backoff(
        self,
        payload: Dict,
        max_retries: Optional[int] = None,
        max_total_sec: Optional[float] = None,
    ) -> Optional[List[Dict]]:
        """
        POST to Hyperliquid with retry/backoff protection.
        """
        retry_cap = PERP_FETCH_MAX_RETRIES if max_retries is None else int(max_retries)
        retry_cap = max(0, retry_cap)
        total_attempts = 1 + retry_cap
        budget_sec = (
            PERP_FETCH_MAX_TOTAL_SEC
            if max_total_sec is None
            else max(0.2, float(max_total_sec))
        )
        deadline = time.monotonic() + budget_sec
        last_error = None

        for attempt in range(total_attempts):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                if not self._respect_global_rate_limit(deadline=deadline):
                    break

                req_timeout = max(0.2, min(PERP_FETCH_TIMEOUT_SEC, remaining))
                resp = self._session.post(
                    HYPERLIQUID_INFO_URL,
                    json=payload,
                    timeout=req_timeout,
                )

                if resp.status_code == 429:
                    if attempt >= total_attempts - 1:
                        break
                    sleep_s = min(PERP_FETCH_RETRY_SLEEP_SEC, max(0.0, deadline - time.monotonic()))
                    if self.debug:
                        print(
                            f"[PERP_DATA] 429 for {self.coin}-PERP ({self.interval})"
                            f" | retry={attempt + 1}/{total_attempts}"
                            f" | sleeping {sleep_s:.1f}s"
                        )
                    if sleep_s > 0:
                        time.sleep(sleep_s)
                    continue

                resp.raise_for_status()
                data = resp.json()

                if not isinstance(data, list):
                    if self.debug:
                        print(f"[PERP_DATA] Unexpected candles response format for {self.coin}: {data}")
                    return []

                return data

            except requests.RequestException as e:
                last_error = e
                if attempt >= total_attempts - 1:
                    break
                sleep_s = min(PERP_FETCH_RETRY_SLEEP_SEC, max(0.0, deadline - time.monotonic()))
                if self.debug:
                    print(
                        f"[PERP_DATA] Request error for {self.coin}-PERP ({self.interval})"
                        f" | retry={attempt + 1}/{total_attempts}"
                        f" | sleeping {sleep_s:.1f}s"
                        f" | err={e}"
                    )
                if sleep_s > 0:
                    time.sleep(sleep_s)

        if self.debug:
            print(f"[PERP_DATA] Error fetching candles for {self.coin}-PERP: {last_error}")
        return []

    # ------------------------------------------------------------------
    # Snapshot fetch
    # ------------------------------------------------------------------

    def _fetch_candles_snapshot(self, lookback_minutes: Optional[int] = None) -> List[Dict]:
        """
        Fetches a snapshot of recent candles from Hyperliquid.

        Uses cache + backoff + shared request spacing.
        """
        now_ts = time.time()
        fetch_started = time.monotonic()
        ttl = self._cache_ttl_sec()

        # Serve recent cached snapshot first
        if self._cached_snapshot and (now_ts - self._cached_snapshot_ts) < ttl:
            self._last_fetch_source = "cache_fresh"
            self._last_fetch_duration_sec = 0.0
            if self.debug:
                age = now_ts - self._cached_snapshot_ts
                print(
                    f"[PERP_DATA] Cache hit for {self.coin}-PERP ({self.interval})"
                    f" | age={age:.1f}s"
                )
            return list(self._cached_snapshot)

        if lookback_minutes is None:
            bar_minutes = self._interval_minutes()
            desired_bars = self.max_candles + 50
            lookback_minutes = bar_minutes * desired_bars
            lookback_minutes = max(lookback_minutes, 600)

        now = datetime.utcnow()
        start = now - timedelta(minutes=lookback_minutes)

        start_ms = int(start.timestamp() * 1000)
        end_ms = int(now.timestamp() * 1000)

        payload = {
            "type": "candleSnapshot",
            "req": {
                "coin": self.coin,
                "interval": self.interval,
                "startTime": start_ms,
                "endTime": end_ms,
            },
        }

        data = self._post_with_backoff(payload)
        if not data:
            if self._cached_snapshot:
                self._last_fetch_source = "cache_fallback"
                self._last_fetch_duration_sec = time.monotonic() - fetch_started
                if self.debug:
                    age = now_ts - self._cached_snapshot_ts
                    print(
                        f"[PERP_DATA] Fail-soft using cached candles for {self.coin}-PERP ({self.interval})"
                        f" | cache_age={age:.1f}s"
                        f" | fetch_time={self._last_fetch_duration_sec:.2f}s"
                    )
                return list(self._cached_snapshot)
            self._last_fetch_source = "failed"
            self._last_fetch_duration_sec = time.monotonic() - fetch_started
            return []

        candles = []
        for c in data:
            try:
                t_ms = c.get("T") or c.get("t")
                ts = datetime.fromtimestamp(t_ms / 1000.0, tz=timezone.utc)

                candles.append(
                    {
                        "time": ts,
                        "open": float(c["o"]),
                        "high": float(c["h"]),
                        "low": float(c["l"]),
                        "close": float(c["c"]),
                        "volume": float(c["v"]),
                    }
                )
            except Exception as e:
                if self.debug:
                    print(f"[PERP_DATA] Error parsing candle for {self.coin}: {e}, raw: {c}")
                continue

        # Update local cache
        self._cached_snapshot = list(candles)
        self._cached_snapshot_ts = time.time()
        self._last_fetch_source = "network"
        self._last_fetch_duration_sec = time.monotonic() - fetch_started

        return candles

    # ------------------------------------------------------------------
    # Public refresh
    # ------------------------------------------------------------------

    def refresh(self, force: bool = False):
        """
        One-shot refresh: fetch recent candles and update the internal buffer.

        Unless force=True, skips network refresh if called too frequently.
        """
        now_ts = time.time()
        min_refresh_gap = self._cache_ttl_sec()

        if not force and (now_ts - self._last_refresh_ts) < min_refresh_gap:
            if self.debug:
                age = now_ts - self._last_refresh_ts
                print(
                    f"[PERP_DATA] Refresh skipped for {self.coin}-PERP ({self.interval})"
                    f" | last_refresh={age:.1f}s ago"
                )
            return

        new_candles = self._fetch_candles_snapshot()
        self._last_refresh_ts = time.time()

        if not new_candles:
            return

        with self._lock:
            self._candles.clear()
            for c in new_candles[-self.max_candles:]:
                self._candles.append(c)

        if self.debug:
            print(
                f"[PERP_DATA] Loaded {len(self._candles)} candles for "
                f"{self.coin}-PERP ({self.interval})"
            )

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------

    def run_poll_loop(self, interval_sec: Optional[int] = None):
        """
        Background loop: periodically refreshes candles.

        If no interval is provided, choose a safer default based on timeframe.
        """
        if interval_sec is None:
            interval_sec = self._recommended_poll_interval_sec()

        if self.debug:
            print(
                f"[PERP_DATA] Starting poll loop for {self.coin}-PERP ({self.interval})"
                f" | every {interval_sec}s"
            )

        while True:
            try:
                self.refresh(force=False)
            except Exception as e:
                print(f"[PERP_DATA] Poll loop error for {self.coin}-PERP: {e}")

            time.sleep(interval_sec)

    def start(self, interval_sec: Optional[int] = None):
        """
        Starts the background polling thread.
        """
        t = threading.Thread(
            target=self.run_poll_loop,
            args=(interval_sec,),
            daemon=True,
        )
        t.start()

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    def get_ohlcv_df(self) -> pd.DataFrame:
        """
        Returns the buffered OHLCV data as a pandas DataFrame.
        Columns: time, open, high, low, close, volume
        """
        with self._lock:
            data = list(self._candles)

        if not data:
            return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

        df = pd.DataFrame(data)
        df = df.sort_values("time").reset_index(drop=True)
        return df

    def get_last_fetch_status(self) -> Tuple[str, float]:
        return self._last_fetch_source, float(self._last_fetch_duration_sec)

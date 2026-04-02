import time
import threading
from collections import deque
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

import requests
import pandas as pd

# Hyperliquid info endpoint (public, no auth needed)
HYPERLIQUID_INFO_URL = "https://api.hyperliquid.xyz/info"


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

    def _respect_global_rate_limit(self):
        """
        Enforce a minimum gap between outbound requests across all
        PerpDataFeed instances in the current process.
        """
        with PerpDataFeed._global_request_lock:
            now = time.time()
            elapsed = now - PerpDataFeed._last_global_request_ts
            wait_needed = PerpDataFeed._min_global_request_gap_sec - elapsed

            if wait_needed > 0:
                time.sleep(wait_needed)

            PerpDataFeed._last_global_request_ts = time.time()

    def _post_with_backoff(self, payload: Dict, max_retries: int = 5) -> Optional[List[Dict]]:
        """
        POST to Hyperliquid with retry/backoff protection.
        """
        last_error = None

        for attempt in range(max_retries):
            try:
                self._respect_global_rate_limit()

                resp = self._session.post(
                    HYPERLIQUID_INFO_URL,
                    json=payload,
                    timeout=12,
                )

                if resp.status_code == 429:
                    sleep_s = min(10.0, 0.8 * (2 ** attempt))
                    if self.debug:
                        print(
                            f"[PERP_DATA] 429 for {self.coin}-PERP ({self.interval})"
                            f" | retry={attempt + 1}/{max_retries}"
                            f" | sleeping {sleep_s:.1f}s"
                        )
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
                sleep_s = min(10.0, 0.8 * (2 ** attempt))
                if self.debug:
                    print(
                        f"[PERP_DATA] Request error for {self.coin}-PERP ({self.interval})"
                        f" | retry={attempt + 1}/{max_retries}"
                        f" | sleeping {sleep_s:.1f}s"
                        f" | err={e}"
                    )
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
        ttl = self._cache_ttl_sec()

        # Serve recent cached snapshot first
        if self._cached_snapshot and (now_ts - self._cached_snapshot_ts) < ttl:
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
            return []

        candles = []
        for c in data:
            try:
                t_ms = c.get("T") or c.get("t")
                ts = datetime.utcfromtimestamp(t_ms / 1000.0)

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
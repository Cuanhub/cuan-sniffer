import time
import threading
from collections import deque
from datetime import datetime, timedelta
from typing import List, Dict, Optional

import requests
import pandas as pd

# Hyperliquid info endpoint (public, no auth needed)
HYPERLIQUID_INFO_URL = "https://api.hyperliquid.xyz/info"


class PerpDataFeed:
    """
    REST-based OHLCV feed for SOL perpetuals on Hyperliquid.

    - Polls recent candles every few seconds
    - Supports multiple intervals ("1m", "5m", "15m", "1h", ...)
    - Keeps a rolling buffer of the latest N candles
    - Exposes a pandas DataFrame for the strategy/TA layer
    """

    def __init__(self, coin: str = "SOL", interval: str = "1m", max_candles: int = 500):
        self.coin = coin
        self.interval = interval
        self.max_candles = max_candles

        # internal buffer: list of dicts {time, open, high, low, close, volume}
        self._candles = deque(maxlen=max_candles)
        self._lock = threading.Lock()

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

        # Fallback: assume 1 minute
        return 1

    # ------------------------------------------------------------------

    def _fetch_candles_snapshot(self, lookback_minutes: Optional[int] = None) -> List[Dict]:
        """
        Fetches a snapshot of recent candles from Hyperliquid.

        Uses the official candleSnapshot info method:
        POST https://api.hyperliquid.xyz/info
        {
          "type": "candleSnapshot",
          "req": {
            "coin": "SOL",
            "interval": "15m",
            "startTime": <ms>,
            "endTime": <ms>
          }
        }

        We choose lookback based on interval and max_candles so that the
        strategy has enough historical bars (e.g. 300+ 15m candles).
        """
        # Dynamically determine lookback if not provided
        if lookback_minutes is None:
            bar_minutes = self._interval_minutes()
            # We want at least `max_candles + margin` bars worth of time.
            # Add a small margin so we always have enough even if a few bars are missing.
            desired_bars = self.max_candles + 50
            lookback_minutes = bar_minutes * desired_bars

            # Also enforce a minimum wall-clock window (e.g. 600 minutes ~ 10h),
            # so very small max_candles still give decent context.
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

        try:
            resp = requests.post(HYPERLIQUID_INFO_URL, json=payload, timeout=10)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"[PERP_DATA] Error fetching candles: {e}")
            return []

        # Expected format: list of candle objects:
        # { "T": 1681923600000, "t": 1681920000000,
        #   "o": "43200.0", "h": "43350.5", "l": "43150.0", "c": "43250.0",
        #   "v": "125.75", "n": 47, "i": "15m", "s": "SOL" }
        if not isinstance(data, list):
            print("[PERP_DATA] Unexpected candles response format:", data)
            return []

        candles = []
        for c in data:
            try:
                # use close time 'T' as candle timestamp (ms)
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
                print(f"[PERP_DATA] Error parsing candle: {e}, raw: {c}")
                continue

        return candles

    # ------------------------------------------------------------------

    def refresh(self):
        """
        One-shot refresh: fetch recent candles and update the internal buffer.
        """
        new_candles = self._fetch_candles_snapshot()

        if not new_candles:
            return

        with self._lock:
            self._candles.clear()
            # Only keep up to max_candles most recent
            for c in new_candles[-self.max_candles:]:
                self._candles.append(c)

        print(
            f"[PERP_DATA] Loaded {len(self._candles)} candles for "
            f"{self.coin}-PERP ({self.interval})"
        )

    # ------------------------------------------------------------------

    def run_poll_loop(self, interval_sec: int = 5):
        """
        Background loop: periodically refreshes candles.
        """
        while True:
            try:
                self.refresh()
            except Exception as e:
                print(f"[PERP_DATA] Poll loop error: {e}")
            time.sleep(interval_sec)

    def start(self, interval_sec: int = 5):
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

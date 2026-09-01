"""
Read-only Hyperliquid venue client for the funding-data research collector.

Hard constraint (see PIVOT_STATUS.md): this module may call info/market-data
endpoints only. It holds no API key, no wallet, no signing capability, and
never imports anything from execution_backend.py, live_execution_backend.py,
or any other write-capable module. There is no method here that could place,
modify, or cancel an order -- the Hyperliquid /info endpoint used throughout
is unauthenticated and read-only by construction.
"""

import time
from typing import Any, Dict, List, Optional

import requests

HYPERLIQUID_INFO_URL = "https://api.hyperliquid.xyz/info"
REQUEST_TIMEOUT_SEC = 15
MAX_RETRIES = 5
BASE_BACKOFF_SEC = 1.0
MAX_BACKOFF_SEC = 30.0
MIN_REQUEST_GAP_SEC = 0.1  # simple global rate limit -- never hammer the venue


class VenueFetchError(Exception):
    """Raised after all retries are exhausted. Caller must log, never interpolate."""


class HyperliquidReadOnlyClient:
    def __init__(self, on_failure=None):
        self._session = requests.Session()
        self._last_request_ts = 0.0
        # Called with (endpoint, error, attempt) on every failed attempt --
        # collector.py uses this to log gaps, never to silently paper over them.
        self._on_failure = on_failure or (lambda *a, **k: None)

    def _respect_rate_limit(self) -> None:
        elapsed = time.time() - self._last_request_ts
        wait = MIN_REQUEST_GAP_SEC - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_request_ts = time.time()

    def _post(self, payload: Dict[str, Any], endpoint_label: str) -> Any:
        last_error: Optional[Exception] = None
        for attempt in range(MAX_RETRIES):
            self._respect_rate_limit()
            try:
                resp = self._session.post(
                    HYPERLIQUID_INFO_URL, json=payload, timeout=REQUEST_TIMEOUT_SEC
                )
                if resp.status_code == 429:
                    raise VenueFetchError(f"429 rate limited on {endpoint_label}")
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                last_error = e
                self._on_failure(endpoint_label, str(e), attempt + 1)
                sleep_s = min(MAX_BACKOFF_SEC, BASE_BACKOFF_SEC * (2 ** attempt))
                time.sleep(sleep_s)
        raise VenueFetchError(f"{endpoint_label} failed after {MAX_RETRIES} retries: {last_error}")

    def get_meta_and_asset_ctxs(self) -> Any:
        """Universe + per-asset ctx (funding, OI, mark/oracle/mid price, 24h volume)."""
        return self._post({"type": "metaAndAssetCtxs"}, "metaAndAssetCtxs")

    def get_l2_book(self, coin: str) -> Any:
        """Order book levels for a single coin -- used for best_bid/best_ask/spread."""
        return self._post({"type": "l2Book", "coin": coin}, f"l2Book:{coin}")

    def get_funding_history(self, coin: str, start_ms: int, end_ms: int) -> List[Dict[str, Any]]:
        """Realized, settled funding history for a coin over [start_ms, end_ms]."""
        data = self._post(
            {"type": "fundingHistory", "coin": coin, "startTime": start_ms, "endTime": end_ms},
            f"fundingHistory:{coin}",
        )
        return data if isinstance(data, list) else []

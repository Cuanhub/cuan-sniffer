"""
Read-only Hyperliquid venue client for the support/resistance level scanner.

Same hard constraint as research/funding/venue_client.py: info/market-data
endpoints only, no API key, no wallet, no signing capability, nothing that
could place/modify/cancel an order. Deliberately a small, standalone
client (not importing the funding or cascade clients) so this package
carries zero risk to the two collectors already running in production.
"""

import time
from typing import Any, Dict, List, Optional

import requests

HYPERLIQUID_INFO_URL = "https://api.hyperliquid.xyz/info"
REQUEST_TIMEOUT_SEC = 15
MAX_RETRIES = 5
BASE_BACKOFF_SEC = 1.0
MAX_BACKOFF_SEC = 30.0
MIN_REQUEST_GAP_SEC = 0.15


class VenueFetchError(Exception):
    pass


class HyperliquidReadOnlyClient:
    def __init__(self, on_failure=None):
        self._session = requests.Session()
        self._last_request_ts = 0.0
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

    def get_candles(self, coin: str, interval: str, start_ms: int, end_ms: int) -> List[Dict[str, Any]]:
        data = self._post(
            {"type": "candleSnapshot", "req": {"coin": coin, "interval": interval,
                                                "startTime": start_ms, "endTime": end_ms}},
            f"candleSnapshot:{coin}:{interval}",
        )
        return data if isinstance(data, list) else []

    def get_meta_and_asset_ctxs(self) -> Any:
        return self._post({"type": "metaAndAssetCtxs"}, "metaAndAssetCtxs")

    def get_all_mids(self) -> Dict[str, float]:
        """Every perp's mid in ONE request.

        Exists so the 4h candle series can be cached without the current price
        going stale with it. A 4h bar's closed history does not change between
        boundaries, but its FORMING bar does, and that bar's close is the price
        the admission gate measures depth against -- so caching the series
        wholesale would quietly shift which levels qualify. This refreshes that
        one number for the whole universe at the cost of a single call, instead
        of thirty candle requests.

        allMids rather than metaAndAssetCtxs: the payload is a flat
        coin -> price map rather than full per-asset context, and the universe
        only needs re-selecting once a day.
        """
        data = self._post({"type": "allMids"}, "allMids")
        if not isinstance(data, dict):
            return {}
        out: Dict[str, float] = {}
        for coin, px in data.items():
            try:
                val = float(px)
            except (TypeError, ValueError):
                continue
            if val > 0:
                out[str(coin).upper()] = val
        return out

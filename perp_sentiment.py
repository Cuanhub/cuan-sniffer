import threading
import time
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, Any

import requests

HYPERLIQUID_INFO_URL = "https://api.hyperliquid.xyz/info"


@dataclass
class PerpSentimentSnapshot:
    coin: str
    funding_rate: float
    open_interest: float
    bias: float
    premium: float = 0.0
    prev_open_interest: float = 0.0


class PerpSentimentFeed:
    """
    Hyperliquid perp sentiment feed using the `metaAndAssetCtxs` endpoint.

    Improvements:
    - shared process-wide cache
    - shared HTTP session
    - global request throttling
    - exponential backoff on 429
    - reuses one full-universe response across all coins
    """

    _session = requests.Session()
    _request_lock = threading.Lock()
    _cache_lock = threading.Lock()

    _last_request_ts = 0.0
    _min_request_gap_sec = 0.35

    _shared_ctx_cache: Optional[Tuple[float, Any]] = None
    _cache_ttl_sec = 15.0

    def __init__(self, coin: str = "SOL", debug: bool = True):
        self.coin = coin.upper()
        self.debug = debug
        self._snapshot = PerpSentimentSnapshot(
            coin=self.coin,
            funding_rate=0.0,
            open_interest=0.0,
            bias=0.0,
            premium=0.0,
            prev_open_interest=0.0,
        )
        self._lock = threading.Lock()
        self._stop_flag = False
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self, interval_sec: int = 45) -> None:
        """
        Start background thread that refreshes funding/OI every interval_sec.
        """
        if self._thread is not None and self._thread.is_alive():
            if self.debug:
                print(f"[PERP_SENTIMENT] {self.coin} already running")
            return

        self._stop_flag = False
        self._thread = threading.Thread(
            target=self._run_loop,
            args=(interval_sec,),
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_flag = True

    def get_snapshot(self) -> PerpSentimentSnapshot:
        with self._lock:
            return self._snapshot

    def refresh_once(self) -> PerpSentimentSnapshot:
        """
        Manual one-shot refresh, useful for debugging or warmup.
        """
        snap = self._fetch_sentiment_once()
        with self._lock:
            self._snapshot = snap
        return snap

    # ------------------------------------------------------------------
    # Internal loop
    # ------------------------------------------------------------------

    def _run_loop(self, interval_sec: int) -> None:
        while not self._stop_flag:
            try:
                snap = self._fetch_sentiment_once()

                with self._lock:
                    self._snapshot = snap

                if self.debug:
                    print(
                        f"[PERP_SENTIMENT] {snap.coin} "
                        f"funding={snap.funding_rate:.5f}, "
                        f"oi={snap.open_interest:.0f}, "
                        f"bias={snap.bias:.3f}"
                    )
            except Exception as e:
                print(f"[PERP_SENTIMENT ERROR] {self.coin}: {e}")

            time.sleep(interval_sec)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @classmethod
    def _respect_rate_limit(cls) -> None:
        with cls._request_lock:
            now = time.time()
            elapsed = now - cls._last_request_ts
            wait_needed = cls._min_request_gap_sec - elapsed

            if wait_needed > 0:
                time.sleep(wait_needed)

            cls._last_request_ts = time.time()

    @classmethod
    def _fetch_meta_and_asset_ctxs(cls, debug: bool = True, max_retries: int = 2):
        """
        Fetch the full universe ctx payload, with process-wide caching.
        """
        now = time.time()

        with cls._cache_lock:
            if cls._shared_ctx_cache is not None:
                cached_ts, cached_data = cls._shared_ctx_cache
                if (now - cached_ts) < cls._cache_ttl_sec:
                    if debug:
                        age = now - cached_ts
                        print(f"[PERP_SENTIMENT] cache hit | age={age:.1f}s")
                    return cached_data

        payload = {"type": "metaAndAssetCtxs"}
        last_error = None

        for attempt in range(max_retries):
            try:
                cls._respect_rate_limit()

                resp = cls._session.post(
                    HYPERLIQUID_INFO_URL,
                    json=payload,
                    timeout=8,
                )

                if resp.status_code == 429:
                    sleep_s = min(10.0, 0.8 * (2 ** attempt))
                    if debug:
                        print(
                            f"[PERP_SENTIMENT] 429 hit"
                            f" | retry={attempt + 1}/{max_retries}"
                            f" | sleeping {sleep_s:.1f}s"
                        )
                    time.sleep(sleep_s)
                    continue

                resp.raise_for_status()
                data = resp.json()

                with cls._cache_lock:
                    cls._shared_ctx_cache = (time.time(), data)

                return data

            except requests.RequestException as e:
                last_error = e
                sleep_s = min(10.0, 0.8 * (2 ** attempt))
                if debug:
                    print(
                        f"[PERP_SENTIMENT] request error"
                        f" | retry={attempt + 1}/{max_retries}"
                        f" | sleeping {sleep_s:.1f}s"
                        f" | err={e}"
                    )
                time.sleep(sleep_s)

        raise RuntimeError(f"metaAndAssetCtxs failed after retries: {last_error}")

    # ------------------------------------------------------------------
    # Parse sentiment
    # ------------------------------------------------------------------

    def _fetch_sentiment_once(self) -> PerpSentimentSnapshot:
        """
        Query shared Hyperliquid metaAndAssetCtxs cache and extract:
        - funding
        - openInterest
        - premium

        Then compress into a simple bias score.
        """
        data = self._fetch_meta_and_asset_ctxs(debug=self.debug)

        if not isinstance(data, list) or len(data) != 2:
            raise RuntimeError("Unexpected metaAndAssetCtxs response shape")

        universe_obj, ctx_list = data
        universe = universe_obj.get("universe", [])

        if not isinstance(universe, list):
            raise RuntimeError("Universe missing or malformed")

        name_to_idx: Dict[str, int] = {}
        for idx, asset in enumerate(universe):
            name = str(asset.get("name", "")).upper()
            if name:
                name_to_idx[name] = idx

        if self.coin not in name_to_idx:
            raise ValueError(f"Coin {self.coin} not found in Hyperliquid universe")

        idx = name_to_idx[self.coin]

        if not isinstance(ctx_list, list) or idx >= len(ctx_list):
            raise RuntimeError("Asset context list shorter than universe")

        ctx = ctx_list[idx]
        if not isinstance(ctx, dict):
            raise RuntimeError("Malformed asset context")

        funding_str: Optional[str] = ctx.get("funding", "0")
        oi_str: Optional[str] = ctx.get("openInterest", "0")
        premium_str: Optional[str] = ctx.get("premium", "0")

        try:
            funding = float(funding_str)
        except (TypeError, ValueError):
            funding = 0.0

        try:
            open_interest = float(oi_str)
        except (TypeError, ValueError):
            open_interest = 0.0

        try:
            premium = float(premium_str)
        except (TypeError, ValueError):
            premium = 0.0

        with self._lock:
            prev_oi = float(getattr(self._snapshot, "open_interest", 0.0) or 0.0)

        # Simple bias heuristic
        # funding > 0 -> long-heavy
        # premium > 0 -> perp above oracle -> bullish skew
        bias = 0.0
        bias += 50.0 * funding
        bias += 2.0 * premium

        if bias > 2.0:
            bias = 2.0
        elif bias < -2.0:
            bias = -2.0

        return PerpSentimentSnapshot(
            coin=self.coin,
            funding_rate=funding,
            open_interest=open_interest,
            bias=bias,
            premium=premium,
            prev_open_interest=prev_oi,
        )
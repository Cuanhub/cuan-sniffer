import json
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any

import requests

from live_data_guard import (
    LIVE_MAX_SENTIMENT_CACHE_AGE_SECONDS,
    api_backoff_status,
    cache_age_seconds,
    record_api_failure,
    record_api_success,
)

# OI history: how many seconds of history to retain and the poll interval.
# 4H at 45s ≈ 320 entries; keep 350 for safety margin.
_OI_HISTORY_MAXLEN: int = 350
_OI_1H_SECS: int = 3600
_OI_4H_SECS: int = 14400

# OI history previously lived only in memory, so every restart threw it away
# and oi_delta_1h/4h silently read 0.0 for the first 1-4h after every boot
# with no flag distinguishing "warming up" from "genuinely flat OI". Persist
# it across restarts so a restart doesn't blind this factor.
_OI_HISTORY_STATE_PATH = os.getenv("OI_HISTORY_STATE_PATH", "oi_history_state.json")
_OI_HISTORY_STATE_LOCK = threading.Lock()

# Env-tunable thresholds for the two OI delta tiers (fraction, not percent).
OI_STRONG_1H = float(os.getenv("OI_STRONG_1H_THRESHOLD", "0.05"))  # 5 % in 1H
OI_MILD_1H   = float(os.getenv("OI_MILD_1H_THRESHOLD",   "0.02"))  # 2 % in 1H
OI_STRONG_4H = float(os.getenv("OI_STRONG_4H_THRESHOLD", "0.10"))  # 10% in 4H

HYPERLIQUID_INFO_URL = "https://api.hyperliquid.xyz/info"


@dataclass
class PerpSentimentSnapshot:
    coin: str
    funding_rate: float
    open_interest: float
    bias: float
    premium: float = 0.0
    prev_open_interest: float = 0.0   # OI from previous poll cycle (~45s ago, legacy)
    oi_delta_1h_pct: float = 0.0      # % OI change vs ~1H ago (primary signal)
    oi_delta_4h_pct: float = 0.0      # % OI change vs ~4H ago (structural conviction)
    fetched_at: float = 0.0
    data_source: str = "init"
    cache_age_sec: float = 0.0
    stale_neutralized: bool = False


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
            oi_delta_1h_pct=0.0,
            oi_delta_4h_pct=0.0,
            fetched_at=0.0,
            data_source="init",
            cache_age_sec=0.0,
            stale_neutralized=False,
        )
        self._lock = threading.Lock()
        self._stop_flag = False
        self._thread: Optional[threading.Thread] = None
        # Ring buffer of (epoch_seconds, oi_value) — retained up to 4H+
        self._oi_history: deque = deque(
            self._load_persisted_oi_history(self.coin), maxlen=_OI_HISTORY_MAXLEN
        )

    @staticmethod
    def _load_persisted_oi_history(coin: str) -> List[Tuple[float, float]]:
        """Load this coin's OI ring buffer from disk, dropping stale entries."""
        cutoff = time.time() - (_OI_4H_SECS + _OI_1H_SECS)
        try:
            with _OI_HISTORY_STATE_LOCK:
                if not os.path.exists(_OI_HISTORY_STATE_PATH):
                    return []
                with open(_OI_HISTORY_STATE_PATH, "r") as f:
                    state = json.load(f)
            entries = state.get(coin.upper(), [])
            return [
                (float(ts), float(oi)) for ts, oi in entries
                if float(ts) >= cutoff
            ]
        except Exception:
            return []

    def _persist_oi_history(self) -> None:
        """Write this coin's current OI ring buffer back to the shared state file."""
        try:
            with _OI_HISTORY_STATE_LOCK:
                state = {}
                if os.path.exists(_OI_HISTORY_STATE_PATH):
                    try:
                        with open(_OI_HISTORY_STATE_PATH, "r") as f:
                            state = json.load(f)
                    except Exception:
                        state = {}
                state[self.coin] = [[ts, oi] for ts, oi in self._oi_history]
                tmp_path = f"{_OI_HISTORY_STATE_PATH}.tmp"
                with open(tmp_path, "w") as f:
                    json.dump(state, f)
                os.replace(tmp_path, _OI_HISTORY_STATE_PATH)
        except Exception:
            pass

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

    def get_snapshot(
        self,
        max_age_sec: float = LIVE_MAX_SENTIMENT_CACHE_AGE_SECONDS,
    ) -> PerpSentimentSnapshot:
        with self._lock:
            snap = self._snapshot
        age = cache_age_seconds(float(getattr(snap, "fetched_at", 0.0) or 0.0))
        if age is None:
            age = float("inf")
        if age <= float(max_age_sec):
            snap.cache_age_sec = float(age)
            return snap

        print(
            f"[PERP_SENTIMENT] sentiment_stale_neutralized {self.coin}"
            f" | cache_age={age:.1f}s"
            f" | max_age={float(max_age_sec):.1f}s"
        )
        return PerpSentimentSnapshot(
            coin=self.coin,
            funding_rate=0.0,
            open_interest=0.0,
            bias=0.0,
            premium=0.0,
            prev_open_interest=0.0,
            oi_delta_1h_pct=0.0,
            oi_delta_4h_pct=0.0,
            fetched_at=0.0,
            data_source="neutralized",
            cache_age_sec=float(age),
            stale_neutralized=True,
        )

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
                        print(
                            f"[PERP_SENTIMENT] cache hit"
                            f" | data_source=cache"
                            f" | cache_age={age:.1f}s"
                        )
                    return cached_data

        backoff_active, backoff_remaining, backoff_reason = api_backoff_status()
        if backoff_active:
            raise RuntimeError(
                f"api_backoff_active remaining={backoff_remaining:.1f}s reason={backoff_reason}"
            )

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
                    record_api_failure("perp_sentiment", "429:metaAndAssetCtxs")
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

                record_api_success()
                return data

            except requests.RequestException as e:
                last_error = e
                record_api_failure("perp_sentiment", e)
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
    # OI history helpers
    # ------------------------------------------------------------------

    def _oi_delta_pct(self, current_oi: float, target_secs_ago: int) -> float:
        """
        Return fractional OI change vs the reading closest to target_secs_ago in the past.
        Returns 0.0 if history is too short or current_oi is zero.

        Walk from oldest → newest and find the entry whose timestamp is closest to
        (now - target_secs_ago).  Using the closest-match entry rather than the
        exact-boundary one avoids off-by-one noise from irregular poll timing.
        """
        if not self._oi_history or current_oi <= 0:
            return 0.0

        now = time.time()
        target_ts = now - target_secs_ago
        best_ts, best_oi = None, None

        for ts, oi_val in self._oi_history:
            if ts <= target_ts:
                best_ts, best_oi = ts, oi_val

        if best_oi is None or best_oi <= 0:
            return 0.0

        return (current_oi - best_oi) / best_oi

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

        # Append current OI to history ring buffer (outside the lock — deque ops are GIL-safe
        # enough for append/iteration at this frequency, and we avoid holding the lock longer).
        now_ts = time.time()
        self._oi_history.append((now_ts, open_interest))
        self._persist_oi_history()

        # Compute real timeframe deltas now that history is updated.
        oi_delta_1h = self._oi_delta_pct(open_interest, _OI_1H_SECS)
        oi_delta_4h = self._oi_delta_pct(open_interest, _OI_4H_SECS)

        # Simple bias heuristic
        # funding > 0 -> long-heavy
        # premium > 0 -> perp above oracle -> bullish skew
        bias = 0.0
        bias += 50.0 * funding
        bias += 2.0 * premium
        bias = max(-2.0, min(2.0, bias))

        now_ts = time.time()
        return PerpSentimentSnapshot(
            coin=self.coin,
            funding_rate=funding,
            open_interest=open_interest,
            bias=bias,
            premium=premium,
            prev_open_interest=prev_oi,
            oi_delta_1h_pct=oi_delta_1h,
            oi_delta_4h_pct=oi_delta_4h,
            fetched_at=now_ts,
            data_source="fresh",
            cache_age_sec=0.0,
            stale_neutralized=False,
        )

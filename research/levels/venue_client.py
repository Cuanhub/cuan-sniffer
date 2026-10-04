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

    def get_perp_contexts(self) -> Dict[str, Dict[str, Any]]:
        """Per-perp positioning and liquidity context, in ONE request.

        Collected, never traded on. Every one of the 43 features the episode
        set already carries is a transform of OHLCV -- a screen of 19 of them
        against expectancy on 1,060 resolved episodes found exactly one that
        survives controlling for target distance (level_pivot_count, rho
        +0.088). The gap is not more price indicators; it is that there is no
        positioning, flow or liquidity data at all.

        Open interest distinguishes a bounce backed by new buying from one that
        is short covering, which no candle transform can do. Funding is a
        crowding proxy.

        impact_spread_bps is NOT the inside spread, and must not be read as
        one. impactPxs is the venue's price to execute a standard notional, so
        it carries the depth behind the quote rather than the quote itself.
        Measured 2026-10-04 against the L2 book: BTC 0.12bps impact vs 0.12
        top-of-book and HYPE 0.17 vs 0.11 -- the same number where the book is
        deep -- but CASHCAT 23.22 vs 3.74 and USELESS 12.75 vs 2.49, six and
        five times wider. For thin names the two diverge sharply, and impact is
        the one that describes what a fill actually costs, which is why it is
        the one collected.

        A point-in-time reading, NOT aligned to the bar the episode is keyed
        to -- the venue serves no historical OI per candle, so this cannot be
        backfilled and the capture time is recorded alongside it.
        """
        data = self._post({"type": "metaAndAssetCtxs"}, "metaAndAssetCtxs")
        if not (isinstance(data, (list, tuple)) and len(data) == 2):
            return {}
        universe_obj, ctxs = data
        names = [str(a.get("name", "")).upper()
                 for a in (universe_obj or {}).get("universe", [])]
        out: Dict[str, Dict[str, Any]] = {}

        def num(d, k):
            try:
                v = float(d.get(k))
                return v if v == v and v not in (float("inf"), float("-inf")) else None
            except (TypeError, ValueError):
                return None

        for name, ctx in zip(names, ctxs or []):
            if not name or not isinstance(ctx, dict):
                continue
            mark = num(ctx, "markPx")
            oi = num(ctx, "openInterest")
            imp = ctx.get("impactPxs") or []
            bid = ask = None
            if isinstance(imp, (list, tuple)) and len(imp) == 2:
                try:
                    bid, ask = float(imp[0]), float(imp[1])
                except (TypeError, ValueError):
                    bid = ask = None
            spread_bps = None
            if bid and ask and bid > 0 and ask > 0:
                mid_imp = (bid + ask) / 2.0
                if mid_imp > 0:
                    spread_bps = (ask - bid) / mid_imp * 10_000.0
            out[name] = {
                "mid_px": num(ctx, "midPx"),
                "mark_px": mark,
                "oracle_px": num(ctx, "oraclePx"),
                "funding_rate": num(ctx, "funding"),
                "premium": num(ctx, "premium"),
                "oi_base": oi,
                "oi_notional_usd": (oi * mark) if (oi is not None and mark) else None,
                "impact_bid": bid,
                "impact_ask": ask,
                "impact_spread_bps": spread_bps,
                "day_ntl_vlm": num(ctx, "dayNtlVlm"),
                "prev_day_px": num(ctx, "prevDayPx"),
            }
        return out

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

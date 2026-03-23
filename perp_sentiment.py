import threading
import time
from dataclasses import dataclass
from typing import Optional

import requests

HYPERLIQUID_INFO_URL = "https://api.hyperliquid.xyz/info"


@dataclass
class PerpSentimentSnapshot:
    coin: str
    funding_rate: float       # current funding rate
    open_interest: float      # current open interest
    bias: float               # simple sentiment score from funding + premium


class PerpSentimentFeed:
    """
    Real perp sentiment feed for a coin from Hyperliquid using the
    `metaAndAssetCtxs` info endpoint.

    Response shape:
      [
        { "universe": [{ "name": "BTC" }, { "name": "ETH" }, ...] },
        [
          { "funding": "...", "openInterest": "...", "premium": "...", ... },  # BTC ctx
          { "funding": "...", "openInterest": "...", "premium": "...", ... },  # ETH ctx
          ...
        ]
      ]
    """

    def __init__(self, coin: str = "SOL"):
        self.coin = coin.upper()
        self._snapshot = PerpSentimentSnapshot(
            coin=self.coin,
            funding_rate=0.0,
            open_interest=0.0,
            bias=0.0,
        )
        self._lock = threading.Lock()
        self._stop_flag = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self, interval_sec: int = 30) -> None:
        """
        Start background thread that refreshes funding/OI every `interval_sec`.
        """
        t = threading.Thread(
            target=self._run_loop,
            args=(interval_sec,),
            daemon=True,
        )
        t.start()

    def stop(self) -> None:
        self._stop_flag = True

    def get_snapshot(self) -> PerpSentimentSnapshot:
        """
        Thread-safe access to latest snapshot.
        """
        with self._lock:
            return self._snapshot

    # ------------------------------------------------------------------
    # Internal loop
    # ------------------------------------------------------------------

    def _run_loop(self, interval_sec: int) -> None:
        while not self._stop_flag:
            try:
                snap = self._fetch_sentiment_once()

                with self._lock:
                    self._snapshot = snap

                print(
                    f"[PERP_SENTIMENT] {snap.coin} "
                    f"funding={snap.funding_rate:.5f}, "
                    f"oi={snap.open_interest:.0f}, "
                    f"bias={snap.bias:.3f}"
                )
            except Exception as e:
                # Keep the bot alive even if HL hiccups
                print(f"[PERP_SENTIMENT ERROR] {e}")

            time.sleep(interval_sec)

    # ------------------------------------------------------------------
    # Hyperliquid query + parsing
    # ------------------------------------------------------------------

    def _fetch_sentiment_once(self) -> PerpSentimentSnapshot:
        """
        Query Hyperliquid `metaAndAssetCtxs` and extract:
        - funding (current funding rate)
        - openInterest
        - premium

        Then compress that into a simple bias score.
        """
        payload = {"type": "metaAndAssetCtxs"}

        resp = requests.post(
            HYPERLIQUID_INFO_URL,
            json=payload,
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()

        # Expected shape: [ { "universe": [...] }, [ctx0, ctx1, ...] ]
        if not isinstance(data, list) or len(data) != 2:
            raise RuntimeError("Unexpected metaAndAssetCtxs response shape")

        universe_obj, ctx_list = data
        universe = universe_obj.get("universe", [])

        # Map coin name -> index in ctx_list
        name_to_idx = {}
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

        # Raw strings from API
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

        # ---- Simple bias heuristic ------------------------------------
        # funding > 0 → market long-heavy
        # premium > 0 → perp trading above oracle → bullish skew
        # Scale so typical values land roughly in [-1, +1]
        bias = 0.0
        bias += 50.0 * funding     # funding is usually tiny (e.g. 0.00001)
        bias += 2.0 * premium      # premium also small (e.g. 0.0003)

        # Clamp to keep it sane
        if bias > 2.0:
            bias = 2.0
        elif bias < -2.0:
            bias = -2.0

        return PerpSentimentSnapshot(
            coin=self.coin,
            funding_rate=funding,
            open_interest=open_interest,
            bias=bias,
        )

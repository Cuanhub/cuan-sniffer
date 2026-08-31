"""
SPL token metadata for per-coin on-chain flow tracking.

Tracked coins: JTO, WIF, PENGU (all Solana SPL tokens).
HYPE, TAO, NEAR, SUI are not tracked on-chain here (HYPE is on Hyperliquid
EVM; the others aren't Solana-ecosystem tokens) — their whale_pressure/
flow_imbalance scores read as untracked, not "confirmed calm".

Price fetching reuses the Hyperliquid allMids endpoint (same source as
config.get_sol_price) so no additional API credentials are needed.
"""

import os
import time
import threading
import requests
from typing import Dict, Optional

# ── Mint addresses ────────────────────────────────────────────────────────────
# Maps coin symbol → SPL token metadata.
TOKEN_MINTS: Dict[str, Dict] = {
    "JTO":   {"mint": "jtojtomepa8beP8AuQc6eXt5FriJwfFMwQx2v2f9mCL", "decimals": 9},
    "WIF":   {"mint": "EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm", "decimals": 6},
    "PENGU": {"mint": "2zMMhcVQEXDtdE6vsFS7S7D5oUodfJHE8vd1gnBouauv", "decimals": 6},
}

# Reverse lookup: mint address → coin symbol
MINT_TO_COIN: Dict[str, str] = {v["mint"]: k for k, v in TOKEN_MINTS.items()}

# Set of all watched mint addresses (for fast membership tests)
WATCHED_MINTS: frozenset = frozenset(v["mint"] for v in TOKEN_MINTS.values())

# ── Alert threshold ───────────────────────────────────────────────────────────
# Minimum USD value of a token move before it is recorded as a FlowEvent.
# Equivalent to MIN_SOL_ALERT in engine.py but denominated in USD.
MIN_TOKEN_FLOW_USD = float(os.getenv("MIN_TOKEN_FLOW_USD", "5000.0"))

# ── Price cache ───────────────────────────────────────────────────────────────
_HYPERLIQUID_INFO_URL = "https://api.hyperliquid.xyz/info"
_PRICE_CACHE_TTL_SEC = float(os.getenv("TOKEN_PRICE_CACHE_TTL_SEC", "60.0"))

_price_cache: Dict[str, float] = {}
_price_cache_ts: float = 0.0
_price_lock = threading.Lock()


def _fetch_all_mids() -> Optional[Dict[str, float]]:
    """Fetch all mid prices from Hyperliquid. Returns None on failure."""
    try:
        resp = requests.post(
            _HYPERLIQUID_INFO_URL,
            json={"type": "allMids"},
            timeout=6,
        )
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict):
            return {k: float(v) for k, v in data.items() if v is not None}
    except Exception:
        pass
    return None


def _refresh_price_cache_if_stale() -> None:
    """Must be called while holding _price_lock."""
    global _price_cache, _price_cache_ts
    now = time.time()
    if now - _price_cache_ts >= _PRICE_CACHE_TTL_SEC or not _price_cache:
        fresh = _fetch_all_mids()
        if fresh:
            _price_cache = fresh
            _price_cache_ts = now


def get_token_price(symbol: str) -> float:
    """
    Return current USD price for a tracked token.

    Uses a process-wide cache refreshed every TOKEN_PRICE_CACHE_TTL_SEC seconds.
    Returns 0.0 on failure so callers can safely multiply without crashing.
    """
    with _price_lock:
        _refresh_price_cache_if_stale()
        return float(_price_cache.get(symbol.upper(), 0.0))


def get_token_prices(symbols: list) -> Dict[str, float]:
    """
    Return USD prices for multiple tokens in one cache hit.
    Missing symbols get price 0.0.
    """
    with _price_lock:
        _refresh_price_cache_if_stale()
        return {s.upper(): float(_price_cache.get(s.upper(), 0.0)) for s in symbols}

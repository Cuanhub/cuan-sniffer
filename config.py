# config.py
import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

# === RPC ===
RPC_URL = os.getenv("RPC_URL", "https://api.mainnet-beta.solana.com")

# === TELEGRAM ===
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# === WALLETS ===
TRACKED_WALLETS = [
    w.strip()
    for w in os.getenv("TRACKED_WALLETS", "").split(",")
    if w.strip()
]

# === POLLING ===
POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL_SECONDS", "60"))

# === ALERT THRESHOLD ===
MIN_SOL_ALERT = float(os.getenv("MIN_SOL_ALERT", "50"))

# === SOL PRICE CACHE ===
SOL_PRICE_CACHE_TTL_SEC = int(os.getenv("SOL_PRICE_CACHE_TTL_SEC", "300"))
DEFAULT_SOL_USD_PRICE = float(os.getenv("SOL_USD_PRICE", "130"))

_sol_price_cache: dict = {"value": None, "fetched_at": 0.0}


def _fetch_sol_price_from_api() -> float | None:
    """
    Fetch live SOL/USD price from Hyperliquid.
    Returns None on failure so caller can decide fallback behavior.
    """
    try:
        resp = requests.post(
            "https://api.hyperliquid.xyz/info",
            json={"type": "allMids"},
            timeout=5,
        )
        resp.raise_for_status()
        mids = resp.json()

        if isinstance(mids, dict) and "SOL" in mids:
            return float(mids["SOL"])
    except Exception:
        pass

    return None


def get_sol_price(force_refresh: bool = False) -> float:
    """
    Return current SOL/USD price using cache-first logic.

    Behavior:
    - Uses cached price if still fresh
    - Refreshes from Hyperliquid when cache is stale or force_refresh=True
    - Falls back to last known cached value
    - Falls back to DEFAULT_SOL_USD_PRICE if no cached value exists
    """
    now = time.time()
    cache = _sol_price_cache

    cache_is_stale = (now - cache["fetched_at"]) >= SOL_PRICE_CACHE_TTL_SEC
    needs_refresh = force_refresh or cache["value"] is None or cache_is_stale

    if needs_refresh:
        fresh = _fetch_sol_price_from_api()
        if fresh is not None:
            cache["value"] = fresh
            cache["fetched_at"] = now
        elif cache["value"] is None:
            cache["value"] = DEFAULT_SOL_USD_PRICE
            cache["fetched_at"] = now

    return float(cache["value"] if cache["value"] is not None else DEFAULT_SOL_USD_PRICE)
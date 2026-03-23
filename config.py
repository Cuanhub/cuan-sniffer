# config.py
import os
import requests
from dotenv import load_dotenv

load_dotenv()

# === RPC ===
# FIX: standardise on RPC_URL — solana_rpc.py (now deleted) used SOLANA_RPC_URL
RPC_URL = os.getenv("RPC_URL", "https://api.mainnet-beta.solana.com")

# === TELEGRAM ===
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")

# === WALLETS ===
TRACKED_WALLETS = [
    w.strip()
    for w in os.getenv("TRACKED_WALLETS", "").split(",")
    if w.strip()
]

# === POLLING ===
POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL_SECONDS", "60"))

# === ALERT THRESHOLD ===
# FIX: lowered default from 200 → 50 SOL (~$6.5k at $130)
# At 200 SOL the bot was silently missing most meaningful whale moves
MIN_SOL_ALERT = float(os.getenv("MIN_SOL_ALERT", "50"))


def _fetch_sol_price() -> float:
    """
    Fetch live SOL/USD price from Hyperliquid (same API we already use).
    Falls back to env var SOL_USD_PRICE, then to 130.0.
    """
    fallback = float(os.getenv("SOL_USD_PRICE", "130"))
    try:
        resp = requests.post(
            "https://api.hyperliquid.xyz/info",
            json={"type": "allMids"},
            timeout=5,
        )
        resp.raise_for_status()
        mids = resp.json()
        # allMids returns {"SOL": "130.5", "BTC": "...", ...}
        if isinstance(mids, dict) and "SOL" in mids:
            return float(mids["SOL"])
    except Exception:
        pass
    return fallback


# FIX: was hardcoded 180 — now fetched live at startup
# Module-level so it's fetched once on import, not every call
SOL_USD_PRICE: float = _fetch_sol_price()
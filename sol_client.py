import itertools
import os
import requests
from typing import List, Dict, Optional
from config import RPC_URL

JSON_RPC_VERSION = "2.0"

# ── Fallback RPC pool (Fix #4) ─────────────────────────────────────────────
# Primary endpoint comes from RPC_URL env var (Helius or custom).
# Fallbacks are tried in order on any hard failure.
# Override via RPC_FALLBACKS env var: comma-separated URLs.
_fallback_env = os.getenv("RPC_FALLBACKS", "")
_FALLBACK_URLS: List[str] = [
    u.strip() for u in _fallback_env.split(",") if u.strip()
] if _fallback_env else [
    "https://api.mainnet-beta.solana.com",
    "https://solana-mainnet.rpc.extrnode.com",
]

# Build the full ordered list: primary first, then fallbacks (deduped)
_ALL_RPC_URLS: List[str] = list(dict.fromkeys([RPC_URL] + _FALLBACK_URLS))

# In-memory request counter for unique JSON-RPC ids
_req_id = itertools.count(1)


# === Core RPC Request Handler (with retry + fallback) =====================

def _rpc_request(method: str, params: list) -> dict:
    """
    Sends a JSON-RPC request, rotating through the RPC pool on failures.
    Each URL is tried once before giving up; transient network errors and
    RPC-level errors both trigger a rotate.
    """
    req_id = next(_req_id)
    payload = {
        "jsonrpc": JSON_RPC_VERSION,
        "id": req_id,
        "method": method,
        "params": params,
    }

    last_err = None
    for url in _ALL_RPC_URLS:
        try:
            resp = requests.post(url, json=payload, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            if "error" in data:
                last_err = f"RPC Error from {url}: {data['error']}"
                print(f"[RPC WARN] {last_err} — trying next endpoint")
                continue

            return data["result"]

        except Exception as e:
            last_err = str(e)
            print(f"[RPC WARN] Method: {method}, URL: {url}, Error: {e} — trying next endpoint")

    print(f"[RPC ERROR] Method: {method} — all endpoints exhausted. Last error: {last_err}")
    return None


# === Fetch recent signatures for a wallet ================================

def get_signatures_for_address(
    address: str,
    before: Optional[str] = None,
    limit: int = 20,
    finalized_only: bool = True,
) -> List[Dict]:
    """
    Fetch recent transaction signatures for `address`.

    finalized_only=True (default, Fix #5): only returns signatures with
    confirmationStatus == 'finalized', guarding against acting on
    transactions that are still in 'processed' or 'confirmed' state and
    could be rolled back under network stress.
    """
    options: Dict = {"limit": limit}
    if before:
        options["before"] = before
    if finalized_only:
        options["commitment"] = "finalized"

    result = _rpc_request("getSignaturesForAddress", [address, options])
    if not result:
        return []

    if finalized_only:
        # Extra guard: filter out any entries that slipped through without
        # a finalized confirmation status (can happen on some RPC providers).
        result = [
            entry for entry in result
            if entry.get("confirmationStatus") in ("finalized", None)
            # None means the RPC didn't return the field — treat as finalized
            # to avoid dropping valid entries from providers that omit it.
        ]

    return result


# === Fetch full transaction details ======================================

def get_transaction(signature: str) -> Optional[dict]:
    result = _rpc_request(
        "getTransaction",
        [signature, {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0}],
    )
    return result


# === Extract actual SOL delta for an address from a parsed transaction ===

def get_sol_transfer_for_address(signature: str, address: str) -> Optional[float]:
    """
    Parse the transaction and return the net SOL change (in SOL, not lamports)
    for `address`.  Uses preBalances / postBalances from tx meta — this is the
    authoritative on-chain amount and is immune to fee/rent noise that plagues
    balance-diff polling.

    Returns None if the transaction cannot be fetched or the address is not
    present in the account list (e.g. inner-instruction-only involvement).
    """
    tx = get_transaction(signature)
    if not tx:
        return None

    meta = tx.get("meta")
    if not meta:
        return None

    # accountKeys may be nested under transaction.message (jsonParsed) or flat
    message = tx.get("transaction", {}).get("message", {})
    account_keys = message.get("accountKeys", [])

    # accountKeys entries are dicts {"pubkey": ..., "signer": ..., "writable": ...}
    # under jsonParsed encoding; normalise to a plain list of address strings.
    key_addresses = []
    for k in account_keys:
        if isinstance(k, dict):
            key_addresses.append(k.get("pubkey", ""))
        else:
            key_addresses.append(str(k))

    if address not in key_addresses:
        return None

    idx = key_addresses.index(address)

    pre_balances = meta.get("preBalances", [])
    post_balances = meta.get("postBalances", [])

    if idx >= len(pre_balances) or idx >= len(post_balances):
        return None

    delta_lamports = post_balances[idx] - pre_balances[idx]
    return delta_lamports / 1_000_000_000  # lamports → SOL


# === Fetch wallet SOL balance ============================================

def get_sol_balance(address: str) -> Optional[float]:
    """Returns current SOL balance, or None on RPC failure."""
    result = _rpc_request("getBalance", [address])

    if not result:
        return None

    lamports = result.get("value", 0)
    return lamports / 1_000_000_000  # convert lamports → SOL

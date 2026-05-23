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


# === Pure parse helpers (no RPC calls) ===================================

def _extract_account_keys(tx: dict) -> List[str]:
    """
    Return a flat list of account address strings from a jsonParsed transaction.
    accountKeys entries may be dicts (jsonParsed) or plain strings.
    """
    message = tx.get("transaction", {}).get("message", {})
    account_keys = message.get("accountKeys", [])
    keys: List[str] = []
    for k in account_keys:
        if isinstance(k, dict):
            keys.append(k.get("pubkey", ""))
        else:
            keys.append(str(k))
    return keys


def _parse_sol_delta(tx: dict, address: str) -> Optional[float]:
    """
    Extract the net SOL change (in SOL, not lamports) for `address` from an
    already-fetched jsonParsed transaction dict.  Uses preBalances/postBalances
    which are authoritative and immune to fee/rent noise.

    Returns None if address is not in the account list.
    """
    meta = tx.get("meta")
    if not meta:
        return None

    key_addresses = _extract_account_keys(tx)
    if address not in key_addresses:
        return None

    idx = key_addresses.index(address)
    pre_balances = meta.get("preBalances", [])
    post_balances = meta.get("postBalances", [])

    if idx >= len(pre_balances) or idx >= len(post_balances):
        return None

    delta_lamports = post_balances[idx] - pre_balances[idx]
    return delta_lamports / 1_000_000_000  # lamports → SOL


def _parse_token_transfers(
    tx: dict,
    address: str,
    watched_mints: frozenset,
) -> List[Dict]:
    """
    Extract SPL token balance changes for `address` across `watched_mints`
    from an already-fetched jsonParsed transaction dict.

    Returns a list of dicts, one per mint that changed:
        {
            "mint":      str,   # token mint address
            "delta_ui":  float, # signed UI-amount delta (positive = IN, negative = OUT)
            "direction": str,   # "IN" or "OUT"
        }

    Uses preTokenBalances/postTokenBalances from tx.meta.  The `owner` field
    identifies which wallet owns each token account — only entries owned by
    `address` are returned.  Missing `owner` fields are silently skipped
    (safe for modern Alchemy/Helius RPCs which always include owner).
    """
    meta = tx.get("meta")
    if not meta:
        return []

    pre_token = meta.get("preTokenBalances", []) or []
    post_token = meta.get("postTokenBalances", []) or []

    # Build pre-balance lookup: (accountIndex, mint) → uiAmount
    pre_map: Dict[tuple, float] = {}
    for entry in pre_token:
        mint = entry.get("mint", "")
        if mint not in watched_mints:
            continue
        owner = entry.get("owner", "")
        if owner != address:
            continue
        idx = entry.get("accountIndex", -1)
        ui_amount = entry.get("uiTokenAmount", {}).get("uiAmount") or 0.0
        pre_map[(idx, mint)] = float(ui_amount)

    # Build post-balance lookup: (accountIndex, mint) → uiAmount
    post_map: Dict[tuple, float] = {}
    for entry in post_token:
        mint = entry.get("mint", "")
        if mint not in watched_mints:
            continue
        owner = entry.get("owner", "")
        if owner != address:
            continue
        idx = entry.get("accountIndex", -1)
        ui_amount = entry.get("uiTokenAmount", {}).get("uiAmount") or 0.0
        post_map[(idx, mint)] = float(ui_amount)

    # Compute deltas across all (accountIndex, mint) keys seen in either map
    all_keys = set(pre_map) | set(post_map)
    results: List[Dict] = []
    for key in all_keys:
        _, mint = key
        pre_val = pre_map.get(key, 0.0)
        post_val = post_map.get(key, 0.0)
        delta = post_val - pre_val
        if delta == 0.0:
            continue
        results.append({
            "mint": mint,
            "delta_ui": delta,
            "direction": "IN" if delta > 0 else "OUT",
        })

    return results


# === Public RPC-backed helpers ============================================

def get_sol_transfer_for_address(signature: str, address: str) -> Optional[float]:
    """
    Parse the transaction and return the net SOL change (in SOL, not lamports)
    for `address`.  Thin wrapper over _parse_sol_delta for callers that only
    need the SOL delta and don't want to manage the transaction fetch.
    """
    tx = get_transaction(signature)
    if not tx:
        return None
    return _parse_sol_delta(tx, address)


def get_all_transfers_for_address(
    signature: str,
    address: str,
    watched_mints: frozenset,
) -> Dict:
    """
    Single RPC fetch returning both the SOL delta and SPL token transfers for
    `address` in one transaction.  Avoids double-fetching when the caller
    needs both.

    Returns:
        {
            "sol_delta":       Optional[float],  # None if address not in tx
            "token_transfers": List[Dict],        # may be empty
        }
    """
    tx = get_transaction(signature)
    if not tx:
        return {"sol_delta": None, "token_transfers": []}

    return {
        "sol_delta": _parse_sol_delta(tx, address),
        "token_transfers": _parse_token_transfers(tx, address, watched_mints),
    }


# === Fetch wallet SOL balance ============================================

def get_sol_balance(address: str) -> Optional[float]:
    """Returns current SOL balance, or None on RPC failure."""
    result = _rpc_request("getBalance", [address])

    if not result:
        return None

    lamports = result.get("value", 0)
    return lamports / 1_000_000_000  # convert lamports → SOL

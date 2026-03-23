import requests
from typing import List, Dict, Optional
from config import RPC_URL

JSON_RPC_VERSION = "2.0"


# === Core RPC Request Handler ===
def _rpc_request(method: str, params: list) -> dict:
    """
    Sends a JSON-RPC request to the Solana RPC endpoint.
    Handles errors safely and ensures clean responses.
    """
    payload = {
        "jsonrpc": JSON_RPC_VERSION,
        "id": 1,
        "method": method,
        "params": params,
    }

    try:
        resp = requests.post(RPC_URL, json=payload, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        if "error" in data:
            raise RuntimeError(f"RPC Error: {data['error']}")

        return data["result"]

    except Exception as e:
        print(f"[RPC ERROR] Method: {method}, Error: {e}")
        return None


# === Fetch recent signatures for a wallet ===
def get_signatures_for_address(
    address: str,
    before: Optional[str] = None,
    limit: int = 20,
) -> List[Dict]:
    params = [address, {"limit": limit}]
    if before:
        params[1]["before"] = before

    result = _rpc_request("getSignaturesForAddress", params)
    return result if result else []


# === Fetch full transaction details ===
def get_transaction(signature: str) -> Optional[dict]:
    result = _rpc_request("getTransaction", [signature, "jsonParsed"])
    return result


# === Fetch wallet SOL balance ===
def get_sol_balance(address: str) -> float:
    result = _rpc_request("getBalance", [address])

    if not result:
        return 0.0

    lamports = result.get("value", 0)
    return lamports / 1_000_000_000  # convert lamports → SOL

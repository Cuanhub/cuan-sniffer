"""
Read-only Etherscan (v2, multi-chain) API client for wallet trade
reconstruction research.

Hard constraint, same as every other research/ package: read-only data
endpoints only. This client has no private key, no signing capability,
and cannot submit a transaction of any kind -- Etherscan's API doesn't
even expose a write path, but the point stands regardless.

Requires ETHERSCAN_API_KEY in .env -- free tier, no payment info needed.
V1 endpoints are fully deprecated (verified 2026-09-04, returns an error
unconditionally); V2 requires a key with no anonymous tier.
"""

import os
import time
from typing import Any, Dict, List, Optional

import requests

ETHERSCAN_V2_URL = "https://api.etherscan.io/v2/api"
ETH_MAINNET_CHAIN_ID = 1
REQUEST_TIMEOUT_SEC = 20
MAX_RETRIES = 5
BASE_BACKOFF_SEC = 1.0
MAX_BACKOFF_SEC = 20.0
MIN_REQUEST_GAP_SEC = 0.25  # free tier is ~5 req/sec; stay well under it

# Etherscan's page*offset pagination silently caps around a 10,000-row
# window -- results beyond that are unreachable via page number alone.
# Cursor pagination via startblock (ascending sort) is the documented
# workaround and is what fetch_all_token_transfers/fetch_all_normal_txs
# use below.
#
# The free tier ALSO silently caps every single response at 1000 rows
# regardless of the requested `offset` -- verified directly (requested
# offset=2000 on a 21k-tx wallet, got exactly 1000 back, and a second
# call from the next startblock returned another full 1000 with a later
# timestamp range, proving more data existed). BATCH_SIZE must equal this
# real cap, not a hoped-for larger one -- otherwise the "stop when
# len(batch) < BATCH_SIZE" pagination end-condition never fires correctly
# and silently truncates every wallet's history at the cap instead of its
# real end. This caused exactly that silent truncation once already
# (2026-09-04) before being caught and fixed.
BATCH_SIZE = 1000


class EtherscanFetchError(Exception):
    pass


class EtherscanReadOnlyClient:
    def __init__(self, api_key: Optional[str] = None, on_failure=None):
        self.api_key = api_key or os.getenv("ETHERSCAN_API_KEY", "")
        if not self.api_key:
            raise ValueError("ETHERSCAN_API_KEY not set (check .env)")
        self._session = requests.Session()
        self._last_request_ts = 0.0
        self._on_failure = on_failure or (lambda *a, **k: None)

    def _respect_rate_limit(self) -> None:
        elapsed = time.time() - self._last_request_ts
        wait = MIN_REQUEST_GAP_SEC - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_request_ts = time.time()

    def _get(self, params: Dict[str, Any], label: str) -> Any:
        params = {**params, "apikey": self.api_key, "chainid": ETH_MAINNET_CHAIN_ID}
        last_error: Optional[Exception] = None
        for attempt in range(MAX_RETRIES):
            self._respect_rate_limit()
            try:
                resp = self._session.get(ETHERSCAN_V2_URL, params=params, timeout=REQUEST_TIMEOUT_SEC)
                resp.raise_for_status()
                data = resp.json()
                if data.get("status") == "0" and data.get("message") not in ("No transactions found",):
                    raise EtherscanFetchError(f"{label}: {data.get('result')}")
                return data.get("result", [])
            except Exception as e:
                last_error = e
                self._on_failure(label, str(e), attempt + 1)
                time.sleep(min(MAX_BACKOFF_SEC, BASE_BACKOFF_SEC * (2 ** attempt)))
        raise EtherscanFetchError(f"{label} failed after {MAX_RETRIES} retries: {last_error}")

    def get_eth_balance(self, address: str) -> float:
        result = self._get(
            {"module": "account", "action": "balance", "address": address, "tag": "latest"},
            f"balance:{address}",
        )
        return int(result) / 1e18

    def get_code(self, address: str) -> str:
        result = self._get(
            {"module": "proxy", "action": "eth_getCode", "address": address, "tag": "latest"},
            f"eth_getCode:{address}",
        )
        return result or "0x"

    def is_contract(self, address: str) -> bool:
        return len(self.get_code(address)) > 2

    def _fetch_page(self, action: str, address: str, startblock: int, offset: int) -> List[Dict[str, Any]]:
        result = self._get(
            {
                "module": "account", "action": action, "address": address,
                "startblock": startblock, "endblock": 99_999_999,
                "page": 1, "offset": offset, "sort": "asc",
            },
            f"{action}:{address}:from_block_{startblock}",
        )
        return result if isinstance(result, list) else []

    def fetch_all_token_transfers(self, address: str, max_rows: Optional[int] = None) -> List[Dict[str, Any]]:
        """All ERC20 transfers in/out of address, cursor-paginated by block
        number to work around the page*offset window cap."""
        return self._fetch_all("tokentx", address, max_rows)

    def fetch_all_normal_txs(self, address: str, max_rows: Optional[int] = None) -> List[Dict[str, Any]]:
        """All normal (top-level) transactions from/to address -- needed to
        see native ETH value moved alongside a token transfer in the same
        tx (a direct ETH<->token swap, as opposed to a WETH<->token one
        which shows up as a second token transfer instead)."""
        return self._fetch_all("txlist", address, max_rows)

    def fetch_all_internal_txs(self, address: str, max_rows: Optional[int] = None) -> List[Dict[str, Any]]:
        """All internal (contract-to-contract/contract-to-wallet) ETH
        transfers involving address. Required, not optional: many DEX
        routers unwrap WETH and forward native ETH to the trader via an
        internal call rather than the top-level tx value -- verified
        directly on a real sell (WETH contract -> router -> wallet, both
        internal calls, zero top-level value). Skipping this silently
        misclassifies real sells as "no quote asset found"."""
        return self._fetch_all("txlistinternal", address, max_rows)

    def _fetch_all(self, action: str, address: str, max_rows: Optional[int]) -> List[Dict[str, Any]]:
        # Known, unmitigated edge case: advancing startblock past the last
        # row's block number would silently drop any further rows in that
        # SAME block if a batch happened to end mid-block. Not fixed here
        # because it requires >BATCH_SIZE (2000) transfers from one address
        # in a single ~12s block, which no wallet in this research set
        # comes close to -- flagging the assumption rather than engineering
        # around a near-zero-probability case.
        all_rows: List[Dict[str, Any]] = []
        startblock = 0
        while True:
            batch = self._fetch_page(action, address, startblock, BATCH_SIZE)
            if not batch:
                break
            all_rows.extend(batch)
            if max_rows and len(all_rows) >= max_rows:
                return all_rows[:max_rows]
            last_block = int(batch[-1]["blockNumber"])
            if len(batch) < BATCH_SIZE:
                break  # reached the tip
            startblock = last_block + 1
        return all_rows

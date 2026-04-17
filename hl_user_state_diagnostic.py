#!/usr/bin/env python3
"""
Standalone Hyperliquid account-state diagnostic.

Purpose:
- Query user_state() for a fixed target address outside the trading bot.
- Use the same Info/base-url setup as live_execution_backend.py.
- Print full raw payload plus extracted margin/account sections.
"""

import json
import os
import sys
from typing import Any, Dict

from dotenv import load_dotenv
from eth_account import Account
from hyperliquid.info import Info
from hyperliquid.utils import constants

TARGET_ADDRESS = "0x66bdB8795F11eDFB4F1c7Ca30A095A12bbc70D53"
_EMPTY_SPOT_META: Dict[str, Any] = {"tokens": [], "universe": []}


def _pretty(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, default=str)


def main() -> int:
    load_dotenv(override=False)

    hl_testnet = os.getenv("HL_TESTNET", "true").lower() == "true"
    hl_account_address = os.getenv("HL_ACCOUNT_ADDRESS", "").strip()
    hl_secret_key = os.getenv("HL_SECRET_KEY", "").strip()
    hl_vault_address = os.getenv("HL_VAULT_ADDRESS", "").strip() or None

    venue = "TESTNET" if hl_testnet else "MAINNET"
    base_url = constants.TESTNET_API_URL if hl_testnet else constants.MAINNET_API_URL
    backend_style_query_address = hl_vault_address if hl_vault_address else hl_account_address

    wallet_address_from_secret = ""
    if hl_secret_key:
        try:
            wallet_address_from_secret = Account.from_key(hl_secret_key).address
        except Exception as e:
            wallet_address_from_secret = f"<invalid_secret_key: {e}>"
    else:
        wallet_address_from_secret = "<missing_HL_SECRET_KEY>"

    print("=== HYPERLIQUID ACCOUNT-STATE DIAGNOSTIC ===")
    print(f"venue={venue}")
    print(f"base_url={base_url}")
    print(f"env_account_address={hl_account_address or '<missing_HL_ACCOUNT_ADDRESS>'}")
    print(f"env_vault_address={hl_vault_address or 'none'}")
    print(f"backend_style_query_address={backend_style_query_address or '<missing>'}")
    print(f"wallet_address_from_secret={wallet_address_from_secret}")
    print(f"user_state_query_address={TARGET_ADDRESS}")
    print("")

    try:
        info = Info(
            base_url,
            skip_ws=True,
            spot_meta=_EMPTY_SPOT_META,
        )
    except Exception as e:
        print(f"[ERROR] Failed to initialize Info client: {e}")
        return 1

    try:
        state = info.user_state(TARGET_ADDRESS)
    except Exception as e:
        print(f"[ERROR] user_state query failed for {TARGET_ADDRESS}: {e}")
        return 1

    print("=== RAW user_state JSON (full) ===")
    print(_pretty(state))
    print("")

    if not isinstance(state, dict):
        print("=== EXTRACTED ===")
        print(f"unexpected_state_type={type(state)}")
        return 0

    margin_summary = state.get("marginSummary")
    cross_margin_summary = state.get("crossMarginSummary")
    withdrawable = state.get("withdrawable")
    asset_positions = state.get("assetPositions")

    print("=== EXTRACTED ===")
    print("marginSummary=")
    print(_pretty(margin_summary))
    print("crossMarginSummary=")
    print(_pretty(cross_margin_summary))
    print("withdrawable=")
    print(_pretty(withdrawable))
    print("assetPositions=")
    print(_pretty(asset_positions))
    print(f"assetPositions_count={len(asset_positions) if isinstance(asset_positions, list) else 'n/a'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

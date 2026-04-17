"""
Backend factory.

Switches between PaperExecutionBackend and LiveExecutionBackend from env.
Keeps executor.py clean — no backend-specific imports needed there.

Usage in executor.py:
    Replace:
        from paper_execution_backend import PaperExecutionBackend
    With:
        from execution_backend_factory import build_execution_backend

    Replace:
        self.backend = backend or PaperExecutionBackend()
    With:
        self.backend = backend or build_execution_backend(debug=True)
"""

import os

from paper_execution_backend import PaperExecutionBackend


def build_execution_backend(debug: bool = True):
    paper_mode = os.getenv("PAPER_MODE", "true").lower() == "true"

    if paper_mode:
        if debug:
            print("[FACTORY] PAPER_MODE=true — using PaperExecutionBackend")
        return PaperExecutionBackend(debug=debug)

    # Only import LiveExecutionBackend when actually going live.
    # This avoids SDK import errors if hyperliquid SDK is not installed
    # in paper-only environments.
    if debug:
        print("[FACTORY] PAPER_MODE=false — using LiveExecutionBackend")

    _validate_live_env()

    from live_execution_backend import LiveExecutionBackend
    return LiveExecutionBackend(debug=debug)


def _validate_live_env():
    """
    Hard check before allowing live backend to init.
    Raises immediately with a clear message if env is misconfigured
    rather than letting the SDK raise an opaque error mid-trade.
    """
    missing = []

    if not os.getenv("HL_ACCOUNT_ADDRESS", "").strip():
        missing.append("HL_ACCOUNT_ADDRESS")

    if not os.getenv("HL_SECRET_KEY", "").strip():
        missing.append("HL_SECRET_KEY")

    if missing:
        raise EnvironmentError(
            f"[FACTORY] Cannot start live backend — missing env vars: {', '.join(missing)}. "
            f"Set PAPER_MODE=true to run in paper mode."
        )

    # Warn but don't block if still pointing at testnet
    if os.getenv("HL_TESTNET", "true").lower() == "true":
        print(
            "[FACTORY] WARNING: PAPER_MODE=false but HL_TESTNET=true. "
            "You are trading on TESTNET with live backend. "
            "Set HL_TESTNET=false when ready for mainnet."
        )
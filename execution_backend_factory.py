"""Live backend factory.

Paper execution is intentionally unsupported. This factory always builds the
live Hyperliquid backend after validating live credentials.
"""

import os


def build_execution_backend(debug: bool = True):
    if os.getenv("PAPER_MODE", "").strip().lower() in {"1", "true", "yes", "on"}:
        raise EnvironmentError(
            "[FACTORY] PAPER_MODE is no longer supported. "
            "Remove PAPER_MODE or set it false and use the live backend."
        )

    if debug:
        print("[FACTORY] Live execution enabled — using LiveExecutionBackend")

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
            "Live execution is the only supported mode."
        )

    # Warn but don't block if still pointing at testnet
    if os.getenv("HL_TESTNET", "true").lower() == "true":
        print(
            "[FACTORY] WARNING: HL_TESTNET=true. "
            "You are using the live backend against TESTNET. "
            "Set HL_TESTNET=false when ready for mainnet."
        )

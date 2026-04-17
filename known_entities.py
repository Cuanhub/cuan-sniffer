"""
Known on-chain entities whose SOL movements are routine operations
(deposits, withdrawals, rebalancing) and should NOT be treated as smart-money
flow signals.

Categories:
  CEX hot/cold wallets  — high-volume, operationally driven
  Bridge programs       — token bridge escrow/relay
  Protocol treasuries   — DAO / team multisigs, grant wallets
  Staking programs      — stake pools and validators that move SOL continuously

Sources: on-chain labels cross-referenced from Solscan, Step Finance explorer,
and public Solana Foundation disclosures.  Add new entries as they are
identified rather than waiting for a mass update.
"""

# Addresses whose flow events are excluded from whale_pressure / imbalance
# calculations in FlowContext.  Also used by SolFlowEngine to suppress alerts
# for routine operational moves.
EXCHANGE_AND_PROGRAM_WALLETS: set[str] = {
    # ── Binance ───────────────────────────────────────────────────────────
    "9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM",
    "5tzFkiKscXHK5ZXCGbXZxdw7gE8Wk6NMnPLSaMaqbANy",
    "GJRs4FwHtemZ5ZE9x3FNvJ8TMwitKTh21yxdRPqn39fd",

    # ── Coinbase ──────────────────────────────────────────────────────────
    "H8sMJSCQxfKiFTCfDR3DUMLPwcRbM61LGFJ8N4dK3WjS",
    "GvpCiTgq9dmEeojCDBivoLoZqc4sFgBDnAJCRujRpVM",

    # ── OKX ───────────────────────────────────────────────────────────────
    "FWznbcNXWQuHTawe9RxvQ2LdCENssh12dsznf4RiouN5",

    # ── Kraken ────────────────────────────────────────────────────────────
    "2AQdpHJ2JpcEgPiATUXjQxA8QmafFegfQwSLWSprPicm",

    # ── Bybit ─────────────────────────────────────────────────────────────
    "AC5RDfQFmDS1deWZos921JfqscXdByf8BKHs5ACWjtW2",

    # ── KuCoin ────────────────────────────────────────────────────────────
    "BmFdpraQhkiDHSdBXWGZSqpiTFRdTxSPfRcVJhWDohBF",

    # ── Marinade Finance (staking) ────────────────────────────────────────
    "MarBmsSgKXdrN1egZf5sqe1TMai9K1rChYNDJgjq7aD",

    # ── Lido / stSOL (staking) ────────────────────────────────────────────
    "CrX7kMhLC3cSsXJdT7wiclwigNmmcm62broQr9CFuCNQ",

    # ── Wormhole bridge ───────────────────────────────────────────────────
    "worm2ZoG2kUd4vFXhvjh93UUH596ayRfgQ2MgjNMTth",

    # ── Allbridge ────────────────────────────────────────────────────────
    "Dn6pGzaLqY6bL9Frdm8pFJDZbDU6XFpJpPqD9NdTRPb",

    # ── Jump Crypto / Firedancer multisig ─────────────────────────────────
    "9itGPMCbMUPEhLGJMRPFdJN3BFJvvnAcaFG3EFWBXJZU",
}


def is_known_entity(address: str) -> bool:
    """Return True if this address is a known exchange/program wallet."""
    return address in EXCHANGE_AND_PROGRAM_WALLETS

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
    # Second Binance hot wallet — confirmed by address prefix match + $15B flow volume
    # (14,492 events, 72% OUT-count: routine exchange op pattern)
    "5tzFkiKscXHK5ZXCGbXZxdw7gTjjD1mBwuoFbhUvuAi9",

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

    # ── Unknown CEX / market-maker hot wallets (identified from DB, 2026-06-14) ───
    # Balanced IN/OUT, $1.37B volume, 3,554 events — classic MM / exchange hot wallet
    "G9X7F4JzLzbSGMCndiBdWNi5YzZZakmtkdwq7xS3Q3FE",
    # Cold-wallet sweep pattern: 1,823 small INs, 89 massive OUTs ($39.7M avg), $7B total
    "7mhcgF1DVsj5iv4CxZDgp51H6MBBwqamsH1KnqXhSRc5",
    # Hot-wallet distributor: 71 large INs, 959 small OUTs ($115K avg) — classic custodian
    "4xTyenBywER1qM8bPYyjNuagPaYTSXFHQqcZiRS5UdUK",

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

    # ══════════════════════════════════════════════════════════════════════
    # Solana core programs + major DeFi protocols
    # Forward-compatible: effective when any of these program accounts are
    # added to TRACKED_WALLETS (engine.py _handle_signature checks
    # is_known_entity against the TRACKED wallet address, not the program
    # that the wallet interacted with).
    # ══════════════════════════════════════════════════════════════════════

    # ── Solana system programs ────────────────────────────────────────────
    "11111111111111111111111111111111",                # System Program
    "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA",   # SPL Token Program
    "TokenzQdBNbequF1TqxBzYVBGnNHQKVsXSR4dTbQ8M",    # Token-2022
    "ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJe1bS",   # Associated Token Account
    "ComputeBudget111111111111111111111111111111",     # Compute Budget
    "MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr",   # Memo Program v2

    # ── Jupiter ───────────────────────────────────────────────────────────
    "JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN",   # Jupiter v6 aggregator
    "jupoNjAxXgZ4rjzxzPMP4XXi1FqVobsAkd69SVyoHNm",   # Jupiter Limit Orders

    # ── Raydium ───────────────────────────────────────────────────────────
    "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8",  # Raydium AMM v4
    "CAMMCzo5YL8w4VFF8KVHrK22GGUsp5VTaW7grrKgrWqK",  # Raydium CLMM
    "CPMMoo8L3F4NbTegBCKVNunggL7H1ZpdTHKxQB5qKP1C",  # Raydium CP-Swap
    "5quBtoiQqxF9Jv6KYKctB59NT3gtJD2Y65kdnB1Uev3h",  # Raydium fee account

    # ── Orca ──────────────────────────────────────────────────────────────
    "whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc",   # Orca Whirlpools
    "DjVE6JNiYqPL2QXyCUUh8rNjHrbz9hXHNYt99MQ59qr1",  # Orca token vault

    # ── Drift Protocol ────────────────────────────────────────────────────
    "dRiftyHA39MWEi3m9aunc5MzRF1JYuBsbn6VPcn33UH",   # Drift v2
    "8UJgxaiQx5nTrdDgph5FiahMmzduuLTLf5WmsPegYA6",   # Drift insurance fund

    # ── MarginFi ─────────────────────────────────────────────────────────
    "MFv2hWf31Z9kbCa1snEPdcgp7BKHKVKtW7cvMjVQSiK",  # MarginFi v2

    # ── Kamino ────────────────────────────────────────────────────────────
    "KLend2g3cP87fffoy8q1mQqGKjrL5kAkGJBgkRrXfSf",   # Kamino Lending
    "KAMT3kdKDsQQZrQPaJLovKcJaTzuQE6kh2bKJECatQg",   # Kamino vaults

    # ── Phoenix DEX ───────────────────────────────────────────────────────
    "PhoeNiXZ8ByJGLkxNfZRnkUfjvmuYqLR89jjFHGqdXY",   # Phoenix

    # ── Meteora ───────────────────────────────────────────────────────────
    "LBUZKhRxPF3XUpBCjp4YzTKgLccjZhTSDM9YuVaPwxo",   # Meteora DLMM
    "Eo7WjKq67rjJQDjFj1bkMhGjYJFNYxHtHMOLbCRaE1u",   # Meteora AMM

    # ── OpenBook (Serum successor) ────────────────────────────────────────
    "opnb2LAfJYbRMAHHvqjCwQxanZn7n1ZRjPa9DkAFMnm",   # OpenBook v2
    "srmqPvymJeFKQ4zGQed1GFppgkRHL9kaELCbyksJejD",    # Serum v3 DEX

    # ── Sanctum / liquid staking ──────────────────────────────────────────
    "stkitrT1Uoy18Dk1fTrgPw8W6MVzoCfYoAFT4MLsmhq",   # Sanctum router
    "SP12tWFxD9oB3nkvmbS65fn3djLH9epnAiMuZeANPFq",   # Sanctum stake pool

    # ── Tensor (NFT/token marketplace) ───────────────────────────────────
    "TSWAPaqyCSx2KABk68Shruf4rp7CxcAi9UTjtKidsZG",    # Tensor swap

    # ── Pyth oracle (price feeds) ─────────────────────────────────────────
    "FsJ3A3u2vn5cTVofAjvy6y5kwABJAqYWpe4975bi2epH",   # Pyth oracle v2
    "rec5EKMGg6MxZYaMdyBfgwp4d5rB9T1VQH5pDv3Nxi",    # Pyth receiver program
}


def is_known_entity(address: str) -> bool:
    """Return True if this address is a known exchange/program wallet."""
    return address in EXCHANGE_AND_PROGRAM_WALLETS

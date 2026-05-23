import time
from datetime import datetime

from sqlalchemy.exc import SQLAlchemyError

from db import SessionLocal, WalletBalance, FlowEvent
from sol_client import (
    get_signatures_for_address,
    get_sol_balance,
    get_sol_transfer_for_address,
    get_all_transfers_for_address,
)
from known_entities import is_known_entity
from config import MIN_SOL_ALERT, get_sol_price
from token_config import WATCHED_MINTS, MINT_TO_COIN, MIN_TOKEN_FLOW_USD, get_token_prices


class SolFlowEngine:
    """
    Core engine:
    - Tracks new signatures for each wallet
    - Recomputes SOL balance after each tx
    - Detects inflows/outflows >= MIN_SOL_ALERT
    - Logs events
    - (No Telegram alerts here; higher-level agent consumes FlowEvent)

    PLUS:
    - Light-but-snappy Helius usage via per-wallet throttling:
        * Wallets with recent big flows → scanned more often
        * Quiet wallets → scanned less often
    """

    def __init__(self, tracked_wallets):
        self.tracked_wallets = tracked_wallets

        # Store last seen signature per wallet
        self.last_signatures = {}  # {wallet: last_sig}

        # Per-wallet throttle state:
        # {
        #   wallet: {
        #       "last_scan": epoch_seconds_of_last_RPC_scan,
        #       "last_big_move": epoch_seconds_of_last_large_flow_event,
        #   }
        # }
        self.wallet_state = {
            addr: {"last_scan": 0.0, "last_big_move": 0.0}
            for addr in tracked_wallets
        }

        # Throttle tuning:
        # If wallet had a big move within ACTIVE_WINDOW_SEC → scan frequently
        # Else → scan more slowly to save Helius credits
        self.ACTIVE_WINDOW_SEC = 30 * 60        # 30 minutes
        self.ACTIVE_SCAN_INTERVAL_SEC = 30      # scan at most every 30s for active wallets
        self.QUIET_SCAN_INTERVAL_SEC = 180      # scan at most every 3min for quiet wallets

    # ---------------------------------------------------

    def _ensure_wallet_record(self, session, address: str):
        """
        Adds wallet to DB if not present.
        Sets initial balance so diffs work correctly.
        """
        record = session.query(WalletBalance).filter_by(address=address).first()
        if record:
            return record

        # First time seeing the wallet — pull initial balance
        initial_balance = get_sol_balance(address) or 0.0

        new_record = WalletBalance(
            address=address,
            sol_balance=initial_balance,
            updated_at=datetime.utcnow(),
        )
        session.add(new_record)
        session.commit()

        print(f"[INIT] {address} → {initial_balance:.4f} SOL")
        return new_record

    # ---------------------------------------------------

    def _should_scan_wallet(self, address: str) -> bool:
        """
        Decide whether we should actually hit Helius for this wallet
        on this cycle, based on last_scan and last_big_move.

        This is where we save credits:
        - Active wallets (recent large flow) → scan more often
        - Quiet wallets → scan less often
        """
        now_ts = time.time()
        state = self.wallet_state.setdefault(
            address, {"last_scan": 0.0, "last_big_move": 0.0}
        )

        since_last_big = now_ts - state["last_big_move"]
        # Choose desired scan interval
        if since_last_big < self.ACTIVE_WINDOW_SEC:
            desired_interval = self.ACTIVE_SCAN_INTERVAL_SEC
            mode = "active"
        else:
            desired_interval = self.QUIET_SCAN_INTERVAL_SEC
            mode = "quiet"

        since_last_scan = now_ts - state["last_scan"]

        if since_last_scan < desired_interval:
            # Too soon to scan again, skip to save credits
            # (You can uncomment the next line if you want to see throttling per wallet)
            # print(f"[THROTTLE] {address}: {mode}, last_scan={since_last_scan:.1f}s < {desired_interval}s, skipping.")
            return False

        # Update last_scan timestamp and allow scan
        state["last_scan"] = now_ts
        return True

    # ---------------------------------------------------

    def process_wallet(self, address: str):
        """
        Main scanning loop for each wallet.
        - Throttle check (light-but-snappy)
        - fetch new signatures
        - detect new events
        """

        # Throttle Helius calls per wallet
        if not self._should_scan_wallet(address):
            return

        session = SessionLocal()

        try:
            wallet_record = self._ensure_wallet_record(session, address)

            last_sig = self.last_signatures.get(address)
            sigs = get_signatures_for_address(address, limit=10)

            if not sigs:
                session.close()
                return

            # Determine which signatures are NEW
            new_sigs = []
            for entry in sigs:
                if entry["signature"] == last_sig:
                    break
                new_sigs.append(entry)

            if not new_sigs:
                session.close()
                return

            # Process from oldest → newest
            for entry in reversed(new_sigs):
                sig = entry["signature"]
                slot = entry["slot"]
                self._handle_signature(session, address, sig, slot)

            # Update pointer
            self.last_signatures[address] = new_sigs[0]["signature"]

        except Exception as e:
            print(f"[ERROR] Engine crashed on wallet {address}: {e}")

        finally:
            session.close()

    # ---------------------------------------------------

    def _handle_signature(self, session, address: str, signature: str, slot: int):
        """
        On each tx:
        - parse actual SOL delta from tx preBalances/postBalances (primary)
        - fall back to balance-diff polling if tx cannot be fetched
        - if >= threshold: store + log big flow event
        """

        # Skip wallets classified as exchanges / programs — their moves are
        # routine operations and would pollute the flow signal.
        if is_known_entity(address):
            print(f"[SKIP] {address} is a known exchange/program wallet — not a signal")
            return

        record = session.query(WalletBalance).filter_by(address=address).first()
        if not record:
            return

        # Single RPC fetch — parse both SOL delta and SPL token transfers at once.
        transfers = get_all_transfers_for_address(signature, address, WATCHED_MINTS)
        delta = transfers["sol_delta"]
        token_transfers = transfers["token_transfers"]

        if delta is None:
            # Fallback: balance diff (noisier — fees/rent included)
            old_balance = record.sol_balance
            new_balance = get_sol_balance(address)
            if new_balance is None:
                return
            delta = new_balance - old_balance
            new_balance_for_db = new_balance
        else:
            new_balance_for_db = record.sol_balance + delta

        # Update base balance regardless
        record.sol_balance = new_balance_for_db
        record.updated_at = datetime.utcnow()

        any_event_written = False

        # ── SOL flow event ────────────────────────────────────────────────────
        if abs(delta) >= MIN_SOL_ALERT:
            direction = "IN" if delta > 0 else "OUT"
            amount = abs(delta)
            usd_val = amount * get_sol_price()

            session.add(FlowEvent(
                address=address,
                direction=direction,
                sol_amount=amount,
                usd_value=usd_val,
                signature=signature,
                slot=slot,
                coin="SOL",
                created_at=datetime.utcnow(),
            ))
            any_event_written = True

            state = self.wallet_state.setdefault(
                address, {"last_scan": 0.0, "last_big_move": 0.0}
            )
            state["last_big_move"] = time.time()

            emoji = "🟢" if direction == "IN" else "🔴"
            print(
                f"[FLOW/SOL] {emoji} {direction} {amount:.2f} SOL "
                f"(~${usd_val:,.0f}) | {address} | slot {slot}"
            )
        else:
            print(f"[DEBUG] {address}: Δ {delta:.3f} SOL (below threshold)")

        # ── SPL token flow events ─────────────────────────────────────────────
        if token_transfers:
            # Fetch token prices in one batch call to avoid N separate requests
            symbols_needed = [
                MINT_TO_COIN[t["mint"]]
                for t in token_transfers
                if t["mint"] in MINT_TO_COIN
            ]
            prices = get_token_prices(symbols_needed) if symbols_needed else {}

            for transfer in token_transfers:
                mint = transfer["mint"]
                coin_symbol = MINT_TO_COIN.get(mint)
                if not coin_symbol:
                    continue

                token_price = prices.get(coin_symbol, 0.0)
                delta_ui = abs(transfer["delta_ui"])
                usd_val = delta_ui * token_price

                if usd_val < MIN_TOKEN_FLOW_USD:
                    print(
                        f"[DEBUG] {address}: {coin_symbol} Δ {delta_ui:.2f} "
                        f"(~${usd_val:,.0f}) below token threshold"
                    )
                    continue

                direction = transfer["direction"]
                # sol_amount stores USD value for token events so FlowContext
                # imbalance calculations stay in comparable units per coin.
                session.add(FlowEvent(
                    address=address,
                    direction=direction,
                    sol_amount=usd_val,   # USD equivalent — see flow_context.py
                    usd_value=usd_val,
                    signature=signature,
                    slot=slot,
                    coin=coin_symbol,
                    created_at=datetime.utcnow(),
                ))
                any_event_written = True

                state = self.wallet_state.setdefault(
                    address, {"last_scan": 0.0, "last_big_move": 0.0}
                )
                state["last_big_move"] = time.time()

                emoji = "🟢" if direction == "IN" else "🔴"
                print(
                    f"[FLOW/{coin_symbol}] {emoji} {direction} {delta_ui:.2f} "
                    f"{coin_symbol} (~${usd_val:,.0f}) | {address} | slot {slot}"
                )

        # ── Persist all events atomically ─────────────────────────────────────
        if any_event_written or True:   # always commit balance update
            try:
                session.commit()
            except SQLAlchemyError as e:
                session.rollback()
                print(f"[DB ERROR] {e}")

import time
from datetime import datetime

from sqlalchemy.exc import SQLAlchemyError

from db import SessionLocal, WalletBalance, FlowEvent
from sol_client import get_signatures_for_address, get_sol_balance, get_sol_transfer_for_address
from known_entities import is_known_entity
from config import MIN_SOL_ALERT, get_sol_price


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

        # Primary: parse authoritative SOL delta straight from the transaction.
        # This is immune to fee/rent noise that corrupts balance-diff polling.
        delta = get_sol_transfer_for_address(signature, address)

        if delta is None:
            # Fallback: balance diff (noisier — fees/rent included)
            old_balance = record.sol_balance
            new_balance = get_sol_balance(address)
            if new_balance is None:
                return
            delta = new_balance - old_balance
            new_balance_for_db = new_balance
        else:
            # Recompute absolute balance from the parsed delta so the DB stays in sync
            new_balance_for_db = record.sol_balance + delta

        # Update base balance regardless
        record.sol_balance = new_balance_for_db
        record.updated_at = datetime.utcnow()

        # Not big enough to alert/log as major event?
        if abs(delta) < MIN_SOL_ALERT:
            try:
                session.commit()
            except SQLAlchemyError:
                session.rollback()
            print(f"[DEBUG] {address}: Δ {delta:.3f} SOL (below threshold)")
            return

        # Big move detected
        direction = "IN" if delta > 0 else "OUT"
        amount = abs(delta)
        usd_val = amount * get_sol_price()

        event = FlowEvent(
            address=address,
            direction=direction,
            sol_amount=amount,
            usd_value=usd_val,
            signature=signature,
            slot=slot,
            created_at=datetime.utcnow(),
        )

        session.add(event)

        try:
            session.commit()
        except SQLAlchemyError as e:
            session.rollback()
            print(f"[DB ERROR] {e}")
            return

        # Mark this wallet as "recently active" for throttle logic
        state = self.wallet_state.setdefault(
            address, {"last_scan": 0.0, "last_big_move": 0.0}
        )
        state["last_big_move"] = time.time()

        # --- Log big flow event (no external alert here) ---
        emoji = "🟢" if direction == "IN" else "🔴"
        print(
            f"[FLOW] {emoji} {direction} {amount:.2f} SOL "
            f"(~${usd_val:,.0f}) | {address} | slot {slot} | {signature}"
        )
        # NOTE:
        # We intentionally do NOT send Telegram alerts here.
        # These events are consumed by the higher-level agent which decides
        # when to propose actual trades based on all context.

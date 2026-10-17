"""
Live execution backend for Hyperliquid.

Implements the ExecutionBackend interface:
execute_entry / execute_exit / get_mid_price / shutdown.

Patched improvements:
- Reason-aware exit aggressiveness
- Exit pricing anchored to the more aggressive of trigger price and mid
- Stronger debug logging for exits
- Exit fees estimated with estimate_exit_fee()

────────────────────────────────────────────────────────────────────────────────
LIVE POSITION MONITORING ARCHITECTURE — READ BEFORE MODIFYING
────────────────────────────────────────────────────────────────────────────────

This backend is STATELESS. It does NOT monitor open positions for stop or TP hits.
It only executes individual order requests when called. The full monitoring stack is:

1. executor._evaluate_live_position()
   Called each agent cycle via executor.update() → _update_live_positions().
   Software-based price polling: checks pos.stop_price and pos.tp_price against
   current mid-price. Fires _live_close_runner / _live_close_full_stop /
   _live_take_partial as appropriate.
   RISK: If executor is down or the agent loop is blocked, software exits do not fire.

2. ProtectionManager (protection_manager.py)
   Places native stop and TP trigger orders directly on Hyperliquid at entry time
   via place_stop_tp(). These orders fire independently of the agent process.
   They are the primary safety net when executor is offline.
   Monitored by executor._audit_open_protection() each update cycle.

3. LivePositionMonitor (live_position_monitor.py)
   Called each agent cycle via executor.update() → live_monitor.update().
   Polls venue position size and reconciles positions that went flat on-venue
   but were not yet finalized locally. Only processes positions explicitly marked
   allow_reconcile_close=True (bootstrap-restored or after executor exit intent).
   Assigns CloseReason.MANUAL by default for externally-closed positions (native
   TP/SL fired, or manual close on venue UI).

INVESTIGATION: "manual" close_reason and reconciled_from_venue=True in trades.csv
   These entries occur when a native TP or SL order placed by ProtectionManager
   fires on the venue. The executor's software exit path was NOT triggered (the
   native order fired first or independently). LivePositionMonitor detected the
   flat state and reconciled with CloseReason.MANUAL.
   Affected trades: positions where protection_status="protected" at entry and
   close_reason="manual" + reconciled_from_venue="true" at close.
   This is EXPECTED BEHAVIOR, not a bug. The trade was exited safely via the
   native protection order. The "manual" label is cosmetically misleading
   (see OPEN ITEM below).

OPEN ITEM: consider assigning CloseReason.TP_FULL or CloseReason.STOP_FULL in
   LivePositionMonitor._finalize_closed_position() based on whether the position
   closed at profit (realized_r > 0) or loss, rather than always using MANUAL.
   This would improve analytics accuracy (strategy filter kill-switch, recap reports).
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING, ROUND_DOWN, ROUND_FLOOR, ROUND_HALF_UP
from math import floor, log10
from typing import Any, Dict, List, Optional, Tuple

from eth_account import Account

from execution_backend import ExecutionBackend, FillResult
from live_data_guard import (
    LIVE_MAX_MIDPRICE_AGE_SECONDS,
    api_backoff_status,
    cache_age_seconds,
    cache_is_fresh,
    record_api_failure,
    record_api_success,
)

from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from hyperliquid.utils import constants

# ── Env config ─────────────────────────────────────────────────────────────

HL_TESTNET = os.getenv("HL_TESTNET", "true").lower() == "true"
HL_ACCOUNT_ADDRESS = os.getenv("HL_ACCOUNT_ADDRESS", "").strip()
HL_SECRET_KEY = os.getenv("HL_SECRET_KEY", "").strip()
HL_VAULT_ADDRESS = os.getenv("HL_VAULT_ADDRESS", "").strip() or None

LIVE_ENTRY_IOC_BPS = float(os.getenv("LIVE_ENTRY_IOC_BPS", "25"))
LIVE_EXIT_SLIPPAGE_BPS = float(os.getenv("LIVE_EXIT_SLIPPAGE_BPS", "10"))

# Reason-aware exit slippage
LIVE_EXIT_SLIPPAGE_BPS_TP = float(os.getenv("LIVE_EXIT_SLIPPAGE_BPS_TP", str(LIVE_EXIT_SLIPPAGE_BPS)))
LIVE_EXIT_SLIPPAGE_BPS_PARTIAL = float(os.getenv("LIVE_EXIT_SLIPPAGE_BPS_PARTIAL", "12"))
LIVE_EXIT_SLIPPAGE_BPS_STOP_RUNNER = float(os.getenv("LIVE_EXIT_SLIPPAGE_BPS_STOP_RUNNER", "20"))
LIVE_EXIT_SLIPPAGE_BPS_STOP_FULL = float(os.getenv("LIVE_EXIT_SLIPPAGE_BPS_STOP_FULL", "35"))
LIVE_EXIT_SLIPPAGE_BPS_MANUAL = float(os.getenv("LIVE_EXIT_SLIPPAGE_BPS_MANUAL", str(LIVE_EXIT_SLIPPAGE_BPS)))
HL_PROTECTION_USE_TRIGGER_MARKET = (
    os.getenv("HL_PROTECTION_USE_TRIGGER_MARKET", "true").lower() == "true"
)
HL_PROTECTION_TRIGGER_REFERENCE = os.getenv("HL_PROTECTION_TRIGGER_REFERENCE", "mark").strip().lower()
HL_SL_TRIGGER_IS_MARKET = os.getenv("HL_SL_TRIGGER_IS_MARKET", "true").lower() == "true"
HL_TP_TRIGGER_IS_MARKET = os.getenv(
    "HL_TP_TRIGGER_IS_MARKET",
    "true" if HL_PROTECTION_USE_TRIGGER_MARKET else "false",
).lower() == "true"
HL_SL_ALLOW_LIMIT_FALLBACK = os.getenv("HL_SL_ALLOW_LIMIT_FALLBACK", "false").lower() == "true"
HL_TP_ALLOW_LIMIT_FALLBACK = os.getenv("HL_TP_ALLOW_LIMIT_FALLBACK", "true").lower() == "true"

TAKER_FEE_BPS = float(os.getenv("TAKER_FEE_BPS", "4.5"))
MID_CACHE_TTL_SEC = float(os.getenv("MID_CACHE_TTL_SEC", "0.50"))
LIVE_LOG_ORDERS = os.getenv("LIVE_LOG_ORDERS", "true").lower() == "true"
LIVE_FLAT_EPSILON_SZ = float(os.getenv("LIVE_FLAT_EPSILON_SZ", "1e-9"))

# Passed to both Info and Exchange to prevent internal spot metadata fetch.
_EMPTY_SPOT_META: Dict[str, Any] = {"tokens": [], "universe": []}


class LiveExecutionBackend(ExecutionBackend):
    def __init__(self, debug: bool = True):
        self.debug = debug

        if not HL_ACCOUNT_ADDRESS:
            raise ValueError("[LIVE_BACKEND] HL_ACCOUNT_ADDRESS not set in env")
        if not HL_SECRET_KEY:
            raise ValueError("[LIVE_BACKEND] HL_SECRET_KEY not set in env")

        self.base_url = (
            constants.TESTNET_API_URL if HL_TESTNET else constants.MAINNET_API_URL
        )

        self.account_address = HL_ACCOUNT_ADDRESS.strip()
        self.vault_address = HL_VAULT_ADDRESS.strip() if HL_VAULT_ADDRESS else None
        self._wallet = Account.from_key(HL_SECRET_KEY)
        self.wallet = self._wallet

        self.info = Info(
            self.base_url,
            skip_ws=True,
            spot_meta=_EMPTY_SPOT_META,
        )

        self.exchange = Exchange(
            self._wallet,
            self.base_url,
            account_address=self.account_address,
            vault_address=self.vault_address,
            spot_meta=_EMPTY_SPOT_META,
        )

        self._mid_cache: Dict[str, float] = {}
        self._mid_cache_ts: float = 0.0
        self._sz_decimals: Dict[str, int] = {}
        self._max_leverage: Dict[str, int] = {}
        self._unified_account_cache: Optional[bool] = None

        self._load_meta()

        venue = "TESTNET" if HL_TESTNET else "MAINNET"
        print(
            f"[LIVE_BACKEND] Initialized | venue={venue} | "
            f"account={self.account_address} | "
            f"assets_loaded={len(self._sz_decimals)}"
        )

    # ── Entry ──────────────────────────────────────────────────────────────

    def execute_entry(
        self,
        coin: str,
        side: str,
        price: float,
        size_usd: float,
        confidence: float,
    ) -> FillResult:
        coin = coin.upper()
        side = side.upper()

        if size_usd <= 0:
            return self._reject("invalid_entry_request", price, size_usd, "entry")
        if not self._ensure_meta():
            # Without szDecimals every size and price is rounded to a guessed
            # precision. Refusing is the only safe answer.
            return self._reject("asset_metadata_unavailable", price, size_usd, "entry")

        ref_price = self.get_mid_price(coin)
        if not ref_price or ref_price <= 0:
            return self._reject("stale_midprice", price, size_usd, "entry")

        sz = self._notional_to_size(coin, size_usd, ref_price)
        if sz <= 0:
            return self._reject("size_rounds_to_zero", ref_price, size_usd, "entry")

        is_buy = side == "LONG"
        limit_px = self.round_to_tick(
            ref_price * (1 + LIVE_ENTRY_IOC_BPS / 10_000) if is_buy
            else ref_price * (1 - LIVE_ENTRY_IOC_BPS / 10_000),
            coin,
            direction="nearest",
        )
        if limit_px <= 0:
            return self._reject("invalid_rounded_entry_price", ref_price, size_usd, "entry")

        try:
            resp = self.exchange.order(
                coin,
                is_buy,
                sz,
                limit_px,
                {"limit": {"tif": "Ioc"}},
                reduce_only=False,
            )
        except Exception as e:
            return self._reject(f"sdk_error:{e}", ref_price, size_usd, "entry")

        result = self._parse_fill(
            resp,
            coin=coin,
            requested_notional=size_usd,
            ref_price=ref_price,
            is_exit=False,
            is_buy=is_buy,
        )
        result.reason = "entry"

        if LIVE_LOG_ORDERS:
            status = "FILLED" if result.filled else f"REJECTED({result.reject_reason})"
            print(
                f"[LIVE_BACKEND] ENTRY {coin} {status} "
                f"side={side} ref_px={ref_price:.4f} limit_px={limit_px:.4f} "
                f"requested_notional={size_usd:.2f} rounded_sz={sz:.8f} "
                f"fill_px={result.fill_price:.4f} "
                f"fill_sz=${result.fill_size_usd:.2f} "
                f"slip={result.slippage_bps:.1f}bps"
            )

        return result

    # ── Exit ───────────────────────────────────────────────────────────────

    def execute_exit(
        self,
        coin: str,
        side: str,
        price: float,
        size_usd: float,
        reason: str = "",
    ) -> FillResult:
        coin = coin.upper()
        side = side.upper()
        reason = (reason or "exit").lower().strip()

        if size_usd <= 0:
            return self._reject("invalid_exit_request", price, size_usd, reason)

        mid_price = self.get_mid_price(coin)
        if not mid_price or mid_price <= 0:
            return self._reject("missing_market_price", price, size_usd, reason)

        # More aggressive reference pricing:
        # For sell exits (close longs), use the lower of trigger price and mid.
        # For buy exits (close shorts), use the higher of trigger price and mid.
        is_buy = side == "SHORT"
        trigger_price = float(price) if price and price > 0 else mid_price
        if is_buy:
            ref_price = max(mid_price, trigger_price)
        else:
            ref_price = min(mid_price, trigger_price)

        signed_venue_sz = self._get_venue_signed_size(coin)
        venue_sz = self._signed_size_for_side(side, signed_venue_sz)
        requested_sz = self._notional_to_size(coin, size_usd, ref_price)
        sz = min(requested_sz, venue_sz) if venue_sz is not None else requested_sz

        if sz <= 0:
            wallet_flat = (
                signed_venue_sz is not None
                and abs(float(signed_venue_sz)) <= LIVE_FLAT_EPSILON_SZ
            )
            reject_meta = {
                "coin": coin,
                "side": side,
                "requested_usd": float(size_usd),
                "requested_coin_size": float(requested_sz),
                "final_send_size": float(sz),
                "venue_side_size": (float(venue_sz) if venue_sz is not None else None),
                "signed_venue_size": (float(signed_venue_sz) if signed_venue_sz is not None else None),
                "user_state_address": self._user_state_query_address(),
                "reject_reason": "size_rounds_to_zero_or_no_position",
                "order_reason": reason,
                "wallet_flat_hint": bool(wallet_flat),
                "unknown_venue_state": bool(signed_venue_sz is None),
            }
            if LIVE_LOG_ORDERS:
                print(
                    f"[LIVE_BACKEND] EXIT {coin} {side} {reason} REJECTED(size_rounds_to_zero_or_no_position) "
                    f"req_usd={size_usd:.2f} req_sz={requested_sz:.8f} send_sz={sz:.8f} "
                    f"venue_side_sz={venue_sz} venue_szi={signed_venue_sz} "
                    f"wallet_flat_hint={wallet_flat} "
                    f"user_state_addr={reject_meta['user_state_address']}"
                )
            return self._reject(
                "size_rounds_to_zero_or_no_position",
                ref_price,
                size_usd,
                reason,
                meta=reject_meta,
            )

        exit_bps = self._exit_slippage_bps_for_reason(reason)
        limit_px = self._round_price(
            coin,
            ref_price * (1 + exit_bps / 10_000) if is_buy
            else ref_price * (1 - exit_bps / 10_000),
        )

        try:
            resp = self.exchange.order(
                coin,
                is_buy,
                sz,
                limit_px,
                {"limit": {"tif": "Ioc"}},
                reduce_only=True,
            )
        except Exception as e:
            return self._reject(f"sdk_error:{e}", ref_price, size_usd, reason)

        result = self._parse_fill(
            resp,
            coin=coin,
            requested_notional=size_usd,
            ref_price=ref_price,
            is_exit=True,
            slip_ref=mid_price,
            is_buy=is_buy,
        )
        result.reason = reason

        if LIVE_LOG_ORDERS:
            status = "FILLED" if result.filled else f"REJECTED({result.reject_reason})"
            print(
                f"[LIVE_BACKEND] EXIT {coin} {side} {reason} {status} "
                f"trigger_px={trigger_price:.4f} mid_px={mid_price:.4f} ref_px={ref_price:.4f} "
                f"limit_px={limit_px:.4f} venue_side_sz={venue_sz} venue_szi={signed_venue_sz} "
                f"req_sz={requested_sz:.8f} send_sz={sz:.8f} "
                f"user_state_addr={self._user_state_query_address()} "
                f"fill_px={result.fill_price:.4f} fill_sz=${result.fill_size_usd:.2f} "
                f"slip={result.slippage_bps:.1f}bps"
            )

        return result

    def flatten(self, coin: str, side: str = "LONG", reason: str = "flatten",
                attempts: int = 3) -> Dict[str, Any]:
        """Close a position COMPLETELY, checking the venue between attempts.

        execute_exit sends one marketable IOC with bounded slippage. A single
        call is not a close: partial fills and no-fills are both normal, and
        on 2026-10-17 a hand-flatten of four positions left three as dust and
        ZRO entirely untouched at $178.70 of exposure, while every caller had
        been told the exit was sent. One more pass closed all four.

        So "make me flat" is a loop that verifies, not a single order. Size
        comes from the VENUE each time rather than from what the caller
        believes it holds, because the gap between those two is exactly the
        condition this exists to resolve.

        Returns {flat, attempts_used, closed_usd, detail}. `flat` is the only
        field a caller should branch on, and a caller that releases a
        position's slot must branch on it: dropping tracking while the venue
        still holds size orphans a live position.
        """
        coin = coin.upper()
        closed_usd = 0.0
        notes = []
        for attempt in range(1, max(1, attempts) + 1):
            szi = self._get_venue_signed_size(coin)
            if szi is None:
                # Cannot read the venue, so cannot claim flat. Saying "flat"
                # here is the one answer that would orphan a position.
                notes.append(f"attempt {attempt}: venue size unreadable")
                continue
            remaining = self._signed_size_for_side(side, szi)
            if remaining is None or abs(float(remaining)) <= LIVE_FLAT_EPSILON_SZ:
                return {"flat": True, "attempts_used": attempt - 1,
                        "closed_usd": closed_usd,
                        "detail": "; ".join(notes) or "already flat"}
            mid = self.get_mid_price(coin)
            if not mid or mid <= 0:
                notes.append(f"attempt {attempt}: no price, not sending blind")
                continue
            notional = abs(float(remaining)) * float(mid)
            res = self.execute_exit(coin=coin, side=side, price=float(mid),
                                    size_usd=notional, reason=reason)
            closed_usd += float(getattr(res, "fill_size_usd", 0.0) or 0.0)
            ratio = getattr(res, "fill_ratio", None)
            notes.append(
                f"attempt {attempt}: "
                + (f"filled {ratio:.0%} of ${notional:.2f}" if ratio is not None
                   else f"not attempted ({getattr(res, 'reject_reason', '') or 'no reason'})"))

        szi = self._get_venue_signed_size(coin)
        remaining = (self._signed_size_for_side(side, szi)
                     if szi is not None else None)
        flat = (remaining is not None
                and abs(float(remaining)) <= LIVE_FLAT_EPSILON_SZ)
        if not flat:
            notes.append(f"STILL OPEN: venue size {remaining}")
        return {"flat": flat, "attempts_used": attempts,
                "closed_usd": closed_usd, "detail": "; ".join(notes)}

    def place_native_protection(
        self,
        coin: str,
        side: str,
        size_coin: float,
        stop_price: float,
        tp_price: float,
    ) -> Dict[str, Any]:
        coin = coin.upper()
        side = side.upper()
        close_is_buy = side == "SHORT"
        send_sz = self._round_size(coin, abs(float(size_coin)))
        ref_px = self.get_mid_price(coin)

        result: Dict[str, Any] = {
            "coin": coin,
            "side": side,
            "size_coin": send_sz,
            "stop_order_id": "",
            "tp_order_id": "",
            "status": "missing",
            "error": "",
            "trigger_type": HL_PROTECTION_TRIGGER_REFERENCE,
            "stop_trigger_price": float(stop_price),
            "tp_trigger_price": float(tp_price),
            # False until the venue confirms the resting order matches what we
            # sent. Every early return below leaves them False, which is the
            # honest answer: nothing was placed, so nothing was verified.
            "stop_verified": False,
            "tp_verified": False,
        }

        if send_sz <= 0:
            result["error"] = "invalid_protection_size"
            return result
        if not self._ensure_meta():
            result["error"] = "asset_metadata_unavailable"
            return result

        # The venue's own position size wins whenever it can be read. It is the
        # only authoritative number: protection has to cover what is actually
        # held, not what the caller believed it bought.
        #
        # This used to override only when the gap exceeded max(one tick, 0.1%),
        # with a strict >. With a one-tick threshold a one-tick discrepancy
        # could never be corrected -- and one tick is exactly what this path
        # produces. PONS on 2026-09-24: filled_usd / fill_px came out as
        # 114.99999999999999, a float artifact of dividing notional by price,
        # and _round_size rounds DOWN, so a 115-coin position was protected for
        # 114. The gap was 1.0 against a threshold of 1.0, so the override
        # declined to fire and one coin sat naked until the take-profit closed
        # the other 114.
        venue_side_sz = self._get_venue_position_size(coin, side)
        if venue_side_sz is not None and venue_side_sz > 0:
            venue_send_sz = self._round_size(coin, float(venue_side_sz))
            if venue_send_sz > 0 and venue_send_sz != send_sz:
                if LIVE_LOG_ORDERS:
                    print(
                        f"[LIVE_BACKEND] PROTECTION size override {coin} {side} "
                        f"req_sz={send_sz:.8f} venue_sz={venue_send_sz:.8f} "
                        f"(the venue is authoritative)"
                    )
                send_sz = venue_send_sz
                result["size_coin"] = send_sz

        stop_round_direction = "up" if side == "LONG" else "down"
        rounded_stop_px = self.round_to_tick(
            float(stop_price),
            coin,
            direction=stop_round_direction,
        )
        rounded_tp_px = self.round_to_tick(
            float(tp_price),
            coin,
            direction="nearest",
        )
        if rounded_stop_px <= 0 or rounded_tp_px <= 0:
            result["error"] = "invalid_protection_price_after_rounding"
            return result
        result["stop_trigger_price"] = rounded_stop_px
        result["tp_trigger_price"] = rounded_tp_px

        relation_error = self._validate_protection_price_relations(
            side=side,
            stop_price=rounded_stop_px,
            tp_price=rounded_tp_px,
            anchor_price=ref_px,
        )
        if relation_error:
            result["error"] = relation_error
            if LIVE_LOG_ORDERS:
                print(
                    f"[LIVE_BACKEND] PROTECTION {coin} {side} REJECTED "
                    f"err={relation_error} stop={rounded_stop_px} tp={rounded_tp_px} "
                    f"anchor={ref_px}"
                )
            return result

        stop_oid, stop_err = self._place_trigger_reduce_only(
            coin=coin,
            close_is_buy=close_is_buy,
            size_coin=send_sz,
            trigger_price=rounded_stop_px,
            tpsl="sl",
            trigger_is_market=HL_SL_TRIGGER_IS_MARKET,
            allow_limit_fallback=HL_SL_ALLOW_LIMIT_FALLBACK,
        )
        tp_oid, tp_err = self._place_trigger_reduce_only(
            coin=coin,
            close_is_buy=close_is_buy,
            size_coin=send_sz,
            trigger_price=rounded_tp_px,
            tpsl="tp",
            trigger_is_market=HL_TP_TRIGGER_IS_MARKET,
            allow_limit_fallback=HL_TP_ALLOW_LIMIT_FALLBACK,
        )

        verification = self._verify_protection_orders(
            coin=coin,
            close_is_buy=close_is_buy,
            expected_size_coin=send_sz,
            expected_stop_px=rounded_stop_px,
            expected_tp_px=rounded_tp_px,
            stop_oid=stop_oid,
            tp_oid=tp_oid,
        )
        stop_hard, stop_soft = verification["stop"]
        tp_hard, tp_soft = verification["tp"]
        stop_err = self._append_verify_notes(stop_err, stop_hard, stop_soft)
        tp_err = self._append_verify_notes(tp_err, tp_hard, tp_soft)

        # The OIDs are kept whatever the verdict. They are what lets a caller
        # cancel an order the venue just contradicted, so discarding them here
        # would destroy the only handle on the problem.
        result["stop_order_id"] = stop_oid or ""
        result["tp_order_id"] = tp_oid or ""
        result["stop_verified"] = not stop_hard
        result["tp_verified"] = not tp_hard

        # An OID proves the venue ACCEPTED something, not that what it accepted
        # is a stop. Until 2026-09-18 status was computed from OID presence
        # alone and verification was advisory, so the sole live protection this
        # account ever placed was recorded "protected" while carrying four
        # verification failures. status is the only field the caller acts on
        # -- tools/live_levels_trader closes any position whose protection is
        # not "protected" -- so a contradicted order must not count here.
        stop_ok = bool(stop_oid) and not stop_hard
        tp_ok = bool(tp_oid) and not tp_hard
        ok_count = int(stop_ok) + int(tp_ok)
        if ok_count == 2:
            result["status"] = "protected"
        elif ok_count == 1:
            result["status"] = "partial"
        else:
            result["status"] = "missing"

        errors = []
        if stop_err:
            errors.append(f"stop:{stop_err}")
        if tp_err:
            errors.append(f"tp:{tp_err}")
        global_verify_err = verification.get("global", "")
        if global_verify_err:
            errors.append(f"verify:{global_verify_err}")
        result["error"] = " | ".join(errors)

        if LIVE_LOG_ORDERS:
            print(
                f"[LIVE_BACKEND] PROTECTION {coin} {side} status={result['status']} "
                f"sz={send_sz:.8f} stop_oid={result['stop_order_id'] or '-'} "
                f"tp_oid={result['tp_order_id'] or '-'} "
                f"stop_px={rounded_stop_px:.8f} tp_px={rounded_tp_px:.8f} "
                f"trigger_type={HL_PROTECTION_TRIGGER_REFERENCE} "
                f"err={result['error'] or '-'}"
            )

        return result

    def place_limit_entry(
        self,
        coin: str,
        side: str,
        price: float,
        size_usd: float,
    ) -> Dict[str, Any]:
        """Rest a GTC limit order at `price`. Does not wait for a fill.

        execute_entry sends a marketable IOC and takes whatever the book
        offers, which is the right shape for "enter now" and the wrong one for
        support that taps and leaves. Measured on 2026-09-19: across 139
        tradeable watched levels not one was inside the 1.5% entry drift
        guard, the nearest 3.10% away -- every tap had resolved before a poll
        could see it. A bid sitting at the level is filled BY the tap instead
        of chasing it.

        Returns {"oid", "price", "size_coin", "error"}. An empty oid means
        nothing is resting.
        """
        coin = coin.upper()
        out: Dict[str, Any] = {"coin": coin, "side": side.upper(), "oid": "",
                               "price": 0.0, "size_coin": 0.0, "error": ""}
        if size_usd <= 0 or price <= 0:
            out["error"] = "invalid_limit_request"
            return out
        if not self._ensure_meta():
            out["error"] = "asset_metadata_unavailable"
            return out

        limit_px = self._round_price(coin, float(price))
        sz = self._notional_to_size(coin, size_usd, limit_px)
        if limit_px <= 0:
            out["error"] = "invalid_rounded_limit_price"
            return out
        if sz <= 0:
            out["error"] = "size_rounds_to_zero"
            return out

        try:
            resp = self.exchange.order(
                coin,
                side.upper() == "LONG",
                sz,
                limit_px,
                {"limit": {"tif": "Gtc"}},
                reduce_only=False,
            )
        except Exception as e:
            out["error"] = f"sdk_error:{str(e)[:160]}"
            return out

        # Same envelope as a trigger order: statuses carry resting.oid.
        oid, err = self._parse_trigger_order_response(resp)
        out.update({"oid": oid or "", "price": limit_px, "size_coin": sz,
                    "error": "" if oid else (err or "no_resting_oid")})
        if LIVE_LOG_ORDERS:
            print(f"[LIVE_BACKEND] LIMIT_ENTRY {coin} {side.upper()} "
                  f"{'RESTING' if oid else 'FAILED'} px={limit_px:.8f} sz={sz:.8f} "
                  f"oid={oid or '-'} err={out['error'] or '-'}")
        return out

    def cancel_order(self, coin: str, oid: str) -> bool:
        """Cancel one resting order. True only when the venue confirms it.

        The oid is sent as an INT. The SDK signature is cancel(name, oid: int)
        and the venue answers a string with 422 "Failed to deserialize the
        JSON body into the target type" -- so every cancel this system ever
        attempted failed, silently, because this returns False and the callers
        only print. That covers the protection orders cancelled after an
        emergency exit and the expiry of a resting bid: exactly the cleanup
        paths whose whole job is to leave nothing behind at the venue.
        """
        coin = str(coin or "").upper().strip()
        oid_str = str(oid or "").strip()
        if not coin or not oid_str:
            return False
        try:
            oid_int = int(oid_str)
        except (TypeError, ValueError):
            if LIVE_LOG_ORDERS:
                print(f"[LIVE_BACKEND] CANCEL bad oid coin={coin} oid={oid_str!r}")
            return False
        try:
            resp = self.exchange.cancel(coin, oid_int)
        except Exception as e:
            if LIVE_LOG_ORDERS:
                print(f"[LIVE_BACKEND] CANCEL failed coin={coin} oid={oid_str} err={e}")
            return False

        ok = isinstance(resp, dict) and str(resp.get("status", "")).lower() == "ok"
        if LIVE_LOG_ORDERS:
            status = "OK" if ok else "NOT_OK"
            print(f"[LIVE_BACKEND] CANCEL {status} coin={coin} oid={oid_str} resp={resp}")
        return ok

    def place_stop_only(
        self,
        coin: str,
        side: str,
        size_coin: float,
        stop_price: float,
    ) -> Dict[str, Any]:
        coin = coin.upper()
        side = side.upper()
        close_is_buy = side == "SHORT"
        send_sz = self._round_size(coin, abs(float(size_coin)))

        result: Dict[str, Any] = {
            "coin": coin,
            "side": side,
            "size_coin": send_sz,
            "stop_order_id": "",
            "status": "missing",
            "error": "",
            "trigger_type": HL_PROTECTION_TRIGGER_REFERENCE,
            "stop_trigger_price": float(stop_price),
            "stop_verified": False,
        }

        if send_sz <= 0:
            result["error"] = "invalid_stop_size"
            return result
        if not self._ensure_meta():
            result["error"] = "asset_metadata_unavailable"
            return result

        stop_round_direction = "up" if side == "LONG" else "down"
        rounded_stop_px = self.round_to_tick(
            float(stop_price),
            coin,
            direction=stop_round_direction,
        )
        if rounded_stop_px <= 0:
            result["error"] = "invalid_stop_price_after_rounding"
            return result
        result["stop_trigger_price"] = rounded_stop_px

        stop_oid, stop_err = self._place_trigger_reduce_only(
            coin=coin,
            close_is_buy=close_is_buy,
            size_coin=send_sz,
            trigger_price=rounded_stop_px,
            tpsl="sl",
            trigger_is_market=HL_SL_TRIGGER_IS_MARKET,
            allow_limit_fallback=HL_SL_ALLOW_LIMIT_FALLBACK,
        )

        # Same rule as place_native_protection: an accepted order ID says the
        # venue took something, not that what it took is a stop. This path had
        # no verification at all, so it would have reported "protected" for an
        # order resting on the wrong side of the market.
        verification = self._verify_protection_orders(
            coin=coin,
            close_is_buy=close_is_buy,
            expected_size_coin=send_sz,
            expected_stop_px=rounded_stop_px,
            expected_tp_px=0.0,
            stop_oid=stop_oid,
            tp_oid=None,
        )
        stop_hard, stop_soft = verification["stop"]
        stop_err = self._append_verify_notes(str(stop_err or ""), stop_hard, stop_soft)
        if verification.get("global"):
            note = f"verify:{verification['global']}"
            stop_err = f"{stop_err}; {note}" if stop_err else note

        result["stop_order_id"] = stop_oid or ""
        result["stop_verified"] = bool(stop_oid) and not stop_hard
        result["status"] = "protected" if result["stop_verified"] else "missing"
        result["error"] = stop_err

        if LIVE_LOG_ORDERS:
            print(
                f"[LIVE_BACKEND] STOP_ONLY {coin} {side} status={result['status']} "
                f"sz={send_sz:.8f} stop_oid={result['stop_order_id'] or '-'} "
                f"stop_px={rounded_stop_px:.8f} err={result['error'] or '-'}"
            )

        return result

    # ── Price feed ─────────────────────────────────────────────────────────

    def get_cached_mid_price(self, coin: str) -> Optional[float]:
        return self._mid_cache.get(str(coin).upper())

    def get_mid_cache_age_sec(self) -> Optional[float]:
        return cache_age_seconds(self._mid_cache_ts)

    def midprice_is_fresh(
        self,
        max_age_sec: float = LIVE_MAX_MIDPRICE_AGE_SECONDS,
    ) -> bool:
        return bool(self._mid_cache) and cache_is_fresh(
            self._mid_cache_ts,
            max_age_seconds=max_age_sec,
        )

    def get_mid_price(
        self,
        coin: str,
        max_age_sec: float = LIVE_MAX_MIDPRICE_AGE_SECONDS,
    ) -> Optional[float]:
        now = time.time()
        coin = str(coin).upper()
        age_before = self.get_mid_cache_age_sec()

        if self._mid_cache and age_before is not None and age_before <= MID_CACHE_TTL_SEC:
            px = self._mid_cache.get(coin)
            if px is not None:
                if self.debug:
                    print(
                        f"[LIVE_BACKEND] midprice data_source=cache coin={coin}"
                        f" midprice_age={age_before:.2f}s"
                    )
                return px

        backoff_active, backoff_remaining, backoff_reason = api_backoff_status()
        if backoff_active:
            if self.midprice_is_fresh(max_age_sec=max_age_sec):
                px = self._mid_cache.get(coin)
                if px is not None:
                    if self.debug:
                        print(
                            f"[LIVE_BACKEND] midprice data_source=cache coin={coin}"
                            f" midprice_age={self.get_mid_cache_age_sec() or 0.0:.2f}s"
                            f" api_backoff_remaining={backoff_remaining:.1f}s"
                        )
                    return px
            print(
                f"[LIVE_BACKEND] live_midprice_stale_block coin={coin}"
                f" midprice_age={age_before if age_before is not None else -1:.2f}s"
                f" max_age={float(max_age_sec):.1f}s"
                f" api_backoff_remaining={backoff_remaining:.1f}s"
                f" reason={backoff_reason}"
            )
            return None

        if now - self._mid_cache_ts > MID_CACHE_TTL_SEC:
            try:
                mids = self.info.all_mids()
                if isinstance(mids, dict):
                    self._mid_cache = {k.upper(): float(v) for k, v in mids.items()}
                    self._mid_cache_ts = time.time()
                    record_api_success()
                    px = self._mid_cache.get(coin)
                    if px is not None:
                        if self.debug:
                            print(
                                f"[LIVE_BACKEND] midprice data_source=fresh coin={coin}"
                                f" midprice_age=0.00s"
                            )
                        return px
            except Exception as e:
                record_api_failure("live_midprice", e)
                if self.debug:
                    print(f"[LIVE_BACKEND] all_mids failed: {e}")

        age_after = self.get_mid_cache_age_sec()
        if self._mid_cache and age_after is not None and age_after <= float(max_age_sec):
            px = self._mid_cache.get(coin)
            if px is not None:
                if self.debug:
                    print(
                        f"[LIVE_BACKEND] midprice data_source=cache coin={coin}"
                        f" midprice_age={age_after:.2f}s"
                    )
                return px

        print(
            f"[LIVE_BACKEND] live_midprice_stale_block coin={coin}"
            f" midprice_age={age_after if age_after is not None else -1:.2f}s"
            f" max_age={float(max_age_sec):.1f}s"
        )
        return None

    # ── Fee / funding (accounting only) ────────────────────────────────────

    def estimate_entry_fee(self, notional_usd: float) -> float:
        return max(0.0, notional_usd) * (TAKER_FEE_BPS / 10_000.0)

    def estimate_exit_fee(self, notional_usd: float) -> float:
        return max(0.0, notional_usd) * (TAKER_FEE_BPS / 10_000.0)

    def accrue_funding(
        self,
        coin: str,
        side: str,
        notional_usd: float,
        last_funding_ts,
        now: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        return {"funding_usd": 0.0, "last_funding_ts": now or datetime.now(timezone.utc)}

    def shutdown(self):
        venue = "TESTNET" if HL_TESTNET else "MAINNET"
        print(f"[LIVE_BACKEND] shutdown ({venue})")

    # ── Account margin query ───────────────────────────────────────────────

    def _is_unified_account(self, addr: str) -> bool:
        """
        Hyperliquid unified accounts hold perp collateral in the spot layer (USDC),
        not in the perp clearinghouse. clearinghouseState always returns 0.0 for
        these accounts. Must query spotClearinghouseState instead.
        """
        cache = getattr(self, "_unified_account_cache", None)
        if cache is not None:
            return bool(cache)

        try:
            result = self.info.query_user_abstraction_state(addr)
            is_unified = str(result).strip().lower() == "unifiedaccount"
            self._unified_account_cache = is_unified
            print(
                f"[LIVE_BACKEND] account_model={'unified' if is_unified else 'standard'} "
                f"addr={addr}"
            )
            return is_unified
        except Exception:
            return False

    def get_margin_summary(self) -> Optional[Tuple[float, float, float]]:
        """
        Returns (account_equity, used_margin, available_margin) from venue.
        Returns None on query failure — callers must treat None as "no data".

        Handles two HL account models:
        - Standard account: margin in clearinghouseState (perps layer)
        - Unified account: margin in spotClearinghouseState as USDC
          (clearinghouseState always returns 0.0 for unified accounts)
        """
        venue = "TESTNET" if HL_TESTNET else "MAINNET"
        addr = self._get_user_state_address()

        # Detect unified account model once per call.
        # unified accounts keep USDC in spot; perp clearinghouse is always zero.
        if self._is_unified_account(addr):
            return self._get_margin_summary_unified(addr, venue)

        return self._get_margin_summary_standard(addr, venue)

    def _get_margin_summary_unified(self, addr: str, venue: str) -> Optional[Tuple[float, float, float]]:
        """
        Unified account: collateral is spot USDC, and the perp clearinghouse
        reports zeros for everything (verified 2026-09-18: accountValue,
        totalMarginUsed and withdrawable were all "0.0" on a funded account).

        equity    = USDC.total
        available = tokenToAvailableAfterMaintenance for the USDC token, which
                    is the venue's own answer to how much is actually free and
                    nets out maintenance margin on open perp positions.
        used      = equity - available

        available was `total - hold` until 2026-09-18, and `hold` is a SPOT
        order lock. Whether perp margin ever appears in it could not be
        established -- the account was flat, so "hold is 0 because nothing is
        held" and "hold never reflects perp margin" are indistinguishable. If
        it is the latter, `total - hold` reports the full balance as free
        while a position is open, and the next trade is sized against
        collateral that is already committed. tokenToAvailableAfterMaintenance
        is correct either way, so the ambiguity stops mattering.
        """
        try:
            spot_state = self.info.spot_user_state(addr)
        except Exception as e:
            record_api_failure("live_margin", e)
            print(f"[LIVE_BACKEND] unified account spot query failed venue={venue} addr={addr} err={e}")
            return None

        balances = spot_state.get("balances", []) if isinstance(spot_state, dict) else []
        usdc = next((b for b in balances if str(b.get("coin", "")).upper() == "USDC"), None)

        print(
            f"[LIVE_BACKEND] get_margin_summary venue={venue} addr={addr} "
            f"account_model=unified spot_usdc={usdc}"
        )

        if usdc is None:
            print(f"[LIVE_BACKEND] get_margin_summary unified: no USDC balance found venue={venue} addr={addr}")
            return None

        try:
            equity = float(usdc.get("total", 0.0))
            hold = float(usdc.get("hold", 0.0))
        except (TypeError, ValueError) as e:
            print(f"[LIVE_BACKEND] get_margin_summary unified: USDC parse error {e}")
            return None

        avail = self._available_after_maintenance(spot_state, usdc.get("token"))
        if avail is None:
            # Older payload, or the field moved. Fall back to the previous
            # reading and say so, because it may over-report what is free.
            available = max(0.0, equity - hold)
            avail_source = "spot.USDC.total-hold(FALLBACK)"
        else:
            available = max(0.0, min(avail, equity))
            avail_source = "spot.tokenToAvailableAfterMaintenance"
        used = max(0.0, equity - available)

        print(
            f"[LIVE_BACKEND] get_margin_summary PARSED "
            f"equity={equity:.4f} used={used:.4f} available={available:.4f} "
            f"account_model=unified equity_source=spot.USDC.total "
            f"avail_source={avail_source} hold={hold:.4f} venue={venue} addr={addr}"
        )
        record_api_success()
        return (equity, used, available)

    def _get_margin_summary_standard(self, addr: str, venue: str) -> Optional[Tuple[float, float, float]]:
        """
        Standard account: collateral is in the perp clearinghouse.
        Reads marginSummary.accountValue / totalMarginUsed / withdrawable.
        """
        try:
            state = self.info.user_state(addr)
        except Exception as e:
            record_api_failure("live_margin", e)
            print(f"[LIVE_BACKEND] get_margin_summary query failed venue={venue} addr={addr} err={e}")
            return None

        if not isinstance(state, dict):
            print(f"[LIVE_BACKEND] get_margin_summary unexpected response type={type(state)} venue={venue} addr={addr}")
            return None

        ms = state.get("marginSummary") or {}
        cms = state.get("crossMarginSummary") or {}

        print(
            f"[LIVE_BACKEND] get_margin_summary venue={venue} addr={addr} "
            f"account_model=standard marginSummary={ms} crossMarginSummary={cms} "
            f"withdrawable={state.get('withdrawable')}"
        )

        def _f(v) -> Optional[float]:
            try:
                return float(v) if v not in (None, "", "null") else None
            except (TypeError, ValueError):
                return None

        equity = (_f(ms.get("accountValue")) or _f(cms.get("accountValue")) or _f(state.get("accountValue")))
        used = (_f(ms.get("totalMarginUsed")) or _f(cms.get("totalMarginUsed")) or 0.0)
        withdrawable = (_f(state.get("withdrawable")) or _f(ms.get("withdrawable")) or _f(cms.get("withdrawable")))

        if equity is None:
            print(
                f"[LIVE_BACKEND] get_margin_summary PARSE FAILURE: accountValue not found "
                f"venue={venue} addr={addr} — check field names above"
            )
            return None

        available = withdrawable if withdrawable is not None else max(0.0, equity - used)

        print(
            f"[LIVE_BACKEND] get_margin_summary PARSED "
            f"equity={equity:.4f} used={used:.4f} available={available:.4f} "
            f"account_model=standard venue={venue} addr={addr}"
        )
        record_api_success()
        return (equity, used, available)

    @staticmethod
    def _available_after_maintenance(spot_state: Any, token_id: Any) -> Optional[float]:
        """Free collateral for one spot token, per the venue.

        tokenToAvailableAfterMaintenance arrives as [[token_id, "amount"], ...].
        Returns None when the field is absent or the token is not listed, which
        the caller must treat as "ask a different way", never as zero free.
        """
        if not isinstance(spot_state, dict) or token_id is None:
            return None
        rows = spot_state.get("tokenToAvailableAfterMaintenance")
        if not isinstance(rows, list):
            return None
        for row in rows:
            if not isinstance(row, (list, tuple)) or len(row) != 2:
                continue
            try:
                if int(row[0]) == int(token_id):
                    return float(row[1])
            except (TypeError, ValueError):
                continue
        return None

    # ── Realised PnL query ─────────────────────────────────────────────────

    def get_realised_pnl(self, coin: str, since_ms: int) -> Optional[float]:
        """Net realised PnL for `coin` from fills at or after since_ms.

        Returns None on query failure, and the distinction matters: a caller
        that reads a failure as 0.0 has a daily loss stop that cannot see
        losses. Fees are subtracted where the venue reports them -- closedPnl
        is gross, and at 3% risk per trade the fees on a churning day are not
        rounding error.
        """
        coin = str(coin or "").upper().strip()
        addr = self._get_user_state_address()
        try:
            fills = self.info.user_fills_by_time(addr, int(since_ms))
        except Exception as e:
            record_api_failure("live_fills", e)
            if self.debug:
                print(f"[LIVE_BACKEND] user_fills_by_time failed coin={coin} "
                      f"since={since_ms}: {e}")
            return None

        if not isinstance(fills, list):
            if self.debug:
                print(f"[LIVE_BACKEND] user_fills_by_time unexpected type={type(fills)}")
            return None

        total = 0.0
        matched = 0
        for f in fills:
            if not isinstance(f, dict):
                continue
            if str(f.get("coin", "")).upper() != coin:
                continue
            if int(f.get("time", 0) or 0) < int(since_ms):
                continue
            try:
                total += float(f.get("closedPnl", 0) or 0)
                total -= float(f.get("fee", 0) or 0)
            except (TypeError, ValueError):
                continue
            matched += 1

        record_api_success()
        if LIVE_LOG_ORDERS:
            print(f"[LIVE_BACKEND] realised_pnl {coin} since={since_ms} "
                  f"fills={matched} net={total:.4f}")
        return total

    # ── Venue position query ───────────────────────────────────────────────

    def _get_venue_signed_size(self, coin: str) -> Optional[float]:
        """
        Returns signed position size for the coin from wallet state:
          >0 long, <0 short, 0 flat.
        Returns None on query failure.
        """
        query_address = "unknown"
        try:
            query_address = self._get_user_state_address()
            print(f"[HL_QUERY] user_state address={query_address}")
            if self.account_address and query_address != self.account_address.lower():
                print(f"[HL_WARNING] Query addr != env account: {query_address} vs {self.account_address}")
            state = self.info.user_state(query_address)
            record_api_success()
            for item in state.get("assetPositions", []):
                pos = item.get("position", {})
                if pos.get("coin", "").upper() != coin.upper():
                    continue
                return float(pos.get("szi", 0.0))
            return 0.0
        except Exception as e:
            if self.debug:
                print(
                    f"[LIVE_BACKEND] user_state query failed for {coin} "
                    f"addr={query_address}: {e}"
                )
            return None

    def _get_venue_position_size(self, coin: str, side: str) -> Optional[float]:
        """
        Returns absolute open size for requested side only.
        Opposite-side exposure is treated as 0 for this side.
        Returns None on query failure.
        """
        szi = self._get_venue_signed_size(coin)
        if szi is None:
            return None
        return self._signed_size_for_side(side, szi)

    @staticmethod
    def _signed_size_for_side(side: str, szi: Optional[float]) -> Optional[float]:
        if szi is None:
            return None

        s = side.upper()
        if s == "LONG":
            return abs(szi) if szi > 0 else 0.0
        return abs(szi) if szi < 0 else 0.0

    def _get_user_state_address(self) -> str:
        if self.vault_address:
            return self.vault_address.lower()
        if self.account_address:
            return self.account_address.lower()
        if self.wallet and hasattr(self.wallet, "address"):
            wallet_addr = str(getattr(self.wallet, "address", "")).strip()
            if wallet_addr:
                return wallet_addr.lower()
        raise RuntimeError("No valid HL account address found")

    def _user_state_query_address(self) -> str:
        return self._get_user_state_address()

    def get_open_orders(self, coin: str = "") -> Optional[List[Dict[str, Any]]]:
        """
        Return normalized open orders for account (optionally filtered by coin).

        frontend_open_orders FIRST, and that ordering is load-bearing. The plain
        openOrders endpoint returns only coin/limitPx/oid/side/sz/timestamp -- no
        reduceOnly, no isTrigger, no triggerPx. Protection verification is this
        method's only caller, and against that payload it cannot confirm a single
        thing it needs to: on 2026-09-18 the one live stop this account has ever
        placed came back "not_trigger_order;missing_trigger_price" for exactly
        this reason, because the richer endpoint was only ever the fallback.

        Normalized fields. None means THE PAYLOAD DID NOT SAY, which is not the
        same as False -- callers must not collapse the two, because "the venue
        told us this is not a stop" and "this endpoint does not report stops"
        warrant opposite responses:
          - coin        (upper string)
          - oid         (string)
          - reduce_only (bool | None)
          - is_trigger  (bool | None)
          - tpsl        ("tp" | "sl" | "")
          - trigger_px  (float | None)
          - size        (float | None)
          - is_buy      (bool | None)
        """
        query_address = self._user_state_query_address()
        raw = None
        try:
            if hasattr(self.info, "frontend_open_orders"):
                raw = self.info.frontend_open_orders(query_address)
            elif hasattr(self.info, "open_orders"):
                raw = self.info.open_orders(query_address)
            else:
                if self.debug:
                    print("[LIVE_BACKEND] open-orders endpoint unavailable on Info client")
                return None
        except Exception as e:
            if self.debug:
                print(
                    f"[LIVE_BACKEND] open-orders query failed "
                    f"addr={query_address}: {e}"
                )
            return None

        rows = self._normalize_open_orders(raw)
        if coin:
            target = coin.upper()
            rows = [row for row in rows if str(row.get("coin", "")).upper() == target]
        return rows

    def _normalize_open_orders(self, payload: Any) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        stack: List[Any] = [payload]

        while stack:
            item = stack.pop()
            if isinstance(item, list):
                stack.extend(item)
                continue
            if not isinstance(item, dict):
                continue

            norm = self._normalize_open_order_item(item)
            if norm is not None:
                out.append(norm)

            for v in item.values():
                if isinstance(v, (dict, list)):
                    stack.append(v)

        dedup: Dict[str, Dict[str, Any]] = {}
        for row in out:
            key = f"{row.get('coin','')}|{row.get('oid','')}|{row.get('tpsl','')}"
            dedup[key] = row
        return list(dedup.values())

    @staticmethod
    def _normalize_open_order_item(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        nested = raw.get("order")
        if not isinstance(nested, dict):
            nested = {}

        coin = str(
            raw.get("coin")
            or nested.get("coin")
            or ""
        ).upper()

        oid = (
            raw.get("oid")
            if raw.get("oid") is not None else
            nested.get("oid")
        )
        oid_str = str(oid).strip() if oid is not None else ""

        order_type_raw = raw.get("orderType")
        if order_type_raw is None:
            order_type_raw = nested.get("orderType")
        order_type = order_type_raw if isinstance(order_type_raw, dict) else {}
        # frontendOpenOrders reports orderType as a STRING ("Stop Market",
        # "Take Profit Limit"), never as the {"trigger": {...}} dict we send.
        order_type_text = (
            order_type_raw.strip().lower() if isinstance(order_type_raw, str) else ""
        )

        trigger = order_type.get("trigger")
        if not isinstance(trigger, dict):
            trigger = {}

        reduce_only = raw.get("reduceOnly")
        if reduce_only is None:
            reduce_only = nested.get("reduceOnly")

        tpsl = str(
            raw.get("tpsl")
            or nested.get("tpsl")
            or trigger.get("tpsl")
            or ""
        ).lower()
        if tpsl not in {"tp", "sl"} and order_type_text:
            # Invert the mapping we send: tpsl="sl" comes back as "Stop Market"
            # or "Stop Limit", tpsl="tp" as "Take Profit Market"/"Take Profit
            # Limit". frontendOpenOrders carries no tpsl field at all, so
            # without this the tpsl check is unfalsifiable on the only endpoint
            # that reports triggers -- and a stop mistakenly placed as a
            # take-profit, which rests on the WRONG SIDE of price, verifies
            # clean. Check "take profit" first: "Stop" never appears in a TP
            # order type, but matching loosely on "stop" would be fragile.
            if "take profit" in order_type_text:
                tpsl = "tp"
            elif "stop" in order_type_text:
                tpsl = "sl"
        trigger_px_raw = (
            raw.get("triggerPx")
            if raw.get("triggerPx") is not None else
            nested.get("triggerPx")
        )
        if trigger_px_raw is None:
            trigger_px_raw = trigger.get("triggerPx")
        trigger_px = None
        if trigger_px_raw not in (None, ""):
            try:
                trigger_px = float(trigger_px_raw)
            except (TypeError, ValueError):
                trigger_px = None

        is_trigger_raw = raw.get("isTrigger")
        if is_trigger_raw is None:
            is_trigger_raw = nested.get("isTrigger")
        if bool(trigger) or tpsl in {"tp", "sl"}:
            # Positive structural evidence: a trigger block, or a tp/sl tag we
            # were given or recovered above.
            is_trigger = True
        elif is_trigger_raw is not None:
            is_trigger = bool(is_trigger_raw)
        else:
            # The payload says nothing either way. Reporting False here would
            # let a field-poor endpoint masquerade as the venue denying that
            # our stop is a stop.
            is_trigger = None

        if not coin and not oid_str and not is_trigger:
            return None

        size_raw = (
            raw.get("sz")
            if raw.get("sz") is not None else
            nested.get("sz")
        )
        if size_raw is None:
            size_raw = raw.get("origSz")
        if size_raw is None:
            size_raw = nested.get("origSz")
        size_val: Optional[float] = None
        if size_raw not in (None, ""):
            try:
                size_val = float(size_raw)
            except (TypeError, ValueError):
                size_val = None

        is_buy_raw = raw.get("isBuy")
        if is_buy_raw is None:
            is_buy_raw = nested.get("isBuy")
        is_buy: Optional[bool]
        if is_buy_raw is None:
            side_txt = str(raw.get("side") or nested.get("side") or "").strip().lower()
            # Hyperliquid encodes side as "A" (ask/sell) or "B" (bid/buy) on
            # BOTH open-orders endpoints. "a" was absent from this set, so every
            # reduce-only stop on a LONG -- which closes SELL, i.e. "A" --
            # normalized to is_buy=None and reported missing_side forever. The
            # wrong-side check, the one that catches a stop resting on the wrong
            # side of the market, could never fire.
            if side_txt in {"buy", "bid", "b"}:
                is_buy = True
            elif side_txt in {"sell", "ask", "s", "a"}:
                is_buy = False
            else:
                is_buy = None
        else:
            is_buy = bool(is_buy_raw)

        return {
            "coin": coin,
            "oid": oid_str,
            "reduce_only": None if reduce_only is None else bool(reduce_only),
            "is_trigger": is_trigger,
            "tpsl": tpsl,
            "trigger_px": trigger_px,
            "size": size_val,
            "is_buy": is_buy,
        }

    def _size_tick(self, coin: str) -> float:
        decimals = self._sz_decimals.get(coin.upper(), 3)
        return 10 ** (-decimals)

    def _build_trigger_order_attempts(
        self,
        trig_px: float,
        tpsl: str,
        trigger_is_market: bool,
        allow_limit_fallback: bool,
    ) -> List[Dict[str, Any]]:
        attempts: List[Dict[str, Any]] = []
        market_modes = [bool(trigger_is_market)]
        if allow_limit_fallback:
            alt = not bool(trigger_is_market)
            if alt not in market_modes:
                market_modes.append(alt)

        trigger_type = HL_PROTECTION_TRIGGER_REFERENCE
        for is_market in market_modes:
            base_trigger = {
                "triggerPx": trig_px,
                "isMarket": bool(is_market),
                "tpsl": tpsl,
            }
            if trigger_type in {"mark", "last", "index"}:
                with_ref = dict(base_trigger)
                with_ref["triggerType"] = trigger_type
                attempts.append(
                    {
                        "order_type": {"trigger": with_ref},
                        "is_market": bool(is_market),
                        "trigger_type": trigger_type,
                    }
                )

            attempts.append(
                {
                    "order_type": {"trigger": dict(base_trigger)},
                    "is_market": bool(is_market),
                    "trigger_type": "default",
                }
            )

        return attempts

    @staticmethod
    def _validate_protection_price_relations(
        side: str,
        stop_price: float,
        tp_price: float,
        anchor_price: Optional[float],
    ) -> str:
        if stop_price <= 0 or tp_price <= 0:
            return "invalid_protection_prices_non_positive"
        if anchor_price is None or anchor_price <= 0:
            return ""

        side = side.upper()
        if side == "LONG":
            if stop_price >= anchor_price:
                return (
                    f"invalid_stop_side_relation:long_stop_not_below_anchor "
                    f"({stop_price} >= {anchor_price})"
                )
            if tp_price <= anchor_price:
                return (
                    f"invalid_tp_side_relation:long_tp_not_above_anchor "
                    f"({tp_price} <= {anchor_price})"
                )
            return ""

        if stop_price <= anchor_price:
            return (
                f"invalid_stop_side_relation:short_stop_not_above_anchor "
                f"({stop_price} <= {anchor_price})"
            )
        if tp_price >= anchor_price:
            return (
                f"invalid_tp_side_relation:short_tp_not_below_anchor "
                f"({tp_price} >= {anchor_price})"
            )
        return ""

    @staticmethod
    def _append_verify_notes(err: str, hard: List[str], soft: List[str]) -> str:
        """Fold verification results into the human-readable error string.

        verify_fail: the venue contradicted us and status was downgraded.
        verify_warn: something went unreported; status was left alone.
        """
        notes = []
        if hard:
            notes.append("verify_fail:" + ";".join(hard))
        if soft:
            notes.append("verify_warn:" + ";".join(soft))
        if not notes:
            return err
        note = " ".join(notes)
        return f"{err}; {note}" if err else note

    def _verify_single_protection_order(
        self,
        order_row: Optional[Dict[str, Any]],
        *,
        expected_tpsl: str,
        expected_size_coin: float,
        expected_trigger_px: float,
        expected_is_buy: bool,
        coin: str,
    ) -> Tuple[List[str], List[str]]:
        """Compare one resting order against what we asked the venue for.

        Returns (contradictions, unknowns), and the split is the whole point.

        A CONTRADICTION is the venue reporting a value that disagrees with what
        we sent: a stop tagged as a take-profit, a trigger 5% from where we put
        it, a close order on the wrong side. These cannot be explained away by
        timing or by a thin payload -- the order that is resting is not the
        order we asked for -- and the caller treats them as unprotected.

        An UNKNOWN is a field the payload simply did not report, including the
        order not being visible yet. Escalating those would re-create the
        2026-09-18 failure where a correctly protected UNI position was
        emergency-closed seconds after entry by a check that fired on success.
        We surface unknowns and keep the position.
        """
        if order_row is None:
            # Usually the open-orders snapshot lagging the ack by a beat, which
            # is why this is an unknown and not a contradiction.
            return [], ["order_not_found_in_open_orders"]

        hard: List[str] = []
        soft: List[str] = []

        reduce_only = order_row.get("reduce_only")
        if reduce_only is None:
            soft.append("reduce_only_unreported")
        elif not reduce_only:
            # A protection order that is NOT reduce-only can open a position
            # rather than close one. That is strictly worse than no stop.
            hard.append("not_reduce_only")

        is_trigger = order_row.get("is_trigger")
        if is_trigger is None:
            soft.append("trigger_flag_unreported")
        elif not is_trigger:
            hard.append("not_trigger_order")

        got_tpsl = str(order_row.get("tpsl", "") or "").strip().lower()
        if not got_tpsl:
            soft.append("tpsl_unreported")
        elif got_tpsl != expected_tpsl:
            hard.append(f"wrong_tpsl:{got_tpsl}!=expected:{expected_tpsl}")

        got_trigger_px = order_row.get("trigger_px")
        if got_trigger_px is None:
            soft.append("trigger_price_unreported")
        else:
            price_tol = max(1e-9, abs(expected_trigger_px) * 0.0002)
            if abs(float(got_trigger_px) - expected_trigger_px) > price_tol:
                hard.append(
                    f"trigger_price_mismatch:{float(got_trigger_px)}!=expected:{expected_trigger_px}"
                )

        got_size = order_row.get("size")
        if got_size is None:
            soft.append("size_unreported")
        else:
            size_tol = max(self._size_tick(coin) * 1.5, abs(expected_size_coin) * 0.001)
            if abs(float(got_size) - expected_size_coin) > size_tol:
                hard.append(
                    f"size_mismatch:{float(got_size)}!=expected:{expected_size_coin}"
                )

        got_is_buy = order_row.get("is_buy")
        if got_is_buy is None:
            soft.append("side_unreported")
        elif bool(got_is_buy) != bool(expected_is_buy):
            expected_side = "BUY" if expected_is_buy else "SELL"
            got_side = "BUY" if bool(got_is_buy) else "SELL"
            hard.append(f"wrong_side:{got_side}!={expected_side}")

        return hard, soft

    def _verify_protection_orders(
        self,
        *,
        coin: str,
        close_is_buy: bool,
        expected_size_coin: float,
        expected_stop_px: float,
        expected_tp_px: float,
        stop_oid: Optional[str],
        tp_oid: Optional[str],
    ) -> Dict[str, Any]:
        """Per leg: (contradictions, unknowns). See _verify_single_protection_order."""
        out: Dict[str, Any] = {"stop": ([], []), "tp": ([], []), "global": ""}
        orders = self.get_open_orders(coin)
        if orders is None:
            # Could not ask. That is an unknown, never a contradiction: failing
            # to reach the venue says nothing about whether the stop is resting.
            out["global"] = "open_orders_verification_unavailable"
            return out

        by_oid: Dict[str, Dict[str, Any]] = {}
        for row in orders:
            oid = str(row.get("oid", "") or "").strip()
            if oid:
                by_oid[oid] = row

        if stop_oid:
            out["stop"] = self._verify_single_protection_order(
                by_oid.get(str(stop_oid).strip()),
                expected_tpsl="sl",
                expected_size_coin=expected_size_coin,
                expected_trigger_px=expected_stop_px,
                expected_is_buy=close_is_buy,
                coin=coin,
            )
        if tp_oid:
            out["tp"] = self._verify_single_protection_order(
                by_oid.get(str(tp_oid).strip()),
                expected_tpsl="tp",
                expected_size_coin=expected_size_coin,
                expected_trigger_px=expected_tp_px,
                expected_is_buy=close_is_buy,
                coin=coin,
            )
        return out

    def _place_trigger_reduce_only(
        self,
        coin: str,
        close_is_buy: bool,
        size_coin: float,
        trigger_price: float,
        tpsl: str,
        trigger_is_market: bool,
        allow_limit_fallback: bool,
    ) -> tuple:
        trig_px = self._round_price(coin, float(trigger_price))
        if trig_px <= 0:
            return None, "invalid_trigger_price"
        if size_coin <= 0:
            return None, "invalid_trigger_size"

        attempts = self._build_trigger_order_attempts(
            trig_px=trig_px,
            tpsl=tpsl,
            trigger_is_market=trigger_is_market,
            allow_limit_fallback=allow_limit_fallback,
        )

        errors: List[str] = []
        for idx, attempt in enumerate(attempts):
            order_type = attempt.get("order_type", {})
            trigger_type = str(attempt.get("trigger_type", "default"))
            is_market = bool(attempt.get("is_market", True))

            if LIVE_LOG_ORDERS:
                side_txt = "BUY" if close_is_buy else "SELL"
                print(
                    f"[LIVE_BACKEND] PROTECTION_SUBMIT coin={coin} tpsl={tpsl} "
                    f"side={side_txt} sz={size_coin:.8f} trigger_px={trig_px:.8f} "
                    f"trigger_type={trigger_type} is_market={is_market} reduce_only=True "
                    f"payload={order_type}"
                )

            try:
                resp = self.exchange.order(
                    coin,
                    close_is_buy,
                    size_coin,
                    trig_px,
                    order_type,
                    reduce_only=True,
                )
            except Exception as e:
                errors.append(f"attempt{idx + 1}_sdk_error:{e}")
                continue

            if LIVE_LOG_ORDERS:
                print(
                    f"[LIVE_BACKEND] PROTECTION_RESPONSE coin={coin} tpsl={tpsl} "
                    f"trigger_px={trig_px:.8f} trigger_type={trigger_type} "
                    f"is_market={is_market} resp={resp}"
                )

            oid, parse_err = self._parse_trigger_order_response(resp)
            if oid:
                return oid, None
            errors.append(f"attempt{idx + 1}:{parse_err}")

        return None, "; ".join(errors) if errors else "trigger_order_failed"

    @staticmethod
    def _parse_trigger_order_response(resp: Dict[str, Any]) -> tuple:
        if not isinstance(resp, dict):
            return None, "invalid_response_type"

        if resp.get("status") != "ok":
            return None, f"response_not_ok:{resp.get('response', resp)}"

        statuses: List[Dict] = (
            resp.get("response", {}).get("data", {}).get("statuses", [])
        )
        if not statuses:
            return None, "empty_statuses"

        errors: List[str] = []
        for status in statuses:
            if not isinstance(status, dict):
                continue
            if "resting" in status:
                oid = status["resting"].get("oid")
                if oid is not None:
                    return str(oid), None
            if "filled" in status:
                oid = status["filled"].get("oid")
                if oid is not None:
                    return str(oid), None
                return None, "filled_without_oid"
            if "error" in status:
                errors.append(str(status["error"]))

        if errors:
            return None, "; ".join(errors)
        return None, f"no_order_id:{statuses}"

    # ── Precision helpers ──────────────────────────────────────────────────

    def max_leverage_for(self, coin: str) -> Optional[int]:
        """The venue's per-coin leverage ceiling, or None if unknown."""
        if not self._ensure_meta():
            return None
        return self._max_leverage.get(coin.upper()) or None

    def set_leverage(self, coin: str, leverage: int, is_cross: bool = True) -> bool:
        """Set the account's leverage for one coin before sizing into it.

        Hyperliquid enforces leverage per asset, so opening 5x notional while
        the account is set to 3x on that coin is simply refused for margin.
        Flat sizing is meaningless unless this is set to match.
        """
        coin = str(coin or "").upper().strip()
        cap = self.max_leverage_for(coin)
        want = int(max(1, leverage))
        if cap:
            want = min(want, int(cap))
        try:
            resp = self.exchange.update_leverage(want, coin, is_cross)
        except Exception as e:
            print(f"[LIVE_BACKEND] set_leverage FAILED {coin} -> {want}x: {str(e)[:140]}")
            return False
        ok = isinstance(resp, dict) and str(resp.get("status", "")).lower() == "ok"
        if LIVE_LOG_ORDERS or not ok:
            print(f"[LIVE_BACKEND] set_leverage {coin} -> {want}x "
                  f"({'cross' if is_cross else 'isolated'}) "
                  f"{'ok' if ok else 'NOT_OK'} resp={str(resp)[:120]}")
        return ok

    def _ensure_meta(self) -> bool:
        """True when asset metadata is available, retrying the load if not.

        _load_meta ran exactly once, in __init__, and swallowed its own
        failure. A backend that started inside a rate-limit window therefore
        stayed empty for the life of the process: on 2026-09-19 a 429 during
        startup produced "assets_loaded=0", and nothing said so again.

        Empty is not harmless. Every coin then falls back to szDecimals=3 and
        a 1e-3 price tick, so sizes and trigger prices are rounded to a
        precision the venue did not ask for -- silently wrong orders rather
        than refused ones, on a path whose whole job is placing stops
        correctly.
        """
        if self._sz_decimals:
            return True
        self._load_meta()
        if self._sz_decimals:
            print(f"[LIVE_BACKEND] asset metadata recovered "
                  f"({len(self._sz_decimals)} assets)")
            return True
        return False

    def _load_meta(self):
        """Load szDecimals per asset from the perp universe."""
        try:
            meta = self.info.meta()
            for asset in meta.get("universe", []):
                name = str(asset.get("name", "")).upper()
                if name:
                    self._sz_decimals[name] = int(asset.get("szDecimals", 3))
                    # 130 of the venue's 234 perps cap at 3x. Asking for more
                    # is rejected, so the ask has to know the ceiling.
                    self._max_leverage[name] = int(asset.get("maxLeverage", 0) or 0)
        except Exception as e:
            print(f"[LIVE_BACKEND] _load_meta failed: {str(e)[:200]}")

    def _notional_to_size(self, coin: str, size_usd: float, ref_price: float) -> float:
        """Convert notional USD to coin units, rounded down to venue precision."""
        if ref_price <= 0:
            return 0.0
        raw = size_usd / ref_price
        return self._round_size(coin, raw)

    def _round_size(self, coin: str, size_coin: float) -> float:
        if size_coin <= 0:
            return 0.0
        decimals = self._sz_decimals.get(coin.upper(), 3)
        quant = Decimal("1").scaleb(-decimals)
        return float(Decimal(str(size_coin)).quantize(quant, rounding=ROUND_DOWN))

    def _round_price(self, coin: str, price: float) -> float:
        return self.round_to_tick(price, coin, direction="nearest")

    def _price_tick(self, coin: str) -> float:
        sz_decimals = int(self._sz_decimals.get(coin.upper(), 3))
        px_decimals = max(0, 6 - sz_decimals)
        return 10 ** (-px_decimals)

    def round_to_tick(self, price: float, coin: str, direction: str = "nearest") -> float:
        if price <= 0:
            return 0.0

        magnitude = floor(log10(abs(price)))
        factor = 10 ** (4 - magnitude)
        sig_price = round(price * factor) / factor

        tick = self._price_tick(coin)
        quant = Decimal(str(tick))
        px = Decimal(str(sig_price))

        mode = str(direction or "nearest").strip().lower()
        if mode == "up":
            rounded = px.quantize(quant, rounding=ROUND_CEILING)
        elif mode == "down":
            rounded = px.quantize(quant, rounding=ROUND_FLOOR)
        else:
            rounded = px.quantize(quant, rounding=ROUND_HALF_UP)

        rounded_f = float(rounded)
        if rounded_f <= 0:
            return float(quant)
        return rounded_f

    def _exit_slippage_bps_for_reason(self, reason: str) -> float:
        reason = (reason or "").lower().strip()
        if reason == "partial":
            return LIVE_EXIT_SLIPPAGE_BPS_PARTIAL
        if reason == "tp_full":
            return LIVE_EXIT_SLIPPAGE_BPS_TP
        if reason == "stop_runner":
            return LIVE_EXIT_SLIPPAGE_BPS_STOP_RUNNER
        if reason == "stop_full":
            return LIVE_EXIT_SLIPPAGE_BPS_STOP_FULL
        if reason == "manual" or reason.startswith("system_exit_"):
            return LIVE_EXIT_SLIPPAGE_BPS_MANUAL
        return LIVE_EXIT_SLIPPAGE_BPS

    # ── Response parsing ───────────────────────────────────────────────────

    def _parse_fill(
        self,
        resp: Dict[str, Any],
        coin: str,
        requested_notional: float,
        ref_price: float,
        is_exit: bool,
        slip_ref: Optional[float] = None,
        is_buy: Optional[bool] = None,
    ) -> FillResult:
        """
        Parse Hyperliquid order response.
        Sums ALL filled status entries (handles multi-fill IOC responses).
        Cancels any resting orders immediately (IOC should never rest).

        `slip_ref` is the price slippage is measured against, when that is not
        the same as `ref_price`. Slippage means "how much worse than the market
        did we fill", so the reference has to be the price the market was
        actually showing. On entry those coincide, because ref_price IS the
        mid. On exit they do not: ref_price is deliberately min(mid, trigger)
        for a sell, which is how the limit is guaranteed to sit at or below the
        book and the fill is assured -- but measuring against it reports the
        gap between the trigger and the market, which for a profitable exit is
        the P&L. PONS closed at the mid exactly on 2026-09-26 and was logged as
        376bps of slippage.
        """
        if not isinstance(resp, dict):
            return self._reject(
                "invalid_response_type",
                ref_price,
                requested_notional,
                "live_order",
            )

        if resp.get("status") != "ok":
            return self._reject(
                f"response_not_ok:{resp.get('response', resp)}",
                ref_price,
                requested_notional,
                "live_order",
            )

        statuses: List[Dict] = (
            resp.get("response", {}).get("data", {}).get("statuses", [])
        )

        if not statuses:
            return self._reject(
                "empty_statuses",
                ref_price,
                requested_notional,
                "live_order",
            )

        total_sz = 0.0
        weighted_px_sum = 0.0
        oids: List[Any] = []
        resting_oids: List[Any] = []
        errors: List[str] = []

        for status in statuses:
            if not isinstance(status, dict):
                continue

            if "filled" in status:
                f = status["filled"]
                sz = float(f.get("totalSz", 0.0))
                px = float(f.get("avgPx", ref_price))
                if sz > 0:
                    total_sz += sz
                    weighted_px_sum += sz * px
                if "oid" in f:
                    oids.append(f["oid"])

            elif "resting" in status:
                oid = status["resting"].get("oid")
                if oid:
                    resting_oids.append(oid)
                    # Through cancel_order, not exchange.cancel directly: that
                    # bypassed the int coercion that the other call site needed,
                    # and the validation and logging with it. It happens to work
                    # only because the response JSON yields an int.
                    if not self.cancel_order(coin, oid):
                        print(f"[LIVE_BACKEND] WARNING: IOC rested and the cancel "
                              f"FAILED oid={oid} -- an unintended resting order "
                              f"is live at the venue")
                    else:
                        print(f"[LIVE_BACKEND] WARNING: IOC rested -- cancelled oid={oid}")

            elif "error" in status:
                errors.append(str(status["error"]))

        if total_sz <= 0:
            reason = (
                "; ".join(errors)
                if errors else
                f"ioc_no_fill_resting={resting_oids}" if resting_oids else
                "ioc_no_fill"
            )
            if LIVE_LOG_ORDERS:
                print(f"[LIVE_BACKEND] NO FILL statuses={statuses}")
            return self._reject(reason, ref_price, requested_notional, "live_order")

        avg_px = weighted_px_sum / total_sz
        fill_notional = total_sz * avg_px
        # Signed against the direction that hurts: a fill better than the
        # market is not slippage, and reporting it as though it were is what
        # made a perfect PONS fill look like a disaster.
        # Which direction hurts depends on whether we bought or sold, not on
        # whether this was an entry: a short entry sells and a long exit sells.
        # `is_buy` is passed by both call sites; None keeps the old unsigned
        # magnitude for any caller that does not know its side.
        slip_base = slip_ref if slip_ref and slip_ref > 0 else ref_price
        if slip_base <= 0:
            slippage_bps = 0.0
        elif is_buy is None:
            slippage_bps = abs(avg_px - slip_base) / slip_base * 10_000
        else:
            adverse = (avg_px - slip_base) if is_buy else (slip_base - avg_px)
            slippage_bps = max(0.0, adverse) / slip_base * 10_000
        fill_ratio = (
            min(fill_notional / requested_notional, 1.0)
            if requested_notional > 0 else 1.0
        )
        est_fee = (
            self.estimate_exit_fee(fill_notional)
            if is_exit else
            self.estimate_entry_fee(fill_notional)
        )

        return FillResult(
            filled=True,
            fill_price=avg_px,
            fill_size_usd=fill_notional,
            slippage_bps=slippage_bps,
            requested_price=ref_price,
            requested_size_usd=requested_notional,
            fill_ratio=fill_ratio,
            reason="live_order",
            meta={
                "estimated_fee_usd": round(est_fee, 8),
                "fee_bps": TAKER_FEE_BPS,
                "oids": oids,
                "total_sz": total_sz,
                "resting_cancelled": resting_oids,
            },
        )

    # ── Internal reject helper ─────────────────────────────────────────────

    def _reject(
        self,
        reason: str,
        ref_price: float,
        requested_notional: float,
        order_reason: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> FillResult:
        payload = dict(meta or {})
        if self.debug:
            if payload:
                print(f"[LIVE_BACKEND] REJECTED reason={reason} meta={payload}")
            else:
                print(f"[LIVE_BACKEND] REJECTED reason={reason}")
        return FillResult(
            filled=False,
            fill_price=0.0,
            fill_size_usd=0.0,
            slippage_bps=0.0,
            reject_reason=reason,
            requested_price=ref_price,
            requested_size_usd=requested_notional,
            fill_ratio=None,
            reason=order_reason,
            meta=payload,
        )

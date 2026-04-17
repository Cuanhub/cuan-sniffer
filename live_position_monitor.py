from __future__ import annotations

import os
import threading
import time
from datetime import datetime, timezone
from typing import Callable, Dict, Optional

from position import Position, PositionState, CloseReason
from trade_log import upsert_trade_row

TRADES_FILE = os.getenv("TRADES_FILE", "trades.csv")
PAPER_MODE = os.getenv("PAPER_MODE", "true").lower() == "true"
TAKER_FEE_BPS = float(os.getenv("TAKER_FEE_BPS", "4.5"))
MAX_FULL_LOSS_R = float(os.getenv("MAX_FULL_LOSS_R", "-1.5"))
LIVE_MONITOR_MIN_AGE_SEC = float(os.getenv("LIVE_MONITOR_MIN_AGE_SEC", "20"))
LIVE_MONITOR_ZERO_CONFIRMATIONS = int(os.getenv("LIVE_MONITOR_ZERO_CONFIRMATIONS", "2"))
LIVE_FLAT_EPSILON_SZ = float(os.getenv("LIVE_FLAT_EPSILON_SZ", "1e-9"))
LIVE_MONITOR_ORPHAN_TP_GRACE_SEC = float(os.getenv("LIVE_MONITOR_ORPHAN_TP_GRACE_SEC", "45"))


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class LivePositionMonitor:
    """
    Wallet-authoritative reconciliation monitor for live positions.

    PATCHED SAFE VERSION:
    - Reconciliation is STRICTLY opt-in
    - Fresh runtime positions are NOT reconciled by default
    - Monitor only finalizes positions that are explicitly marked
      allow_reconcile_close=True

    Intended use:
    - Executor owns normal live exit lifecycle
    - Monitor confirms venue-flat closes only after executor has initiated
      an exit, or for bootstrap-restored positions

    This prevents phantom closes where the monitor marks a live trade closed
    in CSV while it is still open on HL.
    """

    def __init__(
        self,
        backend,
        risk_manager,
        notify_fn: Optional[Callable[[str], None]] = None,
        order_tracker=None,
        signal_engine=None,
        trades_file: str = TRADES_FILE,
    ):
        self.backend = backend
        self.risk = risk_manager
        self.notify = notify_fn or (lambda msg: None)
        self.order_tracker = order_tracker
        self.signal_engine = signal_engine
        self.trades_file = trades_file
        self.positions: Dict[str, Position] = {}
        self._zero_streak: Dict[str, int] = {}
        self._flat_first_seen_ts: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(self, position: Position, allow_reconcile: bool = False) -> None:
        """
        Track a position for venue reconciliation.

        SAFE DEFAULT: allow_reconcile=False

        Use allow_reconcile=True only when:
        - the position was restored from bootstrap, OR
        - executor has already requested an exit and now wants wallet
          confirmation before final local finalization.
        """
        if not hasattr(position, "allow_reconcile_close"):
            setattr(position, "allow_reconcile_close", allow_reconcile)
        else:
            position.allow_reconcile_close = allow_reconcile

        if not hasattr(position, "reconciled_from_venue"):
            setattr(position, "reconciled_from_venue", False)

        self.positions[position.position_id] = position
        self._zero_streak[position.position_id] = 0
        self._flat_first_seen_ts.pop(position.position_id, None)

        tag = "reconcilable" if allow_reconcile else "executor-owned"
        print(
            f"[LIVE_MONITOR] Tracking {position.coin} {position.side} "
            f"— {position.position_id} [{tag}]"
        )

    def unregister(self, position_id: str) -> None:
        self.positions.pop(position_id, None)
        self._zero_streak.pop(position_id, None)
        self._flat_first_seen_ts.pop(position_id, None)

    @staticmethod
    def _position_terminal_lock(pos: Position):
        lock = getattr(pos, "_terminal_lock", None)
        if lock is None:
            lock = threading.Lock()
            setattr(pos, "_terminal_lock", lock)
        return lock

    def update(self) -> None:
        """
        Poll tracked positions and reconcile those that disappeared
        from the venue.

        Safety checks:
        - position must be explicitly reconcilable
        - minimum age
        - N zero-size confirmations
        - opposite-side check
        - runtime positions require executor exit intent before reconcile
        """
        to_finalize = []

        for position_id, pos in list(self.positions.items()):
            if bool(getattr(pos, "_terminalized", False)):
                self.unregister(position_id)
                continue
            if pos.state == PositionState.CLOSED:
                continue
            if bool(getattr(pos, "_terminalizing", False)):
                continue

            if not getattr(pos, "allow_reconcile_close", False):
                continue

            opened_ts = pos.opened_at.timestamp() if pos.opened_at else 0.0
            age_sec = time.time() - opened_ts if opened_ts > 0 else 0.0
            if age_sec < LIVE_MONITOR_MIN_AGE_SEC:
                continue

            venue_size = self._get_venue_size(pos.coin, pos.side)
            if venue_size is None:
                continue

            if venue_size > LIVE_FLAT_EPSILON_SZ:
                self._zero_streak[position_id] = 0
                self._flat_first_seen_ts.pop(position_id, None)
                continue

            if position_id not in self._flat_first_seen_ts:
                self._flat_first_seen_ts[position_id] = time.time()

            opposite = "SHORT" if str(pos.side).upper() == "LONG" else "LONG"
            opposite_size = self._get_venue_size(pos.coin, opposite)
            if opposite_size is not None and opposite_size > LIVE_FLAT_EPSILON_SZ:
                self._zero_streak[position_id] = 0
                self._flat_first_seen_ts.pop(position_id, None)
                print(
                    f"[LIVE_MONITOR] {pos.coin} opposite side still open "
                    f"({opposite} sz={opposite_size:.8f}) — keeping tracked"
                )
                continue

            has_blocking, blocking_meta = self._has_blocking_open_orders(
                pos.coin,
                return_details=True,
            )
            finalize_source = ""
            if has_blocking:
                flat_elapsed = time.time() - float(self._flat_first_seen_ts.get(position_id, time.time()))
                only_tp = bool(blocking_meta.get("only_tp", False))
                only_reduce_like = bool(blocking_meta.get("only_reduce_like", False))
                has_unknown_tpsl = bool(blocking_meta.get("has_unknown_tpsl", False))
                if flat_elapsed >= LIVE_MONITOR_ORPHAN_TP_GRACE_SEC and only_reduce_like:
                    if only_tp:
                        print(
                            f"[LIVE_MONITOR] {pos.coin} orphan TP detected (flat for {flat_elapsed:.0f}s) "
                            "— allowing reconciliation"
                        )
                        finalize_source = "monitor_orphan_tp_reconcile"
                    else:
                        print(
                            f"[LIVE_MONITOR] {pos.coin} orphan reduce-only leftovers detected "
                            f"(flat for {flat_elapsed:.0f}s, unknown_tpsl={has_unknown_tpsl}) "
                            "— allowing reconciliation"
                        )
                        finalize_source = "monitor_orphan_reduce_reconcile"
                else:
                    count = int(blocking_meta.get("count", 0))
                    tpsl = str(blocking_meta.get("tpsl_summary", "")).strip() or "-"
                    print(
                        f"[LIVE_MONITOR] {pos.coin} blocking reduce-only/native orders still open "
                        f"(count={count}, tpsl={tpsl}) — deferring reconciliation"
                    )
                    continue

            streak = self._zero_streak.get(position_id, 0) + 1
            self._zero_streak[position_id] = streak
            if streak < max(1, LIVE_MONITOR_ZERO_CONFIRMATIONS):
                continue

            exit_requested_reason = str(
                getattr(pos, "exit_requested_reason", "") or ""
            ).strip()
            is_bootstrap_restored = bool(getattr(pos, "bootstrap_restored", False))
            has_native_protection = bool(
                bool(getattr(pos, "venue_protection_mode", False))
                and (
                    str(getattr(pos, "stop_order_id", "")).strip()
                    or str(getattr(pos, "tp_order_id", "")).strip()
                )
            )
            if not exit_requested_reason and not is_bootstrap_restored and not has_native_protection:
                print(
                    f"[LIVE_MONITOR] SKIP RECONCILE {pos.coin} {pos.side} "
                    "— no executor exit requested or native protection metadata"
                )
                continue

            to_finalize.append((pos, finalize_source))

        for pos, source in to_finalize:
            self._finalize_closed_position(pos, source=source)

    # ------------------------------------------------------------------
    # Venue helpers
    # ------------------------------------------------------------------

    def _get_venue_size(self, coin: str, side: str) -> Optional[float]:
        signed_fn = getattr(self.backend, "_get_venue_signed_size", None)
        if callable(signed_fn):
            try:
                signed = signed_fn(coin)
                if signed is None:
                    return None
                s = str(side).upper()
                signed = float(signed)
                if s == "LONG":
                    return abs(signed) if signed > 0 else 0.0
                return abs(signed) if signed < 0 else 0.0
            except Exception as e:
                print(f"[LIVE_MONITOR] venue signed size check failed for {coin}: {e}")
                return None

        fn = getattr(self.backend, "_get_venue_position_size", None)
        if fn is None:
            return None
        try:
            size = fn(coin, side)
            if size is None:
                return None
            return float(size)
        except Exception as e:
            print(f"[LIVE_MONITOR] venue size check failed for {coin} {side}: {e}")
            return None

    def _get_exit_price(self, coin: str) -> Optional[float]:
        try:
            px = self.backend.get_mid_price(coin)
            if px is None:
                return None
            return float(px)
        except Exception as e:
            print(f"[LIVE_MONITOR] mid-price lookup failed for {coin}: {e}")
            return None

    def _has_blocking_open_orders(self, coin: str, return_details: bool = False):
        fn = getattr(self.backend, "get_open_orders", None)
        if not callable(fn):
            if return_details:
                return False, {"count": 0, "only_tp": False, "tpsl_summary": ""}
            return False

        try:
            orders = fn(coin)
        except Exception as e:
            print(f"[LIVE_MONITOR] open-orders check failed for {coin}: {e}")
            if return_details:
                return False, {"count": 0, "only_tp": False, "tpsl_summary": ""}
            return False

        if not isinstance(orders, list):
            if return_details:
                return False, {"count": 0, "only_tp": False, "tpsl_summary": ""}
            return False

        blocking = []
        for row in orders:
            if not isinstance(row, dict):
                continue
            reduce_only = bool(row.get("reduce_only", row.get("reduceOnly", False)))
            is_trigger = bool(row.get("is_trigger", row.get("isTrigger", False)))
            tpsl = str(row.get("tpsl", "") or "").strip().lower()
            if reduce_only or is_trigger or tpsl in {"tp", "sl"}:
                blocking.append(row)

        if not blocking:
            if return_details:
                return False, {"count": 0, "only_tp": False, "tpsl_summary": ""}
            return False

        tpsl_values = []
        for row in blocking:
            tpsl = str(row.get("tpsl", "") or "").strip().lower()
            tpsl_values.append(tpsl if tpsl else "other")
            oid = str(row.get("oid", "") or "").strip()
            print(
                f"[LIVE_MONITOR] {coin} blocking open order "
                f"oid={oid or '-'} reduce_only={bool(row.get('reduce_only', row.get('reduceOnly', False)))} "
                f"is_trigger={bool(row.get('is_trigger', row.get('isTrigger', False)))} "
                f"tpsl={tpsl or '-'}"
            )

        only_tp = all(str(v).lower() == "tp" for v in tpsl_values)
        has_unknown_tpsl = any(str(v).lower() == "other" for v in tpsl_values)
        only_reduce_like = all(
            bool(row.get("reduce_only", row.get("reduceOnly", False)))
            or bool(row.get("is_trigger", row.get("isTrigger", False)))
            or str(row.get("tpsl", "") or "").strip().lower() in {"tp", "sl"}
            for row in blocking
        )
        details = {
            "count": len(blocking),
            "only_tp": only_tp,
            "only_reduce_like": only_reduce_like,
            "has_unknown_tpsl": has_unknown_tpsl,
            "tpsl_summary": ",".join(sorted(set(tpsl_values))),
        }
        if return_details:
            return True, details
        return True

    @staticmethod
    def _pending_executor_exit(pos: Position) -> Optional[dict]:
        reason = str(getattr(pos, "pending_exit_reason", "")).strip().lower()
        fill_price = float(getattr(pos, "pending_exit_fill_price", 0.0) or 0.0)
        fill_size_usd = float(getattr(pos, "pending_exit_fill_size_usd", 0.0) or 0.0)
        fee_usd = float(getattr(pos, "pending_exit_fee_usd", 0.0) or 0.0)
        if not reason or fill_price <= 0:
            return None
        return {
            "reason": reason,
            "fill_price": fill_price,
            "fill_size_usd": fill_size_usd,
            "fee_usd": fee_usd,
        }

    @staticmethod
    def _position_entry_size_coin(pos: Position) -> float:
        entry_px = float(getattr(pos, "entry_price", 0.0) or 0.0)
        total_usd = float(getattr(pos, "size_usd", 0.0) or 0.0)
        if entry_px <= 0 or total_usd <= 0:
            return 0.0
        return max(0.0, total_usd / entry_px)

    def _close_fraction_from_coin(self, pos: Position, close_size_coin: float, default: float = 1.0) -> float:
        total_coin = self._position_entry_size_coin(pos)
        if total_coin <= 0:
            return max(0.0, min(1.0, float(default)))
        frac = float(close_size_coin or 0.0) / total_coin
        if frac <= 0:
            frac = float(default)
        return max(0.0, min(1.0, frac))

    # ------------------------------------------------------------------
    # Finalization
    # ------------------------------------------------------------------

    def _finalize_closed_position(
        self,
        pos: Position,
        source: str = "",
        reason: str = "",
    ) -> None:
        """
        Finalize a position that is no longer open on venue.

        Guards:
        - must be allow_reconcile_close=True
        - must be bootstrap-restored OR have executor exit intent
        - venue must still be flat on double-check
        - opposite side must also be flat
        """
        source_norm = str(source or "").strip().lower()
        immediate_wallet_flat = source_norm == "wallet_flat_reconcile"
        bypass_orphan_tp_block = source_norm in {
            "monitor_orphan_tp_reconcile",
            "monitor_orphan_reduce_reconcile",
        }

        if bool(getattr(pos, "_terminalized", False)) or bool(getattr(pos, "_terminalizing", False)):
            return
        if pos.state == PositionState.CLOSED and bool(getattr(pos, "wallet_flat_confirmed", False)):
            setattr(pos, "_terminalized", True)
            self.positions.pop(pos.position_id, None)
            self._zero_streak.pop(pos.position_id, None)
            self._flat_first_seen_ts.pop(pos.position_id, None)
            return

        if not getattr(pos, "allow_reconcile_close", False) and not immediate_wallet_flat:
            print(
                f"[LIVE_MONITOR] SKIP FINALIZE {pos.coin} — "
                "not marked for reconciliation"
            )
            return

        exit_requested_reason = str(getattr(pos, "exit_requested_reason", "") or "").strip()
        if reason and not exit_requested_reason:
            exit_requested_reason = str(reason).strip().lower()
            pos.exit_requested_reason = exit_requested_reason
        is_bootstrap_restored = bool(getattr(pos, "bootstrap_restored", False))
        has_native_protection = bool(
            bool(getattr(pos, "venue_protection_mode", False))
            and (
                str(getattr(pos, "stop_order_id", "")).strip()
                or str(getattr(pos, "tp_order_id", "")).strip()
            )
        )
        if (
            not immediate_wallet_flat
            and not exit_requested_reason
            and not is_bootstrap_restored
            and not has_native_protection
        ):
            print(
                f"[LIVE_MONITOR] SKIP FINALIZE {pos.coin} — "
                "runtime position without executor exit intent/native protection"
            )
            return

        if not immediate_wallet_flat:
            venue_size = self._get_venue_size(pos.coin, pos.side)
            if venue_size is None:
                print(f"[LIVE_MONITOR] SKIP FINALIZE {pos.coin} — venue unknown")
                return
            if venue_size > LIVE_FLAT_EPSILON_SZ:
                print(
                    f"[LIVE_MONITOR] SKIP FINALIZE {pos.coin} — "
                    f"venue still open ({venue_size:.8f})"
                )
                return

            opposite = "SHORT" if str(pos.side).upper() == "LONG" else "LONG"
            opposite_size = self._get_venue_size(pos.coin, opposite)
            if opposite_size is not None and opposite_size > LIVE_FLAT_EPSILON_SZ:
                print(
                    f"[LIVE_MONITOR] SKIP FINALIZE {pos.coin} — opposite side still open "
                    f"({opposite} sz={opposite_size:.8f})"
                )
                return

            if (not bypass_orphan_tp_block) and self._has_blocking_open_orders(pos.coin):
                print(
                    f"[LIVE_MONITOR] SKIP FINALIZE {pos.coin} — "
                    "blocking reduce-only/native orders still open"
                )
                return

        pending_exit = self._pending_executor_exit(pos)
        if pending_exit:
            exit_price = pending_exit["fill_price"]
        else:
            exit_price = self._get_exit_price(pos.coin)
            if exit_price is None or exit_price <= 0:
                exit_price = pos.entry_price

        terminal_lock = self._position_terminal_lock(pos)
        with terminal_lock:
            if bool(getattr(pos, "_terminalized", False)) or bool(getattr(pos, "_terminalizing", False)):
                return
            setattr(pos, "_terminalizing", True)
        try:
            gross_r = pos.current_r(exit_price)
            total_coin = self._position_entry_size_coin(pos)
            close_size_coin = 0.0
            if (
                pending_exit
                and pending_exit["fill_size_usd"] > 0
                and pending_exit["fill_price"] > 0
            ):
                close_size_coin = pending_exit["fill_size_usd"] / pending_exit["fill_price"]
            elif pos.partial_closed:
                partial_frac = float(getattr(pos, "partial_close_fraction", 0.0) or 0.0)
                if partial_frac <= 0:
                    partial_size_usd = float(getattr(pos, "partial_close_size_usd", 0.0) or 0.0)
                    entry_px = float(getattr(pos, "entry_price", 0.0) or 0.0)
                    if partial_size_usd > 0 and entry_px > 0 and total_coin > 0:
                        partial_frac = (partial_size_usd / entry_px) / total_coin
                if partial_frac <= 0:
                    # Legacy fallback for rows created before fractional partial accounting.
                    partial_frac = 0.5
                close_size_coin = total_coin * max(0.0, min(1.0, 1.0 - partial_frac))
            else:
                close_size_coin = total_coin

            default_fraction = 1.0 if not pos.partial_closed else 0.5
            close_fraction = self._close_fraction_from_coin(
                pos,
                close_size_coin,
                default=default_fraction,
            )
            close_notional = (
                close_size_coin * float(exit_price)
                if close_size_coin > 0 and float(exit_price) > 0
                else max(0.0, float(getattr(pos, "size_usd", 0.0) or 0.0) * close_fraction)
            )

            if pending_exit and pending_exit["fee_usd"] > 0:
                exit_fee_usd = pending_exit["fee_usd"]
            else:
                exit_fee_usd = self._estimate_exit_fee(close_notional)

            fee_r = exit_fee_usd / pos.r_value if pos.r_value > 0 else 0.0

            close_reason = self._classify_close_reason(pos, exit_price)
            pos.state = PositionState.CLOSED
            pos.close_reason = close_reason
            pos.closed_at = _utc_now()
            pos.updated_at = pos.closed_at
            pos.current_price = exit_price
            pos.exit_fees_usd = round(float(getattr(pos, "exit_fees_usd", 0.0)) + exit_fee_usd, 8)
            pos.total_fees_usd = round(pos.entry_fee_usd + pos.exit_fees_usd + pos.funding_usd, 8)

            if pos.partial_closed:
                gross_runner_usd = gross_r * pos.r_value * close_fraction
                net_runner_usd = gross_runner_usd - exit_fee_usd
                net_runner_r = net_runner_usd / pos.r_value if pos.r_value > 0 else 0.0
                pos.runner_r += net_runner_r
                pos.pnl_usd = round(float(getattr(pos, "pnl_usd", 0.0)) + net_runner_usd, 2)
                realized_r_for_notify = pos.realized_r
            else:
                gross_pnl_usd = gross_r * pos.r_value * close_fraction
                net_pnl_usd = gross_pnl_usd - exit_fee_usd
                net_r = gross_r * close_fraction - fee_r
                pos.partial_r = 0.0
                pos.runner_r = round(net_r, 4)
                pos.pnl_usd = round(net_pnl_usd, 2)
                realized_r_for_notify = net_r

            pos.reconciled_from_venue = True
            pos.allow_reconcile_close = False
            pos.wallet_flat_confirmed = True
            pos.wallet_flat_confirmed_at = pos.closed_at
            if source:
                pos.exit_trigger_source = source
            elif is_bootstrap_restored and not exit_requested_reason:
                pos.exit_trigger_source = "bootstrap_reconciled"
            elif exit_requested_reason:
                pos.exit_trigger_source = "executor_reconciled"
            elif has_native_protection:
                pos.exit_trigger_source = "venue_native_reconciled"
            else:
                pos.exit_trigger_source = "monitor_reconciled"

            if self.order_tracker is not None:
                try:
                    self.order_tracker.reconcile_position_close(
                        pos,
                        close_reason=close_reason,
                        source=str(pos.exit_trigger_source),
                    )
                except Exception as e:
                    print(f"[LIVE_MONITOR] order_tracker reconcile failed: {e}")

            try:
                self._apply_risk_close(
                    pos,
                    gross_r,
                    exit_fee_usd,
                    partial_closed=pos.partial_closed,
                    close_fraction=close_fraction,
                )
            except Exception as e:
                print(f"[LIVE_MONITOR] risk close apply failed for {pos.coin}: {e}")
            try:
                self._record_strategy_outcome(pos, pos.realized_r)
            except Exception as e:
                print(f"[LIVE_MONITOR] strategy outcome write failed for {pos.coin}: {e}")
            try:
                self._rewrite_trade_row(pos)
            except Exception as e:
                print(f"[LIVE_MONITOR] trade row rewrite failed for {pos.coin}: {e}")
            try:
                self._emit_reconcile_log(pos, realized_r_for_notify)
            except Exception as e:
                print(f"[LIVE_MONITOR] reconcile notify failed for {pos.coin}: {e}")

            setattr(pos, "_terminalized", True)
        finally:
            setattr(pos, "_terminalizing", False)
            if bool(getattr(pos, "_terminalized", False)):
                self.positions.pop(pos.position_id, None)
                self._zero_streak.pop(pos.position_id, None)
                self._flat_first_seen_ts.pop(pos.position_id, None)

    def _record_strategy_outcome(self, pos: Position, realized_r: float) -> None:
        strategy_filter = getattr(self.risk, "strategy_filter", None)
        if strategy_filter is None:
            return

        try:
            strategy_filter.record_outcome(
                pos.coin,
                getattr(pos, "setup_family", ""),
                getattr(pos, "htf_regime", ""),
                won=(realized_r > 0),
            )
        except Exception as e:
            print(f"[LIVE_MONITOR] strategy_filter outcome update failed: {e}")

        try:
            signal_engine = getattr(self, "signal_engine", None)
            recorder = getattr(signal_engine, "record_directional_outcome", None)
            if signal_engine is not None and callable(recorder):
                recorder(
                    coin=str(getattr(pos, "coin", "")).upper(),
                    side=str(getattr(pos, "side", "")).upper(),
                    won=bool(realized_r > 0),
                    setup_family=str(getattr(pos, "setup_family", "")).strip().lower(),
                )
        except Exception as e:
            print(f"[LIVE_MONITOR] directional outcome update failed: {e}")

    def _classify_close_reason(self, pos: Position, exit_price: float) -> CloseReason:
        requested = str(getattr(pos, "exit_requested_reason", "")).strip().lower()
        if requested == CloseReason.TP_FULL.value:
            return CloseReason.TP_FULL
        if requested == CloseReason.STOP_FULL.value:
            return CloseReason.STOP_FULL
        if requested == CloseReason.STOP_RUNNER.value:
            return CloseReason.STOP_RUNNER

        stop = float(pos.stop_price)
        tp = float(pos.tp_price)

        if pos.side == "LONG":
            if exit_price <= stop * 1.002:
                return CloseReason.STOP_FULL
            if exit_price >= tp * 0.998:
                return CloseReason.TP_FULL
        else:
            if exit_price >= stop * 0.998:
                return CloseReason.STOP_FULL
            if exit_price <= tp * 1.002:
                return CloseReason.TP_FULL

        return CloseReason.MANUAL

    def _estimate_exit_fee(self, size_usd: float) -> float:
        return round(float(size_usd) * (TAKER_FEE_BPS / 10000.0), 8)

    # ------------------------------------------------------------------
    # Risk/accounting
    # ------------------------------------------------------------------

    def _apply_risk_close(
        self,
        pos: Position,
        gross_r: float,
        exit_fee_usd: float,
        partial_closed: bool,
        close_fraction: float = 1.0,
    ) -> None:
        if partial_closed:
            record_close = getattr(self.risk, "record_close", None)
            if callable(record_close):
                record_close(
                    pos,
                    runner_r=gross_r,
                    fee_usd=exit_fee_usd,
                    close_fraction=close_fraction,
                )
                return

        record_live_full_close = getattr(self.risk, "record_live_full_close", None)
        if callable(record_live_full_close):
            record_live_full_close(
                pos,
                realized_r=gross_r,
                fee_usd=exit_fee_usd,
                close_fraction=close_fraction,
            )
            return

        realized_r = gross_r * max(0.0, min(1.0, float(close_fraction or 1.0)))
        if realized_r < MAX_FULL_LOSS_R:
            realized_r = MAX_FULL_LOSS_R

        pnl_usd = realized_r * pos.r_value - exit_fee_usd
        fee_r = (exit_fee_usd / pos.r_value) if pos.r_value > 0 else 0.0
        self.risk.balance += pnl_usd
        self.risk.daily_r += realized_r - fee_r
        self.risk.open_positions.pop(pos.coin, None)

        update_hwm = getattr(self.risk, "_update_hwm", None)
        if callable(update_hwm):
            update_hwm()

    # ------------------------------------------------------------------
    # CSV persistence
    # ------------------------------------------------------------------

    def _rewrite_trade_row(self, pos: Position) -> None:
        try:
            upsert_trade_row(self._position_to_csv_row(pos))
        except Exception as e:
            print(f"[LIVE_MONITOR] failed writing {self.trades_file}: {e}")

    def _position_to_csv_row(self, pos: Position) -> dict:
        row = pos.to_dict()
        row["paper_mode"] = "true" if PAPER_MODE else "false"
        row["reconciled_from_venue"] = "true" if getattr(pos, "reconciled_from_venue", False) else "false"
        row["allow_reconcile_close"] = "true" if getattr(pos, "allow_reconcile_close", False) else "false"
        row.setdefault("close_reason", pos.close_reason.value if pos.close_reason else "")
        row.setdefault("closed_at", pos.closed_at.isoformat() if pos.closed_at else "")
        row.setdefault("state", pos.state.value)
        row.setdefault("realized_r", round(pos.realized_r, 4))
        row.setdefault("pnl_usd", round(pos.pnl_usd, 2))

        out = {}
        for k, v in row.items():
            if isinstance(v, bool):
                out[k] = "true" if v else "false"
            elif v is None:
                out[k] = ""
            else:
                out[k] = str(v)
        return out

    # ------------------------------------------------------------------
    # Logging only
    # ------------------------------------------------------------------

    def _emit_reconcile_log(self, pos: Position, realized_r: float) -> None:
        reason = pos.close_reason.value if pos.close_reason else "manual"
        source = str(getattr(pos, "exit_trigger_source", "monitor_reconciled"))
        print(
            f"[LIVE_MONITOR] RECONCILED {pos.coin} {pos.side} {reason.upper()} — "
            f"{realized_r:+.2f}R | pnl=${pos.pnl_usd:+.2f} | source={source}"
        )

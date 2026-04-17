from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Callable, Dict, Iterable, List, Optional, Set

from order_tracker import OrderKind
from position import Position, PositionState


LIVE_FLAT_EPSILON_SZ = float(os.getenv("LIVE_FLAT_EPSILON_SZ", "1e-9"))


class ProtectionManager:
    """
    Owns native venue protection lifecycle:
      - placement after entry fill
      - ongoing audit against venue state/open orders
      - loud persistence of degraded protection
    """

    def __init__(
        self,
        backend,
        notify_fn: Optional[Callable[[str], None]] = None,
        persist_fn: Optional[Callable[[Position], None]] = None,
        order_tracker=None,
    ):
        self.backend = backend
        self.notify = notify_fn or (lambda msg: None)
        self.persist = persist_fn
        self.order_tracker = order_tracker
        self._warned_issue_keys: Set[str] = set()

    @staticmethod
    def _coin_size_from_fill(fill) -> float:
        if fill is None:
            return 0.0
        meta = getattr(fill, "meta", {})
        if isinstance(meta, dict):
            total_sz = float(meta.get("total_sz", 0.0) or 0.0)
            if total_sz > 0:
                return total_sz

        fill_notional = float(getattr(fill, "fill_size_usd", 0.0) or 0.0)
        fill_px = float(getattr(fill, "fill_price", 0.0) or 0.0)
        if fill_notional > 0 and fill_px > 0:
            return fill_notional / fill_px
        return 0.0

    @staticmethod
    def _first_oid(value: str) -> str:
        return str(value or "").strip()

    @staticmethod
    def _extract_prefixed_error(error_blob: str, prefix: str) -> str:
        prefix = f"{prefix.lower()}:"
        for chunk in str(error_blob or "").split("|"):
            text = chunk.strip()
            if text.lower().startswith(prefix):
                return text[len(prefix):].strip()
        return ""

    def _persist_position(self, pos: Position):
        if self.persist is None:
            return
        try:
            self.persist(pos)
        except Exception as e:
            print(f"[PROTECTION] persist failed for {pos.position_id}: {e}")

    def _warn_once(
        self,
        pos: Position,
        issue: str,
        source: str,
        severity: str = "CRITICAL",
        include_details: Optional[Dict[str, object]] = None,
    ):
        key = f"{pos.position_id}:{issue}:{source}"
        if key in self._warned_issue_keys:
            return
        self._warned_issue_keys.add(key)

        detail_parts: List[str] = []
        if include_details:
            for k, v in include_details.items():
                detail_parts.append(f"{k}={v}")
        detail_text = " ".join(detail_parts)

        print(
            f"[PROTECTION][{severity}] {pos.coin} {pos.side} "
            f"pos={pos.position_id} issue={issue} source={source} {detail_text}".rstrip()
        )

        try:
            msg = (
                f"🚨 *PROTECTION {severity}* 🔴 LIVE\n\n"
                f"coin=`{pos.coin}` side=`{pos.side}`\n"
                f"status=`{getattr(pos, 'protection_status', '') or 'missing'}`\n"
                f"issue=`{issue}`\n"
                f"source=`{source}`\n"
                f"position_id=`{pos.position_id}`"
            )
            if detail_text:
                msg += f"\n\n`{detail_text}`"
            self.notify(msg)
        except Exception as e:
            print(f"[PROTECTION] notify failed: {e}")

    def _clear_warnings(self, pos: Position):
        prefix = f"{pos.position_id}:"
        self._warned_issue_keys = {k for k in self._warned_issue_keys if not k.startswith(prefix)}

    def notify_issue(self, pos: Position, issue: str, source: str = "runtime"):
        self._warn_once(pos, issue=issue, source=source, severity="CRITICAL")
        self._persist_position(pos)

    def _resolve_position_size_coin(self, pos: Position, entry_fill=None) -> float:
        signed_fn = getattr(self.backend, "_get_venue_signed_size", None)
        side = str(pos.side).upper()

        if callable(signed_fn):
            try:
                signed = signed_fn(pos.coin)
            except Exception as e:
                print(f"[PROTECTION] venue signed-size lookup failed for {pos.coin}: {e}")
                signed = None
            if signed is not None:
                signed = float(signed)
                if side == "LONG":
                    if signed > LIVE_FLAT_EPSILON_SZ:
                        return abs(signed)
                    if signed < -LIVE_FLAT_EPSILON_SZ:
                        return 0.0
                else:
                    if signed < -LIVE_FLAT_EPSILON_SZ:
                        return abs(signed)
                    if signed > LIVE_FLAT_EPSILON_SZ:
                        return 0.0

        from_fill = self._coin_size_from_fill(entry_fill)
        if from_fill > 0:
            return from_fill

        px = self.backend.get_mid_price(pos.coin)
        if px is None or px <= 0:
            px = float(getattr(pos, "entry_price", 0.0) or 0.0)
        if px <= 0:
            return 0.0
        notional = float(getattr(pos, "size_usd", 0.0) or 0.0)
        return max(0.0, notional / px)

    def _create_tracker_orders(
        self,
        pos: Position,
        size_coin: float,
        source: str,
    ) -> Dict[str, str]:
        out = {"stop": "", "tp": ""}
        if self.order_tracker is None:
            return out

        out["stop"] = self.order_tracker.create_order(
            position_id=pos.position_id,
            coin=pos.coin,
            side=pos.side,
            order_kind=OrderKind.STOP_LOSS,
            requested_price=float(pos.stop_price),
            requested_size_coin=float(size_coin),
            source=f"protection:{source or 'runtime'}",
        )
        out["tp"] = self.order_tracker.create_order(
            position_id=pos.position_id,
            coin=pos.coin,
            side=pos.side,
            order_kind=OrderKind.TAKE_PROFIT,
            requested_price=float(pos.tp_price),
            requested_size_coin=float(size_coin),
            source=f"protection:{source or 'runtime'}",
        )
        return out

    def _mark_tracker_submitted(self, tracker_ids: Dict[str, str]):
        if self.order_tracker is None:
            return
        if tracker_ids.get("stop"):
            self.order_tracker.mark_submitted(tracker_ids["stop"])
        if tracker_ids.get("tp"):
            self.order_tracker.mark_submitted(tracker_ids["tp"])

    def _mark_tracker_failed(self, tracker_ids: Dict[str, str], reason: str):
        if self.order_tracker is None:
            return
        if tracker_ids.get("stop"):
            self.order_tracker.mark_failed(tracker_ids["stop"], reason=reason)
        if tracker_ids.get("tp"):
            self.order_tracker.mark_failed(tracker_ids["tp"], reason=reason)

    def _mark_tracker_ack_or_fail(
        self,
        tracker_ids: Dict[str, str],
        stop_oid: str,
        tp_oid: str,
        error: str,
        source: str,
    ):
        if self.order_tracker is None:
            return

        stop_track_id = tracker_ids.get("stop", "")
        tp_track_id = tracker_ids.get("tp", "")

        if stop_track_id:
            if stop_oid:
                self.order_tracker.mark_acknowledged(
                    stop_track_id,
                    venue_order_id=stop_oid,
                    note=f"protection:{source or 'runtime'}",
                )
            else:
                self.order_tracker.mark_failed(
                    stop_track_id,
                    reason=self._extract_prefixed_error(error, "stop") or "stop_order_not_acknowledged",
                )

        if tp_track_id:
            if tp_oid:
                self.order_tracker.mark_acknowledged(
                    tp_track_id,
                    venue_order_id=tp_oid,
                    note=f"protection:{source or 'runtime'}",
                )
            else:
                self.order_tracker.mark_failed(
                    tp_track_id,
                    reason=self._extract_prefixed_error(error, "tp") or "tp_order_not_acknowledged",
                )

    def place_after_entry(self, pos: Position, entry_fill=None, source: str = "") -> bool:
        source = source or "runtime"
        place_fn = getattr(self.backend, "place_native_protection", None)
        size_coin = self._resolve_position_size_coin(pos, entry_fill=entry_fill)
        tracker_ids = self._create_tracker_orders(pos, size_coin=size_coin, source=source)

        if not callable(place_fn):
            issue = "backend_missing_place_native_protection"
            pos.venue_protection_mode = False
            pos.stop_order_id = ""
            pos.tp_order_id = ""
            pos.protection_status = "missing"
            pos.protection_placed_at = None
            pos.protection_error = issue
            self._mark_tracker_failed(tracker_ids, issue)
            self._warn_once(pos, issue=issue, source=source, severity="CRITICAL")
            self._persist_position(pos)
            return False

        pos.venue_protection_mode = True
        if size_coin <= 0:
            issue = "unconfirmed_or_zero_venue_size"
            pos.stop_order_id = ""
            pos.tp_order_id = ""
            pos.protection_status = "missing"
            pos.protection_placed_at = None
            pos.protection_error = issue
            self._mark_tracker_failed(tracker_ids, issue)
            self._warn_once(pos, issue=issue, source=source, severity="CRITICAL")
            self._persist_position(pos)
            return False

        self._mark_tracker_submitted(tracker_ids)

        try:
            result = place_fn(
                coin=pos.coin,
                side=pos.side,
                size_coin=size_coin,
                stop_price=pos.stop_price,
                tp_price=pos.tp_price,
            )
        except Exception as e:
            issue = f"protection_sdk_error:{e}"
            pos.stop_order_id = ""
            pos.tp_order_id = ""
            pos.protection_status = "missing"
            pos.protection_placed_at = None
            pos.protection_error = issue
            self._mark_tracker_failed(tracker_ids, issue)
            self._warn_once(pos, issue=issue, source=source, severity="CRITICAL")
            self._persist_position(pos)
            return False

        stop_oid = self._first_oid((result or {}).get("stop_order_id", ""))
        tp_oid = self._first_oid((result or {}).get("tp_order_id", ""))
        status = str((result or {}).get("status", "") or "").strip().lower()
        error = str((result or {}).get("error", "") or "").strip()
        self._mark_tracker_ack_or_fail(
            tracker_ids=tracker_ids,
            stop_oid=stop_oid,
            tp_oid=tp_oid,
            error=error,
            source=source,
        )

        pos.stop_order_id = stop_oid
        pos.tp_order_id = tp_oid
        pos.protection_status = status or (
            "protected" if (stop_oid and tp_oid) else ("partial" if (stop_oid or tp_oid) else "missing")
        )
        pos.protection_error = error
        pos.protection_placed_at = datetime.now(timezone.utc) if (stop_oid or tp_oid) else None
        if pos.protection_status in {"protected", "partial"}:
            pos.allow_reconcile_close = True

        if pos.protection_status == "protected":
            self._clear_warnings(pos)
            print(
                f"[PROTECTION] {pos.coin} {pos.side} ACTIVE "
                f"stop_oid={stop_oid} tp_oid={tp_oid} source={source}"
            )
            self._persist_position(pos)
            return True

        issue = error or f"status:{pos.protection_status}"
        self._warn_once(
            pos,
            issue=issue,
            source=source,
            severity="CRITICAL",
            include_details={
                "stop_oid": stop_oid or "-",
                "tp_oid": tp_oid or "-",
                "size_coin": round(float(size_coin), 8),
            },
        )
        self._persist_position(pos)
        return False

    def replace_stop_after_partial(
        self,
        pos: Position,
        fallback_size_coin: float = 0.0,
        source: str = "partial_be",
    ) -> bool:
        source = source or "partial_be"
        stop_px = float(getattr(pos, "stop_price", 0.0) or 0.0)
        if stop_px <= 0:
            issue = "breakeven_stop_replace_invalid_stop_price"
            pos.protection_error = issue
            self._warn_once(pos, issue=issue, source=source, severity="CRITICAL")
            self._persist_position(pos)
            return False

        size_coin = self._resolve_position_size_coin(pos, entry_fill=None)
        if size_coin <= 0 and float(fallback_size_coin or 0.0) > 0:
            size_coin = float(fallback_size_coin)
        if size_coin <= 0:
            issue = "breakeven_stop_replace_missing_size"
            pos.protection_error = issue
            self._warn_once(pos, issue=issue, source=source, severity="CRITICAL")
            self._persist_position(pos)
            return False

        old_stop_oid = str(getattr(pos, "stop_order_id", "") or "").strip()
        cancel_err = ""
        cancel_fn = getattr(self.backend, "cancel_order", None)
        if old_stop_oid:
            if callable(cancel_fn):
                try:
                    cancel_ok = bool(cancel_fn(pos.coin, old_stop_oid))
                    if not cancel_ok:
                        cancel_err = "cancel_not_ok"
                except Exception as e:
                    cancel_err = str(e)
            else:
                cancel_err = "backend_missing_cancel_order"

        place_stop_fn = getattr(self.backend, "place_stop_only", None)
        if not callable(place_stop_fn):
            issue = "backend_missing_place_stop_only"
            pos.protection_error = issue
            self._warn_once(pos, issue=issue, source=source, severity="CRITICAL")
            self._persist_position(pos)
            return False

        try:
            place_result = place_stop_fn(
                coin=pos.coin,
                side=pos.side,
                size_coin=float(size_coin),
                stop_price=stop_px,
            )
        except Exception as e:
            issue = f"breakeven_stop_replace_sdk_error:{e}"
            if old_stop_oid and not cancel_err:
                pos.stop_order_id = old_stop_oid
            else:
                pos.stop_order_id = ""
            pos.protection_status = "partial" if str(getattr(pos, "tp_order_id", "") or "").strip() else "missing"
            pos.protection_error = issue
            self._warn_once(pos, issue=issue, source=source, severity="CRITICAL")
            self._persist_position(pos)
            return False

        new_stop_oid = self._first_oid((place_result or {}).get("stop_order_id", ""))
        place_err = str((place_result or {}).get("error", "") or "").strip()
        if not new_stop_oid:
            issue = place_err or "breakeven_stop_replace_failed"
            if old_stop_oid and not cancel_err:
                pos.stop_order_id = old_stop_oid
            else:
                pos.stop_order_id = ""
            pos.protection_status = "partial" if str(getattr(pos, "tp_order_id", "") or "").strip() else "missing"
            pos.protection_error = issue
            self._warn_once(pos, issue=issue, source=source, severity="CRITICAL")
            self._persist_position(pos)
            return False

        pos.stop_order_id = new_stop_oid
        pos.protection_status = "protected" if str(getattr(pos, "tp_order_id", "") or "").strip() else "partial"
        pos.protection_error = ""
        pos.protection_placed_at = datetime.now(timezone.utc)
        pos.allow_reconcile_close = True

        if cancel_err:
            warn_issue = f"breakeven_stop_replace_cancel_warning:{cancel_err}"
            self._warn_once(pos, issue=warn_issue, source=source, severity="WARNING")

        self._clear_warnings(pos)
        self._persist_position(pos)
        print(
            f"[PROTECTION] BE STOP REPLACED {pos.coin} {pos.side} "
            f"old_stop_oid={old_stop_oid or '-'} new_stop_oid={new_stop_oid} "
            f"stop_px={stop_px:.8f} source={source}"
        )
        return True

    @staticmethod
    def _venue_side_size_from_signed(side: str, signed_size: Optional[float]) -> Optional[float]:
        if signed_size is None:
            return None
        signed = float(signed_size)
        if str(side).upper() == "LONG":
            return abs(signed) if signed > LIVE_FLAT_EPSILON_SZ else 0.0
        return abs(signed) if signed < -LIVE_FLAT_EPSILON_SZ else 0.0

    @staticmethod
    def _collect_oid_set(orders: Iterable[Dict]) -> Set[str]:
        out: Set[str] = set()
        for row in orders:
            oid = str(row.get("oid", "") or "").strip()
            if oid:
                out.add(oid)
        return out

    @staticmethod
    def _count_reduce_only_triggers(orders: Iterable[Dict]) -> int:
        count = 0
        for row in orders:
            is_trigger = bool(row.get("is_trigger", False))
            reduce_only = bool(row.get("reduce_only", False))
            if is_trigger and reduce_only:
                count += 1
        return count

    def audit_position(self, pos: Position, source: str = "runtime_audit") -> Dict[str, object]:
        side = str(pos.side).upper()
        stop_oid = str(getattr(pos, "stop_order_id", "") or "").strip()
        tp_oid = str(getattr(pos, "tp_order_id", "") or "").strip()
        prev_status = str(getattr(pos, "protection_status", "") or "").strip()
        prev_error = str(getattr(pos, "protection_error", "") or "").strip()

        signed_fn = getattr(self.backend, "_get_venue_signed_size", None)
        signed_size = None
        if callable(signed_fn):
            try:
                signed_size = signed_fn(pos.coin)
            except Exception as e:
                print(f"[PROTECTION] signed size audit failed for {pos.coin}: {e}")
                signed_size = None
        if signed_size is not None:
            signed_size = float(signed_size)

        venue_side_size = self._venue_side_size_from_signed(side=side, signed_size=signed_size)
        venue_open_fn = getattr(self.backend, "get_open_orders", None)
        venue_orders: Optional[List[Dict]] = None
        if callable(venue_open_fn):
            try:
                venue_orders = venue_open_fn(pos.coin)
            except Exception as e:
                print(f"[PROTECTION] open-order audit failed for {pos.coin}: {e}")
                venue_orders = None

        venue_order_ids: Set[str] = set()
        trigger_count = 0
        if venue_orders is not None:
            venue_order_ids = self._collect_oid_set(venue_orders)
            trigger_count = self._count_reduce_only_triggers(venue_orders)

        stop_on_venue = bool(stop_oid and stop_oid in venue_order_ids)
        tp_on_venue = bool(tp_oid and tp_oid in venue_order_ids)

        audit = {
            "position_id": pos.position_id,
            "coin": pos.coin,
            "side": side,
            "local_state": pos.state.value if isinstance(pos.state, PositionState) else str(pos.state),
            "signed_venue_size": signed_size,
            "venue_side_size": venue_side_size,
            "venue_orders_known": venue_orders is not None,
            "venue_trigger_reduce_only_count": trigger_count,
            "local_stop_order_id": stop_oid,
            "local_tp_order_id": tp_oid,
            "stop_on_venue": stop_on_venue,
            "tp_on_venue": tp_on_venue,
        }

        if pos.state == PositionState.CLOSED:
            self._clear_warnings(pos)
            return audit

        pos.venue_protection_mode = True

        # No open position on venue; no protection needed for this moment.
        if venue_side_size is not None and venue_side_size <= LIVE_FLAT_EPSILON_SZ:
            if prev_status not in {"", "flat"}:
                pos.protection_status = "flat"
                pos.protection_error = ""
                self._persist_position(pos)
            self._clear_warnings(pos)
            return audit

        derived_status = "missing"
        issue = ""
        if venue_orders is None:
            # Fall back to local metadata when venue order feed is unavailable.
            if stop_oid and tp_oid:
                derived_status = "protected"
            elif stop_oid or tp_oid:
                derived_status = "partial"
                issue = "venue_open_orders_unknown_partial_local_metadata"
            else:
                derived_status = "missing"
                issue = "venue_open_orders_unknown_and_missing_local_metadata"
        else:
            if stop_on_venue and tp_on_venue:
                derived_status = "protected"
            elif stop_on_venue or tp_on_venue:
                derived_status = "partial"
                issue = "only_one_protection_order_live_on_venue"
            else:
                if stop_oid or tp_oid:
                    issue = "local_protection_ids_not_found_on_venue"
                elif trigger_count > 0:
                    issue = "venue_has_trigger_orders_but_local_ids_missing"
                else:
                    issue = "no_live_protection_orders_detected"
                derived_status = "missing"

        pos.protection_status = derived_status
        pos.protection_error = issue
        if derived_status in {"protected", "partial"}:
            pos.allow_reconcile_close = True

        changed = (prev_status != pos.protection_status) or (prev_error != pos.protection_error)
        if changed:
            self._persist_position(pos)

        if derived_status == "protected":
            self._clear_warnings(pos)
            return audit

        self._warn_once(
            pos,
            issue=issue or f"status:{derived_status}",
            source=source,
            severity="CRITICAL",
            include_details={
                "signed_size": signed_size,
                "venue_side_size": venue_side_size,
                "stop_on_venue": stop_on_venue,
                "tp_on_venue": tp_on_venue,
                "trigger_count": trigger_count,
            },
        )
        if not changed:
            # Persist at least once when issue is detected but status did not change.
            self._persist_position(pos)
        return audit

    def audit_open_positions(self, positions: Iterable[Position], source: str = "runtime_audit"):
        for pos in positions:
            if pos.state == PositionState.CLOSED:
                continue
            self.audit_position(pos, source=source)

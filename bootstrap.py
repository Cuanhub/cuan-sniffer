"""
State reconstruction from trades.csv on boot.

v2 — Audited for live mode.

Reads the trade log and rebuilds:
  - realized balance (STARTING_BALANCE + all closed pnl_usd)
  - high-water mark  (peak balance ever reached, walked chronologically)
  - daily R          (realized_r for trades closed today UTC)
  - open positions   (state=open|partial → rebuilt Position objects)
  - strategy filter  (outcome replay → kill-switch state restored)

─── v2 changes ────────────────────────────────────────────────────
LIVE MODE FIXES:
  • paper_mode filter: only restores positions matching current mode.
    Prevents paper ghost positions from blocking live slots.
  • Venue reconciliation: in live mode, verifies each restored position
    actually exists on venue before including it. Prevents phantom
    positions from accumulating in risk.open_positions after restarts.
  • Cooldown seeding: returns opened_at timestamps for restored
    positions so executor can pre-fill cooldown dicts.
  • Fee field restoration from CSV (entry_fee_usd, etc).

CARRIED FORWARD:
  • original_stop_price / original_tp_price restoration.
  • Cancelled/stale trade exclusion.
  • stale_invalidated and first_update_pending restore.
  • Graceful degradation to fresh STARTING_BALANCE.
───────────────────────────────────────────────────────────────────
"""

import csv
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Dict, Optional

from position import Position, PositionState

TRADES_FILE = os.getenv("TRADES_FILE", "trades.csv")
LIVE_BOOTSTRAP_STRICT_WALLET_CLOSE = (
    os.getenv("LIVE_BOOTSTRAP_STRICT_WALLET_CLOSE", "true").lower() == "true"
)


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class BootstrapResult:
    """Everything the Executor needs to reconstruct its state."""
    starting_balance: float
    realized_balance: float
    high_water_mark:  float
    daily_r:          float
    daily_pnl:        float
    total_trades:     int
    open_positions:   List[Position]
    closed_today:     int
    all_time_pnl:     float
    # v2: cooldown seeding — maps (coin, side) → opened_at timestamp
    open_timestamps:  Dict = field(default_factory=dict)


# ── CSV helpers ────────────────────────────────────────────────────────────────

def _load_all_trades() -> List[Dict]:
    if not os.path.exists(TRADES_FILE):
        return []
    try:
        rows = []
        with open(TRADES_FILE, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        return rows
    except Exception as e:
        print(f"[BOOTSTRAP] Could not read {TRADES_FILE}: {e}")
        return []


def _parse_dt(raw: str) -> Optional[datetime]:
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


def _safe_float(val: str, default: float = 0.0) -> float:
    try:
        return float(val) if val else default
    except (ValueError, TypeError):
        return default


def _safe_bool(val: str) -> bool:
    return str(val).lower() in ("true", "1", "yes")


def _sort_key(r: Dict) -> datetime:
    dt = _parse_dt(r.get("opened_at", ""))
    return dt or datetime.min.replace(tzinfo=timezone.utc)


def _is_wallet_authoritative_close_row(row: Dict) -> bool:
    if _safe_bool(row.get("wallet_flat_confirmed", "false")):
        return True
    if _safe_bool(row.get("reconciled_from_venue", "false")):
        return True
    if str(row.get("exit_order_ids", "")).strip():
        return True
    return False


# ── Main bootstrap function ────────────────────────────────────────────────────

def bootstrap_state(
    starting_balance: float,
    paper_mode: Optional[bool] = None,
    venue_checker=None,
) -> BootstrapResult:
    """
    Reconstruct state from trades.csv.

    Args:
        starting_balance: Initial account balance.
        paper_mode: If provided, only restore positions matching this mode.
                    True = only paper positions, False = only live positions,
                    None = restore all (backward-compatible default).
        venue_checker: Optional callable(coin, side) -> bool.
                       If provided, only restore positions that the venue
                       confirms are still open. Used in live mode to prevent
                       phantom position accumulation.
    """
    rows = _load_all_trades()
    today = datetime.now(timezone.utc).date()

    if not rows:
        print("[BOOTSTRAP] No trades.csv found — starting fresh.")
        return BootstrapResult(
            starting_balance=starting_balance,
            realized_balance=starting_balance,
            high_water_mark=starting_balance,
            daily_r=0.0,
            daily_pnl=0.0,
            total_trades=0,
            open_positions=[],
            closed_today=0,
            all_time_pnl=0.0,
        )

    rows_sorted = sorted(rows, key=_sort_key)

    # Deduplicate — keep latest version of each position
    seen_ids: Dict[str, Dict] = {}
    for row in rows_sorted:
        pid = row.get("position_id", "")
        if pid:
            seen_ids[pid] = row

    deduped_sorted = sorted(seen_ids.values(), key=_sort_key)

    running_balance = starting_balance
    high_water_mark = starting_balance
    daily_r = 0.0
    daily_pnl = 0.0
    total_closed = 0
    closed_today = 0
    open_positions: List[Position] = []
    open_timestamps: Dict = {}

    for row in deduped_sorted:
        state = row.get("state", "")
        pnl = _safe_float(row.get("pnl_usd", "0"))

        # Skip cancelled/stale
        stale = _safe_bool(row.get("stale_invalidated", "false"))
        if state == "cancelled" or stale:
            continue

        if state == "closed":
            if (
                paper_mode is False
                and LIVE_BOOTSTRAP_STRICT_WALLET_CLOSE
                and not _is_wallet_authoritative_close_row(row)
            ):
                pid = row.get("position_id", "?")
                print(
                    f"[BOOTSTRAP] Skipping unconfirmed closed row {pid} "
                    "(wallet authority not present)"
                )
                continue

            running_balance += pnl
            if running_balance > high_water_mark:
                high_water_mark = running_balance
            total_closed += 1

            closed_at = _parse_dt(row.get("closed_at", ""))
            if closed_at and closed_at.date() >= today:
                daily_r += _safe_float(row.get("realized_r", "0"))
                daily_pnl += pnl
                closed_today += 1

        elif state in ("open", "partial"):
            # ── Paper mode filter ────────────────────────────────────
            if paper_mode is not None:
                row_paper = _safe_bool(row.get("paper_mode", "true"))
                if paper_mode and not row_paper:
                    # We're in paper mode but this is a live position — skip
                    continue
                if not paper_mode and row_paper:
                    # We're in live mode but this is a paper position — skip
                    pid = row.get("position_id", "?")
                    print(
                        f"[BOOTSTRAP] Skipping paper position {pid} "
                        f"(live mode active)"
                    )
                    continue

            pos = _rebuild_position(row)
            if pos is None:
                continue

            # ── Venue reconciliation (live mode) ─────────────────────
            if venue_checker is not None:
                has_position = False
                try:
                    has_position = venue_checker(pos.coin, pos.side)
                except Exception as e:
                    print(
                        f"[BOOTSTRAP] Venue check failed for {pos.coin} "
                        f"{pos.side}: {e} — keeping position defensively"
                    )
                    has_position = True  # fail-open: keep it if we can't check

                if not has_position:
                    print(
                        f"[BOOTSTRAP] {pos.coin} {pos.side} NOT found on venue "
                        f"— skipping (position may have been closed externally)"
                    )
                    continue

            open_positions.append(pos)

    # ── Deduplicate same-coin positions ──────────────────────────────
    # Multiple restarts can accumulate several open positions for the same
    # coin+side in trades.csv (each restart creates a new one before the
    # previous closes). Only the most recently opened survives in memory;
    # the others become zombies. Keep the latest, warn about duplicates.
    seen_coin: Dict[str, Position] = {}
    deduped_open: List[Position] = []
    for pos in sorted(open_positions, key=lambda p: p.opened_at or datetime.min.replace(tzinfo=timezone.utc)):
        key = f"{pos.coin}:{pos.side}"
        if key in seen_coin:
            older = seen_coin[key]
            print(
                f"[BOOTSTRAP] WARNING: duplicate open position for {pos.coin} {pos.side} "
                f"— dropping older {older.position_id} (opened {older.opened_at}), "
                f"keeping {pos.position_id}"
            )
        seen_coin[key] = pos
    deduped_open = list(seen_coin.values())

    if len(deduped_open) < len(open_positions):
        print(
            f"[BOOTSTRAP] Deduplication removed "
            f"{len(open_positions) - len(deduped_open)} ghost position(s) from memory. "
            f"trades.csv still holds original rows — close or cancel them manually."
        )
    open_positions = deduped_open

    # Seed cooldown data from deduplicated positions
    for pos in open_positions:
        opened_ts = pos.opened_at.timestamp() if pos.opened_at else 0.0
        open_timestamps[(pos.coin, pos.side)] = opened_ts

    all_time_pnl = running_balance - starting_balance

    mode_tag = ""
    if paper_mode is True:
        mode_tag = " [paper only]"
    elif paper_mode is False:
        mode_tag = " [live only]"

    print(
        f"[BOOTSTRAP] Balance: ${running_balance:.2f} "
        f"(start ${starting_balance:.2f}, P&L ${all_time_pnl:+.2f}){mode_tag}"
    )
    print(
        f"[BOOTSTRAP] HWM: ${high_water_mark:.2f} | "
        f"Closed: {total_closed} | "
        f"Today: {closed_today} trades, {daily_r:+.2f}R | "
        f"Open restored: {len(open_positions)}"
    )

    return BootstrapResult(
        starting_balance=starting_balance,
        realized_balance=running_balance,
        high_water_mark=high_water_mark,
        daily_r=daily_r,
        daily_pnl=daily_pnl,
        total_trades=total_closed,
        open_positions=open_positions,
        closed_today=closed_today,
        all_time_pnl=all_time_pnl,
        open_timestamps=open_timestamps,
    )


def replay_strategy_filter(strategy_filter, starting_balance: float) -> None:
    rows = _load_all_trades()
    if not rows:
        return

    rows_sorted = sorted(rows, key=_sort_key)

    seen: Dict[str, Dict] = {}
    for row in rows_sorted:
        pid = row.get("position_id", "")
        if pid:
            seen[pid] = row

    replayed = 0
    apply_pauses_during_replay = True
    if hasattr(strategy_filter, "has_persisted_pause_state"):
        try:
            apply_pauses_during_replay = not bool(strategy_filter.has_persisted_pause_state())
        except Exception:
            apply_pauses_during_replay = True

    for row in sorted(seen.values(), key=_sort_key):
        state = row.get("state", "")
        stale = _safe_bool(row.get("stale_invalidated", "false"))

        if state != "closed" or stale:
            continue
        coin = row.get("coin", "")
        setup_family = row.get("setup_family", "")
        htf_regime = row.get("htf_regime", "")
        realized_r = _safe_float(row.get("realized_r", "0"))
        won = realized_r > 0

        if coin and setup_family:
            strategy_filter.record_outcome(
                coin,
                setup_family,
                htf_regime,
                won,
                apply_pauses=apply_pauses_during_replay,
            )
            replayed += 1

    if replayed:
        mode = "with pauses" if apply_pauses_during_replay else "without pause recompute"
        print(f"[BOOTSTRAP] Strategy filter replayed {replayed} trade outcomes ({mode})")


# ── Position reconstruction ────────────────────────────────────────────────────

def _rebuild_position(row: Dict) -> Optional[Position]:
    """
    Reconstruct a Position from a trades.csv row.

    Restores original_stop_price/original_tp_price from CSV so R
    calculations remain correct for partial positions at breakeven.
    """
    try:
        state_str = row.get("state", "open")
        state = (PositionState.PARTIAL if state_str == "partial"
                 else PositionState.OPEN)

        entry = _safe_float(row.get("entry_price"))
        stop = _safe_float(row.get("stop_price"))
        tp = _safe_float(row.get("tp_price"))

        if entry <= 0 or stop <= 0 or tp <= 0:
            return None

        # Restore original levels
        original_stop = _safe_float(row.get("original_stop_price"))
        original_tp = _safe_float(row.get("original_tp_price"))

        is_partial = _safe_bool(row.get("partial_closed", "false"))
        is_breakeven = _safe_bool(row.get("breakeven_moved", "false"))

        if original_stop <= 0:
            if is_breakeven or is_partial:
                print(
                    f"[BOOTSTRAP] WARNING: {row.get('position_id','?')} "
                    f"partial/BE missing original_stop_price — "
                    f"attempting recovery from r_value/size_usd"
                )
                r_value = _safe_float(row.get("r_value"))
                size_usd = _safe_float(row.get("size_usd"))
                if r_value > 0 and size_usd > 0 and entry > 0:
                    stop_pct = r_value / size_usd
                    side = row.get("side", "LONG")
                    if side == "LONG":
                        original_stop = entry * (1 - stop_pct)
                    else:
                        original_stop = entry * (1 + stop_pct)
                else:
                    original_stop = stop
            else:
                original_stop = stop

        if original_tp <= 0:
            original_tp = tp

        opened_at = _parse_dt(row.get("opened_at", "")) or datetime.now(timezone.utc)

        pos = Position(
            position_id=row.get("position_id", "unknown"),
            coin=row.get("coin", "?"),
            side=row.get("side", "LONG"),
            signal_id=int(_safe_float(row.get("signal_id", "0"))),
            entry_price=entry,
            stop_price=stop,
            tp_price=tp,
            atr=_safe_float(row.get("atr")),
            original_stop_price=original_stop,
            original_tp_price=original_tp,
            size_usd=_safe_float(row.get("size_usd")),
            risk_usd=_safe_float(row.get("risk_usd")),
            r_value=_safe_float(row.get("r_value")),
            size_multiplier=_safe_float(row.get("size_multiplier", "1.0"), 1.0),
            state=state,
            partial_closed=is_partial,
            breakeven_moved=is_breakeven,
            partial_r=_safe_float(row.get("partial_r")),
            runner_r=_safe_float(row.get("runner_r")),
            pnl_usd=_safe_float(row.get("pnl_usd")),
            partial_close_size_usd=_safe_float(row.get("partial_close_size_usd")),
            partial_close_fraction=_safe_float(row.get("partial_close_fraction")),
            # Fee fields (v2) — restore from CSV, default 0.0 if absent
            entry_fee_usd=_safe_float(row.get("entry_fee_usd")),
            exit_fees_usd=_safe_float(row.get("exit_fees_usd")),
            funding_usd=_safe_float(row.get("funding_usd")),
            total_fees_usd=_safe_float(row.get("total_fees_usd")),
            # Metadata
            regime=row.get("regime", ""),
            setup_family=row.get("setup_family", ""),
            htf_regime=row.get("htf_regime", ""),
            confidence=_safe_float(row.get("confidence")),
            total_score=_safe_float(row.get("total_score")),
            session=row.get("session", ""),
            timeframe=row.get("timeframe", "15m") or "15m",
            execution_track=row.get("execution_track", "intraday") or "intraday",
            entry_order_ids=row.get("entry_order_ids", "") or "",
            exit_order_ids=row.get("exit_order_ids", "") or "",
            stop_order_id=row.get("stop_order_id", "") or "",
            tp_order_id=row.get("tp_order_id", "") or "",
            protection_status=row.get("protection_status", "") or "",
            protection_error=row.get("protection_error", "") or "",
            venue_protection_mode=_safe_bool(row.get("venue_protection_mode", "false")),
            allow_reconcile_close=_safe_bool(row.get("allow_reconcile_close", "true")),
            reconciled_from_venue=_safe_bool(row.get("reconciled_from_venue", "false")),
            exit_requested_reason=row.get("exit_requested_reason", "") or "",
            exit_trigger_source=row.get("exit_trigger_source", "") or "",
            wallet_flat_confirmed=_safe_bool(row.get("wallet_flat_confirmed", "false")),
            pending_exit_reason=row.get("pending_exit_reason", "") or "",
            pending_exit_fill_price=_safe_float(row.get("pending_exit_fill_price")),
            pending_exit_fill_size_usd=_safe_float(row.get("pending_exit_fill_size_usd")),
            pending_exit_fee_usd=_safe_float(row.get("pending_exit_fee_usd")),
            pending_exit_slippage_bps=_safe_float(row.get("pending_exit_slippage_bps")),
            opened_at=opened_at,
            current_price=entry,
            peak_price=entry,
            first_update_pending=False,
            stale_invalidated=_safe_bool(row.get("stale_invalidated", "false")),
        )

        # Restore last_funding_ts if present
        lf_ts = _parse_dt(row.get("last_funding_ts", ""))
        if lf_ts:
            pos.last_funding_ts = lf_ts

        exit_requested_at = _parse_dt(row.get("exit_requested_at", ""))
        if exit_requested_at:
            pos.exit_requested_at = exit_requested_at

        wallet_confirmed_at = _parse_dt(row.get("wallet_flat_confirmed_at", ""))
        if wallet_confirmed_at:
            pos.wallet_flat_confirmed_at = wallet_confirmed_at

        pending_exit_recorded_at = _parse_dt(row.get("pending_exit_recorded_at", ""))
        if pending_exit_recorded_at:
            pos.pending_exit_recorded_at = pending_exit_recorded_at

        protection_placed_at = _parse_dt(row.get("protection_placed_at", ""))
        if protection_placed_at:
            pos.protection_placed_at = protection_placed_at

        return pos

    except Exception as e:
        print(f"[BOOTSTRAP] Could not rebuild position {row.get('position_id', '?')}: {e}")
        return None

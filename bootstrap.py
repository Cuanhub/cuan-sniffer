# bootstrap.py
"""
State reconstruction from trades.csv on boot.

Reads the trade log and rebuilds:
  - realized balance (STARTING_BALANCE + all closed pnl_usd)
  - high-water mark  (peak balance ever reached, walked chronologically)
  - daily R          (realized_r for trades closed today UTC)
  - open positions   (state=open|partial → rebuilt Position objects)
  - strategy filter  (outcome replay → kill-switch state restored)

Called once during Executor.__init__() before any signals are processed.
Never raises — degrades gracefully to a fresh STARTING_BALANCE if anything
goes wrong.

FIX: restores original_stop_price and original_tp_price from CSV
     so R calculations remain correct for restored partial positions.
FIX: cancelled/stale trades excluded from balance reconstruction.
FIX: stale_invalidated and first_update_pending restored.
"""

import csv
import os
from dataclasses import dataclass
from datetime import datetime, timezone, date
from typing import List, Dict, Optional, Tuple

from position import Position, PositionState, CloseReason

TRADES_FILE = os.getenv("TRADES_FILE", "trades.csv")


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


# ── CSV loader ─────────────────────────────────────────────────────────────────

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


# ── Main bootstrap function ────────────────────────────────────────────────────

def bootstrap_state(starting_balance: float) -> BootstrapResult:
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

    def sort_key(r: Dict) -> datetime:
        dt = _parse_dt(r.get("opened_at", ""))
        return dt or datetime.min.replace(tzinfo=timezone.utc)

    rows_sorted = sorted(rows, key=sort_key)

    running_balance = starting_balance
    high_water_mark = starting_balance
    daily_r = 0.0
    daily_pnl = 0.0
    total_closed = 0
    closed_today = 0
    open_positions: List[Position] = []

    # Deduplicate — keep latest version of each position
    seen_ids: Dict[str, Dict] = {}
    for row in rows_sorted:
        pid = row.get("position_id", "")
        if pid:
            seen_ids[pid] = row

    deduped_sorted = sorted(seen_ids.values(), key=sort_key)

    for row in deduped_sorted:
        state = row.get("state", "")
        pnl = _safe_float(row.get("pnl_usd", "0"))

        # FIX: skip cancelled/stale — they have zero P&L and shouldn't
        # affect balance or trade counts
        stale = _safe_bool(row.get("stale_invalidated", "false"))

        if state == "cancelled" or stale:
            continue

        if state == "closed":
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
            pos = _rebuild_position(row)
            if pos is not None:
                open_positions.append(pos)

    all_time_pnl = running_balance - starting_balance

    n_open = len(open_positions)
    print(
        f"[BOOTSTRAP] Balance reconstructed: ${running_balance:.2f} "
        f"(start ${starting_balance:.2f}, P&L ${all_time_pnl:+.2f})"
    )
    print(
        f"[BOOTSTRAP] HWM: ${high_water_mark:.2f} | "
        f"Closed all-time: {total_closed} | "
        f"Today: {closed_today} trades, {daily_r:+.2f}R | "
        f"Open positions restored: {n_open}"
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
    )


def replay_strategy_filter(strategy_filter, starting_balance: float) -> None:
    rows = _load_all_trades()
    if not rows:
        return

    def sort_key(r: Dict) -> datetime:
        dt = _parse_dt(r.get("opened_at", ""))
        return dt or datetime.min.replace(tzinfo=timezone.utc)

    rows_sorted = sorted(rows, key=sort_key)

    seen: Dict[str, Dict] = {}
    for row in rows_sorted:
        pid = row.get("position_id", "")
        if pid:
            seen[pid] = row

    replayed = 0
    for row in sorted(seen.values(), key=sort_key):
        state = row.get("state", "")
        # FIX: skip cancelled/stale — they're not real outcomes
        stale = _safe_bool(row.get("stale_invalidated", "false"))

        if state != "closed" or stale:
            continue
        coin = row.get("coin", "")
        setup_family = row.get("setup_family", "")
        htf_regime = row.get("htf_regime", "")
        realized_r = _safe_float(row.get("realized_r", "0"))
        won = realized_r > 0

        if coin and setup_family:
            strategy_filter.record_outcome(coin, setup_family, htf_regime, won)
            replayed += 1

    if replayed:
        print(f"[BOOTSTRAP] Strategy filter replayed {replayed} trade outcomes")


# ── Position reconstruction ────────────────────────────────────────────────────

def _rebuild_position(row: Dict) -> Optional[Position]:
    """
    Reconstruct a Position from a trades.csv row.

    FIX: restores original_stop_price and original_tp_price from CSV.
    Without these, a restored partial position has stop_price=entry (breakeven)
    and __post_init__ would set original_stop_price=entry, making stop_dist=0
    and breaking all R calculations.

    FIX: restores stale_invalidated and first_update_pending.
    Restored positions should NOT be re-checked for staleness.
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

        # FIX: Restore original levels from CSV
        # If the CSV has original_stop_price, use it.
        # If not (legacy rows), fall back to stop_price for non-partial,
        # or we can't recover the original for partial positions (log a warning).
        original_stop = _safe_float(row.get("original_stop_price"))
        original_tp = _safe_float(row.get("original_tp_price"))

        is_partial = _safe_bool(row.get("partial_closed", "false"))
        is_breakeven = _safe_bool(row.get("breakeven_moved", "false"))

        if original_stop <= 0:
            if is_breakeven or is_partial:
                # stop_price is already at entry — we've lost the original
                # This only happens with legacy CSV rows before the fix
                print(
                    f"[BOOTSTRAP] WARNING: {row.get('position_id','?')} "
                    f"partial/BE position missing original_stop_price — "
                    f"R calculations may be incorrect"
                )
                # Best effort: use r_value / size_usd to back-compute stop distance
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
            # FIX: pass original levels explicitly so __post_init__
            # doesn't overwrite them with the mutated stop_price
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
            regime=row.get("regime", ""),
            setup_family=row.get("setup_family", ""),
            htf_regime=row.get("htf_regime", ""),
            confidence=_safe_float(row.get("confidence")),
            total_score=_safe_float(row.get("total_score")),
            session=row.get("session", ""),
            opened_at=opened_at,
            current_price=entry,
            peak_price=entry,
            # FIX: restored positions are NOT new — don't re-check for staleness
            first_update_pending=False,
            stale_invalidated=_safe_bool(row.get("stale_invalidated", "false")),
        )
        return pos

    except Exception as e:
        print(f"[BOOTSTRAP] Could not rebuild position {row.get('position_id', '?')}: {e}")
        return None
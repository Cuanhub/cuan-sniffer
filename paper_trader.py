# paper_trader.py
"""
Paper trading engine — simulates order lifecycle against live Hyperliquid prices.

FIX: partial closes now persisted to trade_log immediately.
FIX: runner_r tracked separately from partial_r (no double-count confusion).
FIX: strategy_filter.record_outcome() called on every close.
"""

import os
import requests
from datetime import datetime, timezone
from typing import Dict, List, Optional, Callable

from position import Position, PositionState, CloseReason
from trade_log import append_trade

HYPERLIQUID_INFO_URL = "https://api.hyperliquid.xyz/info"
TRAIL_ATR_MULT       = float(os.getenv("TRAIL_ATR_MULT", "1.0"))
PAPER_MODE           = os.getenv("PAPER_MODE", "true").lower() == "true"


def _fetch_mid_price(coin: str) -> Optional[float]:
    try:
        resp = requests.post(
            HYPERLIQUID_INFO_URL,
            json={"type": "allMids"},
            timeout=5,
        )
        resp.raise_for_status()
        mids = resp.json()
        if isinstance(mids, dict) and coin in mids:
            return float(mids[coin])
    except Exception as e:
        print(f"[PAPER] price fetch failed {coin}: {e}")
    return None


class PaperTrader:
    def __init__(
        self,
        notify_fn: Optional[Callable[[str], None]] = None,
        strategy_filter=None,
    ):
        self.positions:       Dict[str, Position] = {}
        self.closed:          List[Position]      = []
        self.notify           = notify_fn or (lambda msg: print(f"[PAPER] {msg}"))
        self.strategy_filter  = strategy_filter

    # ── Open ──────────────────────────────────────────────────────────────────

    def open_position(self, position: Position):
        self.positions[position.coin] = position
        mult_tag = f" ({position.size_multiplier:.2f}x)" if position.size_multiplier != 1.0 else ""
        print(f"[PAPER] OPEN {position.coin} {position.side} "
              f"entry={position.entry_price:.4f} "
              f"stop={position.stop_price:.4f} "
              f"tp={position.tp_price:.4f} "
              f"size=${position.size_usd:.0f}{mult_tag}")

        self.notify(
            f"📋 *[PAPER] {position.coin} {position.side} OPENED*\n\n"
            f"💰 Entry: `{position.entry_price:.4f}`\n"
            f"🛑 Stop:  `{position.stop_price:.4f}`\n"
            f"🎯 TP:    `{position.tp_price:.4f}`\n"
            f"📊 R/R:   `{position.rr_planned:.2f}R`\n"
            f"💵 Size:  `${position.size_usd:.0f}` ({position.size_multiplier:.2f}x)  "
            f"Risk: `${position.risk_usd:.2f}`\n"
            f"🧠 Regime: `{position.regime}`\n"
            f"⚡ Score:  `{position.total_score:.2f}`"
        )

    # ── Update cycle ──────────────────────────────────────────────────────────

    def update_all(self, risk_manager=None) -> List[Position]:
        closed_this_cycle: List[Position] = []

        for coin, pos in list(self.positions.items()):
            price = _fetch_mid_price(coin)
            if price is None:
                continue
            pos.update_price(price)
            result = self._evaluate_position(pos, price, risk_manager)
            if result == "closed":
                closed_this_cycle.append(pos)
                del self.positions[coin]
                self.closed.append(pos)

        return closed_this_cycle

    def _evaluate_position(self, pos: Position, price: float, risk_manager=None) -> str:
        side = pos.side

        stop_hit = (
            (side == "LONG"  and price <= pos.stop_price) or
            (side == "SHORT" and price >= pos.stop_price)
        )

        if stop_hit:
            if pos.partial_closed:
                # Runner stopped — partial already banked, runner at breakeven = 0R
                self._close(pos, price, CloseReason.STOP_RUNNER,
                            runner_r=0.0, risk_manager=risk_manager)
            else:
                # Full stop loss before any partial
                self._close_full_loss(pos, price, risk_manager)
            return "closed"

        if not pos.partial_closed:
            if pos.current_r(price) >= 1.0:
                self._take_partial(pos, price, risk_manager)

        if pos.partial_closed:
            tp_hit = (
                (side == "LONG"  and price >= pos.tp_price) or
                (side == "SHORT" and price <= pos.tp_price)
            )
            if tp_hit:
                runner_r = pos.current_r(price)
                self._close(pos, price, CloseReason.TP_FULL,
                            runner_r=runner_r, risk_manager=risk_manager)
                return "closed"
            self._update_trailing_stop(pos, price)

        return "open"

    def _take_partial(self, pos: Position, price: float, risk_manager=None):
        """50% off at +1R, stop to breakeven. FIX: persisted immediately."""
        pos.partial_r    += 1.0 * 0.5           # lock +0.5R
        pos.pnl_usd      += 1.0 * pos.r_value * 0.5
        pos.partial_closed = True
        pos.state          = PositionState.PARTIAL
        pos.move_stop_to_breakeven()

        if risk_manager:
            risk_manager.record_partial(pos, r_gained=1.0)

        # FIX: persist partial state immediately so it's not lost
        append_trade(pos, paper_mode=PAPER_MODE)

        print(f"[PAPER] PARTIAL {pos.coin} +1R at {price:.4f} "
              f"→ stop BE={pos.stop_price:.4f}")
        self.notify(
            f"⚡ *[PAPER] {pos.coin} PARTIAL CLOSE*\n\n"
            f"✅ +1R taken (50% off)\n"
            f"🛑 Stop → breakeven: `{pos.stop_price:.4f}`\n"
            f"🎯 Runner targeting: `{pos.tp_price:.4f}`\n"
            f"💵 Locked: `+${pos.r_value * 0.5:.2f}`"
        )

    def _update_trailing_stop(self, pos: Position, price: float):
        trail_dist = TRAIL_ATR_MULT * pos.atr
        if pos.side == "LONG":
            new_stop = pos.peak_price - trail_dist
            if new_stop > pos.stop_price:
                pos.stop_price = new_stop
        else:
            new_stop = pos.peak_price + trail_dist
            if new_stop < pos.stop_price:
                pos.stop_price = new_stop

    def _close_full_loss(self, pos: Position, price: float, risk_manager=None):
        """Full stop loss — no partial was taken."""
        pos.runner_r   = 0.0
        pos.pnl_usd   -= pos.r_value          # -1R in USD
        pos.state      = PositionState.CLOSED
        pos.close_reason = CloseReason.STOP_FULL
        pos.closed_at  = datetime.now(timezone.utc)
        pos.current_price = price

        if risk_manager:
            risk_manager.record_full_loss(pos)

        if self.strategy_filter:
            self.strategy_filter.record_outcome(
                pos.coin, pos.setup_family, pos.htf_regime, won=False
            )

        append_trade(pos, paper_mode=PAPER_MODE)

        print(f"[PAPER] CLOSE {pos.coin} STOP_FULL at {price:.4f} "
              f"realized_r=-1.0 pnl=${pos.pnl_usd:+.2f}")
        self.notify(
            f"❌ *[PAPER] {pos.coin} STOPPED OUT*\n\n"
            f"🛑 Total R: `-1.00R`\n"
            f"💵 P&L: `${pos.pnl_usd:+.2f}`\n"
            f"📍 Price: `{price:.4f}`\n"
            f"⏱ Duration: `{self._duration(pos)}`"
        )

    def _close(self, pos: Position, price: float, reason: CloseReason,
               runner_r: float, risk_manager=None):
        """Close after a partial was already taken."""
        pos.runner_r   += runner_r * 0.5      # runner is 50% of original size
        pos.pnl_usd    += runner_r * pos.r_value * 0.5
        pos.state       = PositionState.CLOSED
        pos.close_reason = reason
        pos.closed_at   = datetime.now(timezone.utc)
        pos.current_price = price

        if risk_manager:
            risk_manager.record_close(pos, runner_r)

        won = pos.realized_r > 0
        if self.strategy_filter:
            self.strategy_filter.record_outcome(
                pos.coin, pos.setup_family, pos.htf_regime, won=won
            )

        append_trade(pos, paper_mode=PAPER_MODE)

        tag    = reason.value.replace("_", " ").upper()
        emoji  = "✅" if won else "⚠️"
        total_r = pos.realized_r

        print(f"[PAPER] CLOSE {pos.coin} {tag} at {price:.4f} "
              f"total_r={total_r:+.2f}R pnl=${pos.pnl_usd:+.2f}")
        self.notify(
            f"{emoji} *[PAPER] {pos.coin} CLOSED — {tag}*\n\n"
            f"📊 Total R: `{total_r:+.2f}R`  "
            f"(partial: `{pos.partial_r:+.2f}R` + runner: `{pos.runner_r:+.2f}R`)\n"
            f"💵 P&L: `${pos.pnl_usd:+.2f}`\n"
            f"📍 Close: `{price:.4f}`  Entry: `{pos.entry_price:.4f}`\n"
            f"⏱ Duration: `{self._duration(pos)}`"
        )

    def _duration(self, pos: Position) -> str:
        if not pos.closed_at:
            return "?"
        delta = pos.closed_at - pos.opened_at
        h, r  = divmod(int(delta.total_seconds()), 3600)
        return f"{h}h {r // 60}m"

    def open_count(self) -> int:
        return len(self.positions)

    def session_summary(self) -> str:
        if not self.closed:
            return "No closed trades this session."
        wins    = [p for p in self.closed if p.realized_r > 0]
        total_r = sum(p.realized_r for p in self.closed)
        wr      = len(wins) / len(self.closed) * 100
        return (f"{len(self.closed)} trades  "
                f"W/L: {len(wins)}/{len(self.closed)-len(wins)}  "
                f"WR: {wr:.0f}%  Total R: {total_r:+.2f}R")
"""
Paper trading engine — active-path corrected version.

Key fixes:
- fees and funding are now applied in the runtime chain
- entry fee is carried on the position from executor
- funding accrues during update_all()
- partial / close / full-stop subtract exit fee from pnl and risk manager balance
- original-stop-based R logic retained
- stop-before-stale retained
"""

import os
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

from position import Position, PositionState, CloseReason
from trade_log import append_trade

TRAIL_ATR_MULT = float(os.getenv("TRAIL_ATR_MULT", "1.0"))
PAPER_MODE = os.getenv("PAPER_MODE", "true").lower() == "true"
STALE_R_THRESHOLD = float(os.getenv("STALE_R_THRESHOLD", "1.5"))
MAX_FULL_LOSS_R = float(os.getenv("MAX_FULL_LOSS_R", "-1.5"))


def _r_vs_original(pos: Position, price: float) -> float:
    original_stop = getattr(pos, "original_stop_price", None) or pos.stop_price
    stop_dist = abs(pos.entry_price - original_stop)
    if stop_dist == 0:
        return 0.0
    if pos.side == "LONG":
        return (price - pos.entry_price) / stop_dist
    return (pos.entry_price - price) / stop_dist


class PaperTrader:
    def __init__(
        self,
        notify_fn: Optional[Callable[[str], None]] = None,
        strategy_filter=None,
        backend=None,
    ):
        self.positions: Dict[str, Position] = {}
        self.closed: List[Position] = []
        self.notify = notify_fn or (lambda msg: print(f"[PAPER] {msg}"))
        self.strategy_filter = strategy_filter
        self.backend = backend

    def _get_price(self, coin: str) -> Optional[float]:
        if self.backend is not None:
            return self.backend.get_mid_price(coin)
        return None

    def _execute_exit(
        self,
        coin: str,
        side: str,
        price: float,
        size_usd: float,
        reason: str = "",
        executor=None,
    ):
        if self.backend is not None:
            fill = self.backend.execute_exit(
                coin=coin,
                side=side,
                price=price,
                size_usd=size_usd,
                reason=reason,
            )
            if fill.filled:
                if executor is not None and hasattr(executor, "record_exit_slippage"):
                    executor.record_exit_slippage(fill.slippage_bps)
                return fill
        return None

    def _apply_funding_if_due(self, pos: Position, risk_manager=None):
        if self.backend is None or not hasattr(self.backend, "accrue_funding"):
            return

        # After partial, only runner half remains open
        open_notional = pos.size_usd * (0.5 if pos.partial_closed else 1.0)
        side = str(pos.side).upper()
        result = self.backend.accrue_funding(
            coin=pos.coin,
            side=side,
            notional_usd=open_notional,
            last_funding_ts=getattr(pos, "last_funding_ts", None),
        )
        funding_usd = float(result.get("funding_usd", 0.0))
        pos.last_funding_ts = result.get("last_funding_ts", datetime.now(timezone.utc))

        if funding_usd != 0.0:
            pos.funding_usd = float(getattr(pos, "funding_usd", 0.0)) + funding_usd
            pos.pnl_usd -= funding_usd
            if risk_manager:
                risk_manager.apply_funding(pos, funding_usd)

    def open_position(self, position: Position):
        self.positions[position.coin] = position
        mult_tag = f" ({position.size_multiplier:.2f}x)" if position.size_multiplier != 1.0 else ""
        fee_tag = ""
        entry_fee = float(getattr(position, "entry_fee_usd", 0.0) or 0.0)
        if entry_fee > 0:
            fee_tag = f"\n💸 Entry fee: `${entry_fee:.2f}`"

        print(
            f"[PAPER] OPEN {position.coin} {position.side} "
            f"entry={position.entry_price:.4f} "
            f"stop={position.stop_price:.4f} "
            f"tp={position.tp_price:.4f} "
            f"size=${position.size_usd:.0f}{mult_tag}"
        )
        self.notify(
            f"📋 *[PAPER] {position.coin} {position.side} OPENED*\n\n"
            f"💰 Entry: `{position.entry_price:.4f}`\n"
            f"🛑 Stop: `{position.stop_price:.4f}`\n"
            f"🎯 TP: `{position.tp_price:.4f}`\n"
            f"📊 R/R: `{position.rr_planned:.2f}R`\n"
            f"💵 Size: `${position.size_usd:.0f}` ({position.size_multiplier:.2f}x)  "
            f"Risk: `${position.risk_usd:.2f}`\n"
            f"🧠 Regime: `{position.regime}`\n"
            f"⚡ Score: `{position.total_score:.2f}`"
            f"{fee_tag}"
        )

    def update_all(self, risk_manager=None, executor=None) -> List[Position]:
        closed_this_cycle: List[Position] = []

        for coin, pos in list(self.positions.items()):
            price = self._get_price(coin)
            if price is None:
                continue

            self._apply_funding_if_due(pos, risk_manager)
            pos.update_price(price)
            result = self._evaluate_position(pos, price, risk_manager, executor)

            if result in ("closed", "cancelled"):
                closed_this_cycle.append(pos)
                del self.positions[coin]
                self.closed.append(pos)

        return closed_this_cycle

    def _evaluate_position(
        self, pos: Position, price: float, risk_manager=None, executor=None
    ) -> str:
        side = pos.side

        stop_hit = (
            (side == "LONG" and price <= pos.stop_price) or
            (side == "SHORT" and price >= pos.stop_price)
        )

        if stop_hit:
            pos.first_update_pending = False
            if pos.partial_closed:
                self._close(pos, price, CloseReason.STOP_RUNNER, risk_manager, executor)
            else:
                self._close_full_loss(pos, price, risk_manager, executor)
            return "closed"

        if pos.first_update_pending:
            pos.first_update_pending = False
            first_r = _r_vs_original(pos, price)
            if first_r >= STALE_R_THRESHOLD or first_r <= -STALE_R_THRESHOLD:
                self._invalidate_stale_signal(pos, price, first_r, risk_manager)
                return "cancelled"

        if not pos.partial_closed and pos.current_r(price) >= 1.0:
            self._take_partial(pos, price, risk_manager, executor)

        if pos.partial_closed:
            tp_hit = (
                (side == "LONG" and price >= pos.tp_price) or
                (side == "SHORT" and price <= pos.tp_price)
            )
            if tp_hit:
                self._close(pos, price, CloseReason.TP_FULL, risk_manager, executor)
                return "closed"

            self._update_trailing_stop(pos, price)

        return "open"

    def _invalidate_stale_signal(
        self, pos: Position, price: float, first_r: float, risk_manager=None
    ):
        pos.invalidate_as_stale()

        if risk_manager and hasattr(risk_manager, "record_cancelled"):
            risk_manager.record_cancelled(pos)
        elif risk_manager:
            risk_manager.open_positions.pop(pos.coin, None)

        append_trade(pos, paper_mode=PAPER_MODE)

        print(
            f"[PAPER] CANCEL {pos.coin} STALE_SIGNAL "
            f"first_price={price:.4f} first_r={first_r:+.2f}R "
            f"(threshold=±{STALE_R_THRESHOLD}R)"
        )
        self.notify(
            f"🟡 *[PAPER] {pos.coin} SIGNAL INVALIDATED*\n\n"
            f"Reason: `stale_signal`\n"
            f"First price after entry was already `{first_r:+.2f}R` "
            f"(threshold ±`{STALE_R_THRESHOLD}R`)\n"
            f"No P&L counted."
        )

    def _take_partial(self, pos: Position, price: float, risk_manager=None, executor=None):
        partial_size = pos.size_usd * 0.5
        fill = self._execute_exit(
            coin=pos.coin,
            side=pos.side,
            price=price,
            size_usd=partial_size,
            reason="partial",
            executor=executor,
        )
        fill_price = fill.fill_price if fill and fill.filled else price
        exit_fee_usd = float((getattr(fill, "meta", None) or {}).get("estimated_fee_usd", 0.0)) if fill else 0.0

        gross_r = pos.current_r(fill_price)
        gross_partial_usd = gross_r * 0.5 * pos.r_value
        net_partial_usd = gross_partial_usd - exit_fee_usd
        net_partial_r = net_partial_usd / pos.r_value if pos.r_value > 0 else 0.0

        pos.partial_r += net_partial_r
        pos.pnl_usd += gross_partial_usd - exit_fee_usd
        pos.partial_closed = True
        pos.state = PositionState.PARTIAL
        pos.move_stop_to_breakeven()
        pos.exit_fees_usd = float(getattr(pos, "exit_fees_usd", 0.0)) + exit_fee_usd
        pos.total_fees_usd = float(getattr(pos, "total_fees_usd", 0.0)) + exit_fee_usd

        if risk_manager:
            risk_manager.record_partial(pos, gross_r=gross_r, fee_usd=exit_fee_usd)

        append_trade(pos, paper_mode=PAPER_MODE)

        print(
            f"[PAPER] PARTIAL {pos.coin} gross={gross_r:+.2f}R "
            f"(net realized {net_partial_r:+.2f}R) at {fill_price:.4f} "
            f"fee=${exit_fee_usd:.2f} → stop BE={pos.stop_price:.4f}"
        )
        self.notify(
            f"⚡ *[PAPER] {pos.coin} PARTIAL CLOSE*\n\n"
            f"✅ Gross move: `{gross_r:+.2f}R`\n"
            f"💾 Net realized on 50%: `{net_partial_r:+.2f}R`\n"
            f"📍 Fill: `{fill_price:.4f}`\n"
            f"💸 Exit fee: `${exit_fee_usd:.2f}`\n"
            f"🛑 Stop → breakeven: `{pos.stop_price:.4f}`\n"
            f"🎯 Runner targeting: `{pos.tp_price:.4f}`\n"
            f"💵 Locked: `+${net_partial_usd:.2f}`"
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

    def _close_full_loss(self, pos: Position, price: float, risk_manager=None, executor=None):
        fill = self._execute_exit(
            coin=pos.coin,
            side=pos.side,
            price=price,
            size_usd=pos.size_usd,
            reason="stop_full",
            executor=executor,
        )
        fill_price = fill.fill_price if fill and fill.filled else price
        exit_fee_usd = float((getattr(fill, "meta", None) or {}).get("estimated_fee_usd", 0.0)) if fill else 0.0

        actual_r_gross = _r_vs_original(pos, fill_price)
        actual_r_gross = max(actual_r_gross, MAX_FULL_LOSS_R)

        fee_r = exit_fee_usd / pos.r_value if pos.r_value > 0 else 0.0
        actual_r_net = actual_r_gross - fee_r

        pos.runner_r += actual_r_net
        pos.pnl_usd += actual_r_gross * pos.r_value - exit_fee_usd
        pos.state = PositionState.CLOSED
        pos.close_reason = CloseReason.STOP_FULL
        pos.closed_at = datetime.now(timezone.utc)
        pos.current_price = fill_price
        pos.exit_fees_usd = float(getattr(pos, "exit_fees_usd", 0.0)) + exit_fee_usd
        pos.total_fees_usd = float(getattr(pos, "total_fees_usd", 0.0)) + exit_fee_usd

        if risk_manager:
            risk_manager.record_full_loss(pos, realized_r=actual_r_gross, fee_usd=exit_fee_usd)

        if self.strategy_filter:
            self.strategy_filter.record_outcome(
                pos.coin,
                pos.setup_family,
                pos.htf_regime,
                won=False,
            )

        append_trade(pos, paper_mode=PAPER_MODE)

        original_stop = getattr(pos, "original_stop_price", pos.stop_price)
        print(
            f"[PAPER] CLOSE {pos.coin} STOP_FULL at {fill_price:.4f} "
            f"net_r={actual_r_net:+.2f} fee=${exit_fee_usd:.2f} pnl=${pos.pnl_usd:+.2f} "
            f"(original_stop={original_stop:.4f})"
        )
        self.notify(
            f"❌ *[PAPER] {pos.coin} STOPPED OUT*\n\n"
            f"🛑 Net R: `{actual_r_net:+.2f}R`\n"
            f"💸 Exit fee: `${exit_fee_usd:.2f}`\n"
            f"💵 P&L: `${pos.pnl_usd:+.2f}`\n"
            f"📍 Fill: `{fill_price:.4f}` (original stop `{original_stop:.4f}`)\n"
            f"⏱ Duration: `{self._duration(pos)}`"
        )

    def _close(
        self,
        pos: Position,
        price: float,
        reason: CloseReason,
        risk_manager=None,
        executor=None,
    ):
        fill = self._execute_exit(
            coin=pos.coin,
            side=pos.side,
            price=price,
            size_usd=pos.size_usd * 0.5,
            reason=reason.value,
            executor=executor,
        )
        fill_price = fill.fill_price if fill and fill.filled else price
        exit_fee_usd = float((getattr(fill, "meta", None) or {}).get("estimated_fee_usd", 0.0)) if fill else 0.0

        actual_runner_r_gross = _r_vs_original(pos, fill_price)
        gross_runner_usd = actual_runner_r_gross * 0.5 * pos.r_value
        net_runner_usd = gross_runner_usd - exit_fee_usd
        net_runner_r = net_runner_usd / pos.r_value if pos.r_value > 0 else 0.0

        pos.runner_r += net_runner_r
        pos.pnl_usd += gross_runner_usd - exit_fee_usd
        pos.state = PositionState.CLOSED
        pos.close_reason = reason
        pos.closed_at = datetime.now(timezone.utc)
        pos.current_price = fill_price
        pos.exit_fees_usd = float(getattr(pos, "exit_fees_usd", 0.0)) + exit_fee_usd
        pos.total_fees_usd = float(getattr(pos, "total_fees_usd", 0.0)) + exit_fee_usd

        if risk_manager:
            risk_manager.record_close(pos, runner_r=actual_runner_r_gross, fee_usd=exit_fee_usd)

        won = pos.realized_r > 0
        if self.strategy_filter:
            self.strategy_filter.record_outcome(
                pos.coin,
                pos.setup_family,
                pos.htf_regime,
                won=won,
            )

        append_trade(pos, paper_mode=PAPER_MODE)

        tag = reason.value.replace("_", " ").upper()
        emoji = "✅" if won else "⚠️"
        total_r = pos.realized_r

        print(
            f"[PAPER] CLOSE {pos.coin} {tag} at {fill_price:.4f} "
            f"total_r={total_r:+.2f} fee=${exit_fee_usd:.2f} pnl=${pos.pnl_usd:+.2f}"
        )
        self.notify(
            f"{emoji} *[PAPER] {pos.coin} {tag}*\n\n"
            f"📍 Fill: `{fill_price:.4f}`\n"
            f"🧮 Total R: `{total_r:+.2f}R`\n"
            f"💸 Exit fee: `${exit_fee_usd:.2f}`\n"
            f"💵 P&L: `${pos.pnl_usd:+.2f}`\n"
            f"⏱ Duration: `{self._duration(pos)}`"
        )

    def _duration(self, pos: Position) -> str:
        end = pos.closed_at or datetime.now(timezone.utc)
        delta = end - pos.opened_at
        mins = int(delta.total_seconds() // 60)
        hrs = mins // 60
        mins = mins % 60
        return f"{hrs}h {mins}m"

    def status_lines(self) -> List[str]:
        if not self.positions:
            return ["No open positions."]
        lines: List[str] = []
        for coin, pos in self.positions.items():
            r_now = pos.current_r(pos.current_price) if pos.current_price > 0 else 0.0
            lines.append(
                f"{coin} {pos.side} entry={pos.entry_price:.4f} "
                f"stop={pos.stop_price:.4f} tp={pos.tp_price:.4f} "
                f"R={r_now:+.2f} state={pos.state.value}"
            )
        return lines

    def session_summary(self) -> str:
        closed_n = len(self.closed)
        wins = sum(1 for p in self.closed if p.realized_r > 0)
        losses = sum(1 for p in self.closed if p.realized_r < 0)
        total_pnl = sum(p.pnl_usd for p in self.closed)
        return (
            f"Closed: {closed_n} | Wins: {wins} | Losses: {losses} | "
            f"Session P&L: ${total_pnl:+.2f}"
        )
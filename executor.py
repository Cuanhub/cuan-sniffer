# executor.py
"""
Order executor — single entry point for all trade decisions.

Boot-time state reconstruction:
    Reads trades.csv on startup and reconstructs:
      - balance         = STARTING_BALANCE + sum(closed pnl_usd)
      - high_water_mark = peak equity across all historical trades
      - daily_r         = realized R for trades closed today (UTC)
      - open positions  = restored into PaperTrader for continued monitoring
      - closed history  = restored for session_summary accuracy

PAPER_MODE=true  → paper trading (default)
PAPER_MODE=false → live (placeholder — wire Hyperliquid SDK here)
"""

import os
import time
from datetime import datetime, timezone
from typing import Optional, Callable

from position import Position, PositionState
from risk_manager import RiskManager, PAPER_MODE, STARTING_BALANCE, MAX_OPEN_POSITIONS
from paper_trader import PaperTrader
from trade_log import init_trade_log, append_trade
from strategy_filter import StrategyFilter
from bootstrap import bootstrap_state, replay_strategy_filter

PAPER_MODE_ACTIVE = PAPER_MODE


class Executor:
    """
    Wire into agent.py:

        executor = Executor(notify_fn=notify)

        # In main() startup:
        notify(executor.boot_status_message())

        # After signal fires:
        executor.on_signal(sig, sig_id=int(time.time()*1000))

        # Every cycle:
        executor.update()
    """

    def __init__(self, notify_fn: Optional[Callable[[str], None]] = None):
        self.notify          = notify_fn or print
        self.strategy_filter = StrategyFilter()
        self.risk            = RiskManager(strategy_filter=self.strategy_filter)
        self.paper           = PaperTrader(
            notify_fn=notify_fn,
            strategy_filter=self.strategy_filter,
        )
        init_trade_log()

        # ── Boot-time state reconstruction from trades.csv ───────────────────
        _b = bootstrap_state(STARTING_BALANCE)
        replay_strategy_filter(self.strategy_filter, STARTING_BALANCE)

        # Restore RiskManager
        self.risk.balance         = _b.realized_balance
        self.risk.high_water_mark = _b.high_water_mark
        self.risk.daily_r         = _b.daily_r

        # Restore open/partial positions into PaperTrader + RiskManager
        for pos in _b.open_positions:
            self.paper.positions[pos.coin] = pos
            self.risk.open_positions[pos.coin] = pos
            print(f"[BOOTSTRAP] Restored {pos.coin} {pos.side} "
                  f"entry={pos.entry_price:.4f} state={pos.state.value}")

        # Cache boot stats for boot_status_message()
        self._boot_balance        = _b.realized_balance
        self._boot_hwm            = _b.high_water_mark
        self._boot_daily_r        = _b.daily_r
        self._boot_daily_pnl      = _b.daily_pnl
        self._boot_open           = _b.open_positions
        self._boot_all_time_pnl   = _b.all_time_pnl
        self._boot_closed_today   = _b.closed_today
        self._boot_total_closed   = _b.total_trades

        mode = "PAPER" if PAPER_MODE_ACTIVE else "LIVE"
        print(f"[EXECUTOR] {mode} | "
              f"Balance: ${_b.realized_balance:.2f} | "
              f"HWM: ${_b.high_water_mark:.2f} | "
              f"Open restored: {len(_b.open_positions)} | "
              f"Max: {MAX_OPEN_POSITIONS}")

    # ── Startup message ───────────────────────────────────────────────────────

    def boot_status_message(self) -> str:
        """
        Pro-grade startup Telegram message.
        Sent from agent.py main() after executor is initialised.
        Shows reconstructed balance, all open positions, today's P&L,
        and strategy filter state.
        """
        mode      = "📋 PAPER" if PAPER_MODE_ACTIVE else "🔴 LIVE"
        pct       = (self._boot_all_time_pnl / STARTING_BALANCE * 100
                     if STARTING_BALANCE > 0 else 0.0)
        dd_pct    = ((self._boot_hwm - self._boot_balance) / self._boot_hwm * 100
                     if self._boot_hwm > 0 else 0.0)

        lines = [
            f"🚀 *Cuan Sniffer — {mode}*",
            "",
            "```",
            f"  Balance     : ${self._boot_balance:.2f}  ({pct:+.1f}%)",
            f"  All-time P&L: ${self._boot_all_time_pnl:+.2f}",
            f"  HWM         : ${self._boot_hwm:.2f}",
            f"  Drawdown    : {dd_pct:.1f}%",
            f"  Total closed: {self._boot_total_closed} trades",
            f"  Today       : {self._boot_closed_today} closed  "
            f"{self._boot_daily_r:+.2f}R  ${self._boot_daily_pnl:+.2f}",
            "```",
        ]

        # Open positions
        open_list = list(self.paper.positions.values())
        if open_list:
            lines.append("")
            lines.append(f"*📂 {len(open_list)} open position(s) restored:*")
            lines.append("```")
            for pos in open_list:
                state_tag = "⚡partial" if pos.partial_closed else "open"
                lines.append(
                    f"  {pos.coin:<6} {pos.side:<5}  "
                    f"entry={pos.entry_price:.4f}  "
                    f"stop={pos.stop_price:.4f}  "
                    f"[{state_tag}]"
                )
            lines.append("```")
        else:
            lines.append("\n_No open positions_")

        # Strategy filter state
        sf = self.strategy_filter.summary()
        if sf.strip() not in ("No data yet.", ""):
            lines.append(f"\n*🔒 Filters:*\n```\n{sf}\n```")

        lines.append(f"\n🕒 `{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}`")
        return "\n".join(lines)

    # ── Signal intake ─────────────────────────────────────────────────────────

    def on_signal(self, signal, sig_id: int = 0) -> bool:
        """Returns True if a position was opened."""
        decision = self.risk.check_signal(signal)

        if not decision.approved:
            print(f"[EXECUTOR] {signal.coin} {signal.side} REJECTED — {decision.reason}")
            return False

        print(f"[EXECUTOR] {signal.coin} {signal.side} APPROVED — {decision.reason}")
        position = self._build_position(signal, sig_id, decision)

        if PAPER_MODE_ACTIVE:
            self.paper.open_position(position)
            self.risk.record_open(position)
            append_trade(position, paper_mode=True)
            return True
        else:
            # Live execution placeholder
            print("[EXECUTOR] LIVE mode not yet implemented")
            self.paper.open_position(position)
            self.risk.record_open(position)
            append_trade(position, paper_mode=False)
            return True

    # ── Cycle update ──────────────────────────────────────────────────────────

    def update(self):
        """Call every agent cycle — checks stops/TPs on all open positions."""
        self.paper.update_all(risk_manager=self.risk)
        self.risk.reset_daily_halt()

    # ── Status (cycle-level) ──────────────────────────────────────────────────

    def status_message(self) -> str:
        lines = [
            "📋 *Paper Trader Status*\n",
            f"💰 `{self.risk.status_summary()}`\n",
        ]
        open_list = list(self.paper.positions.values())
        if open_list:
            lines.append("*Open positions:*")
            for pos in open_list:
                r_now = pos.current_r(pos.current_price) if pos.current_price > 0 else 0.0
                lines.append(
                    f"`{pos.coin} {pos.side}  "
                    f"entry={pos.entry_price:.4f}  "
                    f"R={r_now:+.2f}  "
                    f"({pos.state.value})`"
                )
        else:
            lines.append("_No open positions_")

        sf = self.strategy_filter.summary()
        if sf.strip() not in ("No data yet.", ""):
            lines.append(f"\n*Filters:*\n```\n{sf}\n```")

        lines.append(f"\n_{self.paper.session_summary()}_")
        return "\n".join(lines)

    # ── Build position ────────────────────────────────────────────────────────

    def _build_position(self, signal, sig_id: int, decision) -> Position:
        meta         = signal.meta or {}
        pos_id       = f"{signal.coin}_{int(datetime.now(timezone.utc).timestamp() * 1000)}"
        setup_family = meta.get("setup_family", meta.get("regime_local", ""))
        htf_regime   = meta.get("regime_htf_1h", "")

        return Position(
            position_id     = pos_id,
            coin            = signal.coin,
            side            = signal.side,
            signal_id       = sig_id,
            entry_price     = float(signal.entry_price),
            stop_price      = float(signal.stop_price),
            tp_price        = float(signal.tp_price),
            atr             = float(meta.get("atr", 0.0)),
            size_usd        = decision.size_usd,
            risk_usd        = decision.risk_usd,
            r_value         = decision.risk_usd,
            size_multiplier = decision.size_multiplier,
            peak_price      = float(signal.entry_price),
            regime          = signal.regime,
            setup_family    = setup_family,
            htf_regime      = htf_regime,
            confidence      = float(signal.confidence),
            total_score     = float(meta.get("total_score", 0.0)),
            session         = meta.get("session", ""),
        )
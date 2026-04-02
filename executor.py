"""
Order executor — single entry point for all trade decisions.

Active-path corrected version:
- entry fee is deducted immediately from risk manager balance
- position stores entry fees / funding / total fees
- continuation cap fix retained
- coin reentry per direction retained
"""

import csv
import os
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Deque, Dict, List, Optional, Tuple

from position import Position
from risk_manager import PAPER_MODE, STARTING_BALANCE, MAX_OPEN_POSITIONS, RiskManager
from paper_trader import PaperTrader
from trade_log import init_trade_log, append_trade
from strategy_filter import StrategyFilter
from bootstrap import bootstrap_state, replay_strategy_filter
from execution_backend import ExecutionBackend
from paper_execution_backend import PaperExecutionBackend

PAPER_MODE_ACTIVE = PAPER_MODE

SIGNAL_COOLDOWN_SEC = int(os.getenv("SIGNAL_COOLDOWN_SEC", "300"))

ENTRY_BUFFER_ATR_MULT = float(os.getenv("ENTRY_BUFFER_ATR_MULT", "0.30"))
MAX_MOVE_STOP_FRAC = float(os.getenv("MAX_MOVE_STOP_FRAC", "0.30"))

BLOCKED_SESSIONS = {
    s.strip().lower()
    for s in os.getenv("BLOCKED_SESSIONS", "dead_zone,ny_pm").split(",")
    if s.strip()
}

COIN_REENTRY_COOLDOWN_SEC = int(os.getenv("COIN_REENTRY_COOLDOWN_SEC", "1800"))
BUCKET_REENTRY_COOLDOWN_SEC = int(os.getenv("BUCKET_REENTRY_COOLDOWN_SEC", "900"))

MAX_RECENT_MOVE_ATR = float(os.getenv("MAX_RECENT_MOVE_ATR", "1.50"))
CONTINUATION_MAX_SIZE_MULT = float(os.getenv("CONTINUATION_MAX_SIZE_MULT", "1.00"))

MISSED_LOG_FILE = os.getenv("MISSED_LOG_FILE", "missed_signals.csv")
LOG_MISSED = os.getenv("LOG_MISSED", "true").lower() == "true"

MISSED_FIELDS = [
    "timestamp", "signal_id", "coin", "side", "entry_price", "stop_price",
    "tp_price", "confidence", "total_score", "session", "regime",
    "reject_reason", "current_price", "price_move_r",
]

MAJORS_BUCKET_COINS = {
    c.strip().upper()
    for c in os.getenv("FACTOR_BUCKET_MAJORS_COINS", "BTC,ETH").split(",")
    if c.strip()
}
SOL_BETA_BUCKET_COINS = {
    c.strip().upper()
    for c in os.getenv("FACTOR_BUCKET_SOL_BETA_COINS", "SOL,JUP,JTO,WIF,PYTH").split(",")
    if c.strip()
}
ALT_BETA_BUCKET_COINS = {
    c.strip().upper()
    for c in os.getenv("FACTOR_BUCKET_ALT_BETA_COINS", "ARB").split(",")
    if c.strip()
}

_MAJORS_TOTAL = int(os.getenv("MAX_BUCKET_MAJORS", "2"))
_SOL_BETA_TOTAL = int(os.getenv("MAX_BUCKET_SOL_BETA", "1"))
_ALT_BETA_TOTAL = int(os.getenv("MAX_BUCKET_ALT_BETA", "1"))
_OTHER_TOTAL = int(os.getenv("MAX_BUCKET_OTHER", "1"))

MAX_BUCKET_MAJORS_LONG = int(os.getenv("MAX_BUCKET_MAJORS_LONG", str(_MAJORS_TOTAL)))
MAX_BUCKET_MAJORS_SHORT = int(os.getenv("MAX_BUCKET_MAJORS_SHORT", str(_MAJORS_TOTAL)))
MAX_BUCKET_SOL_BETA_LONG = int(os.getenv("MAX_BUCKET_SOL_BETA_LONG", "1"))
MAX_BUCKET_SOL_BETA_SHORT = int(os.getenv("MAX_BUCKET_SOL_BETA_SHORT", "1"))
MAX_BUCKET_ALT_BETA_LONG = int(os.getenv("MAX_BUCKET_ALT_BETA_LONG", str(_ALT_BETA_TOTAL)))
MAX_BUCKET_ALT_BETA_SHORT = int(os.getenv("MAX_BUCKET_ALT_BETA_SHORT", str(_ALT_BETA_TOTAL)))
MAX_BUCKET_OTHER_LONG = int(os.getenv("MAX_BUCKET_OTHER_LONG", str(_OTHER_TOTAL)))
MAX_BUCKET_OTHER_SHORT = int(os.getenv("MAX_BUCKET_OTHER_SHORT", str(_OTHER_TOTAL)))


@dataclass
class ExecutorResult:
    traded: bool
    reason: str
    position_id: str = ""
    fill_price: float = 0.0
    fill_slippage_bps: float = 0.0
    fill_ratio: float = 0.0

    def __bool__(self) -> bool:
        return self.traded


class Executor:
    def __init__(
        self,
        notify_fn: Optional[Callable[[str], None]] = None,
        backend: Optional[ExecutionBackend] = None,
    ):
        self.notify = notify_fn or print
        self.backend = backend or PaperExecutionBackend()

        self.strategy_filter = StrategyFilter()
        self.risk = RiskManager(strategy_filter=self.strategy_filter)
        self.paper = PaperTrader(
            notify_fn=notify_fn,
            strategy_filter=self.strategy_filter,
            backend=self.backend,
        )
        init_trade_log()

        self._cooldowns: Dict[Tuple[str, str], float] = {}
        self._coin_last_fill_ts: Dict[Tuple[str, str], float] = {}
        self._bucket_last_fill_ts: Dict[Tuple[str, str], float] = {}
        self._session_entry_slippage_bps: List[float] = []
        self._session_exit_slippage_bps: List[float] = []
        self._recent_prices: Dict[str, Deque[Tuple[float, float]]] = {}

        boot = None
        replay_strategy_filter(self.strategy_filter, STARTING_BALANCE)

        self.risk.balance = STARTING_BALANCE
        self.risk.high_water_mark = STARTING_BALANCE
        self.risk.daily_r = 0.0


        self._boot_balance = STARTING_BALANCE
        self._boot_hwm = STARTING_BALANCE
        self._boot_daily_r = 0.0
        self._boot_daily_pnl = 0.0
        self._boot_open = []
        self._boot_all_time_pnl = 0.0
        self._boot_closed_today = 0
        self._boot_total_closed = 0

        mode = "PAPER" if PAPER_MODE_ACTIVE else "LIVE"
        print(
            f"[EXECUTOR] {mode} | "
            f"Balance: ${STARTING_BALANCE:.2f} | "
            f"HWM: ${STARTING_BALANCE:.2f} | "
            f"Open restored: {0} | "
            f"Max positions: {MAX_OPEN_POSITIONS} | "
            f"Bucket caps (long/short): "
            f"majors={MAX_BUCKET_MAJORS_LONG}/{MAX_BUCKET_MAJORS_SHORT}, "
            f"sol_beta={MAX_BUCKET_SOL_BETA_LONG}/{MAX_BUCKET_SOL_BETA_SHORT}, "
            f"alt_beta={MAX_BUCKET_ALT_BETA_LONG}/{MAX_BUCKET_ALT_BETA_SHORT}, "
            f"other={MAX_BUCKET_OTHER_LONG}/{MAX_BUCKET_OTHER_SHORT} | "
            f"Cooldown: {SIGNAL_COOLDOWN_SEC}s | "
            f"Coin reentry: {COIN_REENTRY_COOLDOWN_SEC}s (per direction) | "
            f"Bucket reentry: {BUCKET_REENTRY_COOLDOWN_SEC}s | "
            f"Entry buffer: {ENTRY_BUFFER_ATR_MULT}xATR | "
            f"Max move: {MAX_MOVE_STOP_FRAC}xstop_dist | "
            f"Blocked sessions: {','.join(sorted(BLOCKED_SESSIONS)) or 'none'}"
        )

    def boot_status_message(self) -> str:
        mode = "📋 PAPER" if PAPER_MODE_ACTIVE else "🔴 LIVE"
        pct = (
            self._boot_all_time_pnl / STARTING_BALANCE * 100
            if STARTING_BALANCE > 0 else 0.0
        )
        dd_pct = (
            (self._boot_hwm - self._boot_balance) / self._boot_hwm * 100
            if self._boot_hwm > 0 else 0.0
        )

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

        if self._boot_open:
            lines.append("")
            lines.append(f"*📂 {len(self._boot_open)} open position(s) restored:*")
            lines.append("```")
            for pos in self._boot_open:
                state_tag = "⚡partial" if pos.partial_closed else "open"
                lines.append(
                    f"  {pos.coin:<6} {str(pos.side):<5} "
                    f"entry={pos.entry_price:.4f} stop={pos.stop_price:.4f} [{state_tag}]"
                )
            lines.append("```")
        else:
            lines.append("\n_No open positions_")

        sf = self.strategy_filter.summary()
        if sf.strip() not in ("No data yet.", ""):
            lines.append(f"\n*🔒 Filters:*\n```\n{sf}\n```")

        lines.append(f"\n🕒 `{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}`")
        return "\n".join(lines)

    def on_signal(self, signal, sig_id: int = 0) -> ExecutorResult:
        signal_side = self._side_str(signal.side)
        meta = signal.meta or {}
        session = str(meta.get("session", "")).strip().lower()
        coin = str(signal.coin).upper()
        bucket = self._factor_bucket(coin)
        now = time.time()

        if session in BLOCKED_SESSIONS:
            reason = f"session_blocked:{session}"
            print(f"[EXECUTOR] {coin} {signal_side} REJECTED — {reason}")
            self._log_missed(signal, sig_id, reason)
            return ExecutorResult(traded=False, reason=reason)

        cooldown_key = (coin, signal_side)
        last_fired = self._cooldowns.get(cooldown_key, 0.0)
        if now - last_fired < SIGNAL_COOLDOWN_SEC:
            remaining = int(SIGNAL_COOLDOWN_SEC - (now - last_fired))
            reason = f"cooldown ({remaining}s remaining)"
            print(f"[EXECUTOR] {coin} {signal_side} COOLDOWN — {remaining}s")
            self._log_missed(signal, sig_id, reason)
            return ExecutorResult(traded=False, reason=reason)

        coin_fill_key = (coin, signal_side)
        recent_coin_fill = self._coin_last_fill_ts.get(coin_fill_key, 0.0)
        if now - recent_coin_fill < COIN_REENTRY_COOLDOWN_SEC:
            remaining = int(COIN_REENTRY_COOLDOWN_SEC - (now - recent_coin_fill))
            reason = f"coin_reentry_cooldown:{coin}:{signal_side} ({remaining}s remaining)"
            print(f"[EXECUTOR] {coin} {signal_side} REJECTED — {reason}")
            self._log_missed(signal, sig_id, reason)
            return ExecutorResult(traded=False, reason=reason)

        bucket_key = (bucket, signal_side)
        recent_bucket_fill = self._bucket_last_fill_ts.get(bucket_key, 0.0)
        if now - recent_bucket_fill < BUCKET_REENTRY_COOLDOWN_SEC:
            remaining = int(BUCKET_REENTRY_COOLDOWN_SEC - (now - recent_bucket_fill))
            reason = f"bucket_reentry_cooldown:{bucket}:{signal_side} ({remaining}s remaining)"
            print(f"[EXECUTOR] {coin} {signal_side} REJECTED — {reason}")
            self._log_missed(signal, sig_id, reason)
            return ExecutorResult(traded=False, reason=reason)

        dir_limit = self._bucket_dir_limit(bucket, signal_side)
        same_bucket_same_dir = sum(
            1
            for p in self.risk.open_positions.values()
            if self._factor_bucket(getattr(p, "coin", "")) == bucket
            and self._side_str(getattr(p, "side", "")) == signal_side
        )
        if same_bucket_same_dir >= dir_limit:
            reason = (
                f"bucket_dir_limit:{bucket}:{signal_side} "
                f"({same_bucket_same_dir}/{dir_limit})"
            )
            print(f"[EXECUTOR] {coin} {signal_side} REJECTED — {reason}")
            self._log_missed(signal, sig_id, reason)
            return ExecutorResult(traded=False, reason=reason)

        decision = self.risk.check_signal(signal)
        if not decision.approved:
            print(f"[EXECUTOR] {coin} {signal_side} REJECTED — {decision.reason}")
            self._log_missed(signal, sig_id, decision.reason)
            return ExecutorResult(traded=False, reason=decision.reason)

        setup_family = str(meta.get("setup_family", meta.get("regime_local", ""))).strip().lower()
        if setup_family == "continuation" and decision.size_multiplier > CONTINUATION_MAX_SIZE_MULT:
            original_mult = decision.size_multiplier
            scale = CONTINUATION_MAX_SIZE_MULT / original_mult
            decision.size_usd = round(decision.size_usd * scale, 2)
            decision.risk_usd = round(decision.risk_usd * scale, 2)
            decision.size_multiplier = CONTINUATION_MAX_SIZE_MULT
            decision.reason += f" | continuation_cap={CONTINUATION_MAX_SIZE_MULT:.2f}x"
            print(
                f"[EXECUTOR] {coin} continuation cap applied: "
                f"{original_mult:.2f}x → {CONTINUATION_MAX_SIZE_MULT:.2f}x "
                f"(scale={scale:.3f})"
            )

        entry_reject = self._validate_entry(signal)
        if entry_reject:
            print(f"[EXECUTOR] {coin} {signal_side} REJECTED — {entry_reject}")
            self._log_missed(signal, sig_id, entry_reject)
            return ExecutorResult(traded=False, reason=entry_reject)

        print(f"[EXECUTOR] {coin} {signal_side} APPROVED — {decision.reason}")

        fill = self.backend.execute_entry(
            coin=coin,
            side=signal_side,
            price=float(signal.entry_price),
            size_usd=decision.size_usd,
            confidence=float(signal.confidence),
        )

        if not fill.filled:
            reason = f"fill_rejected: {fill.reject_reason}"
            print(f"[EXECUTOR] {coin} FILL REJECTED — {fill.reject_reason}")
            self._log_missed(signal, sig_id, reason)
            return ExecutorResult(traded=False, reason=reason)

        self._session_entry_slippage_bps.append(fill.slippage_bps)

        position = self._build_position(
            signal=signal,
            sig_id=sig_id,
            decision=decision,
            fill_price=fill.fill_price,
            fill_size_usd=fill.fill_size_usd,
        )

        entry_fee_usd = 0.0
        if hasattr(fill, "meta") and isinstance(fill.meta, dict):
            entry_fee_usd = float(fill.meta.get("estimated_fee_usd", 0.0))
        elif hasattr(self.backend, "estimate_entry_fee"):
            entry_fee_usd = float(self.backend.estimate_entry_fee(fill.fill_size_usd))

        position.entry_fee_usd = round(entry_fee_usd, 8)
        position.exit_fees_usd = 0.0
        position.funding_usd = 0.0
        position.total_fees_usd = round(entry_fee_usd, 8)
        position.last_funding_ts = datetime.now(timezone.utc)

        # Entry fee is a real realized cost immediately
    
        slip_tag = f" (slip={fill.slippage_bps:.1f}bps)" if fill.slippage_bps > 0 else ""
        partial_tag = ""
        if fill.fill_ratio is not None and fill.fill_ratio < 0.99:
            partial_tag = f" [partial fill: {fill.fill_ratio * 100:.0f}%]"

        print(
            f"[EXECUTOR] FILLED {coin} {signal_side} "
            f"@ {fill.fill_price:.4f}{slip_tag}{partial_tag}"
        )

        self.paper.open_position(position)
        self.risk.record_open(position)
        self.risk.apply_entry_fee(position, entry_fee_usd)
        append_trade(position, paper_mode=PAPER_MODE_ACTIVE)

        self._cooldowns[cooldown_key] = now
        self._coin_last_fill_ts[coin_fill_key] = now
        self._bucket_last_fill_ts[bucket_key] = now

        return ExecutorResult(
            traded=True,
            reason=decision.reason,
            position_id=position.position_id,
            fill_price=fill.fill_price,
            fill_slippage_bps=fill.slippage_bps,
            fill_ratio=fill.fill_ratio if fill.fill_ratio is not None else 1.0,
        )

    def _validate_entry(self, signal) -> Optional[str]:
        meta = signal.meta or {}
        coin = str(signal.coin).upper()
        atr = float(meta.get("atr", 0.0))
        signal_price = float(signal.entry_price)
        stop_price = float(signal.stop_price)
        side = self._side_str(signal.side)

        if signal_price <= 0 or atr <= 0:
            return None

        current_price = self.backend.get_mid_price(coin)
        if current_price is None:
            return None

        self._record_recent_price(coin, current_price)

        price_move = abs(current_price - signal_price)
        stop_dist = abs(signal_price - stop_price)

        atr_buffer = ENTRY_BUFFER_ATR_MULT * atr
        if price_move > atr_buffer:
            return (
                f"entry_outside_atr_buffer "
                f"(move={price_move:.4f} > {atr_buffer:.4f} = {ENTRY_BUFFER_ATR_MULT}xATR)"
            )

        if stop_dist > 0:
            move_frac = price_move / stop_dist
            if move_frac > MAX_MOVE_STOP_FRAC:
                if side == "LONG" and current_price > signal_price:
                    return f"too_late_long (moved {move_frac:.2f}x stop_dist toward TP)"
                if side == "SHORT" and current_price < signal_price:
                    return f"too_late_short (moved {move_frac:.2f}x stop_dist toward TP)"

        recent_move_atr = self._recent_move_in_atr(coin, current_price, atr)
        if recent_move_atr >= MAX_RECENT_MOVE_ATR:
            return (
                f"overextended_move "
                f"({recent_move_atr:.2f} ATR >= {MAX_RECENT_MOVE_ATR:.2f} ATR)"
            )

        return None

    def _record_recent_price(self, coin: str, price: float):
        now = time.time()
        dq = self._recent_prices.setdefault(coin, deque())
        dq.append((now, float(price)))
        cutoff = now - 900
        while dq and dq[0][0] < cutoff:
            dq.popleft()

    def _recent_move_in_atr(self, coin: str, current_price: float, atr: float) -> float:
        if atr <= 0:
            return 0.0
        dq = self._recent_prices.get(coin)
        if not dq:
            return 0.0
        lookback_price = dq[0][1]
        return abs(current_price - lookback_price) / atr

    def _factor_bucket(self, coin: str) -> str:
        c = str(coin).upper()
        if c in MAJORS_BUCKET_COINS:
            return "majors"
        if c in SOL_BETA_BUCKET_COINS:
            return "sol_beta"
        if c in ALT_BETA_BUCKET_COINS:
            return "alt_beta"
        return "other"

    def _bucket_dir_limit(self, bucket: str, side: str) -> int:
        s = side.upper()
        if bucket == "majors":
            return MAX_BUCKET_MAJORS_LONG if s == "LONG" else MAX_BUCKET_MAJORS_SHORT
        if bucket == "sol_beta":
            return MAX_BUCKET_SOL_BETA_LONG if s == "LONG" else MAX_BUCKET_SOL_BETA_SHORT
        if bucket == "alt_beta":
            return MAX_BUCKET_ALT_BETA_LONG if s == "LONG" else MAX_BUCKET_ALT_BETA_SHORT
        return MAX_BUCKET_OTHER_LONG if s == "LONG" else MAX_BUCKET_OTHER_SHORT

    def record_exit_slippage(self, slippage_bps: float):
        self._session_exit_slippage_bps.append(slippage_bps)

    def slippage_summary(self) -> str:
        entry_n = len(self._session_entry_slippage_bps)
        exit_n = len(self._session_exit_slippage_bps)

        if entry_n == 0 and exit_n == 0:
            return "No fills this session."

        entry_avg = sum(self._session_entry_slippage_bps) / entry_n if entry_n else 0.0
        exit_avg = sum(self._session_exit_slippage_bps) / exit_n if exit_n else 0.0
        total_entry_cost = sum(self._session_entry_slippage_bps)
        total_exit_cost = sum(self._session_exit_slippage_bps)

        return (
            f"Entry: {entry_n} fills, avg {entry_avg:.1f}bps "
            f"(total {total_entry_cost:.0f}bps) | "
            f"Exit: {exit_n} fills, avg {exit_avg:.1f}bps "
            f"(total {total_exit_cost:.0f}bps)"
        )

    def update(self):
        self.paper.update_all(risk_manager=self.risk, executor=self)
        self.risk.reset_daily_halt()

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
                    f"`{pos.coin} {self._side_str(pos.side)} entry={pos.entry_price:.4f} "
                    f"R={r_now:+.2f} ({pos.state.value})`"
                )
        else:
            lines.append("_No open positions_")

        sf = self.strategy_filter.summary()
        if sf.strip() not in ("No data yet.", ""):
            lines.append(f"\n*Filters:*\n```\n{sf}\n```")

        lines.append(f"\n_{self.paper.session_summary()}_")
        lines.append(f"📊 _{self.slippage_summary()}_")
        return "\n".join(lines)

    def shutdown(self):
        self.backend.shutdown()
        print(
            f"[EXECUTOR] Shutdown — cooldowns: {len(self._cooldowns)} | "
            f"Slippage: {self.slippage_summary()}"
        )

    def _log_missed(self, signal, sig_id: int, reason: str):
        if not LOG_MISSED:
            return

        meta = signal.meta or {}
        signal_price = float(signal.entry_price)
        stop_dist = abs(signal_price - float(signal.stop_price))

        current_price = self.backend.get_mid_price(signal.coin)
        price_move_r = 0.0
        if current_price is not None and stop_dist > 0:
            if self._side_str(signal.side) == "LONG":
                price_move_r = (current_price - signal_price) / stop_dist
            else:
                price_move_r = (signal_price - current_price) / stop_dist

        file_exists = os.path.exists(MISSED_LOG_FILE)
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "signal_id": sig_id,
            "coin": signal.coin,
            "side": self._side_str(signal.side),
            "entry_price": round(signal_price, 8),
            "stop_price": round(float(signal.stop_price), 8),
            "tp_price": round(float(signal.tp_price), 8),
            "confidence": round(float(signal.confidence), 3),
            "total_score": round(float(meta.get("total_score", 0.0)), 3),
            "session": meta.get("session", ""),
            "regime": signal.regime,
            "reject_reason": reason[:200],
            "current_price": round(current_price, 8) if current_price else "",
            "price_move_r": round(price_move_r, 3),
        }

        try:
            with open(MISSED_LOG_FILE, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=MISSED_FIELDS)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)
        except Exception as e:
            print(f"[EXECUTOR] Missed log write failed: {e}")

    def _build_position(
        self,
        signal,
        sig_id: int,
        decision,
        fill_price: Optional[float] = None,
        fill_size_usd: Optional[float] = None,
    ) -> Position:
        meta = signal.meta or {}
        pos_id = f"{signal.coin}_{int(datetime.now(timezone.utc).timestamp() * 1000)}"
        setup_family = meta.get("setup_family", meta.get("regime_local", ""))

        actual_entry = float(fill_price if fill_price is not None else signal.entry_price)
        actual_size = float(fill_size_usd if fill_size_usd is not None else decision.size_usd)

        stop_dist = abs(actual_entry - float(signal.stop_price))
        if actual_entry > 0 and stop_dist > 0:
            stop_pct = stop_dist / actual_entry
            actual_risk = actual_size * stop_pct
        else:
            actual_risk = decision.risk_usd

        return Position(
            position_id=pos_id,
            coin=signal.coin,
            side=self._side_str(signal.side),
            signal_id=sig_id,
            entry_price=actual_entry,
            stop_price=float(signal.stop_price),
            tp_price=float(signal.tp_price),
            original_stop_price=float(signal.stop_price),
            original_tp_price=float(signal.tp_price),
            atr=float(meta.get("atr", 0.0)),
            size_usd=actual_size,
            risk_usd=round(actual_risk, 2),
            r_value=round(actual_risk, 2),
            size_multiplier=decision.size_multiplier,
            peak_price=actual_entry,
            regime=signal.regime,
            setup_family=setup_family,
            htf_regime=meta.get("regime_htf_1h", ""),
            confidence=float(signal.confidence),
            total_score=float(meta.get("total_score", 0.0)),
            session=meta.get("session", ""),
        )

    @staticmethod
    def _side_str(side) -> str:
        return str(side).upper().replace("ORDERSIDE.", "").replace("POSITIONSIDE.", "")
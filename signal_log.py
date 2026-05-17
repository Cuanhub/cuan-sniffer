# signal_log.py
"""
Signal event log — records every signal from the engine + executor decision.
"""

import csv
import os
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from signal_engine import Signal
from smc_live_log import append_smc_live_event

LOG_PATH = os.getenv("SIGNAL_LOG_PATH", "signals.csv")

HEADER = [
    "coin",
    "timestamp_utc",
    "signal_id",
    "timeframe",
    "side",
    "entry",
    "stop",
    "tp",
    "rr_planned",
    "confidence",
    "regime",
    "total_score",
    "atr",
    "stop_dist",
    "tp_dist",
    "effective_threshold",
    "session",
    "vol_state",
    "vol_ratio",
    "whale_pressure",
    "flow_momentum",
    "funding_rate",
    "open_interest",
    "long_short_bias",
    "regime_local",
    "regime_htf_1h",
    "regime_macro_4h",
    "reason_text",
    "executor_result",
    "reject_reason",
    "position_id",
    "fill_price",
    "fill_slippage_bps",
    "fill_ratio",
]


def _abs_log_path() -> str:
    return os.path.abspath(LOG_PATH)


def init_signal_log():
    abs_path = _abs_log_path()

    if os.path.exists(LOG_PATH):
        print(f"[SIGNAL_LOG] Using existing log at {LOG_PATH}")
        print(f"[SIGNAL_LOG] Absolute path: {abs_path}")
        return

    with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(HEADER)

    print(f"[SIGNAL_LOG] Initialized log at {LOG_PATH}")
    print(f"[SIGNAL_LOG] Absolute path: {abs_path}")


def append_signal(
    coin: str,
    sig_id: int,
    signal: Signal,
    flow_snapshot: Dict[str, Any],
    funding_rate: float,
    open_interest: int,
    long_short_bias: float,
    executor_result: Optional[str] = None,
    reject_reason: str = "",
    position_id: str = "",
    fill_price: float = 0.0,
    fill_slippage_bps: float = 0.0,
    fill_ratio: float = 0.0,
):
    try:
        ts_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")

        meta = signal.meta or {}

        timeframe = meta.get("timeframe", "unknown")
        total_score = float(meta.get("total_score", 0.0))
        atr = float(meta.get("atr", 0.0))
        stop_dist = float(meta.get("stop_dist", 0.0))
        tp_dist = float(meta.get("tp_dist", 0.0))
        eff_thr = float(meta.get("effective_threshold", 0.0))
        session = meta.get("session", "unknown")
        vol_state = meta.get("vol_state", "unknown")
        vol_ratio = float(meta.get("vol_ratio", 0.0))

        regime_local = meta.get("regime_local", "")
        regime_htf_1h = meta.get("regime_htf_1h", "")
        regime_macro_4h = meta.get("regime_macro_4h", "")

        whale_pressure = float(flow_snapshot.get("whale_pressure", 0.0)) if flow_snapshot else 0.0
        flow_momentum = float(flow_snapshot.get("flow_momentum", 0.0)) if flow_snapshot else 0.0

        entry = float(signal.entry_price)
        stop = float(signal.stop_price)
        tp = float(signal.tp_price)

        rr = abs((tp - entry) / (entry - stop)) if entry != stop else 0.0
        final_entry = float(fill_price or meta.get("final_entry", entry) or entry)
        final_stop = stop
        final_tp = tp
        final_rr = (
            abs((final_tp - final_entry) / (final_entry - final_stop))
            if final_entry != final_stop else 0.0
        )
        final_stop_method = str(
            meta.get("final_stop_method", meta.get("stop_method", ""))
        )
        stop_was_redesigned = bool(meta.get("stop_was_redesigned", False))
        original_stop = float(meta.get("original_stop", meta.get("stop_structural", stop)) or stop)
        reason_text = (signal.reason or "")[:300]

        result = (executor_result or "unknown").strip().lower()
        if result not in {"traded", "rejected", "error"}:
            result = "unknown"

        reason_out = (reject_reason or "").strip()
        if result == "rejected" and not reason_out:
            reason_out = "unknown_rejection"
        elif result == "error" and not reason_out:
            reason_out = "executor_error_unknown"
        reason_out = reason_out[:200]

        row = [
            coin,
            ts_utc,
            sig_id,
            timeframe,
            signal.side,
            entry,
            stop,
            tp,
            rr,
            float(signal.confidence),
            signal.regime,
            total_score,
            atr,
            stop_dist,
            tp_dist,
            eff_thr,
            session,
            vol_state,
            vol_ratio,
            whale_pressure,
            flow_momentum,
            float(funding_rate),
            int(open_interest),
            float(long_short_bias),
            regime_local,
            regime_htf_1h,
            regime_macro_4h,
            reason_text,
            result,
            reason_out,
            position_id,
            round(fill_price, 8),
            round(fill_slippage_bps, 2),
            round(fill_ratio, 4),
        ]

        if not os.path.exists(LOG_PATH):
            init_signal_log()

        print(
            f"[SIGNAL_LOG] Writing row to: {_abs_log_path()} | "
            f"signal_id={sig_id} | coin={coin} | result={result}"
        )

        with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)
            f.flush()
            os.fsync(f.fileno())

        append_smc_live_event(
            event_type="executor_result",
            signal_id=sig_id,
            position_id=position_id,
            coin=coin,
            timeframe=timeframe,
            side=signal.side,
            score=total_score,
            confidence=float(signal.confidence),
            accepted=result == "traded",
            reject_reason=reason_out,
            entry=entry,
            stop=stop,
            tp=tp,
            rr=rr,
            stop_method=meta.get("stop_method", ""),
            final_entry=round(final_entry, 8),
            final_stop=round(final_stop, 8),
            final_tp=round(final_tp, 8),
            final_rr=round(final_rr, 4),
            final_stop_method=final_stop_method,
            stop_was_redesigned=stop_was_redesigned,
            original_stop=round(original_stop, 8),
            ob_level=meta.get("ob_level", 0.0),
            price=entry,
            atr=atr,
            stop_dist_atr=meta.get("stop_dist_atr", 0.0),
            bos_bull=meta.get("bos_bull", False),
            bos_bear=meta.get("bos_bear", False),
            choch_bull=meta.get("choch_bull", False),
            choch_bear=meta.get("choch_bear", False),
            ob_bull=meta.get("ob_bull", False),
            ob_bear=meta.get("ob_bear", False),
            in_bull_ob=meta.get("in_bull_ob", False),
            in_bear_ob=meta.get("in_bear_ob", False),
            dist_to_bull_ob=meta.get("dist_to_bull_ob", 0.0),
            dist_to_bear_ob=meta.get("dist_to_bear_ob", 0.0),
            bull_ob_low=meta.get("bull_ob_low", 0.0),
            bull_ob_high=meta.get("bull_ob_high", 0.0),
            bear_ob_low=meta.get("bear_ob_low", 0.0),
            bear_ob_high=meta.get("bear_ob_high", 0.0),
            fvg_bull=meta.get("fvg_bull", False),
            fvg_bear=meta.get("fvg_bear", False),
            in_bull_fvg=meta.get("in_bull_fvg", False),
            in_bear_fvg=meta.get("in_bear_fvg", False),
            dist_to_bull_fvg=meta.get("dist_to_bull_fvg", 0.0),
            dist_to_bear_fvg=meta.get("dist_to_bear_fvg", 0.0),
            bull_fvg_low=meta.get("bull_fvg_low", 0.0),
            bull_fvg_high=meta.get("bull_fvg_high", 0.0),
            bear_fvg_low=meta.get("bear_fvg_low", 0.0),
            bear_fvg_high=meta.get("bear_fvg_high", 0.0),
            sweep_bull=meta.get("sweep_bull", False),
            sweep_bear=meta.get("sweep_bear", False),
            eq_high=meta.get("eq_high", False),
            eq_low=meta.get("eq_low", False),
            htf_regime=regime_htf_1h,
            macro_regime=regime_macro_4h,
            market_regime=meta.get("market_regime", ""),
            session=session,
            order_submitted=result == "traded",
            fill_price=round(fill_price, 8),
            slippage_bps=round(fill_slippage_bps, 2),
            partial_hit=False,
            close_reason="",
            realized_r="",
            realized_pnl="",
        )

    except Exception as e:
        print(f"[SIGNAL_LOG] Error appending signal row: {e}")

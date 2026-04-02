# signal_log.py
"""
Signal event log — records every signal from the engine + executor decision.
"""

import csv
import os
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from signal_engine import Signal

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

    except Exception as e:
        print(f"[SIGNAL_LOG] Error appending signal row: {e}")
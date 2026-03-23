# signal_log.py

import csv
import os
from datetime import datetime, timezone
from typing import Dict, Any

from signal_engine import Signal

LOG_PATH = "signals.csv"


def init_signal_log():
    """
    Initialize the CSV log with a clean, structured header.
    Only creates the file if it doesn't exist.
    """
    if os.path.exists(LOG_PATH):
        print(f"[SIGNAL_LOG] Using existing log at {LOG_PATH}")
        return

    header = [
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
    ]

    with open(LOG_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

    print(f"[SIGNAL_LOG] Initialized log at {LOG_PATH}")


def append_signal(
    coin: str,
    sig_id: int,
    signal: Signal,
    flow_snapshot: Dict[str, Any],
    funding_rate: float,
    open_interest: int,
    long_short_bias: float,
):
    """
    Append a single signal row to the CSV.

    This becomes your ground truth for:
    - win rate
    - regime performance
    - flow + sentiment impact
    """

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

        # 🔥 R/R calculation (IMPORTANT)
        entry = float(signal.entry_price)
        stop = float(signal.stop_price)
        tp = float(signal.tp_price)

        rr = abs((tp - entry) / (entry - stop)) if entry != stop else 0.0

        # Trim reason to keep CSV clean
        reason = (signal.reason or "")[:300]

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
            reason,
        ]

        file_exists = os.path.exists(LOG_PATH)

        with open(LOG_PATH, "a", newline="") as f:
            writer = csv.writer(f)

            # Safety: re-init if file somehow disappears mid-run
            if not file_exists:
                init_signal_log()

            writer.writerow(row)

    except Exception as e:
        print(f"[SIGNAL_LOG] Error appending signal row: {e}")
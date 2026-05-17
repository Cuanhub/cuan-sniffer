"""
Append-only live SMC evaluation log.

This file intentionally does not import the signal engine, executor, or risk
manager so logging can never affect live trading decisions.
"""

import csv
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LOG_PATH = os.getenv("SMC_LIVE_LOG_PATH", "smc_live_log.csv")
_HEADER_INITIALIZED = False

HEADER = [
    "timestamp_utc",
    "event_type",
    "signal_id",
    "position_id",
    "coin",
    "timeframe",
    "side",
    "score",
    "confidence",
    "accepted",
    "reject_reason",
    "reason_family",
    "entry",
    "stop",
    "tp",
    "rr",
    "stop_method",
    "final_entry",
    "final_stop",
    "final_tp",
    "final_rr",
    "final_stop_method",
    "stop_was_redesigned",
    "original_stop",
    "ob_level",
    "price",
    "atr",
    "stop_dist_atr",
    "bos_bull",
    "bos_bear",
    "choch_bull",
    "choch_bear",
    "ob_bull",
    "ob_bear",
    "in_bull_ob",
    "in_bear_ob",
    "dist_to_bull_ob",
    "dist_to_bear_ob",
    "bull_ob_low",
    "bull_ob_high",
    "bear_ob_low",
    "bear_ob_high",
    "fvg_bull",
    "fvg_bear",
    "in_bull_fvg",
    "in_bear_fvg",
    "dist_to_bull_fvg",
    "dist_to_bear_fvg",
    "bull_fvg_low",
    "bull_fvg_high",
    "bear_fvg_low",
    "bear_fvg_high",
    "sweep_bull",
    "sweep_bear",
    "eq_high",
    "eq_low",
    "htf_regime",
    "macro_regime",
    "market_regime",
    "session",
    "order_submitted",
    "fill_price",
    "slippage_bps",
    "partial_hit",
    "close_reason",
    "realized_r",
    "realized_pnl",
]


def _normalize(value: Any) -> Any:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return value


def normalize_reject_reason_family(reason: str) -> str:
    text = str(reason or "").strip().lower()
    if not text:
        return ""
    compact = text.replace("-", "_").replace(" ", "_")
    if (
        "rr_too_low" in compact
        or "fill_rr_below_threshold" in compact
        or "final_rr_below_min" in compact
        or "reward_risk" in compact
        or "risk_reward" in compact
    ):
        return "rr_too_low"
    if (
        "signal_stale_move" in text
        or "entry_outside_atr_buffer" in text
        or "too_late_" in text
        or "adverse_drift_" in text
        or "overextended_move" in text
        or "stale_or_drift" in text
    ):
        return "stale_entry"
    if "tp_already_consumed" in text:
        return "tp_consumed"
    if "session_soft_blocked" in text or "session_blocked" in text:
        return "session_block"
    if "hard_blocked_timeframe" in text or "4h_disabled" in text:
        return "hard_timeframe_block"
    if "strategy_filter" in text:
        return "strategy_filter_block"
    if "daily loss" in text or "daily_halt" in text or "halted: daily" in text:
        return "daily_halt"
    if "max positions" in text or "bucket_dir_limit" in text or "position capacity" in text:
        return "max_positions"
    if "cooldown" in text or "reject_throttled" in text:
        return "cooldown"
    if "margin" in text:
        return "margin"
    if "fill_rejected" in text or "entry_rejected" in text or "sdk_error" in text or "live_order" in text:
        return "live_order_reject"
    if "protection" in text:
        return "protection_fail"
    return "unknown"


def _write_header(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(HEADER)
        f.flush()


def init_smc_live_log() -> None:
    global _HEADER_INITIALIZED
    if _HEADER_INITIALIZED:
        return
    try:
        path = Path(LOG_PATH)
        if not path.exists() or path.stat().st_size == 0:
            _write_header(path)
            _HEADER_INITIALIZED = True
            return

        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_header = reader.fieldnames or []
            if all(field in existing_header for field in HEADER):
                _HEADER_INITIALIZED = True
                return
            rows = list(reader)

        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=HEADER, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({field: _normalize(row.get(field, "")) for field in HEADER})
            f.flush()
        os.replace(tmp_path, path)
        print(f"[SMC_LIVE_LOG] Migrated header at {path}")
        _HEADER_INITIALIZED = True
    except Exception as exc:
        print(f"[SMC_LIVE_LOG] init failed: {exc}")


def append_smc_live_event(**fields: Any) -> None:
    try:
        path = Path(LOG_PATH)
        needs_header = not path.exists() or path.stat().st_size == 0
        reject_reason = str(fields.get("reject_reason", "") or "")
        fields.setdefault("reason_family", normalize_reject_reason_family(reject_reason))
        row = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **fields,
        }

        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=HEADER, extrasaction="ignore")
            if needs_header:
                writer.writeheader()
            writer.writerow({key: _normalize(row.get(key, "")) for key in HEADER})
            f.flush()
    except Exception as exc:
        print(f"[SMC_LIVE_LOG] append failed: {exc}")

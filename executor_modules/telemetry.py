"""
Executor rejection telemetry — CSV logging.

Extracted from executor.py. No trade logic. No circular imports.
"""

import csv
import os
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Deque, Dict


EXECUTOR_REJECTS_PATH = os.getenv("EXECUTOR_REJECTS_PATH", "executor_rejects.csv")
_EXECUTOR_REJECT_LOCK = threading.Lock()
_EXECUTOR_REJECT_FIELDS = [
    "timestamp", "signal_id", "symbol", "coin", "side",
    "entry_price", "stop_price", "tp_price",
    "confidence", "required_confidence", "total_score",
    "active_quality_model", "active_quality_score", "signal_confidence",
    "rr", "required_rr", "reject_reason", "session", "setup_family",
    "market_regime", "regime", "timeframe", "current_price", "price_move_r",
]
_MISSED_CONTEXT_LOCK = threading.Lock()
_MISSED_CONTEXTS: Deque[Dict[str, Any]] = deque(maxlen=256)


def _exec_ensure_csv(path: str, fields: list) -> None:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w", newline="") as fh:
            csv.DictWriter(fh, fieldnames=fields).writeheader()
        return

    try:
        with open(path, "r", newline="") as fh:
            reader = csv.DictReader(fh)
            existing_header = reader.fieldnames or []
            if all(field in existing_header for field in fields):
                return
            rows = list(reader)

        tmp_path = path + ".tmp"
        with open(tmp_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in fields})
        os.replace(tmp_path, path)
        print(f"[EXECUTOR_REJECTS] Migrated header at {path}")
    except Exception:
        pass


def _reject_family_key(reason: str) -> str:
    return str(reason or "").strip().split(":", 1)[0].lower()


def _stage_missed_context(row: Dict[str, Any]) -> None:
    staged = dict(row)
    staged["_staged_at"] = time.time()
    with _MISSED_CONTEXT_LOCK:
        _MISSED_CONTEXTS.append(staged)


def _take_missed_context(symbol: str, side: str, reject_reason: str) -> Dict[str, Any]:
    symbol_norm = str(symbol or "").upper().strip()
    side_norm = str(side or "").upper().strip()
    reason_key = _reject_family_key(reject_reason)
    now = time.time()
    with _MISSED_CONTEXT_LOCK:
        for ctx in reversed(list(_MISSED_CONTEXTS)):
            if now - float(ctx.get("_staged_at", 0.0) or 0.0) > 10:
                continue
            if str(ctx.get("coin", "")).upper().strip() != symbol_norm:
                continue
            if str(ctx.get("side", "")).upper().strip() != side_norm:
                continue
            ctx_reason_key = _reject_family_key(str(ctx.get("reject_reason", "")))
            if reason_key and ctx_reason_key and reason_key != ctx_reason_key:
                if not str(reject_reason or "").startswith(str(ctx.get("reject_reason", ""))):
                    continue
            try:
                _MISSED_CONTEXTS.remove(ctx)
            except ValueError:
                pass
            return ctx
    return {}


def log_executor_reject(
    *,
    symbol: str,
    side: str = "",
    confidence: float = 0.0,
    required_confidence: float = 0.0,
    rr: float = 0.0,
    required_rr: float = 0.0,
    reject_reason: str,
    session: str = "",
    setup_family: str = "",
    market_regime: str = "",
    timeframe: str = "1h",
) -> None:
    try:
        missed = _take_missed_context(symbol, side, reject_reason)
        row = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "signal_id": missed.get("signal_id", ""),
            "symbol": symbol,
            "coin": missed.get("coin", symbol),
            "side": side,
            "entry_price": missed.get("entry_price", ""),
            "stop_price": missed.get("stop_price", ""),
            "tp_price": missed.get("tp_price", ""),
            "confidence": round(float(confidence), 4),
            "required_confidence": round(float(required_confidence), 4),
            "total_score": missed.get("total_score", ""),
            "active_quality_model": missed.get("active_quality_model", ""),
            "active_quality_score": missed.get("active_quality_score", ""),
            "signal_confidence": missed.get("signal_confidence", round(float(confidence), 4)),
            "rr": round(float(rr), 4),
            "required_rr": round(float(required_rr), 4),
            "reject_reason": reject_reason,
            "session": session,
            "setup_family": setup_family,
            "market_regime": market_regime,
            "regime": missed.get("regime", ""),
            "timeframe": timeframe,
            "current_price": missed.get("current_price", ""),
            "price_move_r": missed.get("price_move_r", ""),
        }
        with _EXECUTOR_REJECT_LOCK:
            _exec_ensure_csv(EXECUTOR_REJECTS_PATH, _EXECUTOR_REJECT_FIELDS)
            with open(EXECUTOR_REJECTS_PATH, "a", newline="") as fh:
                csv.DictWriter(fh, fieldnames=_EXECUTOR_REJECT_FIELDS).writerow(row)
    except Exception:
        pass

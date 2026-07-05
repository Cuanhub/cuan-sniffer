"""
Signal engine telemetry — gate rejects + score distribution CSV writers.

Extracted from signal_engine.py. These functions write to:
  - gate_rejects.csv
  - score_distribution.csv

They also delegate to smc_live_log for observability.
All are try/except-wrapped — they must never crash the engine.
"""

import csv
import os
import threading
from datetime import datetime, timezone
from typing import Any, List, Tuple

from smc_live_log import append_smc_live_event

GATE_REJECTS_PATH = os.getenv("GATE_REJECTS_PATH", "gate_rejects.csv")
SCORE_DIST_PATH = os.getenv("SCORE_DIST_PATH", "score_distribution.csv")

_GATE_REJECT_LOCK = threading.Lock()
_SCORE_DIST_LOCK = threading.Lock()

_GATE_REJECT_FIELDS: List[str] = [
    "timestamp", "symbol", "timeframe", "side", "reject_reason",
    "raw_score", "score_v1", "score_v2", "score_v3", "threshold", "confidence", "rr",
    "active_quality_model", "active_quality_score", "signal_confidence",
    "active_quality_score_source",
    "market_regime", "htf_regime", "macro_regime", "session",
    "setup_family", "atr", "price", "metadata",
]
_SCORE_DIST_FIELDS: List[str] = [
    "timestamp", "symbol", "timeframe", "side", "score", "threshold",
    "confidence", "rr", "score_v1", "score_v2", "score_v3",
    "active_quality_model", "active_quality_score", "signal_confidence",
    "active_quality_score_source", "setup_family", "market_regime",
    "htf_regime", "macro_regime", "session",
]


def _telemetry_ensure_csv(path: str, fields: list) -> None:
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
        print(f"[SIGNAL_TELEMETRY] Migrated header at {path}")
    except Exception:
        pass


def _live_quality_model() -> str:
    model = str(os.getenv("LIVE_ELIGIBILITY_MODEL", "v3") or "").strip().lower()
    return model if model in {"v1", "v3"} else ""


def _is_blank(value: Any) -> bool:
    return value is None or str(value).strip() == ""


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if _is_blank(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _canonical_quality_fields(
    *,
    raw_score: Any = 0.0,
    confidence: Any = 0.0,
    score_v1: Any = None,
    score_v2: Any = None,
    score_v3: Any = None,
    active_quality_model: str = "",
    active_quality_score: Any = None,
    signal_confidence: Any = None,
    active_quality_score_source: str = "",
) -> Tuple[str, float, float, str, float, float, float]:
    model = str(active_quality_model or "").strip().lower() or _live_quality_model()
    v1 = _safe_float(score_v1, _safe_float(raw_score, 0.0))
    v2 = _safe_float(score_v2, 0.0)
    v3 = _safe_float(score_v3, 0.0)

    source = str(active_quality_score_source or "").strip().lower()
    if not _is_blank(active_quality_score):
        active_score = _safe_float(active_quality_score, 0.0)
        source = source or "explicit"
    elif model == "v3" and not _is_blank(score_v3):
        active_score = v3
        source = source or "score_v3"
    elif model == "v1":
        active_score = v1
        source = source or "score_v1"
    else:
        active_score = 0.0
        source = source or ("pending_score_v3" if model == "v3" else "unavailable")

    if not _is_blank(signal_confidence):
        sig_conf = _safe_float(signal_confidence, 0.0)
    elif source in {"explicit", "score_v3", "score_v1"}:
        sig_conf = active_score
    else:
        sig_conf = _safe_float(confidence, _safe_float(raw_score, 0.0))

    if source == "pending_score_v3":
        model = ""

    return model, active_score, sig_conf, source, v1, v2, v3


def log_gate_reject(
    *,
    symbol: str,
    timeframe: str = "1h",
    side: str = "",
    reject_reason: str,
    raw_score: float = 0.0,
    threshold: float = 0.0,
    confidence: float = 0.0,
    rr: float = 0.0,
    market_regime: str = "",
    htf_regime: str = "",
    macro_regime: str = "",
    session: str = "",
    setup_family: str = "",
    atr: float = 0.0,
    price: float = 0.0,
    metadata: str = "",
    active_quality_model: str = "",
    active_quality_score: Any = None,
    signal_confidence: Any = None,
    active_quality_score_source: str = "",
    score_v1: Any = None,
    score_v2: Any = None,
    score_v3: Any = None,
) -> None:
    try:
        raw_score_value = _safe_float(raw_score, 0.0)
        threshold_value = _safe_float(threshold, 0.0)
        confidence_value = _safe_float(confidence, 0.0)
        rr_value = _safe_float(rr, 0.0)
        atr_value = _safe_float(atr, 0.0)
        price_value = _safe_float(price, 0.0)
        (
            canonical_model,
            canonical_active_score,
            canonical_signal_conf,
            canonical_source,
            canonical_v1,
            canonical_v2,
            canonical_v3,
        ) = _canonical_quality_fields(
            raw_score=raw_score_value,
            confidence=confidence_value,
            score_v1=score_v1,
            score_v2=score_v2,
            score_v3=score_v3,
            active_quality_model=active_quality_model,
            active_quality_score=active_quality_score,
            signal_confidence=signal_confidence,
            active_quality_score_source=active_quality_score_source,
        )
        row = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "symbol": symbol,
            "timeframe": timeframe,
            "side": side,
            "reject_reason": reject_reason,
            "raw_score": round(raw_score_value, 4),
            "score_v1": round(canonical_v1, 4),
            "score_v2": round(canonical_v2, 4),
            "score_v3": round(canonical_v3, 4),
            "threshold": round(threshold_value, 4),
            "confidence": round(confidence_value, 4),
            "rr": round(rr_value, 4),
            "active_quality_model": canonical_model,
            "active_quality_score": round(canonical_active_score, 4),
            "signal_confidence": round(canonical_signal_conf, 4),
            "active_quality_score_source": canonical_source,
            "market_regime": market_regime,
            "htf_regime": htf_regime,
            "macro_regime": macro_regime,
            "session": session,
            "setup_family": setup_family,
            "atr": round(atr_value, 8),
            "price": round(price_value, 6),
            "metadata": metadata,
        }
        with _GATE_REJECT_LOCK:
            _telemetry_ensure_csv(GATE_REJECTS_PATH, _GATE_REJECT_FIELDS)
            with open(GATE_REJECTS_PATH, "a", newline="") as fh:
                csv.DictWriter(fh, fieldnames=_GATE_REJECT_FIELDS).writerow(row)
        append_smc_live_event(
            event_type="gate_reject",
            coin=symbol,
            symbol=symbol,
            timeframe=timeframe,
            side=side,
            score=round(raw_score_value, 4),
            raw_score=round(raw_score_value, 4),
            score_v1=round(canonical_v1, 4),
            score_v2=round(canonical_v2, 4),
            score_v3=round(canonical_v3, 4),
            threshold=round(threshold_value, 4),
            effective_threshold=round(threshold_value, 4),
            confidence=round(confidence_value, 4),
            active_quality_model=canonical_model,
            active_quality_score=round(canonical_active_score, 4),
            signal_confidence=round(canonical_signal_conf, 4),
            active_quality_score_source=canonical_source,
            accepted=False,
            reject_reason=reject_reason,
            rr=round(rr_value, 4),
            rr_planned=round(rr_value, 4),
            setup_family=setup_family,
            market_regime=market_regime,
            htf_regime=htf_regime,
            macro_regime=macro_regime,
            session=session,
            atr=round(atr_value, 8),
            price=round(price_value, 6),
            metadata=metadata,
        )
    except Exception:
        pass


def log_score_candidate(
    *,
    symbol: str,
    timeframe: str = "1h",
    side: str = "",
    score: float,
    threshold: float,
    rr: float = 0.0,
    setup_family: str = "",
    market_regime: str = "",
    htf_regime: str = "",
    macro_regime: str = "",
    session: str = "",
    score_v1: Any = None,
    score_v2: Any = None,
    score_v3: Any = None,
    active_quality_model: str = "",
    active_quality_score: Any = None,
    signal_confidence: Any = None,
    active_quality_score_source: str = "",
) -> None:
    try:
        score_value = _safe_float(score, 0.0)
        threshold_value = _safe_float(threshold, 0.0)
        rr_value = _safe_float(rr, 0.0)
        confidence = round(min(0.95, max(0.50, score_value)), 4)
        (
            canonical_model,
            canonical_active_score,
            canonical_signal_conf,
            canonical_source,
            canonical_v1,
            canonical_v2,
            canonical_v3,
        ) = _canonical_quality_fields(
            raw_score=score_value,
            confidence=confidence,
            score_v1=score_v1,
            score_v2=score_v2,
            score_v3=score_v3,
            active_quality_model=active_quality_model,
            active_quality_score=active_quality_score,
            signal_confidence=signal_confidence,
            active_quality_score_source=active_quality_score_source,
        )
        row = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "symbol": symbol,
            "timeframe": timeframe,
            "side": side,
            "score": round(score_value, 4),
            "threshold": round(threshold_value, 4),
            "confidence": confidence,
            "rr": round(rr_value, 4),
            "score_v1": round(canonical_v1, 4),
            "score_v2": round(canonical_v2, 4),
            "score_v3": round(canonical_v3, 4),
            "active_quality_model": canonical_model,
            "active_quality_score": round(canonical_active_score, 4),
            "signal_confidence": round(canonical_signal_conf, 4),
            "active_quality_score_source": canonical_source,
            "setup_family": setup_family,
            "market_regime": market_regime,
            "htf_regime": htf_regime,
            "macro_regime": macro_regime,
            "session": session,
        }
        with _SCORE_DIST_LOCK:
            _telemetry_ensure_csv(SCORE_DIST_PATH, _SCORE_DIST_FIELDS)
            with open(SCORE_DIST_PATH, "a", newline="") as fh:
                csv.DictWriter(fh, fieldnames=_SCORE_DIST_FIELDS).writerow(row)
        append_smc_live_event(
            event_type="score_candidate",
            coin=symbol,
            symbol=symbol,
            timeframe=timeframe,
            side=side,
            score=round(score_value, 4),
            raw_score=round(score_value, 4),
            total_score=round(score_value, 4),
            score_v1=round(canonical_v1, 4),
            score_v2=round(canonical_v2, 4),
            score_v3=round(canonical_v3, 4),
            threshold=round(threshold_value, 4),
            effective_threshold=round(threshold_value, 4),
            confidence=confidence,
            active_quality_model=canonical_model,
            active_quality_score=round(canonical_active_score, 4),
            signal_confidence=round(canonical_signal_conf, 4),
            active_quality_score_source=canonical_source,
            accepted="",
            reject_reason="",
            rr=round(rr_value, 4),
            rr_planned=round(rr_value, 4),
            setup_family=setup_family,
            market_regime=market_regime,
            htf_regime=htf_regime,
            macro_regime=macro_regime,
            session=session,
        )
    except Exception:
        pass

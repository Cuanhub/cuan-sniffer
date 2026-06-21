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
from typing import List

from smc_live_log import append_smc_live_event

GATE_REJECTS_PATH = os.getenv("GATE_REJECTS_PATH", "gate_rejects.csv")
SCORE_DIST_PATH = os.getenv("SCORE_DIST_PATH", "score_distribution.csv")

_GATE_REJECT_LOCK = threading.Lock()
_SCORE_DIST_LOCK = threading.Lock()

_GATE_REJECT_FIELDS: List[str] = [
    "timestamp", "symbol", "timeframe", "side", "reject_reason",
    "raw_score", "threshold", "confidence", "rr",
    "market_regime", "htf_regime", "macro_regime", "session",
    "setup_family", "atr", "price", "metadata",
]
_SCORE_DIST_FIELDS: List[str] = [
    "timestamp", "symbol", "timeframe", "side", "score", "threshold",
    "confidence", "rr", "setup_family", "market_regime", "htf_regime", "macro_regime",
]


def _telemetry_ensure_csv(path: str, fields: list) -> None:
    if not os.path.exists(path):
        with open(path, "w", newline="") as fh:
            csv.DictWriter(fh, fieldnames=fields).writeheader()


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
) -> None:
    try:
        row = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "symbol": symbol,
            "timeframe": timeframe,
            "side": side,
            "reject_reason": reject_reason,
            "raw_score": round(float(raw_score), 4),
            "threshold": round(float(threshold), 4),
            "confidence": round(float(confidence), 4),
            "rr": round(float(rr), 4),
            "market_regime": market_regime,
            "htf_regime": htf_regime,
            "macro_regime": macro_regime,
            "session": session,
            "setup_family": setup_family,
            "atr": round(float(atr), 8),
            "price": round(float(price), 6),
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
            score=round(float(raw_score), 4),
            raw_score=round(float(raw_score), 4),
            threshold=round(float(threshold), 4),
            effective_threshold=round(float(threshold), 4),
            confidence=round(float(confidence), 4),
            accepted=False,
            reject_reason=reject_reason,
            rr=round(float(rr), 4),
            rr_planned=round(float(rr), 4),
            setup_family=setup_family,
            market_regime=market_regime,
            htf_regime=htf_regime,
            macro_regime=macro_regime,
            session=session,
            atr=round(float(atr), 8),
            price=round(float(price), 6),
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
) -> None:
    try:
        confidence = round(min(0.95, max(0.50, float(score))), 4)
        row = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "symbol": symbol,
            "timeframe": timeframe,
            "side": side,
            "score": round(float(score), 4),
            "threshold": round(float(threshold), 4),
            "confidence": confidence,
            "rr": round(float(rr), 4),
            "setup_family": setup_family,
            "market_regime": market_regime,
            "htf_regime": htf_regime,
            "macro_regime": macro_regime,
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
            score=round(float(score), 4),
            raw_score=round(float(score), 4),
            total_score=round(float(score), 4),
            threshold=round(float(threshold), 4),
            effective_threshold=round(float(threshold), 4),
            confidence=confidence,
            accepted="",
            reject_reason="",
            rr=round(float(rr), 4),
            rr_planned=round(float(rr), 4),
            setup_family=setup_family,
            market_regime=market_regime,
            htf_regime=htf_regime,
            macro_regime=macro_regime,
        )
    except Exception:
        pass

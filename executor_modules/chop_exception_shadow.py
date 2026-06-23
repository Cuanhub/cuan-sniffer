"""
Chop exception shadow lane — research-only logging.

Evaluates signals rejected by market_regime_block:chop against a
candidate exception recipe. Qualifying signals are logged to
shadow_chop_exception.csv for forward-outcome replay.

Does NOT affect live execution. Does NOT change accept/reject logic.
"""

import csv
import os
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Optional

SHADOW_CHOP_PATH = os.getenv("SHADOW_CHOP_EXCEPTION_PATH", "shadow_chop_exception.csv")
SHADOW_CHOP_LANE_PATH = os.getenv("SHADOW_CHOP_LANE_PATH", "shadow_chop_lane.csv")
_LOCK = threading.Lock()
_LANE_LOCK = threading.Lock()

EXCEPTION_SYMBOLS_BLOCKED = {"ETH", "ZEC", "BNB", "NEAR"}
EXCEPTION_SESSIONS_ALLOWED = {"ny_open", "asia_open"}
EXCEPTION_MIN_CONFIDENCE = 0.80

LANE_SYMBOLS_BLOCKED = {"ETH", "ZEC", "BNB"}

_FIELDS = [
    "timestamp", "symbol", "side", "session",
    "market_regime", "htf_regime", "macro_regime",
    "confidence", "score_v1", "score_v2", "score_v3",
    "entry", "stop", "tp", "rr",
    "setup_family", "reject_reason", "tags",
]


def _has_fvg(meta: Dict[str, Any]) -> bool:
    for key in ("fvg_bull", "fvg_bear", "in_bull_fvg", "in_bear_fvg"):
        val = meta.get(key)
        if val is True or str(val).strip().lower() == "true":
            return True
    triggers = str(meta.get("triggers", "")).lower()
    if "fvg" in triggers:
        return True
    return False


def evaluate_and_log_chop_exception(
    signal: Any,
    reject_reason: str,
) -> None:
    """
    Called AFTER a chop block fires. Evaluates whether the signal
    qualifies for the shadow exception lane and logs if so.

    Never raises. Never affects execution flow.
    """
    try:
        if "market_regime_block:chop" not in reject_reason:
            return

        meta = getattr(signal, "meta", None) or {}
        coin = str(getattr(signal, "coin", "")).upper()
        side = str(getattr(signal, "side", "")).upper()
        confidence = float(getattr(signal, "confidence", 0.0) or 0.0)
        session = str(meta.get("session", "")).strip().lower()
        setup_family = str(
            meta.get("setup_family", meta.get("regime_local", ""))
        ).strip().lower()

        # ── Exception criteria ───────────────────────────────────
        tags = []

        if confidence < EXCEPTION_MIN_CONFIDENCE:
            return
        tags.append(f"conf={confidence:.2f}")

        if coin in EXCEPTION_SYMBOLS_BLOCKED:
            return

        if session not in EXCEPTION_SESSIONS_ALLOWED:
            return
        tags.append(f"session={session}")

        if setup_family != "continuation":
            return
        tags.append("continuation")

        if not _has_fvg(meta):
            return
        tags.append("fvg_present")

        # ── All criteria passed — log ────────────────────────────
        entry = float(getattr(signal, "entry_price", 0) or 0)
        stop = float(getattr(signal, "stop_price", 0) or 0)
        tp = float(getattr(signal, "tp_price", 0) or 0)
        sd = abs(entry - stop)
        rr = abs(tp - entry) / sd if sd > 0 else 0

        htf = str(meta.get("regime_htf_1h", ""))
        macro = str(meta.get("regime_macro_4h", ""))
        mkt = str(meta.get("market_regime", ""))

        row = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "symbol": coin,
            "side": side,
            "session": session,
            "market_regime": mkt,
            "htf_regime": htf,
            "macro_regime": macro,
            "confidence": round(confidence, 4),
            "score_v1": round(float(meta.get("total_score", confidence) or 0), 4),
            "score_v2": round(float(meta.get("score_v2", 0) or 0), 4),
            "score_v3": round(float(meta.get("score_v3", 0) or 0), 4),
            "entry": round(entry, 8),
            "stop": round(stop, 8),
            "tp": round(tp, 8),
            "rr": round(rr, 4),
            "setup_family": setup_family,
            "reject_reason": reject_reason,
            "tags": ",".join(tags),
        }

        with _LOCK:
            file_exists = os.path.exists(SHADOW_CHOP_PATH) and os.path.getsize(SHADOW_CHOP_PATH) > 0
            with open(SHADOW_CHOP_PATH, "a", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=_FIELDS)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)

    except Exception:
        pass


def log_broad_chop_lane(
    signal: Any,
    reject_reason: str,
) -> None:
    """
    Log ALL chop-blocked signals (excluding ETH/ZEC/BNB) to the broad
    shadow chop lane. No confidence, session, family, or FVG filter.
    Discovery-oriented: captures everything for research.
    """
    try:
        if "market_regime_block:chop" not in reject_reason:
            return

        meta = getattr(signal, "meta", None) or {}
        coin = str(getattr(signal, "coin", "")).upper()

        if coin in LANE_SYMBOLS_BLOCKED:
            return

        side = str(getattr(signal, "side", "")).upper()
        confidence = float(getattr(signal, "confidence", 0.0) or 0.0)
        session = str(meta.get("session", "")).strip().lower()
        setup_family = str(
            meta.get("setup_family", meta.get("regime_local", ""))
        ).strip().lower()

        entry = float(getattr(signal, "entry_price", 0) or 0)
        stop = float(getattr(signal, "stop_price", 0) or 0)
        tp = float(getattr(signal, "tp_price", 0) or 0)
        sd = abs(entry - stop)
        rr = abs(tp - entry) / sd if sd > 0 else 0

        htf = str(meta.get("regime_htf_1h", ""))
        macro = str(meta.get("regime_macro_4h", ""))
        mkt = str(meta.get("market_regime", ""))

        tags = []
        if coin in {"WIF", "JTO", "FARTCOIN", "HYPE", "TAO", "SUI"}:
            tags.append("preferred_symbol")
        if session in {"ny_open", "ny_pm"}:
            tags.append("preferred_session")
        if _has_fvg(meta):
            tags.append("fvg_present")

        row = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "symbol": coin, "side": side, "setup_family": setup_family,
            "session": session, "market_regime": mkt,
            "htf_regime": htf, "macro_regime": macro,
            "score_v1": round(float(meta.get("total_score", confidence) or 0), 4),
            "score_v2": round(float(meta.get("score_v2", 0) or 0), 4),
            "score_v3": round(float(meta.get("score_v3", 0) or 0), 4),
            "confidence": round(confidence, 4),
            "entry": round(entry, 8), "stop": round(stop, 8),
            "tp": round(tp, 8), "rr": round(rr, 4),
            "reject_reason": reject_reason,
            "tags": ",".join(tags),
        }

        with _LANE_LOCK:
            file_exists = os.path.exists(SHADOW_CHOP_LANE_PATH) and os.path.getsize(SHADOW_CHOP_LANE_PATH) > 0
            with open(SHADOW_CHOP_LANE_PATH, "a", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=_FIELDS)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)

    except Exception:
        pass

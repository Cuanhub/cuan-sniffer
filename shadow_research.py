"""
Canonical append-only shadow research ledger.

This module is the single shadow-research write path. It joins engine
candidates, executor decisions, and forward candle outcomes without changing
live trading behavior.
"""

from __future__ import annotations

import csv
import hashlib
import os
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import pandas as pd
except Exception:  # pragma: no cover - exercised only in dependency-thin shells.
    pd = None


DEFAULT_CANDIDATES_PATH = "shadow_research_candidates.csv"
DEFAULT_EXECUTIONS_PATH = "shadow_research_executions.csv"
DEFAULT_OUTCOMES_PATH = "shadow_research_outcomes.csv"
DEFAULT_HORIZONS: Tuple[int, ...] = (3, 6, 12, 24)

TRIGGER_KEYS = (
    "bos_bull", "bos_bear", "choch_bull", "choch_bear",
    "ob_bull", "ob_bear", "in_bull_ob", "in_bear_ob",
    "fvg_bull", "fvg_bear", "in_bull_fvg", "in_bear_fvg",
    "sweep_bull", "sweep_bear", "eq_high", "eq_low",
    "chart_pattern_confirmed", "chart_pattern_volume_confirmed",
    "chart_pattern_candle_confirmed",
)

CANDIDATE_FIELDS = [
    "timestamp_utc",
    "shadow_id",
    "setup_key",
    "symbol",
    "timeframe",
    "bar_time",
    "side",
    "entry_price",
    "stop_price",
    "tp_price",
    "rr_planned",
    "score_v1",
    "confidence_v1",
    "score_v2",
    "score_v2_version",
    "score_v2_tags",
    "score_v2_reason",
    "score_v3",
    "score_v3_version",
    "score_v3_tags",
    "score_v3_reason",
    "active_quality_model",
    "active_quality_score",
    "signal_confidence",
    "engine_decision",
    "engine_reject_reason",
    "setup_family",
    "swing_family",
    "session",
    "market_regime",
    "htf_regime",
    "macro_regime",
    "edge_buckets",
    "edge_bucket_count",
    "independent_bucket_count",
    "governance_reason",
    "triggers",
    "stop_method",
    "atr",
    "price",
    "vol_state",
    "vol_ratio",
    "source",
]

EXECUTION_FIELDS = [
    "timestamp_utc",
    "shadow_id",
    "setup_key",
    "symbol",
    "timeframe",
    "side",
    "entry_price",
    "stop_price",
    "tp_price",
    "rr_planned",
    "score_v1",
    "confidence_v1",
    "score_v2",
    "score_v3",
    "active_quality_model",
    "active_quality_score",
    "signal_confidence",
    "executor_decision",
    "executor_reject_reason",
    "executor_reject_family",
    "position_id",
    "fill_price",
    "fill_slippage_bps",
    "fill_ratio",
    "size_usd",
    "risk_usd",
    "entry_fee_usd",
    "protection_status",
    "final_entry",
    "final_stop",
    "final_tp",
    "final_rr",
    "final_stop_method",
    "stop_was_redesigned",
]

OUTCOME_FIELDS = [
    "timestamp_utc",
    "shadow_id",
    "setup_key",
    "symbol",
    "timeframe",
    "side",
    "bar_time",
    "horizon_bars",
    "bars_available",
    "entry_price",
    "stop_price",
    "tp_price",
    "planned_rr",
    "mfe_r",
    "mae_r",
    "close_r",
    "outcome_r",
    "first_touch",
    "bars_to_first_touch",
    "hit_tp",
    "hit_stop",
    "ambiguous_same_bar",
]

_LOCK = threading.Lock()
_SEEN_CANDIDATES_BY_PATH: Dict[str, set] = {}
_OUTCOME_KEYS_BY_PATH: Dict[str, set] = {}


def shadow_research_enabled() -> bool:
    return os.getenv("SHADOW_RESEARCH_ENABLED", "true").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _path(value: Optional[str], env_key: str, default: str) -> Path:
    raw = value or os.getenv(env_key, default)
    return Path(raw)


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        if pd is not None and pd.isna(value):
            return default
    except Exception:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clean(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple, set)):
        return ",".join(str(v) for v in value if v is not None)
    try:
        if pd is not None and pd.isna(value):
            return ""
    except Exception:
        pass
    return value


def _fmt_float(value: Any, places: int = 8) -> str:
    num = _safe_float(value)
    if num == 0.0 and str(value or "").strip() in {"", "nan", "None"}:
        return ""
    return f"{num:.{places}f}"


def normalize_timestamp(value: Any) -> str:
    if value is None or value == "":
        return ""
    if pd is None:
        if isinstance(value, datetime):
            return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        return str(value)
    try:
        if isinstance(value, (int, float)) and value > 10_000_000_000:
            ts = pd.to_datetime(value, unit="ms", utc=True)
        else:
            ts = pd.to_datetime(value, utc=True)
        if pd.isna(ts):
            return ""
        return ts.isoformat().replace("+00:00", "Z")
    except Exception:
        return str(value)


def extract_bar_time(row: Any) -> str:
    if row is None:
        return ""
    for key in ("time", "timestamp", "bar_time"):
        try:
            value = row.get(key, None)
        except AttributeError:
            value = None
        if value is not None and value != "":
            return normalize_timestamp(value)
    return ""


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    text = str(value).strip().lower()
    return text not in {"", "0", "false", "none", "nan", "null"}


def active_triggers(triggers: Optional[Dict[str, Any]]) -> str:
    if not triggers:
        return ""
    active: List[str] = []
    for key in TRIGGER_KEYS:
        if _truthy(triggers.get(key)):
            active.append(key)
    chart_pattern = str(triggers.get("chart_pattern", "") or "").strip()
    if chart_pattern:
        active.append(f"pattern:{chart_pattern}")
    chart_family = str(triggers.get("chart_pattern_family", "") or "").strip()
    if chart_family:
        active.append(f"pattern_family:{chart_family}")
    return ",".join(active)


def _stable_price(value: Any) -> str:
    return _fmt_float(value, 8)


def make_setup_key(
    *,
    symbol: str,
    timeframe: str,
    bar_time: str,
    side: str,
    setup_family: str,
    market_regime: str = "",
    htf_regime: str = "",
    macro_regime: str = "",
) -> str:
    return "|".join([
        str(symbol or "").upper().strip(),
        str(timeframe or "").lower().strip(),
        str(bar_time or "").strip(),
        str(side or "").upper().strip(),
        str(setup_family or "").lower().strip(),
        str(market_regime or "").lower().strip(),
        str(htf_regime or "").lower().strip(),
        str(macro_regime or "").lower().strip(),
    ])


def make_shadow_id(
    *,
    symbol: str,
    timeframe: str,
    bar_time: str,
    side: str,
    setup_family: str,
    entry_price: Any = "",
    stop_price: Any = "",
    tp_price: Any = "",
    market_regime: str = "",
    htf_regime: str = "",
    macro_regime: str = "",
) -> str:
    setup_key = make_setup_key(
        symbol=symbol,
        timeframe=timeframe,
        bar_time=bar_time,
        side=side,
        setup_family=setup_family,
        market_regime=market_regime,
        htf_regime=htf_regime,
        macro_regime=macro_regime,
    )
    payload = "|".join([
        setup_key,
        _stable_price(entry_price),
        _stable_price(stop_price),
        _stable_price(tp_price),
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def build_shadow_candidate(
    *,
    symbol: str,
    timeframe: str,
    side: str = "",
    bar_time: str = "",
    entry_price: Any = "",
    stop_price: Any = "",
    tp_price: Any = "",
    rr_planned: Any = 0.0,
    score_v1: Any = 0.0,
    confidence_v1: Any = 0.0,
    score_v2: Any = 0.0,
    score_v2_version: str = "",
    score_v2_tags: Any = "",
    score_v2_reason: str = "",
    score_v3: Any = 0.0,
    score_v3_version: str = "",
    score_v3_tags: Any = "",
    score_v3_reason: str = "",
    active_quality_model: str = "",
    active_quality_score: Any = "",
    signal_confidence: Any = "",
    engine_decision: str = "",
    engine_reject_reason: str = "",
    setup_family: str = "",
    swing_family: str = "",
    session: str = "",
    market_regime: str = "",
    htf_regime: str = "",
    macro_regime: str = "",
    edge_buckets: Any = "",
    edge_bucket_count: Any = "",
    independent_bucket_count: Any = "",
    governance_reason: str = "",
    triggers: Optional[Dict[str, Any]] = None,
    stop_method: str = "",
    atr: Any = 0.0,
    price: Any = 0.0,
    vol_state: str = "",
    vol_ratio: Any = 0.0,
    source: str = "signal_engine",
) -> Dict[str, Any]:
    symbol_norm = str(symbol or "").upper().strip()
    timeframe_norm = str(timeframe or "").lower().strip()
    side_norm = str(side or "").upper().strip()
    family_norm = str(setup_family or "").lower().strip()
    normalized_bar_time = normalize_timestamp(bar_time)
    setup_key = make_setup_key(
        symbol=symbol_norm,
        timeframe=timeframe_norm,
        bar_time=normalized_bar_time,
        side=side_norm,
        setup_family=family_norm,
        market_regime=market_regime,
        htf_regime=htf_regime,
        macro_regime=macro_regime,
    )
    shadow_id = make_shadow_id(
        symbol=symbol_norm,
        timeframe=timeframe_norm,
        bar_time=normalized_bar_time,
        side=side_norm,
        setup_family=family_norm,
        entry_price=entry_price,
        stop_price=stop_price,
        tp_price=tp_price,
        market_regime=market_regime,
        htf_regime=htf_regime,
        macro_regime=macro_regime,
    )
    return {
        "timestamp_utc": _utc_now(),
        "shadow_id": shadow_id,
        "setup_key": setup_key,
        "symbol": symbol_norm,
        "timeframe": timeframe_norm,
        "bar_time": normalized_bar_time,
        "side": side_norm,
        "entry_price": _fmt_float(entry_price),
        "stop_price": _fmt_float(stop_price),
        "tp_price": _fmt_float(tp_price),
        "rr_planned": round(_safe_float(rr_planned), 6),
        "score_v1": round(_safe_float(score_v1), 6),
        "confidence_v1": round(_safe_float(confidence_v1), 6),
        "score_v2": round(_safe_float(score_v2), 6),
        "score_v2_version": score_v2_version,
        "score_v2_tags": _clean(score_v2_tags),
        "score_v2_reason": score_v2_reason,
        "score_v3": round(_safe_float(score_v3), 6),
        "score_v3_version": score_v3_version,
        "score_v3_tags": _clean(score_v3_tags),
        "score_v3_reason": score_v3_reason,
        "active_quality_model": str(active_quality_model or "").lower().strip(),
        "active_quality_score": round(_safe_float(active_quality_score), 6),
        "signal_confidence": round(_safe_float(signal_confidence), 6),
        "engine_decision": str(engine_decision or "").lower().strip(),
        "engine_reject_reason": str(engine_reject_reason or "")[:240],
        "setup_family": family_norm,
        "swing_family": str(swing_family or "").lower().strip(),
        "session": str(session or "").lower().strip(),
        "market_regime": str(market_regime or "").lower().strip(),
        "htf_regime": str(htf_regime or "").lower().strip(),
        "macro_regime": str(macro_regime or "").lower().strip(),
        "edge_buckets": _clean(edge_buckets),
        "edge_bucket_count": _clean(edge_bucket_count),
        "independent_bucket_count": _clean(independent_bucket_count),
        "governance_reason": str(governance_reason or ""),
        "triggers": active_triggers(triggers),
        "stop_method": str(stop_method or "").lower().strip(),
        "atr": _fmt_float(atr),
        "price": _fmt_float(price),
        "vol_state": str(vol_state or "").lower().strip(),
        "vol_ratio": round(_safe_float(vol_ratio), 6),
        "source": source,
    }


def _repair_shifted_candidate_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Repair rows previously migrated from the 36-column candidate schema with
    DictReader after the four V3 fields had been inserted before engine_decision.
    """
    decision = str(row.get("engine_decision", "") or "").strip().lower()
    shifted_decision = str(row.get("session", "") or "").strip().lower()
    if decision in {"", "accepted", "rejected"}:
        return row
    if shifted_decision not in {"accepted", "rejected"}:
        return row
    if not str(row.get("engine_reject_reason", "") or "").startswith("v3_"):
        return row

    original = dict(row)
    repaired = dict(row)
    repaired.update({
        "score_v3": original.get("engine_decision", ""),
        "score_v3_version": original.get("engine_reject_reason", ""),
        "score_v3_tags": original.get("setup_family", ""),
        "score_v3_reason": original.get("swing_family", ""),
        "engine_decision": original.get("session", ""),
        "engine_reject_reason": original.get("market_regime", ""),
        "setup_family": original.get("htf_regime", ""),
        "swing_family": original.get("macro_regime", ""),
        "session": original.get("edge_buckets", ""),
        "market_regime": original.get("edge_bucket_count", ""),
        "htf_regime": original.get("independent_bucket_count", ""),
        "macro_regime": original.get("governance_reason", ""),
        "edge_buckets": original.get("triggers", ""),
        "edge_bucket_count": original.get("stop_method", ""),
        "independent_bucket_count": original.get("atr", ""),
        "governance_reason": original.get("price", ""),
        "triggers": original.get("vol_state", ""),
        "stop_method": original.get("vol_ratio", ""),
        "atr": original.get("source", ""),
        "price": original.get("entry_price", ""),
        "vol_state": "",
        "vol_ratio": "",
        "source": "signal_engine",
    })
    return repaired


def _ensure_header(path: Path, fields: Sequence[str]) -> None:
    expected = list(fields)
    if path.parent and str(path.parent) not in {"", "."}:
        path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=expected).writeheader()
        return

    try:
        with path.open("r", newline="", encoding="utf-8") as fh:
            raw_rows = list(csv.reader(fh))

        if not raw_rows:
            with path.open("w", newline="", encoding="utf-8") as fh:
                csv.DictWriter(fh, fieldnames=expected).writeheader()
            return

        existing = list(raw_rows[0])
        changed = existing != expected
        missing_expected = [field for field in expected if field not in existing]
        data_rows = []
        for values in raw_rows[1:]:
            migrated = {field: "" for field in expected}
            if missing_expected and len(values) == len(expected):
                # Header was stale but rows were already written with the new
                # canonical field order. Preserve inserted middle columns.
                migrated.update(dict(zip(expected, values)))
            else:
                migrated.update(dict(zip(existing, values)))
            if expected == CANDIDATE_FIELDS:
                repaired = _repair_shifted_candidate_row(migrated)
                if repaired != migrated:
                    changed = True
                migrated = repaired
            data_rows.append(migrated)

        if not changed:
            return

        tmp = path.with_suffix(path.suffix + ".mig")
        with tmp.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=expected, extrasaction="ignore")
            writer.writeheader()
            for row in data_rows:
                writer.writerow({field: row.get(field, "") for field in expected})
        os.replace(str(tmp), str(path))
        print(f"[SHADOW_RESEARCH] migrated header: {path.name} ({len(existing)}→{len(expected)} fields)")
    except Exception as exc:
        print(f"[SHADOW_RESEARCH] header migration failed for {path.name}: {exc}")


def _load_seen_candidate_ids(path: Path) -> set:
    key = str(path)
    if key in _SEEN_CANDIDATES_BY_PATH:
        return _SEEN_CANDIDATES_BY_PATH[key]
    seen = set()
    if path.exists() and path.stat().st_size > 0:
        try:
            with path.open("r", newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    shadow_id = str(row.get("shadow_id", "") or "").strip()
                    if shadow_id:
                        seen.add(shadow_id)
        except Exception:
            seen = set()
    _SEEN_CANDIDATES_BY_PATH[key] = seen
    return seen


def append_shadow_candidate(row: Dict[str, Any], path: Optional[str] = None) -> bool:
    if not shadow_research_enabled():
        return False
    target = _path(path, "SHADOW_RESEARCH_CANDIDATES_PATH", DEFAULT_CANDIDATES_PATH)
    shadow_id = str(row.get("shadow_id", "") or "").strip()
    if not shadow_id:
        return False
    with _LOCK:
        _ensure_header(target, CANDIDATE_FIELDS)
        seen = _load_seen_candidate_ids(target)
        if shadow_id in seen:
            return False
        with target.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=CANDIDATE_FIELDS, extrasaction="ignore")
            writer.writerow({field: _clean(row.get(field, "")) for field in CANDIDATE_FIELDS})
        seen.add(shadow_id)
    return True


def _planned_rr(entry: float, stop: float, tp: float) -> float:
    stop_dist = abs(entry - stop)
    if stop_dist <= 0:
        return 0.0
    return abs(tp - entry) / stop_dist


def _reason_family(reason: str) -> str:
    text = str(reason or "").strip().lower()
    if not text:
        return ""
    if "rr" in text or "risk_reward" in text or "reward_risk" in text:
        return "rr"
    if "stale" in text or "drift" in text or "outside_atr" in text or "overextended" in text:
        return "stale_entry"
    if "session" in text:
        return "session"
    if "cooldown" in text or "dedup" in text:
        return "cooldown"
    if "margin" in text:
        return "margin"
    if "position" in text or "capacity" in text or "bucket_dir_limit" in text:
        return "capacity"
    if "strategy_filter" in text:
        return "strategy_filter"
    if "protection" in text:
        return "protection"
    if "order" in text or "fill" in text or "sdk" in text:
        return "venue"
    return "other"


def append_shadow_execution(signal: Any, result: Any, path: Optional[str] = None) -> bool:
    if not shadow_research_enabled():
        return False
    try:
        meta = getattr(signal, "meta", None) or {}
        symbol = str(getattr(signal, "coin", "") or meta.get("coin", "")).upper().strip()
        timeframe = str(meta.get("timeframe", "") or "").lower().strip()
        side = (
            str(getattr(signal, "side", "") or "")
            .upper()
            .replace("ORDERSIDE.", "")
            .replace("POSITIONSIDE.", "")
        )
        entry = _safe_float(getattr(signal, "entry_price", 0.0))
        stop = _safe_float(getattr(signal, "stop_price", 0.0))
        tp = _safe_float(getattr(signal, "tp_price", 0.0))
        rr = _safe_float(meta.get("rr_planned"), _planned_rr(entry, stop, tp))
        bar_time = str(meta.get("bar_time", "") or "")
        setup_family = str(meta.get("setup_family", meta.get("regime_local", "")) or "").lower()
        market_regime = str(meta.get("market_regime", "") or "").lower()
        htf_regime = str(meta.get("regime_htf_1h", meta.get("htf_regime", "")) or "").lower()
        macro_regime = str(meta.get("regime_macro_4h", meta.get("macro_regime", "")) or "").lower()
        setup_key = str(meta.get("shadow_setup_key", "") or "")
        shadow_id = str(meta.get("shadow_id", "") or "")
        if not shadow_id:
            shadow_id = make_shadow_id(
                symbol=symbol,
                timeframe=timeframe,
                bar_time=bar_time,
                side=side,
                setup_family=setup_family,
                entry_price=entry,
                stop_price=stop,
                tp_price=tp,
                market_regime=market_regime,
                htf_regime=htf_regime,
                macro_regime=macro_regime,
            )
        if not setup_key:
            setup_key = make_setup_key(
                symbol=symbol,
                timeframe=timeframe,
                bar_time=bar_time,
                side=side,
                setup_family=setup_family,
                market_regime=market_regime,
                htf_regime=htf_regime,
                macro_regime=macro_regime,
            )

        final_entry = _safe_float(meta.get("final_entry"), entry)
        final_stop = _safe_float(meta.get("final_stop"), stop)
        final_tp = _safe_float(meta.get("final_tp"), tp)
        final_rr = _safe_float(meta.get("final_rr"), _planned_rr(final_entry, final_stop, final_tp))
        reason = str(getattr(result, "reason", "") or "")
        traded = bool(getattr(result, "traded", False))
        row = {
            "timestamp_utc": _utc_now(),
            "shadow_id": shadow_id,
            "setup_key": setup_key,
            "symbol": symbol,
            "timeframe": timeframe,
            "side": side,
            "entry_price": _fmt_float(entry),
            "stop_price": _fmt_float(stop),
            "tp_price": _fmt_float(tp),
            "rr_planned": round(rr, 6),
            "score_v1": round(_safe_float(meta.get("total_score", getattr(signal, "confidence", 0.0))), 6),
            "confidence_v1": round(_safe_float(meta.get("confidence_v1", meta.get("total_score", 0.0))), 6),
            "score_v2": round(_safe_float(meta.get("score_v2", 0.0)), 6),
            "score_v3": round(_safe_float(meta.get("score_v3", 0.0)), 6),
            "active_quality_model": str(meta.get("active_quality_model", "") or "").lower().strip(),
            "active_quality_score": round(_safe_float(meta.get("active_quality_score", 0.0)), 6),
            "signal_confidence": round(_safe_float(getattr(signal, "confidence", 0.0)), 6),
            "executor_decision": "accepted" if traded else "rejected",
            "executor_reject_reason": "" if traded else reason[:240],
            "executor_reject_family": "" if traded else _reason_family(reason),
            "position_id": str(getattr(result, "position_id", "") or ""),
            "fill_price": _fmt_float(getattr(result, "fill_price", "")),
            "fill_slippage_bps": round(_safe_float(getattr(result, "fill_slippage_bps", 0.0)), 6),
            "fill_ratio": round(_safe_float(getattr(result, "fill_ratio", 0.0)), 6),
            "size_usd": round(_safe_float(getattr(result, "size_usd", 0.0)), 6),
            "risk_usd": round(_safe_float(getattr(result, "risk_usd", 0.0)), 6),
            "entry_fee_usd": round(_safe_float(getattr(result, "entry_fee_usd", 0.0)), 6),
            "protection_status": str(getattr(result, "protection_status", "") or ""),
            "final_entry": _fmt_float(final_entry),
            "final_stop": _fmt_float(final_stop),
            "final_tp": _fmt_float(final_tp),
            "final_rr": round(final_rr, 6),
            "final_stop_method": str(meta.get("final_stop_method", meta.get("stop_method", "")) or ""),
            "stop_was_redesigned": _clean(bool(meta.get("stop_was_redesigned", False))),
        }
        target = _path(path, "SHADOW_RESEARCH_EXECUTIONS_PATH", DEFAULT_EXECUTIONS_PATH)
        with _LOCK:
            _ensure_header(target, EXECUTION_FIELDS)
            with target.open("a", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=EXECUTION_FIELDS, extrasaction="ignore")
                writer.writerow({field: _clean(row.get(field, "")) for field in EXECUTION_FIELDS})
        return True
    except Exception:
        return False


def _frame_with_time(df: pd.DataFrame) -> pd.DataFrame:
    if pd is None:
        return None
    if df is None or df.empty:
        return pd.DataFrame()
    frame = df.copy()
    if "time" in frame.columns:
        frame["_shadow_time"] = pd.to_datetime(frame["time"], utc=True, errors="coerce")
    elif isinstance(frame.index, pd.DatetimeIndex):
        frame["_shadow_time"] = pd.to_datetime(frame.index, utc=True, errors="coerce")
    else:
        return pd.DataFrame()
    return frame.dropna(subset=["_shadow_time"]).sort_values("_shadow_time")


def _future_bars(df: pd.DataFrame, bar_time: Any, horizon_bars: int) -> pd.DataFrame:
    if pd is None:
        return None
    frame = _frame_with_time(df)
    if frame is None or frame.empty:
        return frame
    ts = pd.to_datetime(bar_time, utc=True, errors="coerce")
    if pd.isna(ts):
        return pd.DataFrame()
    future = frame[frame["_shadow_time"] > ts]
    if len(future) < int(horizon_bars):
        return pd.DataFrame()
    return future.iloc[:int(horizon_bars)]


def compute_forward_outcome(
    df: pd.DataFrame,
    *,
    side: str,
    entry_price: Any,
    stop_price: Any,
    tp_price: Any,
    bar_time: Any,
    horizon_bars: int,
) -> Optional[Dict[str, Any]]:
    entry = _safe_float(entry_price)
    stop = _safe_float(stop_price)
    tp = _safe_float(tp_price)
    horizon = int(horizon_bars)
    side_norm = str(side or "").upper().strip()
    if side_norm not in {"LONG", "SHORT"} or horizon <= 0:
        return None
    bars = _future_bars(df, bar_time, horizon)
    if bars is None or bars.empty:
        return None
    if not {"high", "low", "close"}.issubset(set(bars.columns)):
        return None

    if side_norm == "LONG":
        r_dist = entry - stop
        if r_dist <= 0 or tp <= entry:
            return None
        mfe_r = (float(bars["high"].max()) - entry) / r_dist
        mae_r = (float(bars["low"].min()) - entry) / r_dist
        close_r = (float(bars.iloc[-1]["close"]) - entry) / r_dist
        planned_rr = (tp - entry) / r_dist
        stop_hit = lambda bar: float(bar["low"]) <= stop
        tp_hit = lambda bar: float(bar["high"]) >= tp
    else:
        r_dist = stop - entry
        if r_dist <= 0 or tp >= entry:
            return None
        mfe_r = (entry - float(bars["low"].min())) / r_dist
        mae_r = (entry - float(bars["high"].max())) / r_dist
        close_r = (entry - float(bars.iloc[-1]["close"])) / r_dist
        planned_rr = (entry - tp) / r_dist
        stop_hit = lambda bar: float(bar["high"]) >= stop
        tp_hit = lambda bar: float(bar["low"]) <= tp

    first_touch = "none"
    bars_to_touch = ""
    hit_tp = False
    hit_stop = False
    ambiguous = False
    outcome_r = close_r
    for idx, (_, bar) in enumerate(bars.iterrows(), start=1):
        bar_stop = bool(stop_hit(bar))
        bar_tp = bool(tp_hit(bar))
        if bar_stop and bar_tp:
            first_touch = "stop_first_assumed"
            bars_to_touch = idx
            hit_stop = True
            hit_tp = True
            ambiguous = True
            outcome_r = -1.0
            break
        if bar_stop:
            first_touch = "stop"
            bars_to_touch = idx
            hit_stop = True
            outcome_r = -1.0
            break
        if bar_tp:
            first_touch = "tp"
            bars_to_touch = idx
            hit_tp = True
            outcome_r = planned_rr
            break

    return {
        "horizon_bars": horizon,
        "bars_available": len(bars),
        "entry_price": round(entry, 8),
        "stop_price": round(stop, 8),
        "tp_price": round(tp, 8),
        "planned_rr": round(planned_rr, 6),
        "mfe_r": round(mfe_r, 6),
        "mae_r": round(mae_r, 6),
        "close_r": round(close_r, 6),
        "outcome_r": round(outcome_r, 6),
        "first_touch": first_touch,
        "bars_to_first_touch": bars_to_touch,
        "hit_tp": "true" if hit_tp else "false",
        "hit_stop": "true" if hit_stop else "false",
        "ambiguous_same_bar": "true" if ambiguous else "false",
    }


def parse_horizons(value: Optional[str] = None) -> Tuple[int, ...]:
    raw = value if value is not None else os.getenv("SHADOW_RESEARCH_HORIZON_BARS", "")
    if not raw:
        return DEFAULT_HORIZONS
    horizons: List[int] = []
    for part in str(raw).split(","):
        try:
            horizon = int(part.strip())
        except ValueError:
            continue
        if horizon > 0 and horizon not in horizons:
            horizons.append(horizon)
    return tuple(horizons or DEFAULT_HORIZONS)


def _load_outcome_keys(path: Path) -> set:
    key = str(path)
    if key in _OUTCOME_KEYS_BY_PATH:
        return _OUTCOME_KEYS_BY_PATH[key]
    seen = set()
    if path.exists() and path.stat().st_size > 0:
        try:
            with path.open("r", newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    shadow_id = str(row.get("shadow_id", "") or "").strip()
                    horizon = str(row.get("horizon_bars", "") or "").strip()
                    if shadow_id and horizon:
                        seen.add((shadow_id, horizon))
        except Exception:
            seen = set()
    _OUTCOME_KEYS_BY_PATH[key] = seen
    return seen


def _read_candidates(path: Path, max_rows: Optional[int] = None) -> List[Dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        with path.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            header = next(reader, [])
            if not header:
                return []
            if max_rows is not None and max_rows > 0:
                raw_rows = deque(reader, maxlen=max_rows)
            else:
                raw_rows = list(reader)
            return [
                {header[idx]: row[idx] if idx < len(row) else "" for idx in range(len(header))}
                for row in raw_rows
            ]
    except Exception:
        return []


def update_shadow_outcomes_from_df(
    *,
    symbol: str,
    timeframe: str,
    df: pd.DataFrame,
    candidate_path: Optional[str] = None,
    outcome_path: Optional[str] = None,
    horizons: Optional[Iterable[int]] = None,
    max_candidates: Optional[int] = None,
) -> int:
    if not shadow_research_enabled():
        return 0
    if pd is None or df is None or df.empty:
        return 0

    candidates_target = _path(
        candidate_path,
        "SHADOW_RESEARCH_CANDIDATES_PATH",
        DEFAULT_CANDIDATES_PATH,
    )
    outcomes_target = _path(
        outcome_path,
        "SHADOW_RESEARCH_OUTCOMES_PATH",
        DEFAULT_OUTCOMES_PATH,
    )
    horizon_values = tuple(int(h) for h in (horizons or parse_horizons()) if int(h) > 0)
    max_rows = int(
        max_candidates
        if max_candidates is not None
        else os.getenv("SHADOW_RESEARCH_MAX_CANDIDATES_PER_UPDATE", "500")
    )
    symbol_norm = str(symbol or "").upper().strip()
    timeframe_norm = str(timeframe or "").lower().strip()

    with _LOCK:
        candidates = _read_candidates(candidates_target, max_rows=max_rows)
        _ensure_header(outcomes_target, OUTCOME_FIELDS)
        seen = _load_outcome_keys(outcomes_target)

        rows_to_write: List[Dict[str, Any]] = []
        for candidate in candidates:
            if str(candidate.get("symbol", "")).upper().strip() != symbol_norm:
                continue
            if str(candidate.get("timeframe", "")).lower().strip() != timeframe_norm:
                continue
            shadow_id = str(candidate.get("shadow_id", "") or "").strip()
            if not shadow_id:
                continue
            entry = _safe_float(candidate.get("entry_price"))
            stop = _safe_float(candidate.get("stop_price"))
            tp = _safe_float(candidate.get("tp_price"))
            if entry <= 0 or stop <= 0 or tp <= 0:
                continue
            for horizon in horizon_values:
                key = (shadow_id, str(horizon))
                if key in seen:
                    continue
                outcome = compute_forward_outcome(
                    df,
                    side=candidate.get("side", ""),
                    entry_price=entry,
                    stop_price=stop,
                    tp_price=tp,
                    bar_time=candidate.get("bar_time", ""),
                    horizon_bars=horizon,
                )
                if not outcome:
                    continue
                row = {
                    "timestamp_utc": _utc_now(),
                    "shadow_id": shadow_id,
                    "setup_key": candidate.get("setup_key", ""),
                    "symbol": symbol_norm,
                    "timeframe": timeframe_norm,
                    "side": candidate.get("side", ""),
                    "bar_time": candidate.get("bar_time", ""),
                    **outcome,
                }
                rows_to_write.append(row)
                seen.add(key)

        if not rows_to_write:
            return 0
        with outcomes_target.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=OUTCOME_FIELDS, extrasaction="ignore")
            for row in rows_to_write:
                writer.writerow({field: _clean(row.get(field, "")) for field in OUTCOME_FIELDS})
        return len(rows_to_write)

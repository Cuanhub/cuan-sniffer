#!/usr/bin/env python3
"""
Standalone SMC live analyzer for Cuan Sniffer.

Reads smc_live_log.csv, signals.csv, and trades.csv without importing or
touching the live bot. The report is intentionally defensive: missing files,
empty files, and partial historical schemas should warn and keep going.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Optional


IMPORTANT_COLUMNS = {
    "smc": [
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
        "rr",
        "stop_method",
        "final_stop_method",
        "session",
        "slippage_bps",
        "partial_hit",
        "realized_r",
        "realized_pnl",
    ],
    "signals": [
        "timestamp_utc",
        "signal_id",
        "coin",
        "timeframe",
        "side",
        "entry",
        "rr_planned",
        "confidence",
        "total_score",
        "session",
        "executor_result",
        "reject_reason",
        "position_id",
        "fill_price",
        "fill_slippage_bps",
    ],
    "trades": [
        "position_id",
        "coin",
        "side",
        "signal_id",
        "realized_r",
        "pnl_usd",
        "state",
        "close_reason",
        "partial_closed",
        "rr_planned",
        "session",
        "timeframe",
        "opened_at",
        "closed_at",
    ],
}

TIMESTAMP_COLUMNS = (
    "timestamp_utc",
    "opened_at",
    "closed_at",
    "protection_placed_at",
    "pending_exit_recorded_at",
)

TRIGGER_FLAG_COLUMNS = (
    "ob_bull",
    "ob_bear",
    "fvg_bull",
    "fvg_bear",
    "bos_bull",
    "bos_bear",
    "choch_bull",
    "choch_bear",
    "sweep_bull",
    "sweep_bear",
    "eq_low",
    "eq_high",
    "in_bull_ob",
    "in_bear_ob",
    "in_bull_fvg",
    "in_bear_fvg",
)

SESSION_ORDER = {
    "asia": 0,
    "asia_open": 1,
    "asia_late": 2,
    "london": 3,
    "london_open": 4,
    "ny_open": 5,
    "ny_pm": 6,
    "dead_zone": 7,
    "unknown": 99,
    "": 99,
}

LATE_ENTRY_TERMS = (
    "fill_rr_below",
    "tp_consumed",
    "tp_already_consumed",
    "stale_entry",
    "stale_move",
    "signal_stale_move",
    "adverse_drift",
    "entry_outside_buffer",
    "entry_outside_atr_buffer",
    "too_late",
    "overextended_move",
)


@dataclass
class CsvData:
    label: str
    path: Path
    found: bool
    rows: list[dict[str, str]]
    columns: list[str]
    warnings: list[str]


@dataclass
class TradeIndex:
    by_signal: dict[str, list[dict[str, str]]]
    by_position: dict[str, dict[str, str]]
    source_note: str


def clean_text(value: object) -> str:
    return str(value if value is not None else "").strip()


def read_csv_data(label: str, path_arg: str) -> CsvData:
    path = Path(path_arg)
    warnings: list[str] = []
    if not path.exists():
        return CsvData(label, path, False, [], [], [f"{path} not found"])
    if path.stat().st_size == 0:
        return CsvData(label, path, True, [], [], [f"{path} is empty"])

    try:
        with path.open("r", newline="", encoding="utf-8-sig", errors="replace") as f:
            reader = csv.DictReader(f)
            columns = [clean_text(c) for c in (reader.fieldnames or []) if c is not None]
            rows: list[dict[str, str]] = []
            malformed = 0
            for raw in reader:
                if None in raw:
                    malformed += 1
                row = {
                    clean_text(k): clean_text(v)
                    for k, v in raw.items()
                    if k is not None and clean_text(k)
                }
                if any(v for v in row.values()):
                    rows.append(row)
            if malformed:
                warnings.append(f"{malformed} row(s) had extra columns and were trimmed")
            if not columns:
                warnings.append(f"{path} has no CSV header")
            if not rows:
                warnings.append(f"{path} has a header but no data rows")
            return CsvData(label, path, True, rows, columns, warnings)
    except csv.Error as exc:
        return CsvData(label, path, True, [], [], [f"CSV parse error: {exc}"])
    except OSError as exc:
        return CsvData(label, path, True, [], [], [f"read error: {exc}"])


def row_value(row: dict[str, str], names: Iterable[str], default: str = "") -> str:
    for name in names:
        if name in row and clean_text(row.get(name)):
            return clean_text(row.get(name))
    return default


def has_any_column(data: CsvData, names: Iterable[str]) -> bool:
    return any(name in data.columns for name in names)


def to_float(value: object) -> Optional[float]:
    text = clean_text(value)
    if not text:
        return None
    text = text.replace(",", "")
    if text.endswith("%"):
        text = text[:-1]
    try:
        return float(text)
    except ValueError:
        return None


def to_bool(value: object) -> Optional[bool]:
    text = clean_text(value).lower()
    if text in {"true", "t", "1", "yes", "y"}:
        return True
    if text in {"false", "f", "0", "no", "n"}:
        return False
    return None


def parse_datetime(value: object) -> Optional[datetime]:
    text = clean_text(value)
    if not text:
        return None
    text = text.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def find_date_range(rows: list[dict[str, str]]) -> tuple[Optional[datetime], Optional[datetime]]:
    found: list[datetime] = []
    for row in rows:
        for column in TIMESTAMP_COLUMNS:
            dt = parse_datetime(row.get(column, ""))
            if dt:
                found.append(dt)
                break
    if not found:
        return None, None
    return min(found), max(found)


def compact_dt(dt: Optional[datetime]) -> str:
    if not dt:
        return "-"
    return dt.isoformat(timespec="minutes")


def fmt_int(value: object) -> str:
    number = to_float(value)
    if number is None:
        return "-"
    return str(int(round(number)))


def fmt_float(value: object, digits: int = 2) -> str:
    number = to_float(value)
    if number is None:
        return "-"
    return f"{number:.{digits}f}"


def fmt_pct(value: Optional[float], digits: int = 1) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.{digits}f}%"


def fmt_signed(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value:+.{digits}f}"


def pct(numerator: int, denominator: int) -> Optional[float]:
    if denominator <= 0:
        return None
    return numerator / denominator


def mean(values: Iterable[float]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def truncate(value: object, width: int) -> str:
    text = clean_text(value)
    if len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    return text[: width - 3] + "..."


def print_section(title: str) -> None:
    print(f"\n=== {title} ===")


def print_note(text: str) -> None:
    print(f"  {text}")


def print_table(headers: list[str], rows: list[list[object]], max_width: int = 24) -> None:
    if not rows:
        print("  (no rows)")
        return
    rendered = [[truncate(cell, max_width) for cell in row] for row in rows]
    widths = []
    for idx, header in enumerate(headers):
        width = len(header)
        for row in rendered:
            if idx < len(row):
                width = max(width, len(row[idx]))
        widths.append(min(width, max_width))

    def line_for(row: list[object]) -> str:
        cells = []
        for idx, width in enumerate(widths):
            cell = truncate(row[idx] if idx < len(row) else "", width)
            cells.append(cell.ljust(width))
        return "  " + " | ".join(cells)

    print(line_for(headers))
    print("  " + "-+-".join("-" * w for w in widths))
    for row in rendered:
        print(line_for(row))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def normalize_reason_family(reason: str) -> str:
    text = clean_text(reason).lower()
    compact = text.replace("-", "_").replace(" ", "_")
    if not compact:
        return "unknown"
    if "fill_rr_below" in compact:
        return "fill_rr_below"
    if (
        "rr_too_low" in compact
        or "final_rr_below_min" in compact
        or "reward_risk" in compact
        or "risk_reward" in compact
    ):
        return "rr_too_low"
    if (
        "signal_stale_move" in compact
        or "entry_outside_atr_buffer" in compact
        or "entry_outside_buffer" in compact
        or "too_late" in compact
        or "adverse_drift" in compact
        or "overextended_move" in compact
        or "stale_or_drift" in compact
        or "stale_entry" in compact
    ):
        return "stale_entry"
    if "tp_already_consumed" in compact or "tp_consumed" in compact:
        return "tp_consumed"
    if "session_soft_blocked" in compact or "session_blocked" in compact:
        return "session_block"
    if "hard_blocked_timeframe" in compact or "4h_disabled" in compact:
        return "hard_timeframe_block"
    if "strategy_filter" in compact:
        return "strategy_filter_block"
    if "score_below_threshold" in compact:
        return "score_below_threshold"
    if "daily_loss" in compact or "daily_halt" in compact or "halted_daily" in compact:
        return "daily_halt"
    if "max_positions" in compact or "bucket_dir_limit" in compact or "position_capacity" in compact:
        return "max_positions"
    if "cooldown" in compact or "reject_throttled" in compact:
        return "cooldown"
    if "margin" in compact:
        return "margin"
    if "fill_rejected" in compact or "entry_rejected" in compact or "sdk_error" in compact or "live_order" in compact:
        return "live_order_reject"
    if "protection" in compact:
        return "protection_fail"
    if "no_valid_setup" in compact:
        return "no_valid_setup"
    if "insufficient_swing_confluence" in compact:
        return "insufficient_confluence"
    return "unknown"


def rejection_family(row: dict[str, str]) -> str:
    family = row_value(row, ["reason_family"])
    if family and family.lower() != "unknown":
        return family.lower()
    return normalize_reason_family(row_value(row, ["reject_reason", "reason"]))


def raw_reject_reason(row: dict[str, str]) -> str:
    return row_value(row, ["reject_reason", "reason"], "unknown")


def is_rejected(row: dict[str, str]) -> bool:
    accepted = to_bool(row.get("accepted", ""))
    if accepted is False:
        return True
    result = row_value(row, ["executor_result"]).lower()
    if result in {"rejected", "error"}:
        return True
    return bool(raw_reject_reason(row) and raw_reject_reason(row) != "unknown")


def is_accepted(row: dict[str, str]) -> bool:
    accepted = to_bool(row.get("accepted", ""))
    if accepted is True:
        return True
    result = row_value(row, ["executor_result"]).lower()
    return result == "traded"


def accepted_rejected_counts(rows: list[dict[str, str]]) -> tuple[int, int, int]:
    accepted = 0
    rejected = 0
    known = 0
    for row in rows:
        val = to_bool(row.get("accepted", ""))
        if val is not None:
            known += 1
            if val:
                accepted += 1
            else:
                rejected += 1
            continue
        result = row_value(row, ["executor_result"]).lower()
        if result in {"traded", "rejected", "error"}:
            known += 1
            if result == "traded":
                accepted += 1
            else:
                rejected += 1
    return accepted, rejected, known


def is_closed_trade(row: dict[str, str]) -> bool:
    state = row_value(row, ["state"]).lower()
    if state == "closed":
        return True
    if state in {"open", "pending", "active"}:
        return False
    if row_value(row, ["closed_at"]):
        return True
    if row_value(row, ["close_reason"]) and to_float(row_value(row, ["realized_r"])) is not None:
        return True
    return False


def trade_r(row: dict[str, str]) -> Optional[float]:
    return to_float(row_value(row, ["realized_r", "runner_r"]))


def trade_pnl(row: dict[str, str]) -> Optional[float]:
    return to_float(row_value(row, ["pnl_usd", "realized_pnl"]))


def trade_partial(row: dict[str, str]) -> Optional[bool]:
    val = to_bool(row_value(row, ["partial_closed", "partial_hit"]))
    if val is not None:
        return val
    partial_r = to_float(row_value(row, ["partial_r"]))
    if partial_r is not None:
        return partial_r > 0
    return None


def trade_rr(row: dict[str, str]) -> Optional[float]:
    return to_float(row_value(row, ["rr_planned", "final_rr", "rr"]))


def slippage_bps(row: dict[str, str]) -> Optional[float]:
    for name in ("slippage_bps", "fill_slippage_bps", "pending_exit_slippage_bps"):
        value = to_float(row.get(name, ""))
        if value is not None:
            return value
    return None


def trade_stats(rows: list[dict[str, str]]) -> dict[str, object]:
    closed = [row for row in rows if is_closed_trade(row)]
    r_values = [v for v in (trade_r(row) for row in closed) if v is not None]
    pnl_values = [v for v in (trade_pnl(row) for row in closed) if v is not None]
    partial_values = [v for v in (trade_partial(row) for row in closed) if v is not None]
    rr_values = [v for v in (trade_rr(row) for row in rows) if v is not None]
    slip_values = [v for v in (slippage_bps(row) for row in rows) if v is not None]
    wins = sum(1 for value in r_values if value > 0)
    return {
        "count": len(rows),
        "closed": len(closed),
        "open": len(rows) - len(closed),
        "total_r": sum(r_values) if r_values else None,
        "avg_r": mean(r_values),
        "win_rate": pct(wins, len(r_values)),
        "total_pnl": sum(pnl_values) if pnl_values else None,
        "partial_hit_rate": pct(sum(1 for value in partial_values if value), len(partial_values)),
        "avg_rr": mean(rr_values),
        "avg_slippage_bps": mean(slip_values),
    }


def sample_confidence(n: int) -> str:
    if n < 10:
        return "very low"
    if n < 30:
        return "low"
    if n < 100:
        return "medium"
    return "higher"


def recommendation_from_stats(
    evals: int,
    closed: int,
    avg_r: Optional[float],
    win_rate: Optional[float],
    rejection_rate: Optional[float],
    partial_rate: Optional[float] = None,
) -> str:
    if closed < 10:
        if evals >= 100 and rejection_rate is not None and rejection_rate >= 0.98:
            return "reduce"
        return "watch"
    if closed >= 10 and avg_r is not None:
        if avg_r >= 0.20 and (win_rate is None or win_rate >= 0.45):
            return "keep"
        if avg_r <= -0.25 and (win_rate is None or win_rate < 0.45):
            return "avoid"
        if avg_r < 0.0 and (win_rate is None or win_rate < 0.40):
            return "reduce"
    if closed >= 10 and partial_rate is not None and partial_rate < 0.15 and avg_r is not None and avg_r <= 0:
        return "reduce"
    return "watch"


def bool_flag(row: dict[str, str], name: str) -> bool:
    return to_bool(row.get(name, "")) is True


def normal_key(value: str, default: str = "unknown") -> str:
    text = clean_text(value).lower()
    return text if text else default


def side_key(row: dict[str, str]) -> str:
    side = row_value(row, ["side"]).upper()
    return side if side in {"LONG", "SHORT"} else "UNKNOWN"


def timeframe_key(row: dict[str, str]) -> str:
    tf = normal_key(row_value(row, ["timeframe"]), "unknown")
    if tf in {"15", "15min", "15m"}:
        return "15m"
    if tf in {"1h", "60m", "60min"}:
        return "1h"
    if tf in {"4h", "240m", "240min"}:
        return "4h"
    return tf


def session_key(row: dict[str, str]) -> str:
    return normal_key(row_value(row, ["session"]), "unknown")


def coin_key(row: dict[str, str]) -> str:
    return row_value(row, ["coin"], "UNKNOWN").upper()


def group_rows(
    rows: list[dict[str, str]],
    key_fn: Callable[[dict[str, str]], str],
) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[key_fn(row)].append(row)
    return groups


def top_counter_rows(counter: Counter, limit: int = 10) -> list[list[object]]:
    return [[key, count] for key, count in counter.most_common(limit)]


def make_trade_indexes(trades: list[dict[str, str]], smc_rows: list[dict[str, str]]) -> TradeIndex:
    by_signal: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_position: dict[str, dict[str, str]] = {}
    source_note = "trade outcomes sourced from trades.csv"

    for row in trades:
        signal_id = row_value(row, ["signal_id"])
        position_id = row_value(row, ["position_id"])
        if signal_id:
            by_signal[signal_id].append(row)
        if position_id:
            by_position[position_id] = row

    added_from_smc = 0
    for row in smc_rows:
        if row_value(row, ["event_type"]).lower() != "trade_state":
            continue
        if to_float(row_value(row, ["realized_r"])) is None:
            continue
        signal_id = row_value(row, ["signal_id"])
        position_id = row_value(row, ["position_id"])
        if position_id and position_id in by_position:
            continue
        if signal_id and by_signal.get(signal_id):
            continue
        synthetic = {
            "position_id": position_id,
            "signal_id": signal_id,
            "coin": row_value(row, ["coin"]),
            "side": row_value(row, ["side"]),
            "timeframe": row_value(row, ["timeframe"]),
            "session": row_value(row, ["session"]),
            "state": "closed" if row_value(row, ["close_reason"]) else "",
            "close_reason": row_value(row, ["close_reason"]),
            "realized_r": row_value(row, ["realized_r"]),
            "pnl_usd": row_value(row, ["realized_pnl"]),
            "partial_closed": row_value(row, ["partial_hit"]),
            "rr_planned": row_value(row, ["final_rr", "rr"]),
            "slippage_bps": row_value(row, ["slippage_bps"]),
            "source": "smc_live_log",
        }
        if signal_id:
            by_signal[signal_id].append(synthetic)
        if position_id:
            by_position[position_id] = synthetic
        added_from_smc += 1

    if added_from_smc and not trades:
        source_note = "trade outcomes sourced from smc_live_log trade_state rows"
    elif added_from_smc:
        source_note += f"; {added_from_smc} extra smc trade_state outcome(s) filled gaps"
    return TradeIndex(dict(by_signal), by_position, source_note)


def linked_trades(rows: list[dict[str, str]], trade_index: TradeIndex) -> list[dict[str, str]]:
    found: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        position_id = row_value(row, ["position_id"])
        signal_id = row_value(row, ["signal_id"])
        candidates: list[dict[str, str]] = []
        if position_id and position_id in trade_index.by_position:
            candidates.append(trade_index.by_position[position_id])
        elif signal_id and signal_id in trade_index.by_signal:
            candidates.extend(trade_index.by_signal[signal_id])
        for trade in candidates:
            key = row_value(trade, ["position_id"]) or row_value(trade, ["signal_id"]) or str(id(trade))
            if key not in seen:
                seen.add(key)
                found.append(trade)
    return found


def smc_group_summary(rows: list[dict[str, str]], trade_index: TradeIndex) -> dict[str, object]:
    accepted, rejected, known = accepted_rejected_counts(rows)
    trades = linked_trades(rows, trade_index)
    stats = trade_stats(trades)
    rr_values = [v for v in (to_float(row_value(row, ["final_rr", "rr", "rr_planned"])) for row in rows) if v is not None]
    slip_values = [v for v in (slippage_bps(row) for row in rows) if v is not None]
    rejection_rate = pct(rejected, known)
    return {
        "evaluations": len(rows),
        "accepted": accepted,
        "rejected": rejected,
        "accepted_rate": pct(accepted, known),
        "rejection_rate": rejection_rate,
        "closed": stats["closed"],
        "total_r": stats["total_r"],
        "avg_r": stats["avg_r"],
        "win_rate": stats["win_rate"],
        "partial_hit_rate": stats["partial_hit_rate"],
        "avg_rr": mean(rr_values) if rr_values else stats["avg_rr"],
        "avg_slippage_bps": mean(slip_values) if slip_values else stats["avg_slippage_bps"],
        "recommendation": recommendation_from_stats(
            len(rows),
            int(stats["closed"]),
            stats["avg_r"] if isinstance(stats["avg_r"], float) else None,
            stats["win_rate"] if isinstance(stats["win_rate"], float) else None,
            rejection_rate,
            stats["partial_hit_rate"] if isinstance(stats["partial_hit_rate"], float) else None,
        ),
    }


def performance_rows(
    rows_by_key: dict[str, list[dict[str, str]]],
    limit: Optional[int] = None,
) -> list[list[object]]:
    out: list[list[object]] = []
    for key, rows in rows_by_key.items():
        stats = trade_stats(rows)
        out.append([
            key,
            stats["count"],
            stats["closed"],
            fmt_signed(stats["total_r"], 2),
            fmt_signed(stats["avg_r"], 3),
            fmt_pct(stats["win_rate"]),
            fmt_signed(stats["total_pnl"], 2),
            fmt_pct(stats["partial_hit_rate"]),
        ])
    out.sort(key=lambda r: (to_float(r[4]) if to_float(r[4]) is not None else -999), reverse=True)
    return out[:limit] if limit else out


def group_summary_dict(name: str, stats: dict[str, object]) -> dict[str, object]:
    return {
        "group": name,
        "evaluations": stats["evaluations"],
        "accepted": stats["accepted"],
        "rejected": stats["rejected"],
        "accepted_rate": fmt_pct(stats["accepted_rate"]),
        "closed_trades": stats["closed"],
        "total_r": fmt_signed(stats["total_r"], 3),
        "avg_r": fmt_signed(stats["avg_r"], 3),
        "win_rate": fmt_pct(stats["win_rate"]),
        "partial_hit_rate": fmt_pct(stats["partial_hit_rate"]),
        "avg_rr": fmt_float(stats["avg_rr"], 2),
        "avg_slippage_bps": fmt_signed(stats["avg_slippage_bps"], 2),
        "recommendation": stats["recommendation"],
    }


def print_group_summary_table(rows: list[dict[str, object]], limit: Optional[int] = None) -> None:
    clipped = rows[:limit] if limit else rows
    print_table(
        [
            "Group",
            "Eval",
            "Acc",
            "Rej",
            "AccRate",
            "Closed",
            "AvgR",
            "Win",
            "Partial",
            "Rec",
        ],
        [
            [
                row["group"],
                row["evaluations"],
                row["accepted"],
                row["rejected"],
                row["accepted_rate"],
                row["closed_trades"],
                row["avg_r"],
                row["win_rate"],
                row["partial_hit_rate"],
                row["recommendation"],
            ]
            for row in clipped
        ],
    )


def score_bucket(value: Optional[float]) -> str:
    if value is None:
        return "unknown"
    if value < 0.60:
        return "<0.60"
    if value < 0.70:
        return "0.60-0.69"
    if value < 0.80:
        return "0.70-0.79"
    if value < 0.90:
        return "0.80-0.89"
    return ">=0.90"


def score_value(row: dict[str, str]) -> Optional[float]:
    score = to_float(row_value(row, ["score", "total_score"]))
    if score is not None:
        return score
    return to_float(row_value(row, ["confidence"]))


def stop_method_key(row: dict[str, str], column: str) -> str:
    raw = normal_key(row_value(row, [column]), "")
    if not raw:
        return "unknown"
    if "ob" in raw:
        return "ob" if raw == "ob" else raw
    if "atr" in raw:
        return "atr" if raw == "atr" else raw
    return raw


def aligned_with_macro(row: dict[str, str]) -> bool:
    side = side_key(row)
    macro = row_value(row, ["macro_regime", "regime_macro_4h", "htf_regime", "regime_htf_1h"]).lower()
    if side == "LONG":
        return "up" in macro
    if side == "SHORT":
        return "down" in macro
    return False


def regime_contains(row: dict[str, str], word: str) -> bool:
    haystack = " ".join(
        row_value(row, [name])
        for name in ("macro_regime", "htf_regime", "regime_macro_4h", "regime_htf_1h", "regime")
    ).lower()
    return word.lower() in haystack


def trigger_definitions() -> list[tuple[str, Callable[[dict[str, str]], bool]]]:
    return [
        ("ob_bull", lambda r: bool_flag(r, "ob_bull")),
        ("ob_bear", lambda r: bool_flag(r, "ob_bear")),
        ("fvg_bull", lambda r: bool_flag(r, "fvg_bull")),
        ("fvg_bear", lambda r: bool_flag(r, "fvg_bear")),
        ("bos_bull", lambda r: bool_flag(r, "bos_bull")),
        ("bos_bear", lambda r: bool_flag(r, "bos_bear")),
        ("choch_bull", lambda r: bool_flag(r, "choch_bull")),
        ("choch_bear", lambda r: bool_flag(r, "choch_bear")),
        ("sweep_bull", lambda r: bool_flag(r, "sweep_bull")),
        ("sweep_bear", lambda r: bool_flag(r, "sweep_bear")),
        ("eq_low", lambda r: bool_flag(r, "eq_low")),
        ("eq_high", lambda r: bool_flag(r, "eq_high")),
        ("eq_low + sweep_bull", lambda r: bool_flag(r, "eq_low") and bool_flag(r, "sweep_bull")),
        ("eq_high + sweep_bear", lambda r: bool_flag(r, "eq_high") and bool_flag(r, "sweep_bear")),
        (
            "OB + FVG overlap",
            lambda r: (
                (bool_flag(r, "in_bull_ob") and bool_flag(r, "in_bull_fvg"))
                or (bool_flag(r, "in_bear_ob") and bool_flag(r, "in_bear_fvg"))
                or (bool_flag(r, "ob_bull") and bool_flag(r, "fvg_bull"))
                or (bool_flag(r, "ob_bear") and bool_flag(r, "fvg_bear"))
            ),
        ),
        (
            "sweep + CHoCH",
            lambda r: (
                (bool_flag(r, "sweep_bull") and bool_flag(r, "choch_bull"))
                or (bool_flag(r, "sweep_bear") and bool_flag(r, "choch_bear"))
            ),
        ),
    ]


def candidate_definitions() -> list[tuple[str, Callable[[dict[str, str]], bool]]]:
    return [
        (
            "LONG + ob_bull + htf up",
            lambda r: side_key(r) == "LONG" and bool_flag(r, "ob_bull") and regime_contains(r, "up"),
        ),
        (
            "LONG + eq_low + sweep_bull",
            lambda r: side_key(r) == "LONG" and bool_flag(r, "eq_low") and bool_flag(r, "sweep_bull"),
        ),
        (
            "SHORT + eq_high + sweep_bear",
            lambda r: side_key(r) == "SHORT" and bool_flag(r, "eq_high") and bool_flag(r, "sweep_bear"),
        ),
        (
            "1h + OB stop + macro aligned",
            lambda r: timeframe_key(r) == "1h" and "ob" in stop_method_key(r, "final_stop_method")
            and aligned_with_macro(r),
        ),
        (
            "15m + OB/FVG overlap",
            lambda r: timeframe_key(r) == "15m"
            and (
                (bool_flag(r, "in_bull_ob") and bool_flag(r, "in_bull_fvg"))
                or (bool_flag(r, "in_bear_ob") and bool_flag(r, "in_bear_fvg"))
                or (bool_flag(r, "ob_bull") and bool_flag(r, "fvg_bull"))
                or (bool_flag(r, "ob_bear") and bool_flag(r, "fvg_bear"))
            ),
        ),
        (
            "LONG + sweep_bull + CHoCH",
            lambda r: side_key(r) == "LONG" and bool_flag(r, "sweep_bull") and bool_flag(r, "choch_bull"),
        ),
        (
            "SHORT + sweep_bear + CHoCH",
            lambda r: side_key(r) == "SHORT" and bool_flag(r, "sweep_bear") and bool_flag(r, "choch_bear"),
        ),
        ("macro aligned", aligned_with_macro),
    ]


def collect_rejection_rows(smc: CsvData, signals: CsvData) -> tuple[list[dict[str, str]], str]:
    smc_rejections = [row for row in smc.rows if is_rejected(row)]
    if smc_rejections:
        return smc_rejections, "smc_live_log.csv"
    signal_rejections = [row for row in signals.rows if is_rejected(row)]
    return signal_rejections, "signals.csv fallback"


def keyword_match(row: dict[str, str], terms: Iterable[str]) -> bool:
    text = f"{raw_reject_reason(row)} {rejection_family(row)}".lower()
    return any(term.lower() in text for term in terms)


def collect_fill_rows(smc: CsvData, signals: CsvData) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    for source, rows in (("smc", smc.rows), ("signals", signals.rows)):
        for row in rows:
            fill = to_float(row_value(row, ["fill_price"]))
            entry = to_float(row_value(row, ["entry"]))
            slip = slippage_bps(row)
            if fill is None or fill <= 0 or entry is None or entry <= 0:
                continue
            if not is_accepted(row) and not row_value(row, ["position_id"]):
                continue
            enriched = dict(row)
            enriched["_source"] = source
            if slip is None:
                side = side_key(row)
                if side == "SHORT":
                    slip = (entry - fill) / entry * 10000
                else:
                    slip = (fill - entry) / entry * 10000
                enriched["_computed_slippage_bps"] = f"{slip:.6f}"
            candidates.append(enriched)

    deduped: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in candidates:
        key = row_value(row, ["signal_id"]) or f"{coin_key(row)}:{row_value(row, ['timestamp_utc'])}:{row_value(row, ['fill_price'])}"
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def fill_slip(row: dict[str, str]) -> Optional[float]:
    return slippage_bps(row) if slippage_bps(row) is not None else to_float(row.get("_computed_slippage_bps", ""))


def describe_missing_columns(data: CsvData) -> list[str]:
    expected = IMPORTANT_COLUMNS.get(data.label, [])
    return [column for column in expected if column not in data.columns]


def section_data_health(smc: CsvData, signals: CsvData, trades: CsvData) -> int:
    print_section("1. Data Health Summary")
    rows = []
    for data in (smc, signals, trades):
        start, end = find_date_range(data.rows)
        rows.append([
            data.label,
            "yes" if data.found else "no",
            len(data.rows),
            compact_dt(start),
            compact_dt(end),
            len(data.columns),
        ])
    print_table(["File", "Found", "Rows", "Start", "End", "Cols"], rows, max_width=28)

    for data in (smc, signals, trades):
        for warning in data.warnings:
            print_note(f"WARNING [{data.label}]: {warning}")

    if smc.rows and has_any_column(smc, ["event_type"]):
        print_note("Detected smc_live_log event types:")
        print_table(["Event type", "Rows"], top_counter_rows(Counter(row_value(r, ["event_type"], "unknown") for r in smc.rows), 12))

    missing_rows = []
    for data in (smc, signals, trades):
        missing = describe_missing_columns(data)
        if missing:
            missing_rows.append([data.label, ", ".join(missing[:16]) + (" ..." if len(missing) > 16 else "")])
    if missing_rows:
        print_note("Missing important columns:")
        print_table(["File", "Missing columns"], missing_rows, max_width=60)
    else:
        print_note("All important columns for this analyzer were detected.")

    closed_count = len([row for row in trades.rows if is_closed_trade(row)])
    if closed_count < 10:
        print_note(f"WARNING: only {closed_count} closed trade(s); profitability conclusions are very low confidence.")
    elif closed_count < 30:
        print_note(f"Sample warning: {closed_count} closed trades; profitability conclusions are low confidence.")
    return closed_count


def section_overall_performance(trades: CsvData) -> dict[str, object]:
    print_section("2. Overall Live Performance From trades.csv")
    if not trades.rows:
        print_note("No trades.csv rows available; skipping realized performance.")
        return {}

    stats = trade_stats(trades.rows)
    print_table(
        ["Trades", "Closed", "Open", "Total R", "Avg R", "Win", "Total PnL", "Partial"],
        [[
            stats["count"],
            stats["closed"],
            stats["open"],
            fmt_signed(stats["total_r"], 3),
            fmt_signed(stats["avg_r"], 3),
            fmt_pct(stats["win_rate"]),
            fmt_signed(stats["total_pnl"], 2),
            fmt_pct(stats["partial_hit_rate"]),
        ]],
    )

    closed_trades = [row for row in trades.rows if is_closed_trade(row)]
    if not closed_trades:
        print_note("No closed trades found; open-position rows are present but final R is not available.")
        return stats

    print_note("LONG vs SHORT:")
    print_table(
        ["Side", "Rows", "Closed", "Total R", "Avg R", "Win", "PnL", "Partial"],
        performance_rows(group_rows(closed_trades, side_key)),
    )
    print_note("Timeframe performance:")
    print_table(
        ["TF", "Rows", "Closed", "Total R", "Avg R", "Win", "PnL", "Partial"],
        performance_rows(group_rows(closed_trades, timeframe_key)),
    )
    print_note("Coin performance:")
    print_table(
        ["Coin", "Rows", "Closed", "Total R", "Avg R", "Win", "PnL", "Partial"],
        performance_rows(group_rows(closed_trades, coin_key), limit=15),
    )
    return stats


def section_smc_engine_summary(smc: CsvData, signals: CsvData) -> None:
    print_section("3. SMC Engine Evaluation Summary From smc_live_log.csv")
    if not smc.rows:
        print_note("No smc_live_log rows available; skipping SMC engine summary.")
        return

    engine_rows = [row for row in smc.rows if row_value(row, ["event_type"]).lower() == "engine_evaluation"]
    accepted, rejected, known = accepted_rejected_counts(smc.rows)
    print_table(
        ["Metric", "Value"],
        [
            ["Total engine evaluations", len(engine_rows)],
            ["Rows with accepted/rejected state", known],
            ["Accepted", accepted],
            ["Rejected", rejected],
            ["Accepted rate", fmt_pct(pct(accepted, known))],
        ],
    )

    print_note("Event type counts:")
    print_table(["Event", "Count"], top_counter_rows(Counter(row_value(r, ["event_type"], "unknown") for r in smc.rows), 12))

    rejection_rows, source = collect_rejection_rows(smc, signals)
    print_note(f"Rejection source: {source}")
    print_note("Top rejection reason families:")
    print_table(["Family", "Count"], top_counter_rows(Counter(rejection_family(r) for r in rejection_rows), 12))
    print_note("Top raw rejection reasons:")
    print_table(["Raw reason", "Count"], top_counter_rows(Counter(raw_reject_reason(r) for r in rejection_rows), 12), max_width=48)
    print_note("Rejected by timeframe:")
    print_table(["TF", "Count"], top_counter_rows(Counter(timeframe_key(r) for r in rejection_rows), 12))
    print_note("Rejected by session:")
    print_table(["Session", "Count"], top_counter_rows(Counter(session_key(r) for r in rejection_rows), 12))


def section_stop_method(smc: CsvData, trade_index: TradeIndex, out_dir: Optional[Path]) -> list[dict[str, object]]:
    print_section("4. Stop Method Performance")
    if not smc.rows:
        print_note("No SMC rows available; stop method cannot be evaluated.")
        return []
    if not has_any_column(smc, ["stop_method", "final_stop_method"]):
        print_note("No stop_method/final_stop_method columns found; skipping.")
        return []

    print_note(trade_index.source_note)
    print_note("If an engine_evaluation row has no signal_id/position_id, it cannot be linked to closed P&L.")

    summaries: list[dict[str, object]] = []
    for label, column in (("Initial stop_method", "stop_method"), ("Final stop_method", "final_stop_method")):
        method_rows = [row for row in smc.rows if row_value(row, [column])]
        if not method_rows:
            print_note(f"{label}: no populated rows.")
            continue
        grouped = group_rows(method_rows, lambda r, c=column: stop_method_key(r, c))
        table_rows: list[dict[str, object]] = []
        for method, rows in grouped.items():
            stats = smc_group_summary(rows, trade_index)
            summary = group_summary_dict(method, stats)
            summary["kind"] = label
            table_rows.append(summary)
            summaries.append(summary)
        table_rows.sort(key=lambda row: (row["kind"], to_float(row["avg_r"]) if to_float(row["avg_r"]) is not None else -999), reverse=True)
        print_note(label)
        print_table(
            ["Method", "Count", "Closed", "Win", "Avg R", "Total R", "Partial", "Avg RR", "Slip"],
            [
                [
                    row["group"],
                    row["evaluations"],
                    row["closed_trades"],
                    row["win_rate"],
                    row["avg_r"],
                    row["total_r"],
                    row["partial_hit_rate"],
                    row["avg_rr"],
                    row["avg_slippage_bps"],
                ]
                for row in table_rows
            ],
        )

    if out_dir:
        write_csv(out_dir / "summary_by_stop_method.csv", summaries)
    return summaries


def section_trigger_family(smc: CsvData, trade_index: TradeIndex, out_dir: Optional[Path]) -> list[dict[str, object]]:
    print_section("5. Trigger Family Performance")
    if not smc.rows:
        print_note("No SMC rows available; trigger flags cannot be evaluated.")
        return []
    if not any(column in smc.columns for column in TRIGGER_FLAG_COLUMNS):
        print_note("No SMC trigger flag columns found; skipping.")
        return []

    rows_out: list[dict[str, object]] = []
    for name, predicate in trigger_definitions():
        rows = [row for row in smc.rows if predicate(row)]
        if not rows:
            continue
        summary = group_summary_dict(name, smc_group_summary(rows, trade_index))
        rows_out.append(summary)
    rows_out.sort(
        key=lambda row: (
            to_float(row["avg_r"]) if to_float(row["avg_r"]) is not None else -999,
            to_float(row["accepted_rate"]) if to_float(row["accepted_rate"]) is not None else -999,
            int(row["evaluations"]),
        ),
        reverse=True,
    )
    print_group_summary_table(rows_out)
    if out_dir:
        write_csv(out_dir / "summary_by_trigger_family.csv", rows_out)
    return rows_out


def section_score_buckets(smc: CsvData, trade_index: TradeIndex) -> list[dict[str, object]]:
    print_section("6. Score and Confidence Buckets")
    if not smc.rows or not has_any_column(smc, ["score", "total_score", "confidence"]):
        print_note("No score/confidence columns found; skipping.")
        return []

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in smc.rows:
        grouped[score_bucket(score_value(row))].append(row)

    order = ["<0.60", "0.60-0.69", "0.70-0.79", "0.80-0.89", ">=0.90", "unknown"]
    rows_out: list[dict[str, object]] = []
    for bucket in order:
        rows = grouped.get(bucket, [])
        if not rows:
            continue
        summary = group_summary_dict(bucket, smc_group_summary(rows, trade_index))
        rows_out.append(summary)

    print_group_summary_table(rows_out)
    return rows_out


def section_timeframe(smc: CsvData, trades: CsvData, out_dir: Optional[Path]) -> list[dict[str, object]]:
    print_section("7. Timeframe Performance")
    smc_by_tf = group_rows(smc.rows, timeframe_key) if smc.rows else {}
    closed_trades = [row for row in trades.rows if is_closed_trade(row)]
    trade_by_tf = group_rows(closed_trades, timeframe_key) if closed_trades else {}

    keys = sorted(set(smc_by_tf) | set(trade_by_tf), key=lambda k: {"15m": 0, "1h": 1, "4h": 2}.get(k, 99))
    rows_out: list[dict[str, object]] = []
    table_rows: list[list[object]] = []
    for key in keys:
        smc_rows = smc_by_tf.get(key, [])
        accepted, rejected, known = accepted_rejected_counts(smc_rows)
        stats = trade_stats(trade_by_tf.get(key, []))
        row = {
            "timeframe": key,
            "evaluations": len(smc_rows),
            "accepted": accepted,
            "rejected": rejected,
            "accepted_rate": fmt_pct(pct(accepted, known)),
            "closed_trades": stats["closed"],
            "total_r": fmt_signed(stats["total_r"], 3),
            "avg_r": fmt_signed(stats["avg_r"], 3),
            "win_rate": fmt_pct(stats["win_rate"]),
        }
        rows_out.append(row)
        table_rows.append([
            key,
            len(smc_rows),
            accepted,
            stats["closed"],
            row["total_r"],
            row["avg_r"],
            row["win_rate"],
        ])
    print_table(["TF", "Eval", "Accepted", "Closed", "Total R", "Avg R", "Win"], table_rows)

    four_h_active = False
    four_h_rows = smc_by_tf.get("4h", [])
    four_h_accepted = accepted_rejected_counts(four_h_rows)[0]
    four_h_trades = len(trade_by_tf.get("4h", []))
    if four_h_accepted or four_h_trades:
        four_h_active = True
    if four_h_active:
        if four_h_accepted:
            print_note("WARNING: 4H SMC rows are accepted despite SMC_ENABLE_4H_LIVE=false expectation.")
        else:
            print_note("WARNING: trades.csv contains 4H closed trades; verify they are historical if 4H live is disabled.")
    elif four_h_rows:
        print_note("4H evaluation rows exist, but no accepted 4H trades were detected.")

    if out_dir:
        write_csv(out_dir / "summary_by_timeframe.csv", rows_out)
    return rows_out


def section_side(smc: CsvData, trades: CsvData, trade_index: TradeIndex, out_dir: Optional[Path]) -> list[dict[str, object]]:
    print_section("8. Side Performance")
    closed_trades = [row for row in trades.rows if is_closed_trade(row)]
    smc_by_side = group_rows(smc.rows, side_key) if smc.rows else {}
    trade_by_side = group_rows(closed_trades, side_key) if closed_trades else {}
    keys = sorted(set(smc_by_side) | set(trade_by_side))

    rows_out: list[dict[str, object]] = []
    table_rows: list[list[object]] = []
    for key in keys:
        smc_rows = smc_by_side.get(key, [])
        accepted, rejected, known = accepted_rejected_counts(smc_rows)
        stats = trade_stats(trade_by_side.get(key, []))
        row = {
            "side": key,
            "count": len(smc_rows),
            "closed_trades": stats["closed"],
            "total_r": fmt_signed(stats["total_r"], 3),
            "avg_r": fmt_signed(stats["avg_r"], 3),
            "win_rate": fmt_pct(stats["win_rate"]),
            "partial_hit_rate": fmt_pct(stats["partial_hit_rate"]),
            "accepted": accepted,
            "rejected": rejected,
            "accepted_rate": fmt_pct(pct(accepted, known)),
        }
        rows_out.append(row)
        table_rows.append([
            key,
            len(smc_rows),
            stats["closed"],
            row["total_r"],
            row["avg_r"],
            row["win_rate"],
            row["partial_hit_rate"],
        ])
    print_table(["Side", "Count", "Closed", "Total R", "Avg R", "Win", "Partial"], table_rows)

    if closed_trades:
        best_worst_rows: list[list[object]] = []
        for side in ("LONG", "SHORT"):
            grouped = group_rows([row for row in closed_trades if side_key(row) == side], coin_key)
            perf = performance_rows(grouped)
            if perf:
                best_worst_rows.append([side, "best", perf[0][0], perf[0][4], perf[0][5], perf[0][2]])
                best_worst_rows.append([side, "worst", perf[-1][0], perf[-1][4], perf[-1][5], perf[-1][2]])
        if best_worst_rows:
            print_note("Best/worst coins by side:")
            print_table(["Side", "Rank", "Coin", "Avg R", "Win", "Closed"], best_worst_rows)

    if smc.rows and any(column in smc.columns for column in ("macro_regime", "htf_regime", "regime_macro_4h", "regime_htf_1h")):
        mismatch_defs = [
            ("SHORT in macro_up/htf_up", lambda r: side_key(r) == "SHORT" and regime_contains(r, "up")),
            ("LONG in macro_down/htf_down", lambda r: side_key(r) == "LONG" and regime_contains(r, "down")),
        ]
        mismatch_rows = []
        for name, predicate in mismatch_defs:
            rows = [row for row in smc.rows if predicate(row)]
            stats = trade_stats(linked_trades(rows, trade_index))
            mismatch_rows.append([name, len(rows), stats["closed"], fmt_signed(stats["avg_r"], 3), fmt_pct(stats["win_rate"])])
        print_note("Macro/HTF mismatch flags:")
        print_table(["Condition", "Eval", "Closed", "Avg R", "Win"], mismatch_rows, max_width=32)
    else:
        print_note("No macro/HTF regime columns found for side mismatch checks.")

    if out_dir:
        write_csv(out_dir / "summary_by_side.csv", rows_out)
    return rows_out


def section_session(smc: CsvData, trades: CsvData, out_dir: Optional[Path]) -> list[dict[str, object]]:
    print_section("9. Session Performance")
    smc_by_session = group_rows(smc.rows, session_key) if smc.rows else {}
    closed_trades = [row for row in trades.rows if is_closed_trade(row)]
    trade_by_session = group_rows(closed_trades, session_key) if closed_trades else {}
    keys = sorted(set(smc_by_session) | set(trade_by_session), key=lambda k: (SESSION_ORDER.get(k, 50), k))

    rows_out: list[dict[str, object]] = []
    table_rows: list[list[object]] = []
    for key in keys:
        smc_rows = smc_by_session.get(key, [])
        accepted, rejected, known = accepted_rejected_counts(smc_rows)
        stats = trade_stats(trade_by_session.get(key, []))
        row = {
            "session": key,
            "count": len(smc_rows),
            "accepted": accepted,
            "rejected": rejected,
            "accepted_rate": fmt_pct(pct(accepted, known)),
            "closed_trades": stats["closed"],
            "total_r": fmt_signed(stats["total_r"], 3),
            "avg_r": fmt_signed(stats["avg_r"], 3),
            "win_rate": fmt_pct(stats["win_rate"]),
        }
        rows_out.append(row)
        table_rows.append([
            key,
            len(smc_rows),
            accepted,
            stats["closed"],
            row["total_r"],
            row["avg_r"],
            row["win_rate"],
            rejected,
        ])
    print_table(["Session", "Count", "Accepted", "Closed", "Total R", "Avg R", "Win", "Reject"], table_rows)

    if out_dir:
        write_csv(out_dir / "summary_by_session.csv", rows_out)
    return rows_out


def section_rejection_analysis(smc: CsvData, signals: CsvData, out_dir: Optional[Path]) -> list[dict[str, object]]:
    print_section("10. Rejection Reason Analysis")
    rejection_rows, source = collect_rejection_rows(smc, signals)
    if not rejection_rows:
        print_note("No rejection rows found.")
        return []
    print_note(f"Using rejection rows from {source}; rows={len(rejection_rows)}")

    family_counter = Counter(rejection_family(row) for row in rejection_rows)
    raw_counter = Counter(raw_reject_reason(row) for row in rejection_rows)
    print_note("Top 20 rejection families:")
    print_table(["Family", "Count"], top_counter_rows(family_counter, 20))
    print_note("Top 20 raw rejection reasons:")
    print_table(["Raw reason", "Count"], top_counter_rows(raw_counter, 20), max_width=58)
    print_note("Rejection count by coin:")
    print_table(["Coin", "Count"], top_counter_rows(Counter(coin_key(row) for row in rejection_rows), 15))
    print_note("Rejection count by timeframe:")
    print_table(["TF", "Count"], top_counter_rows(Counter(timeframe_key(row) for row in rejection_rows), 10))
    print_note("Rejection count by session:")
    print_table(["Session", "Count"], top_counter_rows(Counter(session_key(row) for row in rejection_rows), 10))

    keyword_defs = [
        ("rr_too_low", ("rr_too_low", "risk_reward", "reward_risk", "final_rr_below")),
        ("stale_entry", ("stale_entry", "stale_move", "too_late", "overextended")),
        ("tp_consumed", ("tp_consumed", "tp_already_consumed")),
        ("fill_rr_below", ("fill_rr_below",)),
        ("session_block", ("session_block", "session_blocked", "session_soft_blocked")),
        ("cooldown", ("cooldown", "reject_throttled")),
        ("hard_timeframe_block", ("hard_timeframe_block", "hard_blocked_timeframe", "4h_disabled")),
        ("strategy_filter_block", ("strategy_filter",)),
    ]
    keyword_rows: list[list[object]] = []
    for name, terms in keyword_defs:
        rows = [row for row in rejection_rows if keyword_match(row, terms)]
        keyword_rows.append([
            name,
            len(rows),
            fmt_pct(pct(len(rows), len(rejection_rows))),
            ", ".join(k for k, _ in Counter(coin_key(r) for r in rows).most_common(3)) or "-",
            ", ".join(k for k, _ in Counter(session_key(r) for r in rows).most_common(3)) or "-",
        ])
    print_note("Specific rejection families of interest:")
    print_table(["Pattern", "Count", "Pct", "Top coins", "Top sessions"], keyword_rows)

    output_rows = [
        {
            "family": family,
            "count": count,
            "pct": fmt_pct(pct(count, len(rejection_rows))),
        }
        for family, count in family_counter.most_common()
    ]
    if out_dir:
        write_csv(out_dir / "rejection_reasons.csv", output_rows)
    return output_rows


def section_fill_slippage(smc: CsvData, signals: CsvData) -> list[dict[str, object]]:
    print_section("11. Fill / Slippage Analysis")
    rows = collect_fill_rows(smc, signals)
    if not rows:
        print_note("No fill rows with entry and fill_price were found.")
        return []

    slip_values = [v for v in (fill_slip(row) for row in rows) if v is not None]
    print_table(
        ["Fills", "Avg slip bps", "Worst abs bps"],
        [[len(rows), fmt_signed(mean(slip_values), 2), fmt_signed(max((abs(v) for v in slip_values), default=0.0), 2)]],
    )

    output_rows: list[dict[str, object]] = []
    for label, key_fn in (("coin", coin_key), ("side", side_key), ("session", session_key)):
        grouped = group_rows(rows, key_fn)
        table_rows = []
        for key, group in grouped.items():
            slips = [v for v in (fill_slip(row) for row in group) if v is not None]
            table_rows.append([key, len(group), fmt_signed(mean(slips), 2), fmt_signed(max((abs(v) for v in slips), default=0.0), 2)])
            output_rows.append({
                "group_type": label,
                "group": key,
                "fills": len(group),
                "avg_slippage_bps": fmt_signed(mean(slips), 3),
                "worst_abs_slippage_bps": fmt_signed(max((abs(v) for v in slips), default=0.0), 3),
            })
        table_rows.sort(key=lambda r: to_float(r[2]) if to_float(r[2]) is not None else -999, reverse=True)
        print_note(f"Average slippage by {label}:")
        print_table([label.title(), "Fills", "Avg bps", "Worst abs"], table_rows[:12])

    worst = sorted(rows, key=lambda row: abs(fill_slip(row) or 0.0), reverse=True)[:10]
    print_note("Worst slippage examples:")
    print_table(
        ["Signal", "Coin", "Side", "Session", "Entry", "Fill", "Slip bps"],
        [
            [
                row_value(row, ["signal_id"], "-"),
                coin_key(row),
                side_key(row),
                session_key(row),
                fmt_float(row_value(row, ["entry"]), 6),
                fmt_float(row_value(row, ["fill_price"]), 6),
                fmt_signed(fill_slip(row), 2),
            ]
            for row in worst
        ],
        max_width=18,
    )
    print_note("Late-entry linkage is inferred from rejection reasons, not fills, unless the same row carries both fields.")
    return output_rows


def section_late_entry(smc: CsvData, signals: CsvData) -> dict[str, object]:
    print_section("12. Late Entry / Execution Leak Analysis")
    rejection_rows, source = collect_rejection_rows(smc, signals)
    if not rejection_rows:
        print_note("No rejection rows available.")
        return {}

    patterns = [
        ("fill_rr_below", ("fill_rr_below",)),
        ("tp_consumed", ("tp_consumed", "tp_already_consumed")),
        ("stale_entry", ("stale_entry", "too_late", "overextended")),
        ("adverse_drift", ("adverse_drift",)),
        ("entry_outside_buffer", ("entry_outside_buffer", "entry_outside_atr_buffer")),
        ("signal_stale_move", ("signal_stale_move", "stale_move")),
    ]
    late_union = [row for row in rejection_rows if keyword_match(row, LATE_ENTRY_TERMS)]
    table_rows = []
    for name, terms in patterns:
        rows = [row for row in rejection_rows if keyword_match(row, terms)]
        table_rows.append([
            name,
            len(rows),
            fmt_pct(pct(len(rows), len(rejection_rows))),
            ", ".join(k for k, _ in Counter(coin_key(row) for row in rows).most_common(3)) or "-",
            ", ".join(k for k, _ in Counter(session_key(row) for row in rows).most_common(3)) or "-",
        ])
    print_note(f"Using rejection rows from {source}; total rejections={len(rejection_rows)}")
    print_table(["Leak pattern", "Count", "Pct", "Top coins", "Top sessions"], table_rows)

    late_pct = pct(len(late_union), len(rejection_rows)) or 0.0
    print_note(f"Late-entry/execution-leak union: {len(late_union)} rows ({fmt_pct(late_pct)} of rejections).")
    if late_pct >= 0.20 and len(late_union) >= 10:
        print_note("Recommendation: prioritize an active setup tracker or entry retrace engine in Sprint 3.")
    else:
        print_note("Recommendation: keep tracking; sample does not yet force an execution-leak rebuild.")
    return {"late_rows": len(late_union), "late_pct": late_pct, "total_rejections": len(rejection_rows)}


def candidate_rows(
    smc: CsvData,
    trade_index: TradeIndex,
    definitions: list[tuple[str, Callable[[dict[str, str]], bool]]],
) -> list[dict[str, object]]:
    rows_out: list[dict[str, object]] = []
    for name, predicate in definitions:
        rows = [row for row in smc.rows if predicate(row)]
        if not rows:
            continue
        stats = smc_group_summary(rows, trade_index)
        closed = int(stats["closed"])
        evidence_n = closed if closed else len(rows)
        rows_out.append({
            "candidate": name,
            "evaluations": len(rows),
            "accepted": stats["accepted"],
            "rejected": stats["rejected"],
            "closed_trades": closed,
            "avg_r": stats["avg_r"],
            "total_r": stats["total_r"],
            "win_rate": stats["win_rate"],
            "partial_hit_rate": stats["partial_hit_rate"],
            "avg_slippage_bps": stats["avg_slippage_bps"],
            "rejection_rate": stats["rejection_rate"],
            "confidence": sample_confidence(evidence_n),
            "recommendation": stats["recommendation"],
        })
    return rows_out


def section_best_candidates(smc: CsvData, trade_index: TradeIndex) -> list[dict[str, object]]:
    print_section("13. Best Setup Candidates")
    if not smc.rows:
        print_note("No SMC rows available.")
        return []
    rows = candidate_rows(smc, trade_index, candidate_definitions())
    rows.sort(
        key=lambda row: (
            row["avg_r"] if isinstance(row["avg_r"], float) else -999,
            row["closed_trades"],
            row["evaluations"],
        ),
        reverse=True,
    )
    print_table(
        ["Candidate", "Eval", "Acc", "Closed", "Avg R", "Win", "Conf", "Rec"],
        [
            [
                row["candidate"],
                row["evaluations"],
                row["accepted"],
                row["closed_trades"],
                fmt_signed(row["avg_r"], 3),
                fmt_pct(row["win_rate"]),
                row["confidence"],
                row["recommendation"],
            ]
            for row in rows[:12]
        ],
        max_width=34,
    )
    return rows


def section_worst_candidates(
    trigger_rows: list[dict[str, object]],
    candidate_stats: list[dict[str, object]],
) -> list[dict[str, object]]:
    print_section("14. Worst Setup Candidates")
    combined: list[dict[str, object]] = []
    for row in trigger_rows:
        combined.append({
            "name": row["group"],
            "evaluations": int(row["evaluations"]),
            "closed": int(row["closed_trades"]),
            "avg_r": to_float(row["avg_r"]),
            "win_rate": to_float(row["win_rate"]) / 100 if to_float(row["win_rate"]) is not None else None,
            "rejection_rate": None,
            "partial": to_float(row["partial_hit_rate"]) / 100 if to_float(row["partial_hit_rate"]) is not None else None,
            "slip": to_float(row["avg_slippage_bps"]),
            "recommendation": row["recommendation"],
        })
    for row in candidate_stats:
        combined.append({
            "name": row["candidate"],
            "evaluations": row["evaluations"],
            "closed": row["closed_trades"],
            "avg_r": row["avg_r"],
            "win_rate": row["win_rate"],
            "rejection_rate": row["rejection_rate"],
            "partial": row["partial_hit_rate"],
            "slip": row["avg_slippage_bps"],
            "recommendation": row["recommendation"],
        })

    worst = sorted(
        [row for row in combined if row["closed"] or row["evaluations"] >= 10],
        key=lambda row: (
            row["avg_r"] if isinstance(row["avg_r"], float) else 0.0,
            -(row["rejection_rate"] if isinstance(row["rejection_rate"], float) else 0.0),
        ),
    )[:12]
    print_table(
        ["Setup", "Eval", "Closed", "Avg R", "Win", "Rej", "Slip", "Rec"],
        [
            [
                row["name"],
                row["evaluations"],
                row["closed"],
                fmt_signed(row["avg_r"], 3),
                fmt_pct(row["win_rate"]),
                fmt_pct(row["rejection_rate"]),
                fmt_signed(row["slip"], 2),
                row["recommendation"],
            ]
            for row in worst
        ],
        max_width=34,
    )
    print_note("Worst ranks are sample-aware; n < 10 remains very low confidence.")
    return worst


def section_action_recommendations(
    overall_stats: dict[str, object],
    stop_rows: list[dict[str, object]],
    trigger_rows: list[dict[str, object]],
    timeframe_rows: list[dict[str, object]],
    side_rows: list[dict[str, object]],
    session_rows: list[dict[str, object]],
    rejection_rows: list[dict[str, object]],
    late_stats: dict[str, object],
    best_rows: list[dict[str, object]],
    worst_rows: list[dict[str, object]],
) -> None:
    print_section("15. Action Recommendations")
    closed = int(overall_stats.get("closed", 0) or 0) if overall_stats else 0
    print_note(f"Sample confidence: {sample_confidence(closed)} for realized trade profitability (closed trades={closed}).")

    env_changes: list[str] = []
    code_changes: list[str] = []
    keep: list[str] = []
    disable: list[str] = []
    more_data: list[str] = []

    for row in session_rows:
        avg_r = to_float(row["avg_r"])
        closed_n = int(row["closed_trades"])
        if closed_n >= 10 and avg_r is not None and avg_r < 0:
            env_changes.append(f"Review session exposure for {row['session']} (closed={closed_n}, avgR={avg_r:.3f}).")
    for row in timeframe_rows:
        avg_r = to_float(row["avg_r"])
        closed_n = int(row["closed_trades"])
        if row["timeframe"] == "4h" and (int(row["accepted"]) > 0 or closed_n > 0):
            env_changes.append("Keep 4H disabled unless explicitly re-enabled after review.")
        elif closed_n >= 10 and avg_r is not None and avg_r < 0:
            env_changes.append(f"Review {row['timeframe']} exposure before changing thresholds.")

    late_pct = late_stats.get("late_pct", 0.0) if late_stats else 0.0
    late_rows = late_stats.get("late_rows", 0) if late_stats else 0
    if isinstance(late_pct, float) and late_pct >= 0.20 and int(late_rows) >= 10:
        code_changes.append("Build an active setup tracker / entry retrace engine to reduce stale-entry and consumed-TP leaks.")
    if rejection_rows:
        unknown = next((row for row in rejection_rows if row["family"] == "unknown"), None)
        if unknown and int(unknown["count"]) >= 20:
            code_changes.append("Improve rejection reason instrumentation for unknown families before tuning filters.")

    for row in best_rows:
        if row["recommendation"] == "keep" and int(row["closed_trades"]) >= 10:
            keep.append(f"{row['candidate']} (avgR={row['avg_r']:.3f}, closed={row['closed_trades']}).")
    for row in worst_rows:
        if row["recommendation"] in {"avoid", "reduce"} and int(row["closed"]) >= 10:
            disable.append(f"{row['name']} ({row['recommendation']}, avgR={row['avg_r']:.3f}, closed={row['closed']}).")

    if not keep:
        more_data.append("No setup has enough positive closed-trade evidence to declare a durable keeper yet.")
    if closed < 30:
        more_data.append("Collect at least 30 closed trades before hard-disabling nuanced SMC features.")
    if not env_changes:
        env_changes.append("No env-only change is justified by sample-aware evidence yet.")
    if not code_changes:
        code_changes.append("No mandatory code change is proven; keep logging and link more outcomes.")
    if not disable:
        disable.append("No feature should be disabled solely from this sample without more closed trades.")

    print_note("Env-only changes to consider:")
    for item in env_changes[:6]:
        print(f"  - {item}")
    print_note("Code changes to consider:")
    for item in code_changes[:6]:
        print(f"  - {item}")
    print_note("Features to keep:")
    for item in (keep[:6] or ["Keep SMC live logging and final stop/TP/R attribution."]):
        print(f"  - {item}")
    print_note("Features to disable/reduce:")
    for item in disable[:6]:
        print(f"  - {item}")
    print_note("Features needing more data:")
    for item in more_data[:6]:
        print(f"  - {item}")
    print_note("Sprint 3 build direction:")
    print("  - Build an active setup tracker / entry retrace analyzer that follows accepted setup geometry after detection,")
    print("    records whether price offers a cleaner retrace, and compares market-entry fills vs delayed/retrace entries.")

    print_section("BOTTOM LINE")
    conclusions: list[str] = []
    if overall_stats:
        conclusions.append(
            f"Closed trades={closed}, totalR={fmt_signed(overall_stats.get('total_r'), 3)}, "
            f"avgR={fmt_signed(overall_stats.get('avg_r'), 3)}, confidence={sample_confidence(closed)}."
        )
    top_best = next((row for row in best_rows if row["closed_trades"]), None)
    if top_best:
        conclusions.append(
            f"Best linked setup candidate: {top_best['candidate']} "
            f"(closed={top_best['closed_trades']}, avgR={fmt_signed(top_best['avg_r'], 3)})."
        )
    top_worst = next((row for row in worst_rows if row["closed"]), None)
    if top_worst:
        conclusions.append(
            f"Worst linked setup candidate: {top_worst['name']} "
            f"(closed={top_worst['closed']}, avgR={fmt_signed(top_worst['avg_r'], 3)})."
        )
    if isinstance(late_pct, float):
        conclusions.append(f"Late-entry/execution-leak rejections are {fmt_pct(late_pct)} of rejection rows.")
    if stop_rows:
        linked_stop_rows = [row for row in stop_rows if int(row["closed_trades"])]
        if linked_stop_rows:
            best_stop = sorted(
                linked_stop_rows,
                key=lambda row: to_float(row["avg_r"]) if to_float(row["avg_r"]) is not None else -999,
                reverse=True,
            )[0]
            conclusions.append(
                f"Best linked stop method: {best_stop['group']} "
                f"(closed={best_stop['closed_trades']}, avgR={best_stop['avg_r']})."
            )
    while len(conclusions) < 5:
        conclusions.append("More closed trades are needed before making aggressive live behavior changes.")
    for idx, item in enumerate(conclusions[:5], start=1):
        print(f"  {idx}. {item}")


def run(args: argparse.Namespace) -> None:
    smc = read_csv_data("smc", args.smc)
    signals = read_csv_data("signals", args.signals)
    trades = read_csv_data("trades", args.trades)
    out_dir = Path(args.out) if args.out else None

    section_data_health(smc, signals, trades)
    trade_index = make_trade_indexes(trades.rows, smc.rows)
    overall_stats = section_overall_performance(trades)
    section_smc_engine_summary(smc, signals)
    stop_rows = section_stop_method(smc, trade_index, out_dir)
    trigger_rows = section_trigger_family(smc, trade_index, out_dir)
    section_score_buckets(smc, trade_index)
    timeframe_rows = section_timeframe(smc, trades, out_dir)
    side_rows = section_side(smc, trades, trade_index, out_dir)
    session_rows = section_session(smc, trades, out_dir)
    rejection_rows = section_rejection_analysis(smc, signals, out_dir)
    section_fill_slippage(smc, signals)
    late_stats = section_late_entry(smc, signals)
    best_rows = section_best_candidates(smc, trade_index)
    worst_rows = section_worst_candidates(trigger_rows, best_rows)
    section_action_recommendations(
        overall_stats=overall_stats,
        stop_rows=stop_rows,
        trigger_rows=trigger_rows,
        timeframe_rows=timeframe_rows,
        side_rows=side_rows,
        session_rows=session_rows,
        rejection_rows=rejection_rows,
        late_stats=late_stats,
        best_rows=best_rows,
        worst_rows=worst_rows,
    )

    if out_dir:
        print_section("Optional CSV Output")
        print_note(f"Wrote requested summaries under {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze Cuan Sniffer SMC live logs without touching live trading behavior."
    )
    parser.add_argument("--smc", default="smc_live_log.csv", help="Path to smc_live_log.csv")
    parser.add_argument("--signals", default="signals.csv", help="Path to signals.csv")
    parser.add_argument("--trades", default="trades.csv", help="Path to trades.csv")
    parser.add_argument("--out", default="", help="Optional output directory for summary CSV files")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

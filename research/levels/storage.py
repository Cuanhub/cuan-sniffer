"""
Append-only, idempotent storage for the level scanner.

Three files, mirroring the two-phase pattern the cascade collector already
uses, plus a third the earlier packages did not need:

  episodes.csv   one row per (coin, level, episode) when the gate fires,
                 carrying every feature computed at that moment
  outcomes.csv   what price did afterwards, written once enough bars exist
  labels.csv     the user's verdict on an episode -- would they have taken it

The label file is the point of the whole package. Forward outcomes tell us
whether a gated setup made money; only labels tell us whether the gate
matches the judgement it is trying to reproduce, and those are different
questions. A gate that is profitable but fires on setups the user would
never take has not captured their strategy, and a gate that misses setups
they would take is leaving the edge on the table. Neither shows up in
outcomes alone.

Schema note: new columns are appended at the end and files are widened in
place on read (see widen_header_if_needed), because a mid-day schema change
in research/cascade/ silently produced ragged CSVs on 2026-09-16 when wider
rows were appended beneath a narrower header.
"""

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set

DATA_DIR = Path(__file__).resolve().parent / "data"

EPISODE_FIELDNAMES = [
    "episode_id", "coin", "detected_at_utc", "bar_ts_ms",
    "level_price", "level_kind", "level_touches", "level_holds", "level_hold_rate",
    "level_pivot_count", "level_bars_since_last_touch",
    "price", "sigma_4h",
    "run_sigma", "pull_sigma", "run_pct", "pull_pct", "bars_since_high",
    "retests", "bars_since_last_retest", "near_sigma", "dist_to_level_pct",
    "cond_run_pullback", "cond_retest", "cond_near", "gate_fires",
    "vol_ratio_recent_ref", "vol_trend_slope", "vol_down_up_ratio",
    "atr_pct", "entry_buffer_sigma", "suggested_entry", "suggested_stop",
    "suggested_tp", "suggested_rr", "opposing_level_price",
    "timeframe", "overhead_sigma", "overhead_pct", "level_origin",
]

OUTCOME_FIELDNAMES = [
    "episode_id", "coin", "resolved_at_utc", "bars_elapsed",
    "reached_level", "bars_to_reach", "fill_price",
    "mae_pct", "mfe_pct", "mae_sigma", "mfe_sigma",
    "close_after_24h_pct", "close_after_72h_pct", "close_after_168h_pct",
    "hit_suggested_stop", "hit_suggested_tp", "bars_to_stop", "bars_to_tp",
    "outcome_r", "level_broke",
]

LABEL_FIELDNAMES = [
    "episode_id", "labelled_at_utc", "verdict", "confidence", "note",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_episode_id(coin: str, bar_ts_ms: int, level_price: float) -> str:
    """Unique per (coin, bar, level). A single bar can gate more than one
    level, so the price has to be part of the key."""
    return f"{coin.upper()}_{int(bar_ts_ms)}_{level_price:.10g}"


def _path(name: str) -> Path:
    return DATA_DIR / name


def _existing_ids(path: Path, id_col: str = "episode_id") -> Set[str]:
    if not path.exists():
        return set()
    with open(path, "r", newline="") as f:
        return {r.get(id_col, "") for r in csv.DictReader(f) if r.get(id_col)}


def widen_header_if_needed(path: Path, fieldnames: List[str]) -> bool:
    """Bring an existing file up to the current schema before appending.
    Same fix as research/cascade/storage.py: new columns are appended at the
    end, so an older header is a strict prefix and the migration is
    positional. A header that is not a prefix raises rather than guessing."""
    if not path.exists():
        return False
    with open(path, "r", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return False
    header, body = rows[0], rows[1:]
    if header == list(fieldnames):
        return False
    if header != list(fieldnames[:len(header)]):
        raise ValueError(f"{path.name}: header is not a prefix of the current schema")
    width = len(fieldnames)
    fixed = [r + [""] * (width - len(r)) if len(r) < width else r for r in body]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fieldnames)
        w.writerows(fixed)
    tmp.replace(path)
    return True


def _append(name: str, fieldnames: List[str], row: Dict[str, Any], id_col: str) -> bool:
    path = _path(name)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    widen_header_if_needed(path, fieldnames)
    if row.get(id_col) and row[id_col] in _existing_ids(path, id_col):
        return False
    is_new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if is_new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fieldnames})
    return True


def append_episode(row: Dict[str, Any]) -> bool:
    return _append("episodes.csv", EPISODE_FIELDNAMES, row, "episode_id")


def append_outcome(row: Dict[str, Any]) -> bool:
    return _append("outcomes.csv", OUTCOME_FIELDNAMES, row, "episode_id")


def append_label(row: Dict[str, Any]) -> bool:
    return _append("labels.csv", LABEL_FIELDNAMES, row, "episode_id")


def load_episodes() -> List[Dict[str, Any]]:
    path = _path("episodes.csv")
    if not path.exists():
        return []
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def resolved_ids() -> Set[str]:
    return _existing_ids(_path("outcomes.csv"))

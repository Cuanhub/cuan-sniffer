"""Retention for the trade log and the launchd-redirected log files.

Nothing is deleted. Rows and lines that fall out of their window are appended
to a monthly gzip under research/levels/data/archive/, so this changes where a
record lives, not whether it exists.

Measured 2026-10-16: live_trades.csv was 55MB over 429,311 rows, 99.3% of them
skips with result=not_triggered -- the same watched level restated on every
pass. trader.log was 80MB because launchd never rotates what it redirects.

TWO THINGS THIS HAS TO GET RIGHT.

1. The trader APPENDS to live_trades.csv while this runs, so a read-filter-
rename loses whatever was appended in between: the writer's fd points at an
inode that rename() has already orphaned. Handled by recording the file size
before reading, and after building the replacement, copying any bytes that
appeared beyond that offset onto the end of it before the rename. The
remaining window is the rename itself.

2. launchd holds the log fds open in append mode. Renaming trader.log would
leave the trader writing to a file nothing can find. So logs are
COPY-TRUNCATED: contents are copied out, then the original is truncated in
place, which an append-mode writer follows correctly.

Dry run is the default. --apply is required to change anything.
"""

import argparse
import csv
import gzip
import pathlib
import shutil
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from research.levels.config import (                              # noqa: E402
    RETAIN_FOREVER_ACTIONS, RETAIN_DECISION_DAYS, RETAIN_WAITING_DAYS,
    LOG_ROTATE_MB, LOG_KEEP_GENERATIONS, ERR_ROTATE_KB, WATCH_TTL_HOURS,
)

DATA = pathlib.Path(__file__).resolve().parent.parent / "research/levels/data"
ARCHIVE = DATA / "archive"
TRADE_LOG = DATA / "live_trades.csv"
LOG_DIR = pathlib.Path.home() / "Library/Logs/cuan-sniffer"
ROTATABLE = ("trader.log", "scanner.log", "protect.log", "telegram.log")
# The stderr streams rotate too, on ERR_ROTATE_KB rather than LOG_ROTATE_MB.
# They are normally empty, so anything in one is a traceback.
ERR_LOGS = ("trader.err.log", "scanner.err.log", "protect.err.log",
            "telegram.err.log", "retention.err.log")
DATA_LOGS = ("_fetch_failures.log",)

# A skip that is a WAIT, not a refusal. Keyed on the result enum the trader
# writes rather than on the wording of `detail`, which gets reworded.
WAITING_RESULT = "not_triggered"


def tier(row: Dict[str, str]) -> str:
    """'forever', 'decision' or 'waiting'."""
    if (row.get("action") or "") in RETAIN_FOREVER_ACTIONS:
        return "forever"
    if (row.get("action") or "") == "skip" \
            and (row.get("result") or "") == WAITING_RESULT:
        return "waiting"
    return "decision"


def cutoffs(now: Optional[datetime] = None) -> Dict[str, str]:
    now = now or datetime.now(timezone.utc)

    def iso(days):
        return (now - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")

    # A full watch lifecycle must stay inspectable in the live file, and
    # seed_watchlist_from_log replays dry_run rows inside WATCH_TTL_HOURS.
    waiting_days = max(RETAIN_WAITING_DAYS, WATCH_TTL_HOURS / 24.0 + 1)
    return {"waiting": iso(waiting_days), "decision": iso(RETAIN_DECISION_DAYS)}


def keep(row: Dict[str, str], cut: Dict[str, str]) -> bool:
    t = tier(row)
    if t == "forever":
        return True
    ts = row.get("ts_utc") or ""
    if not ts:
        # No timestamp means it cannot be aged out safely. Keeping an
        # unreadable row costs a line; discarding one loses evidence.
        return True
    return ts >= cut[t]


def archive_rows(rows: List[Dict[str, str]], cols: List[str],
                 apply: bool) -> Dict[str, int]:
    """Append pruned rows to data/archive/live_trades-YYYY-MM.csv.gz."""
    by_month: Dict[str, List[Dict[str, str]]] = {}
    for r in rows:
        month = (r.get("ts_utc") or "unknown")[:7] or "unknown"
        by_month.setdefault(month, []).append(r)
    if apply:
        ARCHIVE.mkdir(parents=True, exist_ok=True)
        for month, batch in by_month.items():
            path = ARCHIVE / f"live_trades-{month}.csv.gz"
            is_new = not path.exists()
            with gzip.open(path, "at", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
                if is_new:
                    w.writeheader()
                w.writerows(batch)
    return {m: len(v) for m, v in sorted(by_month.items())}


def prune_trade_log(apply: bool, now: Optional[datetime] = None
                    ) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Returns (counts by tier kept/pruned, archived rows by month)."""
    if not TRADE_LOG.exists():
        return {}, {}
    cut = cutoffs(now)
    size_before = TRADE_LOG.stat().st_size
    with TRADE_LOG.open(newline="") as fh:
        reader = csv.DictReader(fh)
        cols = list(reader.fieldnames or [])
        kept: List[Dict[str, Any]] = []
        pruned: List[Dict[str, Any]] = []
        for row in reader:
            (kept if keep(row, cut) else pruned).append(row)
    counts = {"kept": len(kept), "pruned": len(pruned)}
    for t in ("forever", "decision", "waiting"):
        counts[f"kept_{t}"] = sum(1 for r in kept if tier(r) == t)
        counts[f"pruned_{t}"] = sum(1 for r in pruned if tier(r) == t)
    # Checked before a byte moves. A money row reaching the prune list means
    # tier() or RETAIN_FOREVER_ACTIONS is wrong, and the correct response is
    # to do nothing at all rather than to archive it and report afterwards.
    offenders = [r for r in pruned if tier(r) == "forever"]
    if offenders:
        raise AssertionError(
            f"refusing to prune {len(offenders)} audit-trail row(s), "
            f"first: {offenders[0].get('ts_utc')} "
            f"{offenders[0].get('action')} {offenders[0].get('coin')}")
    if not pruned or not apply:
        return counts, archive_rows(pruned, cols, apply=False) if pruned else {}

    months = archive_rows(pruned, cols, apply=True)
    tmp = TRADE_LOG.with_suffix(".csv.retention-tmp")
    with tmp.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(kept)
        # Catch up on anything the trader appended while we were working. Read
        # the raw bytes past where we stopped so a partially written final
        # line is carried across verbatim rather than re-parsed.
        with TRADE_LOG.open("rb") as live:
            live.seek(size_before)
            tail = live.read()
        if tail:
            fh.flush()
            with open(tmp, "ab") as raw:
                raw.write(tail)
            counts["caught_up_bytes"] = len(tail)
    tmp.replace(TRADE_LOG)
    counts["bytes_before"] = size_before
    counts["bytes_after"] = TRADE_LOG.stat().st_size
    return counts, months


def rotate_log(path: pathlib.Path, apply: bool,
               min_bytes: Optional[float] = None) -> Optional[Dict[str, int]]:
    """Copy-truncate `path` into a gzip if it is over the size cap.

    `min_bytes` overrides the cap, for the stderr streams: those rotate on
    64KB rather than 25MB, because a stderr stream with anything in it at all
    is a stream with a traceback in it.

    Copy-truncate, not rename: launchd holds these fds open in append mode,
    so a renamed file leaves the writer appending to an orphaned inode.
    Truncating in place is followed correctly by an append-mode writer.

    VERIFIED against the live trader on 2026-10-16 rather than assumed, since
    the whole approach depends on it: after truncating a 72.7MB trader.log
    under a running process, fd 1 still resolved to the live path, and the
    next pass wrote 958 bytes occupying 8 blocks. Had launchd opened the file
    without O_APPEND the writer's offset would have been stale and that write
    would have produced a 72MB sparse file instead.
    """
    if not path.exists():
        return None
    size = path.stat().st_size
    cap = (LOG_ROTATE_MB * 1024 * 1024) if min_bytes is None else min_bytes
    if size < cap:
        return None
    if not apply:
        return {"bytes": size, "rotated": 0}
    dest_dir = path.parent / "rotated"
    dest_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = dest_dir / f"{path.name}.{stamp}.gz"
    with path.open("rb") as src, gzip.open(dest, "wb") as out:
        shutil.copyfileobj(src, out, length=1 << 20)
    with path.open("r+") as f:
        f.truncate(0)
    # Keep a bounded number of generations per log name.
    gens = sorted(dest_dir.glob(f"{path.name}.*.gz"))
    for old in gens[:-LOG_KEEP_GENERATIONS]:
        old.unlink()
    return {"bytes": size, "rotated": 1, "kept_generations": len(gens[-LOG_KEEP_GENERATIONS:])}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true",
                    help="actually move things; omit for a dry run")
    args = ap.parse_args()
    mode = "APPLY" if args.apply else "DRY RUN (nothing will change)"
    print(f"\n{'='*70}\n  RETENTION -- {mode}\n{'='*70}")
    print(f"  forever  : {sorted(RETAIN_FOREVER_ACTIONS)}")
    print(f"  decisions: {RETAIN_DECISION_DAYS}d")
    print(f"  waiting  : {RETAIN_WAITING_DAYS}d "
          f"(floored at WATCH_TTL_HOURS+1d = "
          f"{max(RETAIN_WAITING_DAYS, WATCH_TTL_HOURS/24+1):.0f}d)")

    counts, months = prune_trade_log(args.apply)
    if counts:
        print("\n  live_trades.csv")
        print(f"    keep  {counts['kept']:>7}  "
              f"(forever {counts['kept_forever']}, "
              f"decision {counts['kept_decision']}, "
              f"waiting {counts['kept_waiting']})")
        print(f"    prune {counts['pruned']:>7}  "
              f"(forever {counts['pruned_forever']}, "
              f"decision {counts['pruned_decision']}, "
              f"waiting {counts['pruned_waiting']})")
        if months:
            print("    -> archive/: " + ", ".join(
                f"{m} {n}" for m, n in months.items()))
        if "bytes_before" in counts:
            print(f"    {counts['bytes_before']/1e6:.1f}MB -> "
                  f"{counts['bytes_after']/1e6:.1f}MB"
                  + (f"  (+{counts['caught_up_bytes']}B appended mid-run, "
                     f"carried over)" if counts.get("caught_up_bytes") else ""))

    print(f"\n  logs (stderr over {ERR_ROTATE_KB:g}KB, stdout over {LOG_ROTATE_MB:g}MB, "
          f"keep {LOG_KEEP_GENERATIONS})")
    any_log = False
    for name in ERR_LOGS:
        r = rotate_log(LOG_DIR / name, args.apply,
                       min_bytes=ERR_ROTATE_KB * 1024)
        if r:
            print(f"    {name:28} {r['bytes'] / 1024:>8.1f}KB  "
                  f"{'rotated' if r['rotated'] else 'would rotate'}")
    for name in ROTATABLE:
        r = rotate_log(LOG_DIR / name, args.apply)
        if r:
            any_log = True
            print(f"    {name:<24} {r['bytes']/1e6:>6.1f}MB  "
                  f"{'rotated' if r['rotated'] else 'would rotate'}")
    for name in DATA_LOGS:
        r = rotate_log(DATA / name, args.apply)
        if r:
            any_log = True
            print(f"    {name:<24} {r['bytes']/1e6:>6.1f}MB  "
                  f"{'rotated' if r['rotated'] else 'would rotate'}")
    if not any_log:
        print("    nothing over the cap")
    print("=" * 70 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

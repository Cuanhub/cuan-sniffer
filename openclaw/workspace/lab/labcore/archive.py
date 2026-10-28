"""An append-only archive of real closed 1h bars that outgrows the venue.

Hyperliquid serves the last 5000 bars and no more, so a lab that refetches
each day can never see past ~208 days. This keeps every closed bar it has
ever fetched. History grows by a day a day, and in a year the lab has a
year -- the only way out-of-sample periods long enough to matter will exist.

Rules, each one a way an archive can quietly stop being real data:
  - only CLOSED bars enter (research.backtest.data.normalise drops the
    forming one);
  - a bar already archived is compared, never overwritten silently. Closed
    bars should not change; if the venue ever serves a different value for
    one, the venue wins (it is the source) and the old value goes to
    revisions.jsonl, so the change is on record rather than invisible;
  - a gap between the archive's end and the new fetch's start (the lab was
    off for longer than the venue's window) stays a gap and is counted;
  - nothing is interpolated, filled or repaired.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from labcore import paths
from research.backtest import data as D

FIELDS = ("o", "h", "l", "c", "v")
INTERVAL_MS = {"1h": D.HOUR_MS, "15m": D.HOUR_MS // 4}
# A 5000-candle request costs about 100 weight against Hyperliquid's per-IP
# budget of roughly 1200 a minute. If this lab runs on the same machine as the
# live bot, they share that budget: at full speed a 45-coin refresh is ~4,500
# weight in seconds, which rate-limited the live trader on 2026-10-27. One
# request every FETCH_GAP_SEC keeps the lab far below the bot's share.
FETCH_GAP_SEC = float(os.getenv("CUAN_LAB_FETCH_GAP_SEC", "10"))


def _dir(interval: str):
    return {"1h": paths.ARCHIVE, "15m": paths.ARCHIVE_15M}[interval]


def _path(coin: str, interval: str = "1h"):
    return _dir(interval) / f"{coin}.json"


def load(coin: str, interval: str = "1h") -> List[Dict[str, Any]]:
    p = _path(coin, interval)
    return json.loads(p.read_text()) if p.exists() else []


def merge(old: List[Dict[str, Any]], new: List[Dict[str, Any]], coin: str = "",
          interval_ms: int = D.HOUR_MS
          ) -> Tuple[List[Dict[str, Any]], Dict[str, int], List[Dict[str, Any]]]:
    """(merged, counts, revisions). Pure, so it can be tested exactly."""
    by_t = {b["t"]: b for b in old}
    added = revised = 0
    revisions = []
    for b in new:
        cur = by_t.get(b["t"])
        if cur is None:
            by_t[b["t"]] = b
            added += 1
        elif any(cur.get(k) != b.get(k) for k in FIELDS):
            revisions.append({"coin": coin, "t": b["t"],
                              "old": {k: cur.get(k) for k in FIELDS},
                              "new": {k: b.get(k) for k in FIELDS}})
            by_t[b["t"]] = b
            revised += 1
    merged = [by_t[t] for t in sorted(by_t)]
    v = D.validate(merged, interval_ms)
    return merged, {"added": added, "revised": revised, "bars": len(merged),
                    "gaps": v["gaps"], "missing_bars": v["missing_bars"]}, revisions


def update(coins: List[str], client=None, log=print, interval: str = "1h",
           gap_sec: Optional[float] = None, sleep=time.sleep) -> Dict[str, Any]:
    if client is None:
        from research.levels.venue_client import HyperliquidReadOnlyClient
        client = HyperliquidReadOnlyClient()
    gap = FETCH_GAP_SEC if gap_sec is None else gap_sec
    ims = INTERVAL_MS[interval]
    _dir(interval).mkdir(parents=True, exist_ok=True)
    report: Dict[str, Any] = {"interval": interval, "coins": {}, "failed": {}}
    for k, coin in enumerate(coins):
        if k:
            sleep(gap)
        fetched_ms = int(time.time() * 1000)
        try:
            raw = client.get_candles(coin, interval,
                                     fetched_ms - (D.MAX_BARS + 2) * ims, fetched_ms)
        except Exception as e:                                  # noqa: BLE001
            report["failed"][coin] = str(e)[:120]
            continue
        rows = D.normalise(raw, fetched_ms, ims)
        v = D.validate(rows, ims)
        if not v["usable"]:
            report["failed"][coin] = f"invalid bars from venue: {v}"
            continue
        merged, counts, revs = merge(load(coin, interval), rows, coin, ims)
        if revs:
            paths.REVISIONS.parent.mkdir(parents=True, exist_ok=True)
            with paths.REVISIONS.open("a") as f:
                for r in revs:
                    f.write(json.dumps(r) + "\n")
        tmp = _path(coin, interval).with_suffix(".tmp")
        tmp.write_text(json.dumps(merged))
        tmp.replace(_path(coin, interval))
        report["coins"][coin] = counts
    report["finished_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if interval == "1h":
        manifest = manifest_now()
        manifest["last_update"] = report
        paths.ARCHIVE_MANIFEST.write_text(json.dumps(manifest, indent=1))
    else:
        (_dir(interval) / "_manifest.json").write_text(json.dumps(
            {"interval": interval, "last_update": report}, indent=1))
    return report


def manifest_now() -> Dict[str, Any]:
    coins = {}
    for p in sorted(paths.ARCHIVE.glob("*.json")):
        bars = json.loads(p.read_text())
        if not bars:
            continue
        v = D.validate(bars)
        coins[p.stem] = {**v, "first_t": bars[0]["t"], "last_t": bars[-1]["t"],
                         "sha256": D.digest(bars)}
    return {"source": D.SOURCE, "interval": "1h", "coins": coins}


def snapshot(coins: Optional[List[str]] = None, until_ms: Optional[int] = None,
             min_bars: int = 1000) -> D.Snapshot:
    """The archive as a verified Snapshot, optionally cut at `until_ms`.

    The cut is how the sealed holdout stays sealed: a development run gets
    bars strictly before the seal and never sees a price after it.
    """
    bars_by: Dict[str, List[Dict[str, Any]]] = {}
    meta: Dict[str, Any] = {}
    names = coins or [p.stem for p in sorted(paths.ARCHIVE.glob("*.json"))]
    for coin in names:
        bars = load(coin)
        if until_ms is not None:
            bars = [b for b in bars if b["t"] + D.HOUR_MS <= until_ms]
        v = D.validate(bars)
        if len(bars) < min_bars or v["duplicates"] or v["bad_ohlc"] or v["nonpositive"]:
            continue
        bars_by[coin] = bars
        meta[coin] = {**v, "first_t": bars[0]["t"], "last_t": bars[-1]["t"],
                      "sha256": D.digest(bars)}
    return D.Snapshot(bars_by, {"source": D.SOURCE, "interval": "1h",
                                "until_ms": until_ms, "coins": meta,
                                "fetched_at_utc": "archive", "excluded": {}})

"""Real 1h candles from Hyperliquid mainnet, with provenance.

Every bar a backtest sees comes through here, and the only source is the
public read-only `candleSnapshot` info endpoint. Nothing is generated,
interpolated or repaired: a missing hour stays missing and is reported, and
a bar whose OHLC is internally inconsistent excludes its coin rather than
being "fixed". A backtest that quietly patches its inputs is reporting on
data that never traded.

Two details that decide whether a backtest is honest:

  - The newest bar candleSnapshot returns is still FORMING. Its high, low
    and close will change before the hour ends, so it is dropped: a closed
    bar is a fact, a forming one is a guess.
  - The venue serves at most 5000 bars per request regardless of the start
    time asked for, so ~208 days of 1h history is the ceiling. Asking for
    more and silently getting less is how a "two-year backtest" turns out
    to be seven months; the manifest records what actually came back.

The manifest stores a SHA-256 per coin over the exact bars used, so a
result can always be tied to the data that produced it.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

HOUR_MS = 3_600_000
SOURCE = "https://api.hyperliquid.xyz/info candleSnapshot (mainnet, public, read-only)"
MAX_BARS = 5000


def normalise(raw: Iterable[Dict[str, Any]], fetched_ms: int,
              interval_ms: int = HOUR_MS) -> List[Dict[str, Any]]:
    """The venue's candle rows as plain floats, closed bars only, oldest first.

    A bar is closed when its full interval has elapsed by `fetched_ms`. The
    venue's own close-time field `T` is the last millisecond of the bar, so
    `t + interval` is used rather than `T` to avoid an off-by-one at the
    boundary.
    """
    out: List[Dict[str, Any]] = []
    for b in raw or []:
        try:
            t = int(b["t"])
            row = {"t": t, "o": float(b["o"]), "h": float(b["h"]),
                   "l": float(b["l"]), "c": float(b["c"]),
                   "v": float(b.get("v") or 0.0), "n": int(b.get("n") or 0)}
        except (KeyError, TypeError, ValueError):
            continue
        if t + interval_ms > fetched_ms:
            continue                      # still forming: not a fact yet
        out.append(row)
    out.sort(key=lambda r: r["t"])
    return out


def validate(bars: List[Dict[str, Any]], interval_ms: int = HOUR_MS
             ) -> Dict[str, Any]:
    """Everything wrong with a bar series, counted. Repairs nothing."""
    dup = gaps = missing = bad_ohlc = nonpos = flat = 0
    max_gap = 0
    for i, b in enumerate(bars):
        if min(b["o"], b["h"], b["l"], b["c"]) <= 0:
            nonpos += 1
        if b["l"] > min(b["o"], b["c"]) or b["h"] < max(b["o"], b["c"]) \
                or b["l"] > b["h"]:
            bad_ohlc += 1
        if b["h"] == b["l"]:
            flat += 1
        if i:
            step = b["t"] - bars[i - 1]["t"]
            if step == 0:
                dup += 1
            elif step != interval_ms:
                gaps += 1
                miss = step // interval_ms - 1
                missing += max(0, miss)
                max_gap = max(max_gap, miss)
    return {"bars": len(bars), "duplicates": dup, "gaps": gaps,
            "missing_bars": missing, "max_gap_bars": max_gap,
            "bad_ohlc": bad_ohlc, "nonpositive": nonpos, "flat_bars": flat,
            "usable": dup == 0 and bad_ohlc == 0 and nonpos == 0}


def digest(bars: List[Dict[str, Any]]) -> str:
    canon = json.dumps([[b["t"], b["o"], b["h"], b["l"], b["c"], b["v"]]
                        for b in bars], separators=(",", ":"))
    return hashlib.sha256(canon.encode()).hexdigest()


@dataclass
class Snapshot:
    bars: Dict[str, List[Dict[str, Any]]]
    manifest: Dict[str, Any] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps({"manifest": self.manifest,
                                   "bars": self.bars}))
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "Snapshot":
        raw = json.loads(Path(path).read_text())
        snap = cls(bars=raw["bars"], manifest=raw["manifest"])
        snap.verify()
        return snap

    def verify(self) -> None:
        """Refuse a snapshot whose bars no longer match their manifest."""
        for coin, meta in self.manifest.get("coins", {}).items():
            if coin not in self.bars:
                continue
            got = digest(self.bars[coin])
            if got != meta["sha256"]:
                raise ValueError(f"{coin}: bars do not match the manifest "
                                 f"digest -- the snapshot was altered")


def fetch(coins: List[str], client=None, interval: str = "1h",
          min_bars: int = 1000, log=print) -> Snapshot:
    """Fetch up to MAX_BARS closed bars per coin and validate every one.

    A coin with fewer than `min_bars` usable bars, or with any duplicate or
    OHLC-inconsistent bar, is left out and the reason recorded. Nothing is
    patched to make it fit.
    """
    if client is None:
        from research.levels.venue_client import HyperliquidReadOnlyClient
        client = HyperliquidReadOnlyClient()
    interval_ms = {"1h": HOUR_MS, "4h": 4 * HOUR_MS, "15m": HOUR_MS // 4}[interval]
    fetched_ms = int(time.time() * 1000)
    bars: Dict[str, List[Dict[str, Any]]] = {}
    coins_meta: Dict[str, Any] = {}
    excluded: Dict[str, str] = {}
    for coin in coins:
        try:
            raw = client.get_candles(coin, interval,
                                     fetched_ms - (MAX_BARS + 2) * interval_ms,
                                     fetched_ms)
        except Exception as e:                                # noqa: BLE001
            excluded[coin] = f"fetch failed: {str(e)[:80]}"
            log(f"  {coin:10} FETCH FAILED {str(e)[:60]}")
            continue
        rows = normalise(raw, fetched_ms, interval_ms)
        v = validate(rows, interval_ms)
        if not v["usable"]:
            excluded[coin] = f"invalid bars: {v}"
        elif len(rows) < min_bars:
            excluded[coin] = f"only {len(rows)} closed bars (< {min_bars})"
        if coin in excluded:
            log(f"  {coin:10} excluded -- {excluded[coin][:70]}")
            continue
        bars[coin] = rows
        coins_meta[coin] = {**v, "first_t": rows[0]["t"], "last_t": rows[-1]["t"],
                            "sha256": digest(rows)}
        log(f"  {coin:10} {len(rows):5} bars  gaps {v['gaps']}  "
            f"missing {v['missing_bars']}")
    manifest = {
        "source": SOURCE,
        "interval": interval,
        "fetched_at_utc": datetime.fromtimestamp(
            fetched_ms / 1000, timezone.utc).isoformat(),
        "fetched_ms": fetched_ms,
        "forming_bar_dropped": True,
        "requested": list(coins),
        "coins": coins_meta,
        "excluded": excluded,
    }
    return Snapshot(bars=bars, manifest=manifest)


def resample(bars: List[Dict[str, Any]], factor: int,
             interval_ms: int = HOUR_MS) -> List[Dict[str, Any]]:
    """Aggregate COMPLETE groups of real bars into a higher timeframe.

    Groups are aligned to UTC multiples of `factor * interval_ms`, and a
    group missing any of its bars is dropped rather than built from a
    partial set -- a 4h bar made of three hours is not a 4h bar.
    """
    span = factor * interval_ms
    groups: Dict[int, List[Dict[str, Any]]] = {}
    for b in bars:
        groups.setdefault(b["t"] - b["t"] % span, []).append(b)
    out = []
    for start in sorted(groups):
        g = groups[start]
        if len(g) != factor:
            continue
        out.append({"t": start, "o": g[0]["o"], "h": max(x["h"] for x in g),
                    "l": min(x["l"] for x in g), "c": g[-1]["c"],
                    "v": sum(x.get("v", 0.0) for x in g),
                    "n": sum(x.get("n", 0) for x in g)})
    return out

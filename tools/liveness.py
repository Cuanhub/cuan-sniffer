"""Tell the operator when the system goes quiet.

Run from the Telegram control loop, because that is the process that can already
reach the operator. It reads what the services already leave behind, the mtime
of their logs, so nothing that trades or scans needed changing.

It cannot report its own death, or the Mac's. On 2026-11-06 the Mac slept for 37
minutes with a bid resting, and nothing said so. So it does two things:

  - On our own bot: a service whose log stopped moving, and afterwards any gap
    in this loop itself, which means the Mac slept, lost power or this process
    was down. That alert is late by nature: it arrives when the Mac is back.
  - Off this machine: if HEARTBEAT_PING_URL is set, a ping while everything is
    fresh. The service behind that URL alerts when the pings stop, which is the
    only thing that can speak while the Mac is off. Unset, this does nothing.

Quiet is counted in AWAKE time and carried across sleeps and restarts of this
loop. It used to be wall time since the last wake, so every wake started the
count again -- and on battery, where the Mac sleeps every few minutes, no
service ever got a full period to be judged in. The trader froze on 2026-11-08
and logged nothing for five days without a single alert.
"""

from __future__ import annotations

import os
import pathlib
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import requests

import notify
from core import books
from core.meta import DATA, read_json, write_json

LOG_DIR = pathlib.Path(os.getenv("CUAN_LOG_DIR",
                                 str(pathlib.Path.home() / "Library/Logs/cuan-sniffer")))
STATE = DATA / "_liveness.json"

# Awake seconds a log may sit still. Each is well over that service's normal
# silence: protect logs every 60s, the trader every 450s plus a pass of about 3
# minutes, the scanner when a pass completes, every 900s or longer if outcomes
# are due.
STALE_SEC: Dict[str, float] = {"protect": 300.0, "trader": 1200.0, "scanner": 2700.0}
# The live trend pass runs hourly while armed and logs every run.
TREND_STALE_SEC = 3 * 3600.0


def services() -> Dict[str, float]:
    out = dict(STALE_SEC)
    if books.trend_armed():
        out["trendlive"] = TREND_STALE_SEC
    return out
# A longer gap between ticks of this loop means nothing here was running. Its
# worst normal gap is a long poll plus the failure backoff, about a minute.
GAP_SEC = float(os.getenv("LIVENESS_GAP_SEC", "300"))
REPEAT_SEC = 3600
PING_EVERY_SEC = 300.0


def _human(sec: float) -> str:
    if sec >= 2 * 86400:
        return f"{sec / 86400:.1f} days"
    return f"{sec / 3600:.1f}h" if sec >= 5400 else f"{sec / 60:.0f} min"


def _gap_text(owed: Dict[str, Any]) -> str:
    n, sec = int(owed["n"]), float(owed["sec"])
    if n == 1:
        head = f"System was down for {_human(sec)}"
    else:
        since = datetime.fromtimestamp(float(owed["since"]), timezone.utc)
        head = (f"System was down {n} times, {_human(sec)} in all, "
                f"since {since:%m-%d %H:%M} UTC")
    return (f"⚠️ *{head}*\nThe Mac slept, lost power or the control loop "
            f"stopped. Stops and targets on the venue kept working, but a bid "
            f"that filled in that time had no stop until protection came back. "
            f"Check /positions.")


class Watch:
    def __init__(self, now: Callable[[], float] = time.time,
                 logs: Optional[pathlib.Path] = None,
                 state: Optional[pathlib.Path] = None) -> None:
        self.now = now
        self.logs = logs or LOG_DIR
        self.state = state or STATE
        self.down: Set[str] = set()
        self.last_ping = 0.0

    def _load(self) -> Dict[str, Any]:
        row = read_json(self.state, {})
        return row if isinstance(row, dict) else {}

    def _mtime(self, name: str) -> Optional[float]:
        try:
            return (self.logs / f"{name}.log").stat().st_mtime
        except OSError:
            return None

    def tick(self) -> None:
        t = self.now()
        row = self._load()
        last = float(row.get("last_tick") or 0.0)
        gap = t - last if last else 0.0
        # Time asleep, or with this loop down, counts toward nothing: no
        # service can log while the Mac sleeps.
        awake = gap if 0.0 < gap <= GAP_SEC else 0.0
        owed = row.get("owed") if isinstance(row.get("owed"), dict) else None
        if gap > GAP_SEC:
            prev = owed or {}
            owed = {"n": int(prev.get("n", 0)) + 1,
                    "sec": float(prev.get("sec", 0.0)) + gap,
                    "since": float(prev.get("since", last))}
        quiet, seen = self._count(awake, row)
        row = {"last_tick": t, "owed": owed, "quiet": quiet, "seen": seen}
        write_json(self.state, row)
        # Kept, and added to, until it lands: the network is often not back in
        # the first seconds after a wake, and this is the one message that says
        # so. Each gap used to replace the last one that had not been sent.
        if owed and notify.send(_gap_text(owed)):
            row["owed"] = None
            write_json(self.state, row)
        # Silence is the signal the outside service acts on, so a stale service
        # stops the pings as well as alerting here.
        if not self._stale(t, quiet, seen):
            self._ping(t)

    def _count(self, awake: float, row: Dict[str, Any]
               ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Each service's awake seconds since its log last moved."""
        quiet = {k: float(v) for k, v in (row.get("quiet") or {}).items()}
        seen = {k: float(v) for k, v in (row.get("seen") or {}).items()}
        for name in services():
            m = self._mtime(name)
            if m is not None and m != seen.get(name):
                quiet[name], seen[name] = 0.0, m
            else:
                quiet[name] = quiet.get(name, 0.0) + awake
        return quiet, seen

    def _stale(self, t: float, quiet: Dict[str, float],
               seen: Dict[str, float]) -> List[str]:
        out = []
        for name, limit in services().items():
            q = quiet.get(name, 0.0)
            if q < limit:
                continue
            out.append(name)
            last = seen.get(name)
            said = ("Its log file is missing." if last is None else
                    f"Nothing logged for {_human(t - last)}, {_human(q)} of it "
                    f"with the Mac awake.")
            notify.send(
                f"🚨 *{name} has gone quiet*\n{said}"
                + (" Stops already on the venue still work, but new fills "
                   "are not being protected." if name == "protect" else ""),
                key=f"liveness_{name}", cooldown_sec=REPEAT_SEC)
        for name in sorted(self.down - set(out)):
            notify.send(f"✅ *{name} is logging again*")
        self.down = set(out)
        return out

    def _ping(self, t: float) -> None:
        url = os.getenv("HEARTBEAT_PING_URL", "").strip()
        if not url or t - self.last_ping < PING_EVERY_SEC:
            return
        self.last_ping = t
        try:
            requests.get(url, timeout=10)
        except Exception as e:                                 # noqa: BLE001
            # The URL is a credential, and requests quotes it in its errors.
            print(f"[LIVENESS] ping failed: {type(e).__name__}")

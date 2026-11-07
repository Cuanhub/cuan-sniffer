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
"""

from __future__ import annotations

import os
import pathlib
import time
from typing import Callable, Dict, List, Optional, Set

import requests

import notify
from core.meta import DATA, read_json, write_json

LOG_DIR = pathlib.Path(os.getenv("CUAN_LOG_DIR",
                                 str(pathlib.Path.home() / "Library/Logs/cuan-sniffer")))
STATE = DATA / "_liveness.json"

# Seconds a log may sit still. Each is well over that service's normal silence:
# protect logs every 60s, the trader every 450s plus a pass of about 3 minutes,
# the scanner when a pass completes, every 900s or longer if outcomes are due.
STALE_SEC: Dict[str, float] = {"protect": 300.0, "trader": 1200.0, "scanner": 2700.0}
# A longer gap between ticks of this loop means nothing here was running. Its
# worst normal gap is a long poll plus the failure backoff, about a minute.
GAP_SEC = float(os.getenv("LIVENESS_GAP_SEC", "300"))
REPEAT_SEC = 3600
PING_EVERY_SEC = 300.0


def _human(sec: float) -> str:
    return f"{sec / 3600:.1f}h" if sec >= 5400 else f"{sec / 60:.0f} min"


class Watch:
    def __init__(self, now: Callable[[], float] = time.time,
                 logs: Optional[pathlib.Path] = None,
                 state: Optional[pathlib.Path] = None) -> None:
        self.now = now
        self.logs = logs or LOG_DIR
        self.state = state or STATE
        self.awake_since = now()
        self.down: Set[str] = set()
        self.last_ping = 0.0
        self.owed: Optional[str] = None     # a gap alert not yet delivered

    def _check_gap(self, t: float) -> None:
        row = read_json(self.state, {})
        last = float(row.get("last_tick", 0.0)) if isinstance(row, dict) else 0.0
        write_json(self.state, {"last_tick": t})
        if last and t - last > GAP_SEC:
            self.awake_since = t
            self.owed = (
                f"⚠️ *System was down for {_human(t - last)}*\nThe Mac slept, "
                f"lost power or the control loop stopped. Stops and targets on "
                f"the venue kept working, but a bid that filled in that time "
                f"had no stop until protection came back. Check /positions.")
        # Kept until it lands: the network is often not back in the first
        # seconds after a wake, and this is the one message that says so.
        if self.owed and notify.send(self.owed):
            self.owed = None

    def _stale(self, t: float) -> List[str]:
        out = []
        for name, limit in STALE_SEC.items():
            # After a wake every log looks old. Give each service one full
            # period to show it came back before calling it dead.
            if t - self.awake_since < limit:
                continue
            try:
                age = t - (self.logs / f"{name}.log").stat().st_mtime
            except OSError:
                age = float("inf")
            if age > limit:
                out.append(name)
                notify.send(
                    f"🚨 *{name} has gone quiet*\nNothing logged for "
                    f"{'too long' if age == float('inf') else _human(age)}."
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

    def tick(self) -> None:
        t = self.now()
        self._check_gap(t)
        # Silence is the signal the outside service acts on, so a stale service
        # stops the pings as well as alerting here.
        if not self._stale(t):
            self._ping(t)

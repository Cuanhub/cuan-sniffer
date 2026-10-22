"""Run the protection cycle every 60 seconds.

Replaces tools/reconcile_loop.py. The difference is not the interval: the old
loop compared the venue against the trader's state file and queued a halt on
any disagreement, so every normal exit and every resting-bid fill produced a
halt within 45s (REVIEW.md finding 2). This one reads the venue, fixes what is
wrong, and never compares two beliefs -- so it has no false-divergence class to
produce.

It may place protective orders and close an unprotectable position. It never
writes trade state and never opens a position.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv                                    # noqa: E402

load_dotenv(pathlib.Path(__file__).resolve().parents[1] / ".env", override=False)

from core import protect, safety                                  # noqa: E402
from core.meta import MetaStore                                   # noqa: E402
from core.venue import Venue                                      # noqa: E402

POLL_SEC = float(os.getenv("PROTECT_POLL_SEC", "60"))

# Retry venue construction before giving the process up.
#
# The SDK fetches perp metadata inside Info.__init__, so construction hits the
# network and raises there before any of our code runs. This loop used to
# return 1 on the first failure, and with KeepAlive plus a 30s
# ThrottleInterval that is a crash loop: during the 2026-10-22 outage it
# logged "no venue ... cannot run" 117 times, about an hour of cycling.
#
# The trader has had start_venue() for exactly this since 2026-09-19, and its
# docstring says why: "a crash loop against a rate limit is the worst possible
# way to wait, because every restart is another request, which is what caused
# the limit." The protection loop -- the safety-critical one -- did not have
# it. Five attempts at 15/30/45/60s ride out a short outage without exiting at
# all, and a longer one cycles every ~3 minutes instead of every 30 seconds.
#
# Kept here rather than shared with the trader's version because that one
# honours the trader's SIGTERM flag mid-wait and this process has none.
START_ATTEMPTS = 5
START_BACKOFF_SEC = 15


def start_venue(sleep=time.sleep) -> "Venue":
    for attempt in range(1, START_ATTEMPTS + 1):
        try:
            return Venue(debug=False)
        except Exception as e:                                 # noqa: BLE001
            if attempt == START_ATTEMPTS:
                raise
            wait = START_BACKOFF_SEC * attempt
            print(f"[PROTECT] venue start attempt {attempt} failed "
                  f"({str(e)[:90]}); retrying in {wait}s")
            sleep(wait)
    raise RuntimeError("unreachable")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    try:
        venue = start_venue()
    except Exception as e:
        print(f"[PROTECT] no venue after {START_ATTEMPTS} attempts "
              f"({str(e)[:120]}) -- exiting for launchd to retry")
        return 1
    store = MetaStore()
    print(f"[PROTECT] protecting every {POLL_SEC:g}s")

    while True:
        try:
            results = protect.run_once(venue, store)
            if results:
                print("[PROTECT] " + "; ".join(
                    f"{r.coin}={r.action}" for r in results))
        except Exception as e:                                   # noqa: BLE001
            # A protection loop that dies leaves positions unprotected, so it
            # logs and keeps going rather than exiting on an unexpected error.
            safety.record_error("protect_cycle", e)
            print(f"[PROTECT] cycle raised: {str(e)[:160]}")
        if args.once:
            return 0
        # Self-heal. A loop that has not managed a single venue read for
        # READ_DEAD_SEC is not protecting anything, and some causes outlive
        # the process: on 2026-10-19 this and the trader lost DNS after days
        # of uptime while `dig` resolved the host instantly, and nothing
        # recovered them. Backing off cannot fix a dead resolver.
        #
        # Exiting is safe HERE in a way it is not for an unexpected error --
        # see the handler above, which deliberately keeps going. The
        # difference is that this process is already doing nothing, and every
        # stop it placed is resting at the venue where it fires without us.
        # KeepAlive restarts it after ThrottleInterval (30s).
        why = safety.give_up_if_unreadable("Protect loop")
        if why:
            print(f"[PROTECT] exiting so launchd can restart me: {why}")
            return 1
        time.sleep(POLL_SEC)


if __name__ == "__main__":
    raise SystemExit(main())

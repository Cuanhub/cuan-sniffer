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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    try:
        venue = Venue(debug=False)
    except Exception as e:
        print(f"[PROTECT] no venue ({str(e)[:120]}) -- cannot run")
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
        time.sleep(POLL_SEC)


if __name__ == "__main__":
    raise SystemExit(main())

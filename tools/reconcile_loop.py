"""Compare the venue against what the trader believes, every 45 seconds.

AUDIT B-1, interim. There is no websocket anywhere in this project: both SDK
clients are constructed with skip_ws=True, so all market and account data
arrives on the trader's 450-second poll. A gap through a stop, a partial
fill, or a venue-side liquidation is invisible for up to seven and a half
minutes. Websockets are the real answer and a later batch.

WHAT THIS PROCESS MAY DO: read the venue, read the trader's state file, queue
a halt, send an alert.

WHAT IT MAY NOT DO: place any order, size anything, or write
_live_state.json. The trader owns that file and stays its only writer --
two writers on a read-modify-write JSON blob is a lost halt waiting to
happen. This process raises the alarm; the trader acts on it through the same
command queue the Telegram channel uses.

A query that FAILS is not a divergence. Reporting an unreadable venue as a
missing position would halt the account on a rate limit, so an unreadable
read returns None and the loop simply tries again.
"""

import argparse
import json
import pathlib
import sys
import time
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv                                    # noqa: E402

load_dotenv(pathlib.Path(__file__).resolve().parents[1] / ".env",
            override=False)

import notify                                                     # noqa: E402
from research.levels.config import RECONCILE_POLL_SEC             # noqa: E402

DATA = pathlib.Path(__file__).resolve().parent.parent / "research/levels/data"
STATE = DATA / "_live_state.json"
QUEUE = DATA / "_control_queue.json"

# Size disagreements below this fraction are rounding, not divergence.
SIZE_TOLERANCE = 0.02

# What has already been alarmed about, so a standing problem does not alert
# every 45 seconds and train the reader to ignore it.
_alarmed: set = set()


def _f(v) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def venue_positions(backend) -> Optional[Dict[str, float]]:
    """coin -> signed size. None if the venue could not be read."""
    try:
        st = backend.info.user_state(backend._user_state_query_address())
    except Exception as e:
        print(f"[RECONCILE] venue unreadable: {str(e)[:90]}")
        return None
    if not isinstance(st, dict):
        return None
    out = {}
    for p in st.get("assetPositions") or []:
        pos = (p or {}).get("position") or {}
        coin = str(pos.get("coin", "")).upper()
        sz = _f(pos.get("szi"))
        if coin and sz is not None and abs(sz) > 0:
            out[coin] = sz
    return out


def has_protection(backend, coin: str) -> Optional[bool]:
    """True/False, or None when the order book could not be read."""
    try:
        orders = backend.get_open_orders(coin)
    except Exception:
        return None
    if orders is None:
        return None
    for o in orders:
        if str(o.get("coin", "")).upper() != coin.upper():
            continue
        if (o.get("reduceOnly") or o.get("reduce_only")
                or o.get("isTrigger") or o.get("is_trigger")
                or o.get("triggerPx") or o.get("trigger_px")):
            return True
    return False


def check(backend, state: Dict[str, Any]) -> Optional[List[str]]:
    """Divergences between the venue and `state`. None if unreadable."""
    live = venue_positions(backend)
    if live is None:
        return None
    believed = {c.upper(): v for c, v in (state.get("owned") or {}).items()}
    problems: List[str] = []

    for coin in sorted(set(live) - set(believed)):
        prot = has_protection(backend, coin)
        problems.append(
            f"{coin} is untracked at the venue ({live[coin]:g})"
            + ("" if prot is not False else " and has NO protective order"))

    for coin in sorted(set(believed) - set(live)):
        problems.append(f"{coin} is gone at the venue but the trader still "
                        f"holds the slot")

    for coin in sorted(set(believed) & set(live)):
        want = _f(believed[coin].get("size_coin"))
        got = abs(live[coin])
        if want and want > 0 and abs(got - want) / want > SIZE_TOLERANCE:
            problems.append(f"{coin} size disagrees: venue {got:g}, "
                            f"trader {want:g}")
        if has_protection(backend, coin) is False:
            problems.append(f"{coin} is open and unprotected -- no stop is "
                            f"resting at the venue")
    return problems


def raise_alarm(problems: List[str]) -> None:
    """Queue a halt for the trader and alert. Never writes trader state."""
    key = "|".join(sorted(problems))
    if key in _alarmed:
        return
    _alarmed.add(key)

    try:
        q = json.loads(QUEUE.read_text()) if QUEUE.exists() else []
        if not isinstance(q, list):
            q = []
    except (OSError, ValueError):
        q = []
    q.append({"cmd": "halt",
              "arg": "reconcile loop found a venue divergence: "
                     + "; ".join(problems)[:300],
              "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    tmp = QUEUE.with_suffix(QUEUE.suffix + ".tmp")
    tmp.write_text(json.dumps(q, indent=2))
    tmp.replace(QUEUE)

    for p in problems:
        print(f"[RECONCILE] !! {p}")
    notify.send("🚨 *Venue disagrees with the bot*\n"
                + "\n".join(f"• {p}" for p in problems)
                + "\n\nA halt has been queued. Check `/positions`.",
                key="reconcile_divergence", cooldown_sec=900)


def read_state() -> Dict[str, Any]:
    """Read-only. This process never writes here."""
    try:
        s = json.loads(STATE.read_text())
        return s if isinstance(s, dict) else {}
    except (OSError, ValueError):
        return {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    try:
        from live_execution_backend import LiveExecutionBackend
        backend = LiveExecutionBackend(debug=False)
    except Exception as e:
        print(f"[RECONCILE] no venue backend ({str(e)[:120]}) -- cannot run")
        return 1

    print(f"[RECONCILE] watching every {RECONCILE_POLL_SEC}s")
    while True:
        problems = check(backend, read_state())
        if problems is None:
            print("[RECONCILE] venue unreadable this cycle")
        elif problems:
            raise_alarm(problems)
        else:
            _alarmed.clear()      # agreement clears the alarm memory
        if args.once:
            return 0
        time.sleep(RECONCILE_POLL_SEC)


if __name__ == "__main__":
    raise SystemExit(main())

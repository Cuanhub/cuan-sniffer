"""Telegram control channel for the level trader.

Why this is a separate process, and what that buys and costs.

The trader polls every POLL_SEC (450s). Routing commands through it would put
up to seven and a half minutes between "/close HYPE" and the order landing,
which in a moving market is the difference between a command and a wish. So
this runs on its own and long-polls Telegram.

That means two processes with order-placing capability against one account,
which is a real hazard and is handled by splitting commands three ways:

  READ-ONLY  (/status, /positions)  answered here, from the venue and from a
             read-only load of the state file. Writes nothing.

  RISK-REDUCING AT THE VENUE  (/close, /closeall)  executed here, directly.
             Safe because the trader already treats a position that has
             vanished as a first-class case: reconciliation logs "no longer
             open at venue" and its comment says in so many words "closed by
             a venue stop, a take-profit, or by hand". Closing from here is
             that case.

  STATE-CHANGING  (/halt, /resume, /pause, /take)  appended to a command
             queue, which the trader drains at the start of its next pass.
             _live_state.json keeps exactly one writer. The cost is latency:
             a halt can take a pass to apply. That is acceptable because a
             halt only blocks NEW entries, and entry spacing already bounds
             those to one an hour.

What /open is not. There is no "/open COIN" that market-buys, because a long
without a level has no structural stop and no target, and inventing them is
precisely what produces the trades this system is trying to stop taking.
`/take COIN` instead forces a coin the scanner ALREADY has as a live candidate
past the rails that are blocking it -- real level, real stop, real target,
operator's judgement substituted for one veto. If the scanner has nothing on
that coin, /take refuses and says so.

Authorisation is a single chat id. See notify.authorised.
"""

import argparse
import json
import pathlib
import secrets
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv                                   # noqa: E402

# Same as the trader: launchd hands a job almost no environment, so the
# credentials have to be loaded here rather than inherited. override=False so
# a variable set deliberately in the plist still wins.
load_dotenv(pathlib.Path(__file__).resolve().parents[1] / ".env",
            override=False)

import notify                                                    # noqa: E402
from research.levels.config import (                             # noqa: E402
    ACCOUNT_DD_HALT_PCT, ACCOUNT_DD_WINDOW_HOURS, DAILY_LOSS_STOP_PCT,
    MAX_CONCURRENT, MAX_NEW_PER_DAY, LOSS_STREAK_TRIGGER,
)

DATA = pathlib.Path(__file__).resolve().parent.parent / "research/levels/data"
STATE = DATA / "_live_state.json"
QUEUE = DATA / "_control_queue.json"
OFFSET = DATA / "_control_offset.json"

# How long a confirmation token stays good. Short, because the thing it
# guards is a market order: a token that is still valid ten minutes later is
# one the operator may have forgotten the context of.
CONFIRM_TTL_SEC = 120

RISKY = {"resume", "close", "closeall", "take"}


def now() -> datetime:
    return datetime.now(timezone.utc)


def read_json(path: pathlib.Path, default):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def write_json(path: pathlib.Path, obj) -> None:
    """Atomic, same as the trader's save_state -- rename, never truncate."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def enqueue(cmd: str, arg: str = "") -> None:
    """Append a command for the trader to drain.

    Read-modify-write on a file the trader only ever truncates, never
    appends to, so the two cannot interleave destructively: the trader
    replaces the queue with [] once it has taken everything.
    """
    q = read_json(QUEUE, [])
    if not isinstance(q, list):
        q = []
    q.append({"cmd": cmd, "arg": arg,
              "at": now().isoformat().replace("+00:00", "Z")})
    write_json(QUEUE, q)


# ---- rendering ------------------------------------------------------------

def _f(v, d=0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def render_status(state: Dict[str, Any], equity: Optional[float]) -> str:
    owned = state.get("owned") or {}
    pending = state.get("pending") or {}
    lines = ["*Cuan Sniffer status*"]

    if state.get("halted"):
        kind = state.get("halt_kind") or "unknown"
        lines.append(f"🛑 HALTED ({kind})\n`{state.get('halt_reason','')}`")
    else:
        lines.append("🟢 live")

    if equity is not None:
        lines.append(f"equity *${equity:,.2f}*")
        samples = [_f(s[1]) for s in (state.get("equity_samples") or [])
                   if isinstance(s, (list, tuple)) and len(s) > 1]
        ref = max(samples + [equity]) if samples else equity
        if ref > 0:
            dd = (ref - equity) / ref * 100.0
            lines.append(f"drawdown {dd:.2f}% of a ${ref:,.2f} "
                         f"{ACCOUNT_DD_WINDOW_HOURS:g}h high "
                         f"(halts at {ACCOUNT_DD_HALT_PCT:g}%)")
        lines.append(f"day realised *${_f(state.get('realised_today')):+,.2f}*"
                     f"  (stop at -{DAILY_LOSS_STOP_PCT:g}% = "
                     f"${-equity*DAILY_LOSS_STOP_PCT/100:,.2f})")

    lines.append(f"positions {len(owned)}/{MAX_CONCURRENT}"
                 + (f", {len(pending)} resting bid(s)" if pending else "")
                 + f"  |  opened today {state.get('opened_today', 0)}"
                 f"/{MAX_NEW_PER_DAY}")

    streak = int(_f(state.get("loss_streak")))
    if streak:
        lines.append(f"losing run {streak}/{LOSS_STREAK_TRIGGER}")
    until = state.get("streak_cooldown_until")
    if until:
        try:
            left = (datetime.fromisoformat(str(until).replace("Z", "+00:00"))
                    - now()).total_seconds() / 3600.0
            if left > 0:
                lines.append(f"⏸ standing down {left:.1f}h "
                             f"(after {LOSS_STREAK_TRIGGER} losses)")
        except ValueError:
            pass

    cooled = state.get("cooled") or {}
    if cooled:
        lines.append("cooling off: " + ", ".join(sorted(cooled))[:200])
    return "\n".join(lines)


def render_positions(state: Dict[str, Any], backend) -> str:
    owned = state.get("owned") or {}
    if not owned:
        return "Flat -- no open positions."
    out = ["*Open positions*"]
    for coin, info in sorted(owned.items()):
        entry = _f(info.get("entry"))
        stop, tp = _f(info.get("stop")), _f(info.get("tp"))
        mid = None
        if backend is not None:
            try:
                mid = backend.get_mid_price(coin)
            except Exception:
                mid = None
        size = _f(info.get("size_usd"))
        row = [f"*{coin}*  entry `{entry:.6g}`"]
        if mid:
            pct = (mid / entry - 1) * 100 if entry else 0.0
            row.append(f"now `{mid:.6g}` ({pct:+.2f}%)")
            if size:
                row.append(f"≈ *${size*pct/100:+,.2f}*")
        row.append(f"stop `{stop:.6g}`  tp `{tp:.6g}`")
        if size:
            row.append(f"${size:,.0f} notional")
        out.append("  ".join(row))
    return "\n".join(out)


HELP = """*Cuan Sniffer controls*

_read-only_
`/status` — rails, equity, drawdown, day P&L
`/positions` — open positions with live P&L
`/help` — this

_immediate_
`/halt` — stop opening anything (queued, applies next pass)
`/close COIN` — market-close one position
`/closeall` — market-close everything

_needs confirming_
`/resume` — clear a halt
`/pause H` — no new entries for H hours
`/take COIN` — force a coin the scanner already likes past the rails

Anything that risks money replies with a token; send it back within \
2 minutes to go ahead.

There is deliberately no `/open COIN`: a long with no level has no \
structural stop or target, and inventing them is what this system is \
trying to stop doing."""


# ---- command handling -----------------------------------------------------

def parse(text: str):
    parts = text.strip().split()
    if not parts:
        return "", ""
    cmd = parts[0].lstrip("/").lower()
    if "@" in cmd:                      # /status@MyBot in a group
        cmd = cmd.split("@", 1)[0]
    return cmd, (parts[1] if len(parts) > 1 else "")


class Controller:
    def __init__(self, backend=None):
        self.backend = backend
        self.pending: Dict[str, Any] = {}

    # -- venue helpers, each tolerant of a backend that is absent or angry --
    def equity(self) -> Optional[float]:
        if self.backend is None:
            return None
        try:
            ms = self.backend.get_margin_summary()
            return None if ms is None else float(ms[0])
        except Exception:
            return None

    def close_one(self, coin: str, state: Dict[str, Any]) -> str:
        info = (state.get("owned") or {}).get(coin)
        if info is None:
            return f"No open position in {coin}."
        if self.backend is None:
            return "No venue connection -- cannot close."
        size = _f(info.get("size_usd"))
        mid = None
        try:
            mid = self.backend.get_mid_price(coin)
        except Exception:
            pass
        if not mid or size <= 0:
            return f"Could not price {coin} -- not sending a blind exit."
        try:
            res = self.backend.execute_exit(coin=coin, side="LONG",
                                            price=float(mid),
                                            size_usd=size,
                                            reason="telegram")
        except Exception as e:
            return f"{coin} exit raised: {str(e)[:160]}"
        status = getattr(res, "status", None) or "unknown"
        ok = str(status).lower() in ("filled", "ok", "success")
        # The trader reconciles the vanished position on its next pass and is
        # the thing that records the realised PnL. Saying so here stops the
        # operator wondering why /status still shows the slot for a few
        # minutes.
        return (f"{'✅' if ok else '⚠️'} {coin} exit: {status}. "
                f"The trader will book the result on its next pass.")

    def handle(self, text: str, state: Dict[str, Any]) -> str:
        cmd, arg = parse(text)
        if not cmd:
            return ""

        # A confirmation is a command in its own right so that a stale or
        # wrong token fails loudly instead of silently doing nothing.
        if cmd == "confirm":
            p = self.pending
            if not p:
                return "Nothing is waiting to be confirmed."
            if time.time() > p.get("expires", 0):
                self.pending = {}
                return "That confirmation expired. Send the command again."
            if arg != p.get("token"):
                return "Wrong token. Send the command again to get a new one."
            self.pending = {}
            return self.execute(p["cmd"], p["arg"], state)

        if cmd in ("help", "start"):
            return HELP
        if cmd == "status":
            return render_status(state, self.equity())
        if cmd == "positions":
            return render_positions(state, self.backend)
        if cmd == "halt":
            # Not confirmed: halting only ever reduces exposure, and a halt
            # the operator did not mean costs nothing but a /resume.
            enqueue("halt", "requested from Telegram")
            return ("🛑 Halt queued. It applies on the trader's next pass "
                    "(within ~7 min) and blocks new entries only -- open "
                    "positions keep their stops. Use `/closeall` to flatten.")
        if cmd in RISKY:
            if cmd in ("close", "take") and not arg:
                return f"Which coin? `/{cmd} HYPE`"
            if cmd == "pause" and not arg:
                return "How many hours? `/pause 6`"
            token = secrets.token_hex(3)
            self.pending = {"cmd": cmd, "arg": arg.upper() if cmd != "pause"
                            else arg, "token": token,
                            "expires": time.time() + CONFIRM_TTL_SEC}
            what = self.describe(cmd, self.pending["arg"], state)
            return (f"{what}\n\nSend `/confirm {token}` within "
                    f"{CONFIRM_TTL_SEC // 60} minutes.")
        if cmd == "pause":
            return "How many hours? `/pause 6`"
        return f"Unknown command `{cmd}`. Try `/help`."

    def describe(self, cmd: str, arg: str, state: Dict[str, Any]) -> str:
        owned = state.get("owned") or {}
        if cmd == "resume":
            if not state.get("halted"):
                return "Not halted -- resuming would change nothing."
            return (f"Clear the halt?\n`{state.get('halt_reason','')}`\n"
                    f"The trader may open a position on its next pass.")
        if cmd == "close":
            info = owned.get(arg)
            if info is None:
                return f"{arg} is not open. Confirming will do nothing."
            return (f"Market-close *{arg}* at whatever the book gives?\n"
                    f"entry `{_f(info.get('entry')):.6g}`, "
                    f"${_f(info.get('size_usd')):,.0f} notional. "
                    f"This realises the result now.")
        if cmd == "closeall":
            if not owned:
                return "Flat already. Confirming will do nothing."
            return (f"Market-close *all {len(owned)}* positions "
                    f"({', '.join(sorted(owned))})?\nThis realises every "
                    f"result now, at market.")
        if cmd == "take":
            return (f"Force an entry on *{arg}*, past whichever rail is "
                    f"blocking it?\nThe trader will only do this if the "
                    f"scanner already has a live level for {arg} -- it will "
                    f"not invent a stop. Normal sizing and protection apply.")
        if cmd == "pause":
            return f"No new entries for *{arg}h*?"
        return f"Confirm `{cmd} {arg}`?"

    def execute(self, cmd: str, arg: str, state: Dict[str, Any]) -> str:
        if cmd == "resume":
            enqueue("resume", "cleared from Telegram")
            return "✅ Resume queued -- applies on the next pass."
        if cmd == "pause":
            try:
                hours = float(arg)
            except ValueError:
                return f"`{arg}` is not a number of hours."
            if not 0 < hours <= 168:
                return "Pick between 0 and 168 hours."
            enqueue("pause", str(hours))
            return f"⏸ Queued: no new entries for {hours:g}h."
        if cmd == "close":
            return self.close_one(arg, state)
        if cmd == "closeall":
            owned = list((state.get("owned") or {}))
            if not owned:
                return "Flat already."
            return "\n".join(self.close_one(c, state) for c in sorted(owned))
        if cmd == "take":
            enqueue("take", arg)
            return (f"Queued: the trader will try *{arg}* on its next pass, "
                    f"or say why it still refuses.")
        return f"Nothing to do for `{cmd}`."


# ---- loop -----------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true",
                    help="drain whatever is waiting, then exit")
    ap.add_argument("--no-venue", action="store_true",
                    help="read-only: no backend, so /close cannot be used")
    args = ap.parse_args()

    if not notify.configured():
        print("[CONTROL] TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID not set -- "
              "nothing to listen to")
        return 1

    backend = None
    if not args.no_venue:
        try:
            from live_execution_backend import LiveExecutionBackend
            backend = LiveExecutionBackend()
        except Exception as e:
            print(f"[CONTROL] no venue backend ({str(e)[:120]}) -- "
                  f"running read-only")

    ctl = Controller(backend)
    offset = int(read_json(OFFSET, {}).get("offset", 0) or 0)
    print(f"[CONTROL] listening, offset {offset}, "
          f"venue={'yes' if backend else 'no'}")

    # Longer than notify.POLL_TIMEOUT_SEC on purpose. Telegram answers a
    # second concurrent getUpdates with 409 while the first is still hanging,
    # so a short retry after a failure collides with this process's own open
    # long-poll and 409s forever -- which is what it did on first install.
    fail_backoff = notify.POLL_TIMEOUT_SEC + 5

    while True:
        updates, nxt, ok = notify.get_updates(offset)
        for u in updates:
            if not notify.authorised(u["chat_id"]):
                # Not answered on purpose. Replying would confirm to a
                # stranger that this bot exists and takes commands.
                print(f"[CONTROL] ignored message from chat "
                      f"{u['chat_id']} user {u['user_id']}: "
                      f"{u['text'][:60]!r}")
                continue
            state = read_json(STATE, {})
            print(f"[CONTROL] {u['text'][:80]!r}")
            try:
                reply = ctl.handle(u["text"], state)
            except Exception as e:
                reply = f"That command raised: {str(e)[:200]}"
                print(f"[CONTROL] handler raised: {e}")
            if reply:
                notify.send(reply)
        if nxt != offset:
            offset = nxt
            # Persisted so a restart does not replay commands Telegram still
            # considers unacknowledged.
            write_json(OFFSET, {"offset": offset})
        if args.once:
            return 0
        time.sleep(1 if ok else fail_backoff)


if __name__ == "__main__":
    raise SystemExit(main())

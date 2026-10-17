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

  STATE-CHANGING  (/halt, /resume, /pause)  appended to a command
             queue, which the trader drains at the start of its next pass.
             _live_state.json keeps exactly one writer. The cost is latency:
             a halt can take a pass to apply. That is acceptable because a
             halt only blocks NEW entries, and entry spacing already bounds
             those to one an hour.

What /open is not. There is no "/open COIN" that market-buys, because a long
without a level has no structural stop and no target, and inventing them is
precisely what produces the trades this system is trying to stop taking.

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
    MAX_CONCURRENT, MAX_NEW_PER_DAY,
)

DATA = pathlib.Path(__file__).resolve().parent.parent / "research/levels/data"
STATE = DATA / "_live_state.json"
QUEUE = DATA / "_control_queue.json"
OFFSET = DATA / "_control_offset.json"

# How long a confirmation token stays good. Short, because the thing it
# guards is a market order: a token that is still valid ten minutes later is
# one the operator may have forgotten the context of.
CONFIRM_TTL_SEC = 120

RISKY = {"resume", "close", "closeall"}


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


def render_status(state: Dict[str, Any], equity: Optional[float],
                  backend=None) -> str:
    """What the account is doing. Positions come from the VENUE.

    This used to count state["owned"], so a position the state file did not
    know about was invisible to /status -- the same blindness as finding 5,
    on the screen the operator checks first.
    """
    from core import safety, streak
    lines = ["*Cuan Sniffer status*"]

    active = safety.HaltState().active()
    if active:
        lines.append(f"🛑 HALTED ({active.get('kind')})\n"
                     f"`{active.get('reason', '')}`")
    else:
        lines.append("🟢 live")

    if equity is not None:
        lines.append(f"equity *${equity:,.2f}*")
        day_start, high = safety.EquityLog().observe(equity)
        if day_start > 0:
            lost = (day_start - equity) / day_start * 100.0
            lines.append(f"today {-lost:+.2f}% from ${day_start:,.2f} "
                         f"(halts at -{safety.DAILY_LOSS_PCT:g}%)")
        if high > 0:
            dd = (high - equity) / high * 100.0
            lines.append(f"drawdown {dd:.2f}% of a ${high:,.2f} "
                         f"{safety.DRAWDOWN_WINDOW_HOURS:g}h high "
                         f"(halts at {safety.DRAWDOWN_PCT:g}%)")

    held = backend.positions() if backend is not None else None
    orders = backend.open_orders() if backend is not None else None
    if held is None:
        lines.append("positions: venue unreadable")
    else:
        bids = [o for o in (orders or []) if o.is_entry]
        lines.append(f"positions {len(held)}/{MAX_CONCURRENT}"
                     + (f", {len(bids)} resting bid(s)" if bids else "")
                     + f"  |  opened today {state.get('opened_today', 0)}"
                     f"/{MAX_NEW_PER_DAY}")

    if backend is not None:
        sd = streak.current(backend,
                            safety.StandDownOverride().ignore_before_ms())
        if sd is not None:
            lines.append(f"⏸ standing down {sd.hours_left:.1f}h -- {sd.reason}")

    return "\n".join(lines)


def render_positions(state: Dict[str, Any], backend) -> str:
    """Every position the VENUE holds, and whether each one has a stop."""
    if backend is None:
        return "No venue connection."
    from core.meta import MetaStore
    held = backend.positions()
    if held is None:
        return "⚠️ Could not read the venue."
    orders = backend.open_orders() or []
    if not held:
        bids = [o for o in orders if o.is_entry]
        return ("Flat -- no open positions."
                + (f"\n{len(bids)} resting bid(s): "
                   + ", ".join(f"`{o.coin}` @ {o.limit_px:.6g}" for o in bids)
                   if bids else ""))
    meta = MetaStore()
    out = ["*Open positions*"]
    for coin, p in sorted(held.items()):
        stops = [o for o in orders
                 if o.coin == coin and o.is_protective and o.is_stop]
        plan = meta.get(coin)
        row = [f"*{coin}*  {p.size:g} @ `{p.entry_px:.6g}`",
               f"mark `{p.mark_px:.6g}`",
               f"*{p.unrealized_pnl:+,.2f}*"]
        if stops:
            row.append(f"stop `{stops[0].trigger_px:.6g}`"
                       + ("" if abs(stops[0].size - p.size)
                          <= p.size * 0.01 else
                          f" ⚠️ covers only {stops[0].size:g}"))
        else:
            row.append("🚨 *NO STOP*")
        if plan and plan.planned_tp:
            row.append(f"tp `{plan.planned_tp:.6g}`")
        out.append("  ".join(row))
    bids = [o for o in orders if o.is_entry]
    if bids:
        out.append("resting: " + ", ".join(
            f"`{o.coin}` @ {o.limit_px:.6g} ({o.age_hours():.1f}h)"
            for o in bids))
    return "\n".join(out)


HELP = """*Cuan Sniffer controls*

_read-only_
`/status` — rails, equity, drawdown, stand-down
`/positions` — open positions, live P&L, and whether each has a stop
`/help` — this

_immediate_
`/halt` — stop opening anything (queued, applies next pass)
`/close COIN` — market-close one position
`/closeall` — cancel every order and close everything the venue holds

_needs confirming_
`/resume` — clear a halt
`/pause H` — no new entries for H hours (`/pause 0` lifts a stand-down)

Anything that risks money replies with a token; send it back within \
2 minutes to go ahead.

There is deliberately no `/open COIN`: a long with no level has no \
structural stop or target, and inventing them is what this system is \
trying to stop doing. There is no `/take` either -- nothing can force an \
entry past a rail."""


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
            eq = self.backend.equity()
            return None if eq is None else eq.usdc_total
        except Exception:
            return None

    def close_one(self, coin: str, state: Dict[str, Any]) -> str:
        """Flatten one position and report what the VENUE says.

        `state` is no longer consulted to decide whether a position exists.
        It used to be, and that is REVIEW.md finding 5: a position the state
        file did not know about -- which is exactly what a timeout-after-fill
        leaves -- could not be closed from Telegram at all. This answered
        "No open position in X" while the venue held it.
        """
        if self.backend is None:
            return "No venue connection -- cannot close."
        from core import safety
        held = self.backend.positions(urgent=True)
        if held is None:
            return "⚠️ Could not read the venue. Nothing sent; try again."
        if coin not in held:
            return f"{coin} is not open at the venue."
        res = safety.flatten_all(self.backend, reason=f"telegram /close {coin}")
        if res.flat:
            return (f"✅ Flat"
                    + (f" (closed `{res.closed}`)" if res.closed else "")
                    + ".")
        return (f"🚨 STILL OPEN after {res.seconds:.0f}s\n"
                f"`{res.still_open}`\n`{str(res.detail)[:300]}`\n"
                f"Close it at the venue.")

    def close_all(self) -> str:
        """The kill switch. Everything the VENUE holds, not what state lists."""
        if self.backend is None:
            return "No venue connection -- cannot close."
        from core import safety
        res = safety.flatten_all(self.backend, reason="telegram /closeall")
        if res.flat:
            return (f"✅ Flat"
                    + (f" (closed `{res.closed}`)" if res.closed else
                       " -- nothing was open") + ".")
        return (f"🚨 NOT FLAT after {res.seconds:.0f}s\n"
                f"Still open: `{res.still_open}`\n"
                f"Closed: `{res.closed or '{}'}`\n`{str(res.detail)[:300]}`")

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
            return render_status(state, self.equity(), self.backend)
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
            if cmd == "close" and not arg:
                return f"Which coin? `/{cmd} HYPE`"
            if cmd == "pause" and not arg:
                return "How many hours? `/pause 6`, or `/pause 0` to lift one"
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
            held = (self.backend.positions(urgent=True)
                    if self.backend is not None else None)
            if held is None:
                return (f"Cannot read the venue. Confirming will try to close "
                        f"*{arg}* anyway.")
            p = held.get(arg)
            if p is None:
                return f"{arg} is not open at the venue. Confirming does nothing."
            return (f"Market-close *{arg}* at whatever the book gives?\n"
                    f"{p.size:g} @ mark `{p.mark_px:.6g}`, entry "
                    f"`{p.entry_px:.6g}`, unrealised `{p.unrealized_pnl:+.2f}`.\n"
                    f"This realises the result now.")
        if cmd == "closeall":
            held = (self.backend.positions(urgent=True)
                    if self.backend is not None else None)
            if held is None:
                return ("Cannot read the venue, so cannot say what is open. "
                        "Confirming will cancel every order and close "
                        "everything the venue reports.")
            if not held:
                return "The venue holds nothing. Confirming will do nothing."
            return (f"Cancel every resting order and market-close *all "
                    f"{len(held)}* positions ({', '.join(sorted(held))})?\n"
                    f"This realises every result now, at market.")
        if cmd == "pause":
            try:
                if float(arg) <= 0:
                    return ("Lift the stand-down and let it open positions "
                            "again on the next pass?")
            except ValueError:
                pass
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
            if hours <= 0:
                enqueue("pause", "0")
                return ("▶️ Queued: lifting the stand-down. Applies on the "
                        "next pass.")
            if hours > 168:
                return "168 hours is the most. Use `/halt` for longer."
            enqueue("pause", str(hours))
            return f"⏸ Queued: no new entries for {hours:g}h."
        if cmd == "close":
            return self.close_one(arg, state)
        if cmd == "closeall":
            return self.close_all()
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
            from core.venue import Venue
            backend = Venue(debug=False)
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

    # A sequential loop cannot conflict with itself, so repeated 409s mean
    # ANOTHER process is polling this same bot token. That is not a transient
    # fault and it is not safe to sit quietly through: two pollers steal each
    # other's updates, so a command sent here can be consumed by the other
    # process and never arrive. Found on first install -- the same token was
    # in use by another project on this machine, which had been polling for
    # four days. The fix is a second bot from BotFather, not a retry.
    conflicts = 0

    while True:
        updates, nxt, ok = notify.get_updates(offset)
        if ok:
            conflicts = 0
        else:
            conflicts += 1
            if conflicts == 3:
                msg = ("⚠️ *Control channel cannot start*\nAnother process is "
                       "polling this same bot token, so commands sent here "
                       "may be swallowed by it instead.\n\nGive Cuan Sniffer "
                       "its own bot from @BotFather and set "
                       "TELEGRAM_BOT_TOKEN to it.")
                print("[CONTROL] three consecutive poll failures -- another "
                      "process is almost certainly polling this same bot "
                      "token. Commands may be delivered to it instead of "
                      "here. Use a separate bot for this trader.")
                # Sent rather than only logged: the whole point of this
                # module is that nobody reads the log.
                notify.send(msg, key="control_conflict", cooldown_sec=21600)
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

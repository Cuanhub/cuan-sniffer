"""One kill switch, two halt states, and limits measured from equity.

`flatten_all` is the ONLY way this system closes everything. The halt paths,
the Telegram commands and both loss limits all call it, and it reads the venue
rather than any belief about what is held -- which is REVIEW.md finding 5, where
`/closeall` could not see an orphan because it iterated a state file.

There is no `backend=None` path. A halt that cannot verify or act is worse than
one that refuses to run, which is REVIEW.md finding 4: every queued halt ran its
protection check against `None` and silently found nothing.
"""

from __future__ import annotations

import os
import pathlib
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import notify
from core import execute, streak
from core.meta import DATA, now_iso, read_json, write_json
from core.venue import Venue

MAX_FLATTEN_SEC = float(os.getenv("SAFETY_MAX_FLATTEN_SEC", "120"))
FLATTEN_PAUSE_SEC = float(os.getenv("SAFETY_FLATTEN_PAUSE_SEC", "3"))
DAILY_LOSS_PCT = float(os.getenv("SAFETY_DAILY_LOSS_PCT", "5.0"))
DRAWDOWN_PCT = float(os.getenv("SAFETY_DRAWDOWN_PCT", "10.0"))
# Rolling, not all-time. See EquityLog.
DRAWDOWN_WINDOW_HOURS = float(os.getenv("SAFETY_DRAWDOWN_WINDOW_HOURS", "24"))
ERROR_WINDOW_SEC = 600.0
ERROR_LIMIT = 5

# Rule 6 acts.
#
# Equity on this account is spot USDC `total` (the perp clearinghouse reports
# accountValue "0.0" -- it is a unifiedAccount). The open question was whether
# `total` already moves with unrealised perp PnL, because reading it wrong
# either double-counts open losses or ignores them entirely.
#
# It does. Across 265 equity readings in the live trader's own log, every
# consecutive pair with a position open and no fill between them has
# d(total)/d(hold) = 5.0000, the flat leverage this system trades at -- so
# `total` moves by the whole change in position notional. It marks to market,
# and `usdc_total` is the equity figure on its own. The evidence is kept as
# tests/fixtures/hyperliquid/equity_hold_series.txt and asserted by
# test_core_safety.py::test_equity_already_includes_unrealised_pnl.
LIMITS_ENFORCE = os.getenv("SAFETY_LIMITS_ENFORCE", "true").lower() == "true"

AUTO = "auto"        # clears itself on the next UTC day
MANUAL = "manual"    # needs --resume


@dataclass(frozen=True)
class FlattenResult:
    flat: bool                  # the venue confirmed nothing is open
    closed: Dict[str, float]    # coin -> size closed
    still_open: Dict[str, float]
    seconds: float
    detail: str


@dataclass(frozen=True)
class Verdict:
    """A limit's reading. `breached` says what it found, not what was done."""
    name: str
    breached: bool
    detail: str
    halt_kind: str = AUTO


# ── the kill switch ────────────────────────────────────────────────────────

def flatten_all(venue: Venue, reason: str = "flatten") -> FlattenResult:
    """Cancel every resting order and close every venue position.

    Loops until the venue confirms flat or MAX_FLATTEN_SEC elapses, because one
    marketable IOC is not a close: on 2026-10-17 a hand-flatten of four
    positions left three as dust and one untouched at $178.70 while every
    caller had been told the exit was sent.

    Reads `urgent`, so the 429 brake cannot throttle the thing it exists to
    escape, and prices from each position's own mark, so a dead price feed
    cannot block it either.
    """
    started = time.time()
    closed: Dict[str, float] = {}
    notes: List[str] = []
    rounds = 0
    stalled = 0
    last_seen: Optional[Dict[str, float]] = None

    while time.time() - started < MAX_FLATTEN_SEC:
        rounds += 1
        _cancel_all_orders(venue, notes)

        positions = venue.positions(urgent=True)
        if positions is None:
            notes.append(f"round {rounds}: venue unreadable")
            time.sleep(FLATTEN_PAUSE_SEC)
            continue
        if not positions:
            took = time.time() - started
            print(f"[SAFETY] flatten_all FLAT after {rounds} round(s), "
                  f"{took:.1f}s ({reason})")
            return FlattenResult(True, closed, {}, took,
                                 "; ".join(notes) or "flat")

        sizes = {c: p.szi for c, p in positions.items()}
        progressed = False
        for coin in sorted(positions):
            out = execute.close(venue, coin)
            if out.filled:
                closed[coin] = closed.get(coin, 0.0) + out.filled_size
                progressed = True
            notes.append(f"round {rounds} {coin}: "
                         f"{'closed ' + format(out.filled_size, 'g') if out.filled else out.note}")

        # Give up early on provable deadlock. A residual below the asset's
        # size precision cannot be expressed as an order, so retrying it until
        # the deadline burns the whole window and tells the operator nothing
        # it did not know after round two. The deadline stays the outer bound;
        # this only reports the same failure sooner.
        stalled = 0 if (progressed or sizes != last_seen) else stalled + 1
        last_seen = sizes
        if stalled >= 2:
            notes.append(f"no progress for {stalled + 1} rounds -- giving up")
            break
        time.sleep(FLATTEN_PAUSE_SEC)

    final = venue.positions(urgent=True)
    took = time.time() - started
    still = ({c: p.szi for c, p in final.items()} if final
             else {"?": 0.0})
    if final is None:
        notes.append("venue unreadable at the deadline -- cannot claim flat")
    flat = final is not None and not final
    if not flat:
        notes.append(f"STILL OPEN: {still}")
        notify.send(
            f"🚨 *Flatten did not finish*\n{reason}\n\n"
            f"Still open: `{still}`\nClosed: `{closed or '{}'}`\n"
            f"Gave up after {took:.0f}s. Close these at the venue.",
            key="flatten_failed", cooldown_sec=300)
    return FlattenResult(flat, closed, {} if flat else still, took,
                         "; ".join(notes))


def _cancel_all_orders(venue: Venue, notes: List[str]) -> None:
    """Cancel everything resting, so a stop cannot fire mid-flatten and a
    resting entry cannot re-open what we just closed."""
    orders = venue.open_orders(urgent=True)
    if orders is None:
        notes.append("open orders unreadable -- cancelled nothing")
        return
    for o in orders:
        resp = venue.cancel(o.coin, o.oid)
        if not (isinstance(resp, dict) and str(resp.get("status")) == "ok"):
            notes.append(f"cancel refused {o.coin} oid={o.oid}")


# ── halt state: exactly two kinds ──────────────────────────────────────────

class HaltState:
    """Persisted halt. AUTO clears on the next UTC day; MANUAL needs --resume."""

    def __init__(self, path: Optional[pathlib.Path] = None) -> None:
        self.path = path or (DATA / "_halt.json")

    def _read(self) -> Dict[str, Any]:
        raw = read_json(self.path, {})
        return raw if isinstance(raw, dict) else {}

    @staticmethod
    def _today() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def active(self) -> Optional[Dict[str, Any]]:
        """The halt in force, or None. An AUTO halt from a previous UTC day
        clears itself here -- the 2026-10-13 daily-loss halt was still blocking
        trading on 10-15 because nothing re-checked it.
        """
        row = self._read()
        if not row.get("kind"):
            return None
        if row.get("kind") == AUTO and row.get("day") != self._today():
            print(f"[SAFETY] AUTO halt from {row.get('day')} cleared "
                  f"(was: {row.get('reason')})")
            self.clear()
            notify.send(f"✅ *Resumed*\nThe automatic halt from "
                        f"{row.get('day')} has cleared with the new UTC day.\n"
                        f"Was: {row.get('reason')}", key="halt_cleared",
                        cooldown_sec=600)
            return None
        return row

    def set(self, kind: str, reason: str) -> None:
        if kind not in (AUTO, MANUAL):
            raise ValueError(f"halt kind must be {AUTO!r} or {MANUAL!r}")
        write_json(self.path, {"kind": kind, "reason": reason,
                               "day": self._today(), "at": now_iso()})

    def clear(self) -> None:
        write_json(self.path, {})


def halt(venue: Venue, state: HaltState, reason: str, kind: str = AUTO,
         flatten: bool = True) -> FlattenResult:
    """Stop trading. Flattens by default, because a halt that leaves exposure
    on venue stops nothing re-verifies is not a halt.

    `venue` is required. The old halt took an optional backend and every queued
    or Telegram halt passed None, so its protection check silently found
    nothing (REVIEW.md finding 4).
    """
    already = state.active()
    state.set(kind, reason)
    print(f"[SAFETY] !! HALT ({kind}) -- {reason}")
    result = (flatten_all(venue, reason=f"halt: {reason}") if flatten
              else FlattenResult(False, {}, {}, 0.0, "flatten not requested"))
    if not already:
        notify.send(
            f"🛑 *Halted* ({kind})\n{reason}\n\n"
            + (f"Flattened: `{result.closed or '{}'}`\n" if flatten else "")
            + ("It clears on the next UTC day." if kind == AUTO
               else "It needs `--resume`."),
            key="halt", cooldown_sec=1800)
    return result


# ── limits, measured from equity ───────────────────────────────────────────

class EquityLog:
    """Day-start equity, and a ROLLING-window high for the drawdown rail.

    The window keeps the REFERENCE fresh. An all-time peak never decays, so
    a drawdown measured against it grows for ever: the account falls 10%,
    stabilises, and every later reading still reads as a breach. The
    reference was deliberately moved off an all-time high (`7764dee`) for
    this reason, and on 2026-10-13 a stale reference kept the rail firing for
    two days against a high the account could no longer reach.

    Nothing older than DRAWDOWN_WINDOW_HOURS survives here, so the reference
    follows the account within a day. The drawdown halt itself stays MANUAL
    per the safety rules -- the window is what stops the rail re-firing the
    moment a human resumes it, not a way of releasing it unattended.

    Measured over 3,193 readings across 23 days, a 10% limit on this window
    fires exactly once -- the 2026-10-12/13 collapse at 10.85% -- and the
    next worst day reaches 4.97%.
    """

    def __init__(self, path: Optional[pathlib.Path] = None) -> None:
        self.path = path or (DATA / "_equity.json")

    def _read(self) -> Dict[str, Any]:
        row = read_json(self.path, {})
        return row if isinstance(row, dict) else {}

    def observe(self, usdc_total: float,
                now: Optional[datetime] = None) -> Tuple[float, float]:
        """Record this reading. Returns (day_start, rolling_high).

        Called on every pass INCLUDING ones that refuse to trade. The old
        version sat below the halted check, so a halted trader took no
        readings at all -- and from 2026-10-13 to 10-15 it therefore spent
        two days measured against a 10-12 reference it could no longer reach.
        A rail whose reference only updates when trading is allowed cannot
        release itself.
        """
        ref = now or datetime.now(timezone.utc)
        row = self._read()
        today = ref.strftime("%Y-%m-%d")
        if row.get("day") != today:
            row["day"] = today
            row["day_start"] = usdc_total
        raw = [s for s in (row.get("samples") or [])
               if isinstance(s, (list, tuple)) and len(s) == 2]
        raw.append([ref.isoformat().replace("+00:00", "Z"), usdc_total])
        cutoff = ref - timedelta(hours=DRAWDOWN_WINDOW_HOURS)
        kept: List[List[Any]] = []
        for ts, eq in raw:
            try:
                when = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                value = float(eq)
            except (TypeError, ValueError):
                continue
            if when.tzinfo is None:
                when = when.replace(tzinfo=timezone.utc)
            if when >= cutoff and value > 0:
                kept.append([str(ts), value])
        row["samples"] = kept
        write_json(self.path, row)
        # Current equity is folded into the reference so a fresh log, or one
        # whose window just pruned empty, cannot report a drawdown it has no
        # evidence for.
        high = max([float(v) for _, v in kept] + [usdc_total])
        return float(row.get("day_start", usdc_total)), float(high)


def check_limits(venue: Venue, log: EquityLog) -> List[Verdict]:
    """Read equity and report what the two limits see.

    Equity is `usdc_total` alone: it already marks to market, so it carries
    the open position's unrealised PnL. `unrealized_pnl` is printed beside it
    for the log, never added to it.
    """
    eq = venue.equity()
    if eq is None:
        record_error("equity_read", "venue returned None")
        return []
    day_start, high = log.observe(eq.usdc_total)

    print(f"[SAFETY] equity {eq.usdc_total:.4f} "
          + (f"(incl. unrealised {eq.unrealized_pnl:+.4f}) "
             if eq.unrealized_pnl != 0.0 else "")
          + f"day_start={day_start:.4f} "
            f"{DRAWDOWN_WINDOW_HOURS:g}h-high={high:.4f}")

    out: List[Verdict] = []
    if day_start > 0:
        lost_pct = (day_start - eq.usdc_total) / day_start * 100.0
        out.append(Verdict(
            "daily_loss", lost_pct >= DAILY_LOSS_PCT,
            f"down {lost_pct:.2f}% today ({eq.usdc_total:.2f} from "
            f"{day_start:.2f}), limit {DAILY_LOSS_PCT:g}%", AUTO))
    if high > 0:
        dd_pct = (high - eq.usdc_total) / high * 100.0
        out.append(Verdict(
            "drawdown", dd_pct >= DRAWDOWN_PCT,
            f"{dd_pct:.2f}% below the {high:.2f} high over the last "
            f"{DRAWDOWN_WINDOW_HOURS:g}h ({eq.usdc_total:.2f}), limit "
            f"{DRAWDOWN_PCT:g}%", MANUAL))
    for v in out:
        if v.breached:
            print(f"[SAFETY] {'!! LIMIT BREACHED' if LIMITS_ENFORCE else 'limit breached (LOG ONLY)'}"
                  f" {v.name}: {v.detail}")
    return out


def enforce_limits(venue: Venue, state: HaltState, log: EquityLog) -> List[Verdict]:
    """Act on a breach, if enforcement is on. Returns every verdict either way."""
    verdicts = check_limits(venue, log)
    if not LIMITS_ENFORCE:
        return verdicts
    for v in verdicts:
        if v.breached:
            halt(venue, state, f"{v.name}: {v.detail}", kind=v.halt_kind)
            break
    return verdicts


class StandDownOverride:
    """The operator's "I disagree, resume now" for the loss-streak stand-down.

    Stores one timestamp, not a flag: trades closed at or before it stop
    counting, so clearing a stand-down cannot be undone by the same trades
    while a fresh loss afterwards starts a new run. The old stand-down needed
    --clear-standdown because its counter got stuck; this one expires on its
    own, and the override exists only for the case where the operator wants it
    gone sooner.
    """

    def __init__(self, path: Optional[pathlib.Path] = None) -> None:
        self.path = path or (DATA / "_standdown.json")

    def ignore_before_ms(self) -> int:
        row = read_json(self.path, {})
        try:
            return int((row or {}).get("ignore_before_ms") or 0)
        except (TypeError, ValueError):
            return 0

    def clear_now(self) -> None:
        write_json(self.path, {"ignore_before_ms": int(time.time() * 1000),
                               "at": now_iso()})


def standing_down(venue: Venue,
                  override: Optional[StandDownOverride] = None
                  ) -> Optional[streak.StandDown]:
    """The loss-streak stand-down in force, or None. Blocks entries only."""
    ov = override or StandDownOverride()
    sd = streak.current(venue, ignore_before_ms=ov.ignore_before_ms())
    if sd is not None:
        print(f"[SAFETY] standing down {sd.hours_left:.1f}h more -- {sd.reason}")
        notify.send(
            f"⏸ *Standing down*\n{sd.reason}.\n\nNo new entries for "
            f"{sd.hours_left:.1f}h. Open positions keep their stops.",
            key="standdown", cooldown_sec=3600)
    return sd


# ── error rate ─────────────────────────────────────────────────────────────

_errors: List[Tuple[float, str]] = []
_alerted = False


def reset_errors() -> None:
    global _alerted
    _errors.clear()
    _alerted = False


def record_error(source: str, detail: object = "") -> int:
    """Note a failure. Alerts over ERROR_LIMIT in ERROR_WINDOW_SEC, once.

    Never raises into its caller: an error reporter that becomes the error is
    worse than none. It does not halt -- the venue having a bad minute is not a
    safety violation.
    """
    global _alerted
    try:
        cutoff = time.time() - ERROR_WINDOW_SEC
        while _errors and _errors[0][0] < cutoff:
            _errors.pop(0)
        _errors.append((time.time(), str(source)[:40]))
        n = len(_errors)
        print(f"[SAFETY] error {n}/{ERROR_LIMIT} in "
              f"{ERROR_WINDOW_SEC / 60:.0f}min [{str(source)[:40]}] "
              f"{str(detail)[:120]}")
        if n <= ERROR_LIMIT:
            _alerted = False
        elif not _alerted:
            _alerted = True
            counts: Dict[str, int] = {}
            for _, src in _errors:
                counts[src] = counts.get(src, 0) + 1
            notify.send(
                f"⚠️ *Error rate is high*\n{n} errors in the last "
                f"{ERROR_WINDOW_SEC / 60:.0f} minutes.\n\n"
                + "\n".join(f"• `{k}` x{v}" for k, v in
                            sorted(counts.items(), key=lambda kv: -kv[1])[:5])
                + "\n\nTrading is NOT halted -- this is a degradation warning.",
                key="error_rate", cooldown_sec=1800)
        return n
    except Exception as e:                                       # noqa: BLE001
        print(f"[SAFETY] record_error itself failed: {str(e)[:80]}")
        return 0

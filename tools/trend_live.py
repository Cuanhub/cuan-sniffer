"""The trend book, live: one pass a UTC day trades ten coins to the targets of
the pre-registered daily trend rule, through core.execute.

Pre-registered in research/trend/PREREG_live.md: the operator chose a live run
at small size over the paper run's 90 days first. The paper pass keeps running
beside it as the model this is measured against. The rule is
research.trend.paper's -- decide() and rebalance() are imported, not copied --
so the backtest, the paper run and this share one definition of the strategy.

Runs hourly (deploy/launchd/com.cuansniffer.trendlive.plist) and acts once per
UTC day, on the first run after 00:00 UTC that can read everything:

  1. reads the venue: positions, open orders, the daily closes and the mids;
  2. computes targets with paper.decide and applies the band rule with
     paper.rebalance to weights read from the VENUE (signed notional over the
     book's capital), never to a stored belief about what is held;
  3. closes and reduces first, then opens and adds, one IOC each through
     core.execute, which reports what the venue did rather than what was sent;
  4. rests a disaster stop STOP_PCT beyond each position's mark. It is not part
     of the tested strategy, which has no stop; it is there because this host
     has been down for days at a time;
  5. books the day from the venue's own fills and funding, appends a ledger
     row, and applies the pre-registered kill rule.

The other hourly runs only check that each position still has its stop. A pass
changes nothing it cannot read, and a coin that still carries a band plan or a
band bid is left to the band book (core/books.py).

    python -m tools.trend_live              # one pass; needs TREND_LIVE_ARMED=true
    python -m tools.trend_live --dry-run    # what today's pass would send; sends nothing
    python -m tools.trend_live --status     # the run so far, against PREREG_live.md
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)

import numpy as np                                                    # noqa: E402

import notify                                                         # noqa: E402
from core import books, config_audit, execute, safety                 # noqa: E402
from core.meta import MetaStore                                       # noqa: E402
from core.protect import _cancel_stale, _matches, _place_stop        # noqa: E402
from core.venue import Order, Position                                # noqa: E402
from research.backtest.experiments import trend_daily as T            # noqa: E402
from research.trend import paper as P                                 # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "research" / "trend" / "data" / "live"
STATE_NAME = "state.json"
COINS: Tuple[str, ...] = P.COINS
DAY_MS = P.DAY_MS
CAPITAL_USD = float(os.getenv("TREND_CAPITAL_USD", "0") or 0)

# Pre-registered in research/trend/PREREG_live.md. Not knobs: changing one is an
# amendment and starts a new run.
STOP_PCT = 25.0                # disaster stop, beyond the mark at each pass
MIN_TRADE_USD = 11.0           # the venue refuses orders under $10 of notional
LEVERAGE = 5                   # cross; it sets margin, not exposure
K1_MAX_DRAWDOWN_PCT = 30.0     # book value below its peak: close and stop
R2_MAX_COST_BPS = 20.0
R3_MAX_FUNDING_PCT_YEAR = 9.0
R4_MIN_TRACKING_BPS_DAY = -5.0
R1_MIN_ROWS = 80
# Fills and funding are re-read across each pass boundary and de-duplicated,
# so one that the venue shows late is still counted once.
OVERLAP_MS = 3_600_000
KEEP_SEEN_MS = 2 * DAY_MS
FUNDING_PAGE = 500


class Unreadable(Exception):
    """Something the pass needs could not be read. Nothing was sent."""


class Blocked(Exception):
    """The pass must not trade now: halted, stopped, or misconfigured."""


# ---------------------------------------------------------------- state --
def fresh_state(capital: float) -> Dict[str, Any]:
    return {"capital": capital, "first_day": None, "last_day": None,
            "realized": 0.0, "fees": 0.0, "funding": 0.0, "peak": capital,
            "accounted_ms": None, "seen": [], "stop_px": {}, "held": {},
            "stopped": None}


def load_state(data_dir: Path, capital: float) -> Dict[str, Any]:
    p = data_dir / STATE_NAME
    if not p.exists():
        return fresh_state(capital)
    state = json.loads(p.read_text())
    if abs(float(state.get("capital") or 0.0) - capital) > 1e-9:
        raise Blocked(
            f"TREND_CAPITAL_USD is {capital:g} but this run started at "
            f"{state.get('capital')}. The capital is fixed for a run "
            f"(PREREG_live.md): amend the pre-registration and move {data_dir} "
            f"aside to start a new one.")
    return state


def save_state(state: Dict[str, Any], data_dir: Path) -> None:
    P._write_json(data_dir / STATE_NAME, state)


def save(state: Dict[str, Any], row: Dict[str, Any], data_dir: Path) -> None:
    """Ledger first, then state. A crash between the two repeats the day, and
    a repeated day trades only what the venue still shows as different."""
    data_dir.mkdir(parents=True, exist_ok=True)
    with open(data_dir / P.LEDGER_NAME, "a") as fh:
        fh.write(json.dumps(row, separators=(",", ":")) + "\n")
    save_state(state, data_dir)


# ------------------------------------------------------------ ownership --
def band_held(store: MetaStore, orders: Sequence[Order]) -> Dict[str, str]:
    """Trend coins the band book still holds, and why. Left alone this pass."""
    out: Dict[str, str] = {}
    for c in COINS:
        if store.get(c) is not None:
            out[c] = "carries a band plan"
    for o in orders:
        if o.coin in COINS and o.is_entry:
            out.setdefault(o.coin, "has a band bid resting")
    return out


# ----------------------------------------------------------------- plan --
@dataclass
class Leg:
    coin: str
    kind: str                       # close | reduce | open | add
    from_w: float
    to_w: float
    usd: float = 0.0                # open / add: notional to trade
    size: Optional[float] = None    # reduce: coin units; close: None, all of it

    @property
    def is_buy(self) -> bool:
        if self.kind in ("open", "add"):
            return self.to_w > 0
        return self.from_w < 0      # closing or reducing a short buys


def plan_legs(held: np.ndarray, new: np.ndarray, mids: Dict[str, float],
              capital: float, blocked: Dict[str, str]
              ) -> Tuple[List[Leg], List[Dict[str, Any]]]:
    """The orders that take `held` to `new`: every close and reduction first,
    so margin is freed before anything opens. A trade under the venue's minimum
    is skipped and recorded; a close is never skipped, because a sign change
    must not leave the old side on."""
    first: List[Leg] = []
    then: List[Leg] = []
    skipped: List[Dict[str, Any]] = []

    def skip(c: str, h: float, n: float, why: str) -> None:
        skipped.append({"coin": c, "from": h, "to": n, "why": why})

    for j, c in enumerate(COINS):
        h, n = float(held[j]), float(new[j])
        if n == h:
            continue                                  # the band rule holds it
        if c in blocked:
            skip(c, h, n, blocked[c])
            continue
        if h and (n == 0.0 or np.sign(n) != np.sign(h)):
            first.append(Leg(c, "close", h, n))
            if n:
                usd = abs(n) * capital
                if usd >= MIN_TRADE_USD:
                    then.append(Leg(c, "open", h, n, usd=usd))
                else:
                    skip(c, h, n, f"opening ${usd:.2f} is under ${MIN_TRADE_USD:g}")
            continue
        delta = (abs(n) - abs(h)) * capital
        if abs(delta) < MIN_TRADE_USD:
            skip(c, h, n, f"${abs(delta):.2f} is under ${MIN_TRADE_USD:g}")
        elif delta > 0:
            then.append(Leg(c, "add" if h else "open", h, n, usd=delta))
        else:
            first.append(Leg(c, "reduce", h, n, size=-delta / mids[c]))
    return first + then, skipped


def run_legs(venue: Any, legs: Sequence[Leg]) -> List[Dict[str, Any]]:
    out = []
    levered: set[str] = set()
    for leg in legs:
        if leg.kind in ("open", "add"):
            if leg.coin not in levered:
                venue.set_leverage(leg.coin, LEVERAGE)
                levered.add(leg.coin)
            o = execute.enter(venue, leg.coin, leg.usd, is_long=leg.to_w > 0)
        else:
            o = execute.close(venue, leg.coin, leg.size)
        out.append({**asdict(leg), "is_buy": leg.is_buy, "filled": o.filled_size,
                    "note": o.note, "unknown": o.unknown})
    return out


# ---------------------------------------------------------------- stops --
def disaster_px(venue: Any, pos: Position) -> float:
    """STOP_PCT beyond the mark, on the losing side after rounding, or 0."""
    mark = pos.mark_px
    if mark <= 0:
        return 0.0
    if pos.is_long:
        px = venue.round_price(pos.coin, mark * (1 - STOP_PCT / 100), direction="up")
        return px if 0 < px < mark else 0.0
    px = venue.round_price(pos.coin, mark * (1 + STOP_PCT / 100), direction="down")
    return px if px > mark else 0.0


def keep_stops(venue: Any, positions: Dict[str, Position], orders: Sequence[Order],
               state: Dict[str, Any], blocked: Dict[str, str], reprice: bool
               ) -> Dict[str, str]:
    """One disaster stop per position, none on a flat coin. Place-then-cancel,
    confirmed by reading the book, with protect's own helpers. `reprice` moves
    each stop to the current mark (the daily pass); otherwise a stop is only
    replaced when it is missing or the wrong size."""
    out: Dict[str, str] = {}
    stop_px: Dict[str, float] = state.setdefault("stop_px", {})
    for c in COINS:
        if c in blocked:
            continue
        pos = positions.get(c)
        mine = [o for o in orders if o.coin == c and o.is_protective]
        if pos is None:
            stop_px.pop(c, None)
            if mine:
                _cancel_stale(venue, c, mine)
                out[c] = "cancelled"
            continue
        want = 0.0 if reprice else float(stop_px.get(c) or 0.0)
        if not (want > 0 and ((want < pos.mark_px) if pos.is_long else (want > pos.mark_px))):
            want = disaster_px(venue, pos)
        if want <= 0:
            out[c] = "no stop price"
            continue
        good = [o for o in mine if o.is_stop and _matches(o, pos.size, want)]
        if good:
            extra = [o for o in mine if o.oid != good[0].oid]
            if extra:
                _cancel_stale(venue, c, extra)
            stop_px[c] = want
            out[c] = "ok"
            continue
        placed, _ = _place_stop(venue, pos, want, {o.oid for o in orders if o.coin == c})
        if placed is None:
            out[c] = "unconfirmed"
            continue
        _cancel_stale(venue, c, [o for o in mine if o.oid != placed.oid])
        stop_px[c] = want
        out[c] = "placed"
    return out


# ----------------------------------------------------------- accounting --
def fetch_funding(ro: Any, user: str, since_ms: int, until_ms: int
                  ) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    start = since_ms
    while True:
        page = ro._post({"type": "userFunding", "user": user, "startTime": start,
                         "endTime": until_ms}, "userFunding")
        if not isinstance(page, list):
            raise Unreadable("funding history is not a list")
        rows += [r for r in page if isinstance(r, dict)]
        if len(page) < FUNDING_PAGE:
            return rows
        start = max(int(r.get("time") or 0) for r in page) + 1


def account(state: Dict[str, Any], fills: Sequence[Dict[str, Any]],
            funding: Sequence[Dict[str, Any]], positions: Dict[str, Position],
            blocked: Dict[str, str], now_ms: int) -> Dict[str, float]:
    """Book value from the venue's own numbers: capital, plus closed P&L less
    fees on the trend coins, plus their funding, plus their open P&L. Each fill
    and funding payment is counted once however often it is read."""
    seen = {str(k) for k, _ in state.get("seen") or []}
    kept = [[k, t] for k, t in state.get("seen") or [] if int(t) >= now_ms - KEEP_SEEN_MS]
    realized = fees = fund = 0.0
    for f in fills:
        c = str(f.get("coin", "")).upper()
        if c not in COINS or c in blocked:
            continue
        key = f"f:{f.get('tid') or (str(f.get('hash')) + ':' + str(f.get('oid')))}"
        if key in seen:
            continue
        seen.add(key)
        kept.append([key, int(f.get("time") or 0)])
        realized += float(f.get("closedPnl") or 0.0)
        fees += float(f.get("fee") or 0.0)
    for r in funding:
        d = r.get("delta") or {}
        c = str(d.get("coin", "")).upper()
        if c not in COINS or c in blocked:
            continue
        key = f"u:{r.get('time')}:{c}"
        if key in seen:
            continue
        seen.add(key)
        kept.append([key, int(r.get("time") or 0)])
        fund += float(d.get("usdc") or 0.0)            # negative when paid
    unreal = sum(p.unrealized_pnl for c, p in positions.items()
                 if c in COINS and c not in blocked)
    state["realized"] = float(state["realized"]) + realized
    state["fees"] = float(state["fees"]) + fees
    state["funding"] = float(state["funding"]) + fund
    state["seen"] = kept
    state["accounted_ms"] = now_ms
    value = (float(state["capital"]) + state["realized"] - state["fees"]
             + state["funding"] + unreal)
    state["peak"] = max(float(state["peak"]), value)
    return {"realized": realized, "fees": fees, "funding": fund, "unrealized": unreal,
            "value": value, "peak": state["peak"],
            "drawdown_pct": (value / state["peak"] - 1.0) * 100.0}


def executions(fills: Sequence[Dict[str, Any]], since_ms: int,
               closes: Dict[str, float], mids: Dict[str, float]
               ) -> Dict[str, Dict[str, float]]:
    """Per coin, this pass's real fills against the decision close and the
    decision mid. Positive bps are adverse: paid up on a buy, sold down."""
    acc: Dict[str, Dict[str, float]] = {}
    for f in fills:
        c = str(f.get("coin", "")).upper()
        if c not in COINS or int(f.get("time") or 0) < since_ms:
            continue
        px, sz = float(f.get("px") or 0.0), float(f.get("sz") or 0.0)
        if px <= 0 or sz <= 0:
            continue
        side = 1.0 if f.get("side") == "B" else -1.0
        n = px * sz
        a = acc.setdefault(c, {"notional": 0.0, "adv": 0.0, "slip": 0.0, "fees": 0.0})
        a["notional"] += n
        a["adv"] += side * (px / closes[c] - 1.0) * n
        a["slip"] += side * (px / mids[c] - 1.0) * n
        a["fees"] += float(f.get("fee") or 0.0)
    return {c: {"notional": a["notional"], "fees": a["fees"],
                "adverse_bps": a["adv"] / a["notional"] * 1e4,
                "slip_bps": a["slip"] / a["notional"] * 1e4,
                "fee_bps": a["fees"] / a["notional"] * 1e4}
            for c, a in acc.items()}


def stop_book(venue: Any, state: Dict[str, Any], positions: Dict[str, Position],
              blocked: Dict[str, str], reason: str) -> Dict[str, Any]:
    """Close every trend position and stop the book until the operator starts
    a new run. Each coin is closed on its own; the band book is not touched."""
    closed = {}
    for c in COINS:
        if c in positions and c not in blocked:
            closed[c] = safety.flatten_coin(venue, c, reason=f"trend book: {reason}").flat
    state["stopped"] = reason
    return {"reason": reason, "flat": closed}


# ----------------------------------------------------------------- pass --
def run_pass(venue: Any, ro: Any, now_ms: int, capital: float,
             data_dir: Path = DATA_DIR, store: Optional[MetaStore] = None,
             halt: Optional[safety.HaltState] = None, dry_run: bool = False
             ) -> Optional[Dict[str, Any]]:
    """Today's pass, or None when it is already done (the hourly stop check
    still runs then). Raises Unreadable or Blocked having sent nothing."""
    started = time.time()
    store = store if store is not None else MetaStore()
    halt = halt if halt is not None else safety.HaltState()
    state = load_state(data_dir, capital)
    if state.get("stopped"):
        raise Blocked(f"the book is stopped: {state['stopped']}")
    today = now_ms - now_ms % DAY_MS
    done = state["last_day"] is not None and T.day_ms(state["last_day"]) >= today
    if done and dry_run:
        return None
    active = halt.active()
    if active:
        raise Blocked(f"halted ({active.get('kind')}): {active.get('reason', '')}")
    positions = venue.positions()
    orders = venue.open_orders()
    if positions is None or orders is None:
        raise Unreadable("positions or open orders unreadable")
    blocked = band_held(store, orders)
    if done:
        keep_stops(venue, positions, orders, state, blocked, reprice=False)
        save_state(state, data_dir)
        return None

    try:
        bars = P.fetch_bars(ro, now_ms)
    except Exception as exc:                                   # noqa: BLE001
        raise Unreadable(f"daily candles: {exc}") from exc
    if not dry_run:
        P.note_missing(P.missing_bars(bars, today), today, data_dir)
    try:
        _, close = P.closes_matrix(bars, today)
    except P.Unreadable as exc:
        raise Unreadable(str(exc)) from exc
    try:
        raw = ro.get_all_mids()
    except Exception as exc:                                   # noqa: BLE001
        raise Unreadable(f"mids: {exc}") from exc
    mids = {c: float((raw or {}).get(c) or 0.0) for c in COINS}
    if not all(m > 0 for m in mids.values()):
        raise Unreadable("a mid is missing")
    if state["first_day"] is None and not dry_run:
        eq = venue.equity()
        if eq is None:
            raise Unreadable("equity unreadable before the first pass")
        if capital > eq.usdc_total:
            raise Blocked(f"capital ${capital:g} exceeds account equity "
                          f"${eq.usdc_total:.2f} (PREREG_live.md)")

    target = P.decide(close)
    held = np.array([positions[c].szi * mids[c] / capital if c in positions else 0.0
                     for c in COINS])
    new = P.rebalance(held, target)
    legs, skipped = plan_legs(held, new, mids, capital, blocked)
    closes = {c: float(close[-1, j]) for j, c in enumerate(COINS)}
    row: Dict[str, Any] = {
        "day": T.iso(today), "ts_ms": now_ms, "decision_close": T.iso(today - DAY_MS),
        "capital": capital, "closes": closes, "mids": mids,
        "target": {c: float(target[j]) for j, c in enumerate(COINS)},
        "held_before": {c: float(held[j]) for j, c in enumerate(COINS)},
        "blocked": blocked, "skipped": skipped}
    if dry_run:
        return {**row, "legs": [asdict(leg) for leg in legs], "dry_run": True}

    stopped_out = sorted(c for c, w in (state.get("held") or {}).items()
                         if w and c not in positions and c not in blocked)
    row["legs"] = run_legs(venue, legs)
    after = venue.positions()
    book = venue.open_orders()
    after_ms = now_ms + int((time.time() - started) * 1000)
    acct: Optional[Dict[str, float]] = None
    killed = None
    if after is None or book is None:
        row["stops"] = {"*": "venue unreadable after trading"}
    else:
        row["stops"] = keep_stops(venue, after, book, state, blocked, reprice=True)
        # From the first pass itself: an hour before it, a coin may still have
        # been the band book's.
        since = (int(state["accounted_ms"]) - OVERLAP_MS if state.get("accounted_ms")
                 else now_ms)
        fills = venue.fills(since)
        try:
            funding = fetch_funding(ro, venue.address, since, after_ms)
        except Exception:                                      # noqa: BLE001
            funding = None
        if fills is not None and funding is not None:
            row["executions"] = executions(fills, now_ms, closes, mids)
            acct = account(state, fills, funding, after, blocked, after_ms)
            if acct["drawdown_pct"] <= -K1_MAX_DRAWDOWN_PCT:
                killed = stop_book(venue, state, after, blocked,
                                   f"K1: book value {acct['drawdown_pct']:.1f}% "
                                   f"below its peak")
    held_after = {c: p.szi * mids[c] / capital for c, p in (after or {}).items()
                  if c in COINS and c not in blocked}
    row.update({"held_after": held_after, "stopped_out": stopped_out,
                "gross": float(sum(abs(w) for w in held_after.values())),
                "net": float(sum(held_after.values())),
                "accounting": acct, "killed": killed})
    state["last_day"] = row["day"]
    state["first_day"] = state["first_day"] or row["day"]
    state["held"] = held_after
    save(state, row, data_dir)
    return row


# --------------------------------------------------------------- status --
def status(rows: List[Dict[str, Any]], today_ms: int,
           model_net: Optional[np.ndarray] = None,
           missing_since: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """The run so far against PREREG_live.md. `model_net` is paper.replay()'s
    net per ledger row on the venue's daily closes."""
    if not rows:
        return {"started": False}
    first = T.day_ms(rows[0]["day"])
    elapsed = int((today_ms - first) // DAY_MS) + 1
    cap = float(rows[-1]["capital"])
    accts = [r["accounting"] for r in rows if r.get("accounting")]
    ex = [e for r in rows for e in (r.get("executions") or {}).values()]
    notional = sum(e["notional"] for e in ex)
    adverse = sum(e["adverse_bps"] * e["notional"] for e in ex) / notional if notional else None
    fee_bps = sum(e["fees"] for e in ex) / notional * 1e4 if notional else None
    paid = -sum(a["funding"] for a in accts) / cap / max(elapsed - 1, 1) * 365 * 100
    worst = min((a["drawdown_pct"] for a in accts), default=0.0)
    tracking = None
    if model_net is not None and len(accts) > 1 and len(accts) == len(rows):
        live = (accts[-1]["value"] - accts[0]["value"]) / cap
        tracking = (live - float(model_net[:-1].sum())) / (len(rows) - 1) * 1e4
    cost = (adverse or 0.0) + (fee_bps or 0.0) if notional else None
    return {
        "started": True, "first_day": rows[0]["day"], "last_day": rows[-1]["day"],
        "days_elapsed": elapsed, "rows": len(rows), "capital": cap,
        "value": accts[-1]["value"] if accts else None,
        "peak": accts[-1]["peak"] if accts else None,
        "worst_drawdown_pct": worst,
        "orders_filled": sum(1 for r in rows for leg in r.get("legs") or [] if leg["filled"]),
        "skipped": sum(len(r.get("skipped") or []) for r in rows),
        "stop_outs": sum(len(r.get("stopped_out") or []) for r in rows),
        "traded_usd": notional, "mean_adverse_bps": adverse, "mean_fee_bps": fee_bps,
        "funding_paid_pct_year": paid, "tracking_bps_day": tracking,
        "stopped": next((r["killed"]["reason"] for r in rows if r.get("killed")), None),
        "kill": {"K1_drawdown": worst <= -K1_MAX_DRAWDOWN_PCT,
                 "S3_candles": bool(P.s3_fired(missing_since or {}, today_ms))},
        "review_on_day_90": {
            "R1_rows": len(rows) >= R1_MIN_ROWS,
            "R2_cost": cost is not None and cost <= R2_MAX_COST_BPS,
            "R3_funding": paid <= R3_MAX_FUNDING_PCT_YEAR,
            "R4_tracking": tracking is not None and tracking >= R4_MIN_TRACKING_BPS_DAY},
        "held": rows[-1].get("held_after"),
    }


# ---------------------------------------------------------------- talk --
def line(row: Dict[str, Any]) -> str:
    a = row.get("accounting") or {}
    head = f"TREND LIVE {row['day']}: "
    if a:
        head += (f"book ${a['value']:.2f} ({a['drawdown_pct']:+.1f}% from its "
                 f"${a['peak']:.2f} peak), ")
    head += (f"gross {row['gross'] * 100:.0f}% net {row['net'] * 100:+.0f}% "
             f"of ${row['capital']:g}")
    legs = row.get("legs") or []
    done = [f"{leg['coin']} {leg['kind']}" for leg in legs if leg["filled"]]
    failed = [f"{leg['coin']} {leg['kind']} ({leg['note'] or 'no fill'})"
              for leg in legs if not leg["filled"]]
    parts = [head, "; ".join(done) or "no trades"]
    if failed:
        parts.append("not filled: " + ", ".join(failed))
    if row.get("skipped"):
        parts.append(f"{len(row['skipped'])} skipped")
    bad = sorted(c for c, s in (row.get("stops") or {}).items()
                 if s not in ("ok", "placed", "cancelled"))
    if bad:
        parts.append("⚠️ stop not confirmed: " + ", ".join(bad))
    if row.get("stopped_out"):
        parts.append("closed since the last pass: " + ", ".join(row["stopped_out"]))
    if row.get("blocked"):
        parts.append("left to the band book: " + ", ".join(sorted(row["blocked"])))
    return "\n".join(parts)


def arm_refusal(capital: float, dry_run: bool) -> str:
    """Why this process must not trade, or ''."""
    if not dry_run and not books.trend_armed():
        return "TREND_LIVE_ARMED is not true -- not trading (PREREG_live.md)"
    if capital <= 0:
        return "TREND_CAPITAL_USD is not set -- the book has nothing to size against"
    return config_audit.refusal(config_audit.dead_keys(ROOT / ".env"))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--capital", type=float, default=None,
                    help="dry run only: preview at this capital")
    a = ap.parse_args(argv)
    from research.levels.venue_client import HyperliquidReadOnlyClient
    ro = HyperliquidReadOnlyClient()
    now_ms = int(time.time() * 1000)
    today = now_ms - now_ms % DAY_MS
    if a.status:
        rows = P.read_ledger(DATA_DIR)
        model = None
        if rows:
            try:
                model = P.model_for(rows, ro, now_ms)
            except Exception as exc:                           # noqa: BLE001
                print(f"model unreadable ({exc}); R4 not computed")
        print(json.dumps(status(rows, today, model, P.read_missing(DATA_DIR)), indent=2))
        return 0
    capital = a.capital if (a.dry_run and a.capital) else CAPITAL_USD
    why = arm_refusal(capital, a.dry_run)
    if why:
        print(f"TREND LIVE: {why}")
        return 0
    from core.venue import Venue
    try:
        venue = Venue()
    except Exception as exc:                                   # noqa: BLE001
        print(f"TREND LIVE: venue unavailable, nothing sent -- {type(exc).__name__}")
        return 1
    try:
        row = run_pass(venue, ro, now_ms, capital, dry_run=a.dry_run)
    except Blocked as exc:
        print(f"TREND LIVE: not trading -- {exc}")
        return 0
    except Unreadable as exc:
        print(f"TREND LIVE: unreadable, nothing sent -- {exc}")
        msg = None if a.dry_run else P.s3_due(today, DATA_DIR)
        if msg and notify.send(msg.replace("TREND PAPER", "TREND LIVE")):
            P.mark_s3_alerted(today, DATA_DIR)
        return 1
    if row is None:
        print("TREND LIVE: today's pass is done; stops checked")
        return 0
    if a.dry_run:
        print(json.dumps({k: row[k] for k in ("day", "capital", "target", "held_before",
                                               "legs", "skipped", "blocked")}, indent=2))
        return 0
    text = line(row)
    print(text)
    notify.send(text)
    if row.get("killed"):
        notify.send(f"🛑 *Trend book stopped*\n{row['killed']['reason']}\n"
                    f"Closed: `{row['killed']['flat']}`. It stays stopped until a new "
                    f"run is pre-registered (research/trend/PREREG_live.md).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

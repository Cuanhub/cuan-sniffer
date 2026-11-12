"""The trend book on paper, on Hyperliquid's own perps. Places no orders.

Pre-registered in research/trend/PREREG_forward.md; the book itself is
H-TREND's (research/backtest/PREREG_trend_daily.md) and its trend and
volatility come from the same `features` function the backtest ran, so there
is one definition of the strategy, not two.

Run hourly. It acts once per UTC day, on the first run that can read
everything it needs; otherwise it changes nothing and exits 1.

    python -m research.trend.paper              # one pass (idempotent per UTC day)
    python -m research.trend.paper --dry-run    # compute and print, write nothing
    python -m research.trend.paper --status     # the run so far, against its criteria
    python -m research.trend.paper --notify     # also send the day's line to Telegram
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from research.backtest import data as D
from research.backtest.experiments import trend_daily as T

DATA_DIR = Path(__file__).resolve().parent / "data"
STATE_NAME = "state.json"
LEDGER_NAME = "ledger.jsonl"
COINS: Tuple[str, ...] = T.COINS
DAY_MS = T.DAY_MS
HISTORY_DAYS = 200                      # fetched; the rule needs 120
RUN_DAYS = 90
# The pre-registered criteria (PREREG_forward.md). Not knobs.
G1_MIN_ROWS = 80
G2_MAX_DRIFT_BPS = 10.68
G3_MAX_FUNDING_PCT_YEAR = 9.0
G4_MIN_TRACKING_BPS_DAY = -5.0
S1_MAX_DRAWDOWN_PCT = 30.0
S2_MAX_GAP_DAYS = 10


class Unreadable(Exception):
    """Something the pass needs could not be read. Nothing is changed."""


# ------------------------------------------------------------- the rule --
def closes_matrix(bars_by: Dict[str, List[Dict[str, Any]]], today_ms: int,
                  coins: Sequence[str] = COINS) -> Tuple[np.ndarray, np.ndarray]:
    """(day timestamps, closes [day x coin]) from closed daily bars. Every
    coin must end on yesterday's bar and be unbroken for the rule's whole
    reach; anything less is unreadable, never a shorter history."""
    need = T.MIN_HISTORY + 1
    last = today_ms - DAY_MS
    times = np.arange(last - (need - 1) * DAY_MS, last + DAY_MS, DAY_MS, dtype=np.int64)
    close = np.full((need, len(coins)), np.nan)
    for j, c in enumerate(coins):
        by_t = {int(b["t"]): float(b["c"]) for b in bars_by.get(c) or []}
        if last not in by_t:
            raise Unreadable(f"{c}: yesterday's daily bar is not closed on the venue yet")
        for i, t in enumerate(times):
            px = by_t.get(int(t))
            if px is None or not px > 0:
                raise Unreadable(f"{c}: no daily close for {T.iso(int(t))}")
            close[i, j] = px
    return times, close


def decide(close: np.ndarray) -> np.ndarray:
    """Target weights from closes up to and including the last row."""
    s, v, elig = T.features(close)
    live = elig[-1]
    n = int(live.sum())
    if not n:
        return np.zeros(close.shape[1])
    cap = T.CAP_GROSS_PER_COIN / n
    vol = np.where(live, v[-1], 1.0)
    sig = np.where(live, np.nan_to_num(s[-1]), 0.0)
    return np.clip(sig * (T.VOL_TARGET / vol) / n, -cap, cap)


def rebalance(held: np.ndarray, target: np.ndarray) -> np.ndarray:
    """The band: a weight moves to its target only on a change of sign, a
    zero target, or a gap of more than BAND of the target."""
    move = ((np.sign(target) != np.sign(held)) | (target == 0.0)
            | (np.abs(target - held) > T.BAND * np.abs(target)))
    return np.where(move, target, held)


def replay(close: np.ndarray, decisions: Sequence[int],
           cost_bps: float = T.COST_BPS, fund_long_bps: float = T.FUND_LONG_BPS
           ) -> np.ndarray:
    """The MODEL: the backtest's rules on a close matrix, deciding at the
    rows in `decisions` and holding to the next one (the last decision earns
    nothing). Net return per decision. With a decision every day this is
    T.simulate, and a test holds it to that."""
    held = np.zeros(close.shape[1])
    out = np.zeros(len(decisions))
    for k, i in enumerate(decisions):
        new = rebalance(held, decide(close[: i + 1]))
        cost = float(np.abs(new - held).sum()) * cost_bps / 1e4
        held = new
        gross = fund = 0.0
        if k + 1 < len(decisions):
            j = decisions[k + 1]
            gross = float(held @ (close[j] / close[i] - 1.0))
            fund = float(np.clip(held, 0, None).sum()) * fund_long_bps / 1e4 * (j - i)
        out[k] = gross - cost - fund
    return out


# ------------------------------------------------------------- one pass --
def fresh_state() -> Dict[str, Any]:
    return {"equity": 1.0, "peak": 1.0, "held": {}, "marks": {}, "last_day": None,
            "first_day": None, "funding_through_ms": None}


def step(state: Dict[str, Any], now_ms: int, close: np.ndarray, mids: Dict[str, float],
         funding: Dict[str, float], coins: Sequence[str] = COINS,
         cost_bps: float = T.COST_BPS) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """One day's pass, pure: (new state, ledger row). `funding[coin]` is the
    sum of the venue's hourly rates since the last pass; `mids` must hold
    every coin. The caller has already checked this UTC day is new."""
    today = now_ms - now_ms % DAY_MS
    for c in coins:
        if not (mids.get(c) or 0) > 0:
            raise Unreadable(f"{c}: no mid")
    held = np.array([float(state["held"].get(c, 0.0)) for c in coins])
    marks = state["marks"]
    pnl = fund = 0.0
    for j, c in enumerate(coins):
        if held[j] and marks.get(c):
            pnl += held[j] * (mids[c] / float(marks[c]) - 1.0)
            if c not in funding:
                raise Unreadable(f"{c}: no funding history for a held position")
            fund += held[j] * float(funding[c])        # long pays a positive rate
    target = decide(close)
    new = rebalance(held, target)
    trades = []
    for j, c in enumerate(coins):
        dw = float(new[j] - held[j])
        if dw:
            side = 1.0 if dw > 0 else -1.0
            trades.append({"coin": c, "from": float(held[j]), "to": float(new[j]),
                           "close": float(close[-1, j]), "mid": float(mids[c]),
                           "adverse_bps": side * (mids[c] / float(close[-1, j]) - 1.0) * 1e4})
    cost = float(np.abs(new - held).sum()) * cost_bps / 1e4
    before = float(state["equity"])
    after = before * (1.0 + pnl - fund - cost)
    last = state["last_day"]
    missed = 0 if last is None else int((today - T.day_ms(last)) // DAY_MS) - 1
    row = {"day": T.iso(today), "ts_ms": int(now_ms), "decision_close": T.iso(today - DAY_MS),
           "equity_before": before, "pnl_price": pnl, "funding": fund, "cost": cost,
           "equity_after": after, "gross": float(np.abs(new).sum()),
           "net": float(new.sum()), "missed_days": missed,
           "target": {c: float(target[j]) for j, c in enumerate(coins)},
           "held": {c: float(new[j]) for j, c in enumerate(coins)}, "trades": trades}
    nxt = {"equity": after, "peak": max(float(state["peak"]), after),
           "held": row["held"], "marks": {c: float(mids[c]) for c in coins},
           "last_day": row["day"], "first_day": state["first_day"] or row["day"],
           "funding_through_ms": int(now_ms)}
    return nxt, row


# ------------------------------------------------------------------ I/O --
def load_state(data_dir: Path = DATA_DIR) -> Dict[str, Any]:
    p = data_dir / STATE_NAME
    return json.loads(p.read_text()) if p.exists() else fresh_state()


def save(state: Dict[str, Any], row: Dict[str, Any], data_dir: Path = DATA_DIR) -> None:
    """Ledger first, then state by atomic replace. A crash between the two
    repeats the day on the next run; readers keep a day's last row."""
    data_dir.mkdir(parents=True, exist_ok=True)
    with open(data_dir / LEDGER_NAME, "a") as fh:
        fh.write(json.dumps(row, separators=(",", ":")) + "\n")
    tmp = data_dir / (STATE_NAME + ".tmp")
    tmp.write_text(json.dumps(state, indent=1))
    os.replace(tmp, data_dir / STATE_NAME)


def read_ledger(data_dir: Path = DATA_DIR) -> List[Dict[str, Any]]:
    p = data_dir / LEDGER_NAME
    if not p.exists():
        return []
    by_day: Dict[str, Dict[str, Any]] = {}
    for line in p.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            by_day[r["day"]] = r
    return [by_day[d] for d in sorted(by_day)]


def funding_sum(rows: Iterable[Dict[str, Any]], since_ms: int, until_ms: int) -> float:
    """Sum of hourly funding rates stamped in (since, until]."""
    total = 0.0
    for r in rows or []:
        t = int(r["time"])
        if since_ms < t <= until_ms:
            total += float(r["fundingRate"])
    return total


def fetch_bars(client: Any, now_ms: int, coins: Sequence[str] = COINS
               ) -> Dict[str, List[Dict[str, Any]]]:
    start = now_ms - HISTORY_DAYS * DAY_MS
    return {c: D.normalise(client.get_candles(c, "1d", start, now_ms), now_ms, DAY_MS)
            for c in coins}


def fetch_funding(client: Any, coin: str, since_ms: int, until_ms: int) -> float:
    total, start = 0.0, since_ms + 1
    while True:
        rows = client._post({"type": "fundingHistory", "coin": coin, "startTime": start,
                             "endTime": until_ms}, f"fundingHistory:{coin}")
        if not isinstance(rows, list):
            raise Unreadable(f"{coin}: funding history is not a list")
        total += funding_sum(rows, since_ms, until_ms)
        if len(rows) < 500:
            return total
        start = max(int(r["time"]) for r in rows) + 1


def run_pass(client: Any, now_ms: int, data_dir: Path = DATA_DIR, dry_run: bool = False
             ) -> Optional[Dict[str, Any]]:
    """One pass against the venue. None when today is already done."""
    state = load_state(data_dir)
    today = now_ms - now_ms % DAY_MS
    if state["last_day"] is not None and T.day_ms(state["last_day"]) >= today:
        return None
    try:
        _, close = closes_matrix(fetch_bars(client, now_ms), today)
        mids = client.get_all_mids()
        since = state["funding_through_ms"]
        funding = {c: fetch_funding(client, c, int(since), now_ms)
                   for c, w in state["held"].items() if w and since is not None}
    except Unreadable:
        raise
    except Exception as exc:                       # the client's own fetch errors
        raise Unreadable(f"venue read failed: {exc}") from exc
    nxt, row = step(state, now_ms, close, mids, funding)
    if not dry_run:
        save(nxt, row, data_dir)
    return row


# --------------------------------------------------------------- status --
def status(rows: List[Dict[str, Any]], today_ms: int,
           model_net: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """The run so far against PREREG_forward.md. `model_net` is replay()'s
    net per ledger row, on the venue's daily closes."""
    if not rows:
        return {"started": False}
    first = T.day_ms(rows[0]["day"])
    elapsed = int((today_ms - first) // DAY_MS) + 1
    eq = np.array([r["equity_after"] for r in rows])
    peak = np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:]
    dd = float((eq / peak - 1.0).min()) * 100
    drifts = [t["adverse_bps"] for r in rows for t in r["trades"]]
    span = max(elapsed - 1, 1)
    fund_year = float(sum(r["funding"] for r in rows)) / span * 365 * 100
    days = [T.day_ms(r["day"]) for r in rows] + [today_ms]
    gap = max(int((b - a) // DAY_MS) - 1 for a, b in zip(days, days[1:]))
    out: Dict[str, Any] = {
        "started": True, "first_day": rows[0]["day"], "last_day": rows[-1]["day"],
        "days_elapsed": elapsed, "rows": len(rows), "decision_day_reached": elapsed >= RUN_DAYS,
        "equity": float(eq[-1]), "max_drawdown_pct": dd, "trades": len(drifts),
        "mean_adverse_bps": float(np.mean(drifts)) if drifts else None,
        "funding_pct_year": fund_year, "longest_gap_days": gap,
        "held": rows[-1]["held"], "gross": rows[-1]["gross"], "net": rows[-1]["net"]}
    tracking = None
    if model_net is not None and len(rows) > 1:
        paper = sum(r["pnl_price"] - r["funding"] for r in rows) - sum(r["cost"] for r in rows[:-1])
        tracking = float(paper - model_net[:-1].sum()) / (len(rows) - 1) * 1e4
    out["tracking_bps_day"] = tracking
    out["go"] = {
        "G1_rows": len(rows) >= G1_MIN_ROWS,
        "G2_drift": bool(drifts) and float(np.mean(drifts)) <= G2_MAX_DRIFT_BPS,
        "G3_funding": fund_year <= G3_MAX_FUNDING_PCT_YEAR,
        "G4_tracking": tracking is not None and tracking >= G4_MIN_TRACKING_BPS_DAY}
    out["stop"] = {"S1_drawdown": dd <= -S1_MAX_DRAWDOWN_PCT, "S2_gap": gap >= S2_MAX_GAP_DAYS}
    return out


def model_for(rows: List[Dict[str, Any]], client: Any, now_ms: int) -> Optional[np.ndarray]:
    """replay() on the venue's daily closes at the ledger's own decisions."""
    bars = fetch_bars(client, now_ms)
    need = T.MIN_HISTORY + 1 + int((T.day_ms(rows[-1]["day"]) - T.day_ms(rows[0]["day"])) // DAY_MS)
    if need > HISTORY_DAYS - 2:
        bars = {c: D.normalise(client.get_candles(c, "1d", now_ms - (need + 5) * DAY_MS, now_ms),
                               now_ms, DAY_MS) for c in COINS}
    last = T.day_ms(rows[-1]["decision_close"])
    times = np.arange(last - (need - 1) * DAY_MS, last + DAY_MS, DAY_MS, dtype=np.int64)
    close = np.full((need, len(COINS)), np.nan)
    for j, c in enumerate(COINS):
        by_t = {int(b["t"]): float(b["c"]) for b in bars[c]}
        close[:, j] = [by_t.get(int(t), np.nan) for t in times]
    if not np.isfinite(close).all():
        return None
    at = {int(t): i for i, t in enumerate(times)}
    return replay(close, [at[T.day_ms(r["decision_close"])] for r in rows])


def line(row: Dict[str, Any]) -> str:
    moved = ", ".join(f"{t['coin']} {t['from'] * 100:+.1f}->{t['to'] * 100:+.1f}%"
                      for t in row["trades"]) or "no trades"
    day_pct = (row["equity_after"] / row["equity_before"] - 1.0) * 100
    return (f"TREND PAPER {row['day']}: equity {row['equity_after']:.4f} ({day_pct:+.2f}% today), "
            f"net {row['net'] * 100:+.0f}% gross {row['gross'] * 100:.0f}%; {moved}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--notify", action="store_true")
    a = ap.parse_args(argv)
    from research.levels.venue_client import HyperliquidReadOnlyClient
    client = HyperliquidReadOnlyClient()
    now_ms = int(time.time() * 1000)
    if a.status:
        rows = read_ledger()
        model = None
        if rows:
            try:
                model = model_for(rows, client, now_ms)
            except Exception as exc:
                print(f"model unreadable ({exc}); G4 not computed", file=sys.stderr)
        print(json.dumps(status(rows, now_ms - now_ms % DAY_MS, model), indent=2))
        return 0
    try:
        row = run_pass(client, now_ms, dry_run=a.dry_run)
    except Unreadable as exc:
        print(f"TREND PAPER: unreadable, nothing changed -- {exc}", file=sys.stderr)
        return 1
    if row is None:
        print("TREND PAPER: today's pass is already recorded")
        return 0
    text = line(row) + (" [dry run]" if a.dry_run else "")
    print(text)
    if a.notify and not a.dry_run:
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        import notify
        notify.send(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

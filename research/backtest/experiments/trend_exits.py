"""H-EXITS: would the trend book do better with a different exit?

Pre-registered in research/backtest/PREREG_trend_exits.md; read that first.

The live trend book (research/trend/PREREG_live.md) exits only when its trend
rule turns, with a disaster stop 25% beyond each day's mark. This compares
four single changes to that exit -- a tighter daily stop, a stop-loss from
entry, a stop moved to entry once a trade is in profit, and half the position
taken off at a profit -- with the live rule, day by day, on H-TREND's own
archive with each day's open, high and low added.

    python -m research.backtest.experiments.trend_exits --build-archive SRC
    python -m research.backtest.experiments.trend_exits --calibrate
    python -m research.backtest.experiments.trend_exits
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from research.backtest import stats as S
from research.backtest.experiments import trend_daily as T

ROOT = T.ROOT
RESULTS = ROOT / "results" / "trend_exits"
ARCHIVE = ROOT / "cache" / "daily_binance_ohlc"
PREREG = ROOT / "PREREG_trend_exits.md"
H_TREND_SUMMARY = T.RESULTS / "summary.json"
COINS = T.COINS
DAY_MS = T.DAY_MS
START = T.HYPOTHESES["T_all"]["start"]
END = T.HYPOTHESES["T_all"]["end"]

PASS_COST_BPS = T.COST_BPS        # a trade at the daily pass: 4.32 fee + 5 slippage
FEE_BPS = 4.32                    # the taker fee on a stop or take-profit fill
STOP_SLIP_BPS = 25.0              # a stop fills this far beyond its level (or the open)
TP_THROUGH_BPS = 5.0              # a take-profit needs the day to trade this far through it
FUND_LONG_BPS = T.FUND_LONG_BPS   # a day, on long notional; shorts receive nothing
BASE_STOP = 0.25
CAPITAL_USD = 350.0               # the live book's capital, for dollars only
K1_PCT = 30.0
SEED = 20261115
BOOT_B = 10_000
BOOT_BLOCK = T.BOOT_BLOCK
NULLS = 20
ALPHA = 0.05
HARSH_SLIP_BPS = 50.0
HARSH_THROUGH_BPS = 20.0


# ----------------------------------------------------------------- rules --
@dataclass(frozen=True)
class Rule:
    """One exit rule. Distances are fractions of price: 0.10 is a 10% move,
    which at a 5x margin setting is 50% of the margin."""
    name: str
    daily_stop: Optional[float] = BASE_STOP   # beyond the pass's mark, reset every pass
    entry_stop: Optional[float] = None        # beyond the average entry; sits out after
    breakeven_at: Optional[float] = None      # stop to the average entry once a pass sees this
    take_at: Optional[float] = None           # take_frac of the trade off at this profit
    take_frac: float = 0.5
    stop_slip_bps: float = STOP_SLIP_BPS
    tp_through_bps: float = TP_THROUGH_BPS


BASE = Rule("BASE")
NOSTOP = Rule("NOSTOP", daily_stop=None)
HYPOTHESES: Dict[str, Rule] = {
    "DS10": Rule("DS10", daily_stop=0.10),
    "SL10": Rule("SL10", entry_stop=0.10),
    "BE10": Rule("BE10", breakeven_at=0.10),
    "TP10": Rule("TP10", take_at=0.10),
}
NEIGHBOURS: Dict[str, Tuple[Rule, Rule]] = {
    "DS10": (Rule("DS05", daily_stop=0.05), Rule("DS15", daily_stop=0.15)),
    "SL10": (Rule("SL05", entry_stop=0.05), Rule("SL20", entry_stop=0.20)),
    "BE10": (Rule("BE05", breakeven_at=0.05), Rule("BE20", breakeven_at=0.20)),
    "TP10": (Rule("TP05", take_at=0.05), Rule("TP20", take_at=0.20)),
}


def harsh(rule: Rule) -> Rule:
    return replace(rule, name=rule.name + "_harsh", stop_slip_bps=HARSH_SLIP_BPS,
                   tp_through_bps=HARSH_THROUGH_BPS)


# ------------------------------------------------------------------ data --
def _sha(body: str) -> str:
    return hashlib.sha256(body.encode()).hexdigest()


def close_hashes(summary: Path = H_TREND_SUMMARY) -> Dict[str, str]:
    """H-TREND's SHA-256 of each coin's [t, close] series."""
    coins = json.loads(summary.read_text())["archive"]["coins"]
    return {c: str(coins[c]["sha256"]) for c in COINS}


def build_archive(src: Path, dst: Path = ARCHIVE,
                  expected: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Each coin's [t, open, high, low, close] from the mirror's kline files.
    Refused unless its closes, serialised exactly as H-TREND's archive was,
    reproduce H-TREND's SHA-256 -- the same data, byte for byte."""
    want = expected if expected is not None else close_hashes()
    dst.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Any] = {"source": T.SOURCE,
                                "closes_verified_against": "results/trend_daily/summary.json",
                                "coins": {}}
    for coin in COINS:
        rows: Dict[int, Tuple[float, float, float, float]] = {}
        for f in sorted((src / f"{coin}USDT" / "1d").glob("*.json")):
            for k in json.loads(f.read_text()):
                rows[int(k[0])] = (float(k[1]), float(k[2]), float(k[3]), float(k[4]))
        ts = sorted(rows)
        if not ts:
            raise ValueError(f"{coin}: no klines under {src}")
        closes = json.dumps([[t, rows[t][3]] for t in ts], separators=(",", ":"))
        if _sha(closes) != want[coin]:
            raise ValueError(f"{coin}: closes do not reproduce H-TREND's archive")
        body = json.dumps([[t, *rows[t]] for t in ts], separators=(",", ":"))
        (dst / f"{coin}.json").write_text(body)
        manifest["coins"][coin] = {"days": len(ts), "first": T.iso(ts[0]), "last": T.iso(ts[-1]),
                                   "close_sha256": _sha(closes), "sha256": _sha(body)}
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


Arrays = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def load(coins: Sequence[str] = COINS, archive: Path = ARCHIVE) -> Arrays:
    """(day timestamps, open, high, low, close [day x coin]); NaN before a coin
    lists. Verified against the manifest; no missing day inside a history."""
    manifest = json.loads((archive / "manifest.json").read_text())
    series: Dict[str, List[List[float]]] = {}
    for c in coins:
        body = (archive / f"{c}.json").read_text()
        if _sha(body) != manifest["coins"][c]["sha256"]:
            raise ValueError(f"{c}: archive does not match its manifest")
        series[c] = json.loads(body)
    t0 = min(int(s[0][0]) for s in series.values())
    t1 = max(int(s[-1][0]) for s in series.values())
    times = np.arange(t0, t1 + DAY_MS, DAY_MS, dtype=np.int64)
    shape = (len(times), len(coins))
    o, h, lo, c_ = (np.full(shape, np.nan) for _ in range(4))
    for j, c in enumerate(coins):
        ts = np.asarray([int(r[0]) for r in series[c]], dtype=np.int64)
        v = np.asarray([r[1:5] for r in series[c]], dtype=float)
        idx = (ts - t0) // DAY_MS
        o[idx, j], h[idx, j], lo[idx, j], c_[idx, j] = v[:, 0], v[:, 1], v[:, 2], v[:, 3]
        seen = np.isfinite(c_[:, j])
        if not seen[int(np.argmax(seen)):].all():
            raise ValueError(f"{c}: a day is missing inside its history")
    return times, o, h, lo, c_


def integrity(times: np.ndarray, o: np.ndarray, h: np.ndarray, lo: np.ndarray,
              c: np.ndarray, coins: Sequence[str] = COINS) -> Dict[str, Any]:
    """Counts only: no return of any rule is computed here."""
    out: Dict[str, Any] = {}
    for j, coin in enumerate(coins):
        ok = np.isfinite(c[:, j])
        oj, hj, lj, cj = o[ok, j], h[ok, j], lo[ok, j], c[ok, j]
        gap = np.abs(oj[1:] / cj[:-1] - 1.0) * 1e4
        out[coin] = {"days": int(ok.sum()), "first": T.iso(int(times[ok][0])),
                     "last": T.iso(int(times[ok][-1])),
                     "high_below_open_or_close": int((hj < np.maximum(oj, cj)).sum()),
                     "low_above_open_or_close": int((lj > np.minimum(oj, cj)).sum()),
                     "non_positive": int(((oj <= 0) | (hj <= 0) | (lj <= 0) | (cj <= 0)).sum()),
                     "open_off_prev_close_over_100bps": int((gap > 100).sum()),
                     "open_off_prev_close_max_bps": float(gap.max())}
    return out


def targets(close: np.ndarray) -> np.ndarray:
    """The rule's target weight every day [day x coin], from closes up to and
    including that day: research.trend.paper.decide, for all days at once."""
    s, v, elig = T.features(close)
    n = elig.sum(axis=1)
    cap = np.where(n > 0, T.CAP_GROSS_PER_COIN / np.maximum(n, 1), 0.0)
    vol = np.where(elig, v, 1.0)
    sig = np.where(elig, np.nan_to_num(s), 0.0)
    w = sig * (T.VOL_TARGET / vol) / np.maximum(n, 1)[:, None]
    return np.clip(w, -cap[:, None], cap[:, None])


def pnl_days(times: np.ndarray, close: np.ndarray) -> np.ndarray:
    """Decision rows whose P&L day lies in the window and has a bar."""
    days = T.window(times, START, END)
    return days[np.isfinite(close[days + 1]).any(axis=1)]


# ---------------------------------------------------------------- engine --
def run(rule: Rule, days: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
        C: np.ndarray, W: np.ndarray) -> Dict[str, Any]:
    """K books over the decision rows `days`; the price and target arrays are
    [K x day x coin]. At each pass (the close of day t) the book trades to the
    rule's weights at that close; its stop and take-profit then rest through
    day t+1's bar. Units are held between passes, so weights drift with price
    as the live book reads them. Capital is 1 and fixed.

    Returns net, gross, cost, funding, long and short [K x len(days)], and
    event counts [K]."""
    K, _, N = C.shape
    D = len(days)
    u = np.zeros((K, N))
    a = np.full((K, N), np.nan)
    be = np.zeros((K, N), dtype=bool)
    took = np.zeros((K, N), dtype=bool)
    lock = np.zeros((K, N))
    out = {k: np.zeros((K, D)) for k in ("net", "gross", "cost", "funding", "long", "short")}
    ev = {k: np.zeros(K, dtype=np.int64)
          for k in ("pass_trades", "daily_stops", "exit_stops", "takes", "pass_takes")}
    pass_cost, fee = PASS_COST_BPS / 1e4, FEE_BPS / 1e4
    slip, thru = rule.stop_slip_bps / 1e4, rule.tp_through_bps / 1e4
    fund_rate, f = FUND_LONG_BPS / 1e4, rule.take_frac
    nan = np.full((K, N), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        for p, t in enumerate(days):
            m, tgt = C[:, t], W[:, t]
            # the pass: the band rule, then no adds in a locked direction
            w = np.where(u != 0, u * m, 0.0)
            lock = np.where((lock != 0) & (np.sign(tgt) != lock), 0.0, lock)
            move = ((np.sign(tgt) != np.sign(w)) | (tgt == 0.0)
                    | (np.abs(tgt - w) > T.BAND * np.abs(tgt)))
            new = np.where(move, tgt, w)
            allowed = np.where(np.sign(w) == lock, np.abs(w), 0.0)
            new = np.where((lock != 0) & (np.sign(new) == lock),
                           lock * np.minimum(np.abs(new), allowed), new)
            traded = np.abs(new - w)
            cost = traded.sum(axis=1) * pass_cost
            ev["pass_trades"] += (traded > 0).sum(axis=1)
            un = np.where(new != 0, new / m, 0.0)
            same = (np.sign(un) == np.sign(u)) & (un != 0)
            grow = same & (np.abs(un) > np.abs(u))
            fresh = (un != 0) & ~same
            a = np.where(grow, (np.abs(u) * a + (np.abs(un) - np.abs(u)) * m) / np.abs(un), a)
            a = np.where(fresh, m, a)
            a = np.where(un == 0, np.nan, a)
            ended = fresh | (un == 0)
            be &= ~ended
            took &= ~ended
            u = un
            side = np.sign(u)
            # a take-profit already reached at the pass is taken there
            tp = nan
            if rule.take_at is not None:
                tp = np.where((side != 0) & ~took, a * (1 + side * rule.take_at), np.nan)
                now = np.isfinite(tp) & (side * (m - tp) >= 0)
                cut = np.where(now, u * f, 0.0)
                cost = cost + np.where(now, np.abs(cut * m), 0.0).sum(axis=1) * pass_cost
                u = u - cut
                took |= now
                lock = np.where(now, side, lock)
                tp = np.where(now, np.nan, tp)
                ev["pass_takes"] += now.sum(axis=1)
            if rule.breakeven_at is not None:
                be |= (side != 0) & (side * (m - a * (1 + side * rule.breakeven_at)) >= 0)
            # the day's stop: the nearest of the rule's levels
            stop = nan
            if rule.daily_stop is not None:
                stop = np.where(side != 0, m * (1 - side * rule.daily_stop), np.nan)
            kind = np.zeros((K, N), dtype=bool)          # True: the coin sits out after
            levels = []
            if rule.entry_stop is not None:
                levels.append(np.where(side != 0, a * (1 - side * rule.entry_stop), np.nan))
            if rule.breakeven_at is not None:
                levels.append(np.where(be, a, np.nan))
            for lv in levels:
                nearer = np.isfinite(lv) & (~np.isfinite(stop) | (side * lv > side * stop))
                stop = np.where(nearer, lv, stop)
                kind |= nearer
            fund = np.where(u > 0, u * m, 0.0).sum(axis=1) * fund_rate
            # the next day's bar: a stop first, whenever its range reaches one
            o, h, lo, c = O[:, t + 1], H[:, t + 1], L[:, t + 1], C[:, t + 1]
            hit = (side != 0) & np.isfinite(stop) & (((side > 0) & (lo <= stop))
                                                     | ((side < 0) & (h >= stop)))
            fill = np.where(side > 0, np.minimum(stop, o), np.maximum(stop, o)) * (1 - side * slip)
            take = (~hit & np.isfinite(tp) & (((side > 0) & (h >= tp * (1 + thru)))
                                              | ((side < 0) & (lo <= tp * (1 - thru)))))
            pnl = np.where(hit, u * (fill - m),
                           np.where(take, u * f * (tp - m) + u * (1 - f) * (c - m),
                                    np.where(u != 0, u * (c - m), 0.0)))
            fees = (np.where(hit, np.abs(u) * fill, 0.0)
                    + np.where(take, np.abs(u) * f * tp, 0.0)).sum(axis=1) * fee
            gross = pnl.sum(axis=1)
            out["gross"][:, p] = gross
            out["long"][:, p] = np.where(u > 0, pnl, 0.0).sum(axis=1)
            out["short"][:, p] = np.where(u < 0, pnl, 0.0).sum(axis=1)
            out["cost"][:, p] = cost + fees
            out["funding"][:, p] = fund
            out["net"][:, p] = gross - cost - fees - fund
            ev["daily_stops"] += (hit & ~kind).sum(axis=1)
            ev["exit_stops"] += (hit & kind).sum(axis=1)
            ev["takes"] += take.sum(axis=1)
            lock = np.where((hit & kind) | take, side, lock)
            u = np.where(hit, 0.0, np.where(take, u * (1 - f), u))
            took |= take
            a = np.where(hit, np.nan, a)
            be &= ~hit
            took &= ~hit
    if not np.isfinite(out["net"]).all():
        raise ValueError(f"{rule.name}: a non-finite daily net")
    return {**out, "events": ev}


def run_plain(rule: Rule, days: Sequence[int] | np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
              C: np.ndarray, W: np.ndarray) -> np.ndarray:
    """The same rules as run(), written again as a plain loop over days and
    coins, one book ([day x coin] arrays), as a check on the vectorised
    engine. Returns the daily net."""
    N = C.shape[1]
    u = [0.0] * N
    a = [math.nan] * N
    be = [False] * N
    took = [False] * N
    lock = [0.0] * N
    slip, thru = rule.stop_slip_bps / 1e4, rule.tp_through_bps / 1e4

    def sgn(x: float) -> float:
        return 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)

    net: List[float] = []
    for t in days:
        t = int(t)
        gross = cost = fees = fund = 0.0
        for j in range(N):
            m, tgt = float(C[t, j]), float(W[t, j])
            w = u[j] * m if u[j] != 0 else 0.0
            if lock[j] != 0 and sgn(tgt) != lock[j]:
                lock[j] = 0.0
            if (sgn(tgt) != sgn(w) or tgt == 0.0 or abs(tgt - w) > T.BAND * abs(tgt)):
                new = tgt
            else:
                new = w
            if lock[j] != 0 and sgn(new) == lock[j]:
                new = lock[j] * min(abs(new), abs(w) if sgn(w) == lock[j] else 0.0)
            cost += abs(new - w) * PASS_COST_BPS / 1e4
            un = new / m if new != 0 else 0.0
            if un != 0 and sgn(un) == sgn(u[j]):
                if abs(un) > abs(u[j]):
                    a[j] = (abs(u[j]) * a[j] + (abs(un) - abs(u[j])) * m) / abs(un)
            elif un != 0:
                a[j], be[j], took[j] = m, False, False
            if un == 0:
                a[j], be[j], took[j] = math.nan, False, False
            u[j] = un
            side = sgn(un)
            tp = math.nan
            if rule.take_at is not None and side != 0 and not took[j]:
                tp = a[j] * (1 + side * rule.take_at)
                if side * (m - tp) >= 0:
                    cut = u[j] * rule.take_frac
                    cost += abs(cut * m) * PASS_COST_BPS / 1e4
                    u[j] -= cut
                    took[j], lock[j], tp = True, side, math.nan
            if (rule.breakeven_at is not None and side != 0
                    and side * (m - a[j] * (1 + side * rule.breakeven_at)) >= 0):
                be[j] = True
            stop, sits_out = math.nan, False
            if side != 0:
                if rule.daily_stop is not None:
                    stop = m * (1 - side * rule.daily_stop)
                cands = []
                if rule.entry_stop is not None:
                    cands.append(a[j] * (1 - side * rule.entry_stop))
                if rule.breakeven_at is not None and be[j]:
                    cands.append(a[j])
                for lv in cands:
                    if math.isnan(stop) or side * lv > side * stop:
                        stop, sits_out = lv, True
            if u[j] > 0:
                fund += u[j] * m * FUND_LONG_BPS / 1e4
            if side == 0:
                continue
            o, h, lo, c = (float(X[t + 1, j]) for X in (O, H, L, C))
            if not math.isnan(stop) and ((side > 0 and lo <= stop) or (side < 0 and h >= stop)):
                fill = (min(stop, o) if side > 0 else max(stop, o)) * (1 - side * slip)
                gross += u[j] * (fill - m)
                fees += abs(u[j]) * fill * FEE_BPS / 1e4
                if sits_out:
                    lock[j] = side
                u[j], a[j], be[j], took[j] = 0.0, math.nan, False, False
                continue
            if not math.isnan(tp) and ((side > 0 and h >= tp * (1 + thru))
                                       or (side < 0 and lo <= tp * (1 - thru))):
                fr = rule.take_frac
                gross += u[j] * fr * (tp - m) + u[j] * (1 - fr) * (c - m)
                fees += abs(u[j]) * fr * tp * FEE_BPS / 1e4
                u[j] *= 1 - fr
                took[j], lock[j] = True, side
            else:
                gross += u[j] * (c - m)
        net.append(gross - cost - fees - fund)
    return np.asarray(net)


# ----------------------------------------------------------------- stats --
def boot_diff(d: np.ndarray, B: int = BOOT_B, block: int = BOOT_BLOCK,
              seed: int = SEED) -> Dict[str, float]:
    """Mean of a paired daily difference; two-sided p from a circular block
    bootstrap of the difference centred on zero (+1 correction); 95%
    percentile CI from the same draws uncentred."""
    if not np.isfinite(d).all():
        raise ValueError("a non-finite difference would read as significant")
    n = len(d)
    mean = float(d.mean())
    dc = d - mean
    rng = np.random.default_rng(seed)
    nb = math.ceil(n / block)
    step = np.arange(block)
    null, boot = np.empty(B), np.empty(B)
    for lo in range(0, B, 1000):
        b = min(1000, B - lo)
        starts = rng.integers(0, n, (b, nb))
        idx = ((starts[:, :, None] + step[None, None, :]) % n).reshape(b, -1)[:, :n]
        null[lo:lo + b] = dc[idx].mean(axis=1)
        boot[lo:lo + b] = d[idx].mean(axis=1)
    p = (1 + int((np.abs(null) >= abs(mean)).sum())) / (B + 1)
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    return {"mean_bps_day": mean * 1e4, "p": float(p),
            "ci_lo_bps_day": float(ci_lo) * 1e4, "ci_hi_bps_day": float(ci_hi) * 1e4}


def fixed_drawdown(x: np.ndarray) -> float:
    """Worst fall of the book's value from its peak, capital fixed at 1 (K1's
    measure): value = 1 + the running sum of daily net."""
    value = 1.0 + np.cumsum(x)
    return float((value / np.maximum.accumulate(np.maximum(value, 1.0)) - 1.0).min())


def describe(x: np.ndarray) -> Dict[str, Any]:
    dd = fixed_drawdown(x)
    return {"ann_return_pct": float(x.mean() * 365 * 100),
            "ann_vol_pct": float(x.std(ddof=1) * math.sqrt(365) * 100),
            "sharpe": T.sharpe(x), "max_drawdown_pct": dd * 100,
            "k1_would_fire": bool(dd * 100 <= -K1_PCT),
            "total_pnl_pct_of_capital": float(x.sum() * 100),
            "usd_per_year_at_350": float(x.mean() * 365 * CAPITAL_USD)}


# ----------------------------------------------------------- calibration --
def null_ohlc(O: np.ndarray, H: np.ndarray, L: np.ndarray, C: np.ndarray,
              rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Drift without trend, for whole bars (H-TREND's null, extended): each
    day, all coins together, is kept or mirrored about the previous close by
    one random sign -- the high becomes the mirrored low and the low the
    mirrored high -- then each coin's own mean daily log return is added back
    to the whole bar. A coin's first bar is its real one."""
    with np.errstate(invalid="ignore", divide="ignore"):
        prev = C[:-1]
        lo_, lh, ll, lc = (np.log(X[1:] / prev) for X in (O, H, L, C))
    mu = np.nanmean(lc, axis=0)
    up = (rng.choice([-1.0, 1.0], size=len(lc)) > 0)[:, None]
    no = np.where(up, lo_, -lo_) + mu
    nh = np.where(up, lh, -ll) + mu
    nl = np.where(up, ll, -lh) + mu
    nc = np.where(up, lc, -lc) + mu
    outs = [np.full(C.shape, np.nan) for _ in range(4)]
    for j in range(C.shape[1]):
        seen = np.where(np.isfinite(C[:, j]))[0]
        a = int(seen[0])
        for X, src in zip(outs, (O, H, L, C)):
            X[a, j] = src[a, j]
        closes = C[a, j] * np.exp(np.cumsum(nc[a:, j]))
        prevs = np.concatenate([[C[a, j]], closes[:-1]])
        outs[3][a + 1:, j] = closes
        outs[0][a + 1:, j] = prevs * np.exp(no[a:, j])
        outs[1][a + 1:, j] = prevs * np.exp(nh[a:, j])
        outs[2][a + 1:, j] = prevs * np.exp(nl[a:, j])
    return outs[0], outs[1], outs[2], outs[3]


def prereg_sha() -> str:
    return hashlib.sha256(PREREG.read_bytes()).hexdigest()


def four_tests(nets: Dict[str, np.ndarray], k: int, seed: int) -> Dict[str, Dict[str, Any]]:
    """The four registered tests on book k: each hypothesis minus BASE, Holm
    across the four."""
    res = {h: boot_diff(nets[h][k] - nets["BASE"][k], seed=seed) for h in HYPOTHESES}
    holm = S.holm({h: r["p"] for h, r in res.items()}, ALPHA)
    for h in res:
        res[h]["p_holm"] = float(holm[h]["p_holm"])
    return res


def calibrate(archive: Path = ARCHIVE, n: int = NULLS, seed: int = SEED) -> Dict[str, Any]:
    times, O, H, L, C = load(COINS, archive)
    days = pnl_days(times, C)
    rng = np.random.default_rng(seed + 1)
    fakes = [null_ohlc(O, H, L, C, rng) for _ in range(n)]
    Ok, Hk, Lk, Ck = (np.stack([fk[i] for fk in fakes]) for i in range(4))
    Wk = np.stack([targets(fk[3]) for fk in fakes])
    nets = {r.name: run(r, days, Ok, Hk, Lk, Ck, Wk)["net"]
            for r in (BASE, *HYPOTHESES.values())}
    runs: List[Dict[str, Any]] = []
    for k in range(n):
        res = four_tests(nets, k, seed + 100 + k)
        runs.append({"dataset": k, "tests": res,
                     "better": any(r["p_holm"] < ALPHA and r["mean_bps_day"] > 0 for r in res.values()),
                     "worse": any(r["p_holm"] < ALPHA and r["mean_bps_day"] < 0 for r in res.values())})
    mech = {}
    for h in HYPOTHESES:
        x = np.asarray([r["tests"][h]["mean_bps_day"] for r in runs])
        mech[h] = {"mean_bps_day": float(x.mean()), "se": float(x.std(ddof=1) / math.sqrt(n)),
                   "p_raw_below_alpha": int(sum(r["tests"][h]["p"] < ALPHA for r in runs))}
    fired = int(sum(r["better"] for r in runs))
    return {"datasets": n, "days": int(len(days)), "fired_better": fired,
            "fired_worse": int(sum(r["worse"] for r in runs)), "passed": bool(fired <= 2),
            "mechanical_effect": mech, "prereg_sha256": prereg_sha(), "seed": seed, "runs": runs}


# -------------------------------------------------------------- the test --
def real_run(archive: Path = ARCHIVE) -> Dict[str, Any]:
    cal_path = RESULTS / "null_calibration.json"
    if not cal_path.exists():
        raise RuntimeError("run --calibrate first (PREREG_trend_exits.md)")
    cal = json.loads(cal_path.read_text())
    if not cal["passed"]:
        raise RuntimeError("the null calibration failed: amend before any real run")
    if cal["prereg_sha256"] != prereg_sha():
        raise RuntimeError("the pre-registration changed after the calibration: re-calibrate")
    times, O, H, L, C = load(COINS, archive)
    W = targets(C)
    days = pnl_days(times, C)
    half = len(days) // 2
    one = (O[None], H[None], L[None], C[None], W[None])

    def go(rule: Rule) -> Dict[str, Any]:
        return run(rule, days, *one)

    rules: List[Rule] = [BASE, NOSTOP, *HYPOTHESES.values(),
                         *[r for pair in NEIGHBOURS.values() for r in pair],
                         harsh(BASE), *[harsh(r) for r in HYPOTHESES.values()]]
    books = {r.name: go(r) for r in rules}
    net = {k: b["net"][0] for k, b in books.items()}

    # the vectorised engine against the plain loop, on the real days
    check = {}
    for r in (BASE, *HYPOTHESES.values()):
        plain = run_plain(r, days, O, H, L, C, W)
        check[r.name] = float(np.abs(plain - net[r.name]).max())
    if not all(math.isfinite(x) and x <= 1e-12 for x in check.values()):
        raise RuntimeError(f"engine and plain loop disagree: {check}")

    tests = four_tests({k: v[None] for k, v in net.items()}, 0, SEED)
    base_dd = fixed_drawdown(net["BASE"])
    verdicts: Dict[str, Any] = {}
    for h, rule in HYPOTHESES.items():
        d = net[h] - net["BASE"]
        halves = [float(d[:half].mean() * 1e4), float(d[half:].mean() * 1e4)]
        nb = {r.name: float((net[r.name] - net["BASE"]).mean() * 1e4) for r in NEIGHBOURS[h]}
        hd = float((net[h + "_harsh"] - net["BASE_harsh"]).mean() * 1e4)
        t = tests[h]
        checks = {"holm_p_below_0.05_and_positive": t["p_holm"] < ALPHA and t["mean_bps_day"] > 0,
                  "positive_in_both_halves": all(x > 0 for x in halves),
                  "drawdown_no_deeper_than_base": fixed_drawdown(net[h]) >= base_dd,
                  "both_neighbours_positive": all(x > 0 for x in nb.values()),
                  "positive_under_harsh_fills": hd > 0}
        if all(checks.values()):
            word = "better"
        elif t["p_holm"] < ALPHA and t["mean_bps_day"] < 0:
            word = "worse"
        else:
            word = "not established"
        verdicts[h] = {"rule": asdict(rule), **t, "halves_bps_day": halves,
                       "neighbours_bps_day": nb, "harsh_bps_day": hd,
                       "usd_per_year_at_350": t["mean_bps_day"] / 1e4 * 365 * CAPITAL_USD,
                       "checks": checks, "verdict": word}

    yrs = len(days) / 365.0
    years = np.asarray([int(T.iso(int(times[d]) + DAY_MS)[:4]) for d in days])
    table = {}
    for k, b in books.items():
        x = b["net"][0]
        table[k] = {**describe(x),
                    "cost_pct_per_year": float(b["cost"][0].sum() / yrs * 100),
                    "funding_pct_per_year": float(b["funding"][0].sum() / yrs * 100),
                    "long_pct_per_year": float(b["long"][0].sum() / yrs * 100),
                    "short_pct_per_year": float(b["short"][0].sum() / yrs * 100),
                    "per_year": {e: float(v[0] / yrs) for e, v in b["events"].items()}}
    by_year = {int(y): {k: float(net[k][years == y].sum() * 100)
                        for k in ("BASE", "NOSTOP", *HYPOTHESES)} for y in sorted(set(years))}
    s, v, e = T.features(C)
    h_trend = T.simulate(s, v, e, T.next_returns(C), days)["net"][0]
    return {"source": T.SOURCE, "archive": json.loads((archive / "manifest.json").read_text()),
            "integrity": integrity(times, O, H, L, C), "prereg_sha256": prereg_sha(),
            "calibration": {k: cal[k] for k in ("datasets", "fired_better", "fired_worse",
                                                "passed", "mechanical_effect")},
            "days": int(len(days)), "first_pnl_day": T.iso(int(times[days[0]]) + DAY_MS),
            "last_pnl_day": T.iso(int(times[days[-1]]) + DAY_MS),
            "split_day": T.iso(int(times[days[half]]) + DAY_MS),
            "engine_vs_plain_max_abs": check, "verdicts": verdicts, "rules": table,
            "by_year_pnl_pct_of_capital": by_year,
            "h_trend_simulate_same_days": describe(h_trend)}


# ---------------------------------------------------------------- report --
def _f(x: float, d: int = 2, sign: bool = True) -> str:
    if x is None or not math.isfinite(x):
        return "n/a"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def render(res: Dict[str, Any]) -> str:
    L_: List[str] = ["# Exits for the trend book -- results", "",
                     f"Generated by `research/backtest/experiments/trend_exits.py`. "
                     f"Pre-registration: `PREREG_trend_exits.md` "
                     f"(SHA-256 `{res['prereg_sha256'][:16]}...`).", "",
                     f"P&L days {res['first_pnl_day']} to {res['last_pnl_day']} "
                     f"({res['days']}), split at {res['split_day']}. Capital fixed; "
                     f"dollars at ${CAPITAL_USD:.0f}.", ""]
    c = res["calibration"]
    L_ += ["## Null calibration (run first)", "",
           f"{c['datasets']} trend-free archives: {c['fired_better']} produced a 'better' "
           f"(pass needs <= 2), {c['fired_worse']} a 'worse'. **{'Passed' if c['passed'] else 'FAILED'}.**", "",
           "| rule | mean difference on the nulls, bps a day | se |", "|---|---|---|"]
    for h, m in c["mechanical_effect"].items():
        L_.append(f"| {h} | {_f(m['mean_bps_day'])} | {_f(m['se'], sign=False)} |")
    L_ += ["", "## The registered test: each rule minus the live rule (BASE)", "",
           "| rule | bps a day | 95% CI | $ a year at $350 | p | Holm p | halves | neighbours | harsh fills | verdict |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for h, v in res["verdicts"].items():
        nb = ", ".join(f"{k} {_f(x)}" for k, x in v["neighbours_bps_day"].items())
        L_.append(f"| {h} | {_f(v['mean_bps_day'])} | {_f(v['ci_lo_bps_day'])} to {_f(v['ci_hi_bps_day'])} "
                  f"| {_f(v['usd_per_year_at_350'], 0)} | {v['p']:.4f} | {v['p_holm']:.4f} "
                  f"| {_f(v['halves_bps_day'][0])}, {_f(v['halves_bps_day'][1])} | {nb} "
                  f"| {_f(v['harsh_bps_day'])} | **{v['verdict']}** |")
    L_ += ["", "## Every rule (descriptive)", "",
           "| rule | return %/yr | vol % | Sharpe | worst drawdown % | K1 fires | $ a year at $350 "
           "| stops/yr | exit stops/yr | takes/yr | cost %/yr |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, r in res["rules"].items():
        e = r["per_year"]
        L_.append(f"| {k} | {_f(r['ann_return_pct'], 1)} | {_f(r['ann_vol_pct'], 1, False)} "
                  f"| {_f(r['sharpe'])} | {_f(r['max_drawdown_pct'], 1)} | {'yes' if r['k1_would_fire'] else 'no'} "
                  f"| {_f(r['usd_per_year_at_350'], 0)} | {_f(e['daily_stops'], 1, False)} "
                  f"| {_f(e['exit_stops'], 1, False)} | {_f(e['takes'] + e['pass_takes'], 1, False)} "
                  f"| {_f(r['cost_pct_per_year'], 2, False)} |")
    ht = res["h_trend_simulate_same_days"]
    L_ += ["", f"H-TREND's own simulation of the same days (weights constant between trades, "
               f"no stop): {_f(ht['ann_return_pct'], 1)}% a year, Sharpe {_f(ht['sharpe'])}.", "",
           "## Year by year, P&L as % of capital (descriptive)", "",
           "| year | " + " | ".join(next(iter(res["by_year_pnl_pct_of_capital"].values())).keys()) + " |",
           "|---" * (1 + len(next(iter(res["by_year_pnl_pct_of_capital"].values())))) + "|"]
    for y, row in res["by_year_pnl_pct_of_capital"].items():
        L_.append(f"| {y} | " + " | ".join(_f(x, 1) for x in row.values()) + " |")
    L_ += ["", f"Engine against the plain loop, largest daily difference: "
               f"{max(res['engine_vs_plain_max_abs'].values()):.1e}.", ""]
    return "\n".join(L_)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--build-archive", type=Path, metavar="SRC",
                    help="the mirror's .klines-cache directory")
    ap.add_argument("--calibrate", action="store_true")
    args = ap.parse_args(argv)
    RESULTS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if args.build_archive:
        man = build_archive(args.build_archive)
        times, O, H, L, C = load()
        print(json.dumps({"manifest": man, "integrity": integrity(times, O, H, L, C)}, indent=1))
        return 0
    if args.calibrate:
        cal = calibrate()
        (RESULTS / "null_calibration.json").write_text(json.dumps(cal, indent=1))
        print(json.dumps({k: cal[k] for k in ("datasets", "days", "fired_better", "fired_worse",
                                              "passed", "mechanical_effect")}, indent=1))
        return 0 if cal["passed"] else 1
    res = real_run()
    res["seconds"] = round(time.time() - t0, 1)
    (RESULTS / "summary.json").write_text(json.dumps(res, indent=1))
    (RESULTS / "report.md").write_text(render(res))
    print(render(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

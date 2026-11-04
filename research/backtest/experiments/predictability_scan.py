"""EXPLORATORY: where, if anywhere, these bars are predictable at all.

Decides nothing. A strategy can only earn what the price path makes
predictable: if E[r(t, t+H) | anything known at t] = 0, every rule built on
that information -- bands, grids, trend, the mirror -- has zero expected
gross P&L and loses its costs. So instead of wrapping another rule around the
bars, this measures the conditional mean directly:

  market    the equal-weighted universe's own past h hours -> its next h
            (time-series momentum, or reversal, of the whole market);
  relative  each coin's past h hours against the universe -> the next h, as a
            tercile long-short book (cross-sectional momentum or reversal,
            market-neutral);
  shock     after a coin moves more than 3x its trailing RMS against the
            universe in one hour, that move's direction -> the next H hours
            of its relative return (continuation or reversal);
  premium   (with funding history) each coin's hourly perp-over-oracle
            premium -> its next h hours, as a tercile book short the rich.

In basis points per period, gross, beside the one-way fee that would eat it.
The null is a randomization test that keeps everything but predictability:
each hour's returns, all coins together, get one random sign. Volatility
clustering, fat tails, the cross-coin correlation and every shock's size
survive; every conditional mean becomes zero. Holm across all cells.

A cell that passes is a lead to pre-register on bars that did not exist when
this ran -- never a result.

    python -m research.backtest.experiments.predictability_scan [--funding DIR]
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from research.backtest import data as D
from research.backtest import stats as S

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "predictability_scan"
SNAPSHOT = ROOT / "cache" / "snapshot_1h_fresh.json"
HORIZONS = (1, 4, 24, 72, 168)
SHOCK_HORIZONS = (1, 4, 24)
SKIP_HORIZONS = (1, 4)      # the bounce diagnostic; outside the Holm family
SHOCK_SIGMAS = 3.0
SHOCK_LOOKBACK = 168
MIN_COINS = 20
DRAWS = 1000
SEED = 20261104
MAKER_BPS, TAKER_BPS = 1.44, 4.32          # this account's one-way fees (costs.MEASURED)

Cells = Dict[str, float]


def panel(bars_by: Dict[str, List[Dict[str, Any]]], min_bars: int = 4000
          ) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """(hour timestamps, coins, hourly log returns [hour x coin]); NaN where
    a bar is missing on either side of the hour. Nothing is filled in."""
    coins = sorted(c for c, b in bars_by.items() if len(b) >= min_bars)
    times = np.asarray(sorted({b["t"] for c in coins for b in bars_by[c]}), dtype=np.int64)
    row = {int(t): k for k, t in enumerate(times)}
    close = np.full((len(times), len(coins)), np.nan)
    for j, c in enumerate(coins):
        for b in bars_by[c]:
            close[row[b["t"]], j] = b["c"]
    R = np.full_like(close, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        R[1:] = np.log(close[1:] / close[:-1])
    return times, coins, R


def window_sum(R: np.ndarray, L: int, forward: bool = False) -> np.ndarray:
    """Sum of R over the L hours ending at t (past) or starting after t
    (forward), per column; NaN if any hour in the window is missing."""
    V = np.isfinite(R)
    cs = np.vstack([np.zeros((1, R.shape[1])), np.cumsum(np.where(V, R, 0.0), axis=0)])
    cv = np.vstack([np.zeros((1, R.shape[1])), np.cumsum(V, axis=0)])
    T = R.shape[0]
    out = np.full(R.shape, np.nan)
    if L <= 0 or L >= T:
        return out
    if forward:
        s, n = cs[L + 1:] - cs[1:T - L + 1], cv[L + 1:] - cv[1:T - L + 1]
        out[:T - L] = np.where(n == L, s, np.nan)
    else:
        s, n = cs[L:] - cs[:T - L + 1], cv[L:] - cv[:T - L + 1]
        out[L - 1:] = np.where(n == L, s, np.nan)
    return out


def market(R: np.ndarray) -> np.ndarray:
    """Equal-weighted universe return per hour; NaN with fewer than MIN_COINS."""
    ok = np.isfinite(R)
    n = ok.sum(axis=1)
    m = np.where(ok, R, 0.0).sum(axis=1) / np.maximum(n, 1)
    return np.where(n >= MIN_COINS, m, np.nan)


def trailing_rms(X: np.ndarray, L: int = SHOCK_LOOKBACK, min_n: int = 100) -> np.ndarray:
    """RMS of X over the L hours BEFORE t. RMS rather than SD so a sign flip
    leaves it -- and so every shock -- exactly where it was."""
    V = np.isfinite(X)
    sq = np.vstack([np.zeros((1, X.shape[1])), np.cumsum(np.where(V, X * X, 0.0), axis=0)])
    cv = np.vstack([np.zeros((1, X.shape[1])), np.cumsum(V, axis=0)])
    T = X.shape[0]
    out = np.full(X.shape, np.nan)
    if T <= L:
        return out
    s, n = sq[L:T] - sq[:T - L], cv[L:T] - cv[:T - L]     # rows t = L..T-1, hours t-L..t-1
    with np.errstate(invalid="ignore", divide="ignore"):
        out[L:] = np.where(n >= min_n, np.sqrt(s / np.maximum(n, 1)), np.nan)
    return out


def tercile_book(signal: np.ndarray, fwd: np.ndarray, rows: np.ndarray, short_high: bool = True
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """At each sampled row: long the bottom third by `signal`, short the top
    third (short_high), equal weight, held for `fwd`. Returns (spread per row,
    weights [row x coin]) -- the spread is long-leg minus short-leg return."""
    A = np.where(np.isfinite(signal[rows]) & np.isfinite(fwd[rows]), signal[rows], np.nan)
    F = fwd[rows]
    nv = np.isfinite(A).sum(axis=1)
    order = np.argsort(np.where(np.isfinite(A), A, np.inf), axis=1, kind="stable")
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.arange(A.shape[1])[None, :].repeat(A.shape[0], 0), axis=1)
    leg = nv // 3
    low = ranks < leg[:, None]
    high = (ranks >= (nv - leg)[:, None]) & (ranks < nv[:, None])
    ok = nv >= 9
    w = np.zeros(A.shape)
    with np.errstate(invalid="ignore", divide="ignore"):
        w = np.where(low, 1.0, 0.0) - np.where(high, 1.0, 0.0)
        w = w / np.maximum(leg, 1)[:, None]
        if not short_high:
            w = -w
        spread = np.nansum(np.where(w != 0, w * np.nan_to_num(F), 0.0), axis=1)
    spread = np.where(ok, spread, np.nan)
    w[~ok] = 0.0
    return spread, w


def row_corr(A: np.ndarray, B: np.ndarray) -> float:
    """Mean over rows of the cross-sectional rank correlation of A and B."""
    out = []
    for a, b in zip(A, B):
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() >= 9:
            ra, rb = np.argsort(np.argsort(a[ok])), np.argsort(np.argsort(b[ok]))
            out.append(float(np.corrcoef(ra, rb)[0, 1]))
    return float(np.mean(out)) if out else math.nan


def cell_stats(R: np.ndarray, premium: Optional[np.ndarray] = None,
               shocks: Optional[np.ndarray] = None, rate: Optional[np.ndarray] = None,
               describe: bool = False) -> Tuple[Cells, Dict[str, Any]]:
    """Every cell's statistic (bps per period) on returns R, plus what the
    cost comparison needs. Called on the real returns and on each flip;
    `describe` adds the premium book's funding carry and its overlap with
    past returns, which only the real call needs."""
    out: Cells = {}
    extra: Dict[str, Any] = {}
    M = market(R)
    X = R - M[:, None]
    for h in HORIZONS:
        rows = np.arange(h - 1, R.shape[0] - h, h)
        pm, fm = window_sum(M[:, None], h)[:, 0], window_sum(M[:, None], h, forward=True)[:, 0]
        v = pm[rows] * fm[rows]
        ok = np.isfinite(v)
        out[f"market_{h}h"] = float(np.mean(np.sign(pm[rows][ok]) * fm[rows][ok]) * 1e4) if ok.any() else math.nan
        extra[f"market_{h}h"] = {"n": int(ok.sum()),
                                 "flips": float(np.mean(np.diff(np.sign(pm[rows][ok])) != 0)) if ok.sum() > 1 else math.nan}
        # long the coins that LAGGED, short those that LED: positive = cross-sectional reversal
        spread, w = tercile_book(window_sum(R, h), window_sum(R, h, forward=True), rows)
        okk = np.isfinite(spread)
        out[f"relative_{h}h"] = float(np.mean(spread[okk]) * 1e4) if okk.any() else math.nan
        traded = np.abs(np.diff(w, axis=0)).sum(axis=1)
        extra[f"relative_{h}h"] = {"n": int(okk.sum()), "traded_per_rebalance": float(np.mean(traded)) if len(traded) else math.nan}
        if h in SKIP_HORIZONS:
            # diagnostic: rank on the h hours ending one hour EARLIER. A close-to-close
            # bounce between bid and ask reverses within the next hour and nothing more,
            # so if the effect is that bounce, skipping the latest hour removes it
            past = window_sum(R, h)
            lagged = np.vstack([np.full((1, R.shape[1]), np.nan), past[:-1]])
            sk, wk = tercile_book(lagged, window_sum(R, h, forward=True), rows)
            oks = np.isfinite(sk)
            out[f"relative_skip_{h}h"] = float(np.mean(sk[oks]) * 1e4) if oks.any() else math.nan
            ts = np.abs(np.diff(wk, axis=0)).sum(axis=1)
            extra[f"relative_skip_{h}h"] = {"n": int(oks.sum()), "traded_per_rebalance": float(np.mean(ts)) if len(ts) else math.nan,
                                            "diagnostic": True}
        if premium is not None:
            sp, wp = tercile_book(premium, window_sum(R, h, forward=True), rows)
            okp = np.isfinite(sp)
            out[f"premium_{h}h"] = float(np.mean(sp[okp]) * 1e4) if okp.any() else math.nan
            tp = np.abs(np.diff(wp, axis=0)).sum(axis=1)
            extra[f"premium_{h}h"] = {"n": int(okp.sum()), "traded_per_rebalance": float(np.mean(tp)) if len(tp) else math.nan}
            if describe and rate is not None:
                # funding the book collects over the hold at the rate known at entry: a long
                # pays a positive rate, a short receives it
                carry = -(wp * np.nan_to_num(rate[rows])).sum(axis=1) * h
                extra[f"premium_{h}h"]["carry_bps"] = float(np.mean(carry[okp]) * 1e4) if okp.any() else math.nan
                extra[f"premium_{h}h"]["corr_with_past_return"] = row_corr(premium[rows], window_sum(R, h)[rows])
    if shocks is not None:
        sgn = np.sign(X)
        for H in SHOCK_HORIZONS:
            fx = window_sum(X, H, forward=True)
            v = np.where(shocks, sgn * fx, np.nan)
            ok = np.isfinite(v)
            out[f"shock_{H}h"] = float(np.mean(v[ok]) * 1e4) if ok.any() else math.nan
            extra[f"shock_{H}h"] = {"n": int(ok.sum())}
    return out, extra


def shock_mask(R: np.ndarray) -> np.ndarray:
    X = R - market(R)[:, None]
    rms = trailing_rms(X)
    with np.errstate(invalid="ignore"):
        return np.isfinite(X) & np.isfinite(rms) & (np.abs(X) > SHOCK_SIGMAS * rms)


def scan(R: np.ndarray, premium: Optional[np.ndarray] = None, draws: int = DRAWS,
         seed: int = SEED, log=print, rate: Optional[np.ndarray] = None,
         only: Optional[str] = None) -> Dict[str, Any]:
    """The cells and their null. `only` keeps the cells whose name starts
    with it -- and so the Holm family -- to one family."""
    shocks = shock_mask(R)
    real, extra = cell_stats(R, premium, shocks, rate=rate, describe=True)
    rng = np.random.default_rng(seed)
    null: Dict[str, List[float]] = {k: [] for k in real}
    t0 = time.time()
    for d in range(draws):
        eps = rng.choice([-1.0, 1.0], size=R.shape[0])
        got, _ = cell_stats(R * eps[:, None], premium, shocks)
        for k, v in got.items():
            null[k].append(v)
        if d and d % 200 == 0:
            log(f"  [null] {d}/{draws} draws ({time.time() - t0:.0f}s)")
    cells: Dict[str, Any] = {}
    for k, obs in real.items():
        if only is not None and not k.startswith(only):
            continue
        nv = np.asarray([x for x in null[k] if math.isfinite(x)])
        sd = float(nv.std(ddof=1)) if len(nv) > 1 else math.nan
        p = float((1 + np.sum(np.abs(nv) >= abs(obs))) / (1 + len(nv))) if math.isfinite(obs) else math.nan
        cells[k] = {"bps": obs, "null_sd": sd, "z": obs / sd if sd and math.isfinite(obs) else math.nan,
                    "null_mean": float(nv.mean()) if len(nv) else math.nan, "p": p, **extra[k]}
        traded = extra[k].get("traded_per_rebalance")
        if k.startswith("shock"):
            traded = 2.0                    # one round trip per event, as a fraction of notional
        elif k.startswith("market"):
            fl = extra[k].get("flips")
            traded = 2.0 * fl if fl is not None and math.isfinite(fl) else math.nan
        cells[k]["traded"] = traded
        cells[k]["break_even_one_way_bps"] = (abs(obs) / traded if traded and math.isfinite(obs)
                                              and math.isfinite(traded) and traded > 0 else math.nan)
    holm = S.holm({k: v["p"] for k, v in cells.items() if not v.get("diagnostic")})
    for k in cells:
        hk = holm.get(k)
        cells[k]["p_holm"] = hk["p_holm"] if hk else math.nan
        cells[k]["holm_reject"] = bool(hk["reject"]) if hk else False
    return {"cells": cells, "draws": draws, "seed": seed, "shock_events": int(shocks.sum())}


INTEREST_HOURLY = 0.01 / 100 / 8     # the funding formula's fixed interest component, per hour


def funding_profile(funding_dir: Path, coins: Sequence[str]) -> Dict[str, Any]:
    """What the funding history says a short collected, and how often the
    rate was nothing but the formula's fixed interest component (the premium
    inside the clamp), which is the same for every coin."""
    per: Dict[str, np.ndarray] = {}
    for c in coins:
        p = funding_dir / f"{c}.json"
        if p.exists():
            per[c] = np.asarray([float(r["fundingRate"]) for r in json.loads(p.read_text())])
    if not per:
        return {}
    allr = np.concatenate(list(per.values()))
    q = np.percentile(allr, [0.1, 1, 50, 99, 99.9]) * 1e4
    return {"coins": len(per), "coin_hours": int(len(allr)),
            "share_at_interest_baseline": float(np.mean(np.abs(allr - INTEREST_HOURLY) < 1e-12)),
            "share_negative": float(np.mean(allr < 0)),
            "mean_bps_per_hour": float(allr.mean() * 1e4),
            "annualised_pct": float(allr.mean() * 24 * 365 * 100),
            "bps_per_hour_p0.1_p1_p50_p99_p99.9": [float(x) for x in q],
            "per_coin_annualised_pct": {c: float(r.mean() * 24 * 365 * 100) for c, r in per.items()}}


def carry_vs_drift(funding_dir: Path, bars_by: Dict[str, List[Dict[str, Any]]], coins: Sequence[str],
                   start_ms: int, end_ms: int) -> Dict[str, Any]:
    """A perp-only short held through the funding window, per coin: funding
    collected against the price move it was exposed to. Both in % of notional."""
    rows = []
    for c in coins:
        p = funding_dir / f"{c}.json"
        bars = [b for b in bars_by.get(c, []) if start_ms <= b["t"] < end_ms]
        if not p.exists() or len(bars) < 24 * 30:
            continue
        rates = [float(r["fundingRate"]) for r in json.loads(p.read_text()) if start_ms <= int(r["time"]) < end_ms]
        rows.append((c, sum(rates) * 100, (bars[-1]["c"] / bars[0]["o"] - 1) * 100))
    if not rows:
        return {}
    f = np.asarray([r[1] for r in rows])
    m = np.asarray([r[2] for r in rows])
    return {"coins": len(rows), "days": (end_ms - start_ms) / 864e5,
            "funding_collected_pct_mean": float(f.mean()), "price_move_pct_mean": float(m.mean()),
            "price_move_pct_median": float(np.median(m)), "short_net_pct_mean": float((f - m).mean()),
            "share_of_coins_where_short_made_money": float(np.mean(f - m > 0)),
            "price_move_sd_across_coins_pct": float(m.std(ddof=1)) if len(m) > 1 else math.nan}


def regime_persistence(R: np.ndarray, block: int = 168, draws: int = 500, seed: int = SEED
                       ) -> Dict[str, Any]:
    """Does a week's character predict the next week's? Per coin, over
    non-overlapping blocks of `block` hours:
      vol         realised volatility, sqrt(sum r^2);
      trendiness  efficiency ratio |sum r| / sum |r| (1 = one straight line);
      trend pays  sign(this block's return) x next block's return, after the
                  most vs least trendy third of blocks -- the regime overlay's
                  premise, that trend-following pays more after trending weeks.
    Rank correlations of consecutive blocks, pooled over coins; the null
    shuffles the blocks in time -- the SAME order for every coin, so each
    block keeps its cross-coin correlation and only the sequence is lost.
    (Shuffling coins independently would make them independent too, and give
    a null far narrower than the data's own noise.)"""
    T, C = R.shape
    nb = T // block
    if nb < 6:
        return {}
    Rb = R[:nb * block].reshape(nb, block, C)
    ok = np.isfinite(Rb).all(axis=1)                       # complete blocks only
    Z = np.where(np.isfinite(Rb), Rb, 0.0)
    vol = np.where(ok, np.sqrt((Z ** 2).sum(axis=1)), np.nan)
    net = np.where(ok, Z.sum(axis=1), np.nan)
    er = np.where(ok, np.abs(net) / np.maximum(np.abs(Z).sum(axis=1), 1e-12), np.nan)

    def rank_corr(A: np.ndarray) -> float:
        a, b = A[:-1].ravel(), A[1:].ravel()
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 10:
            return math.nan
        return float(np.corrcoef(np.argsort(np.argsort(a[m])), np.argsort(np.argsort(b[m])))[0, 1])

    def trend_gap(E: np.ndarray, N: np.ndarray) -> float:
        pay = np.sign(N[:-1]) * N[1:]
        e = E[:-1]
        m = np.isfinite(pay) & np.isfinite(e)
        if m.sum() < 30:
            return math.nan
        lo, hi = np.percentile(e[m], [100 / 3, 200 / 3])
        return float((pay[m & (e >= hi)].mean() - pay[m & (e <= lo)].mean()) * 1e4)

    real = {"vol": rank_corr(vol), "trendiness": rank_corr(er), "trend_pays_gap_bps": trend_gap(er, net)}
    rng = np.random.default_rng(seed + 2)
    null: Dict[str, List[float]] = {k: [] for k in real}
    for _ in range(draws):
        order = rng.permutation(nb)                         # one order for all coins
        take = lambda A: A[order]   # noqa: E731
        null["vol"].append(rank_corr(take(vol)))
        null["trendiness"].append(rank_corr(take(er)))
        null["trend_pays_gap_bps"].append(trend_gap(take(er), take(net)))
    out: Dict[str, Any] = {"blocks": nb, "block_hours": block}
    for k, obs in real.items():
        nv = np.asarray([x for x in null[k] if math.isfinite(x)])
        out[k] = {"value": obs, "null_mean": float(nv.mean()) if len(nv) else math.nan,
                  "null_sd": float(nv.std(ddof=1)) if len(nv) > 1 else math.nan,
                  "p": float((1 + np.sum(np.abs(nv - nv.mean()) >= abs(obs - nv.mean()))) / (1 + len(nv)))
                  if len(nv) and math.isfinite(obs) else math.nan}
    return out


WEEK_FORMATIONS = (7, 28)       # days of past return; every cell holds 7 days


def weekly_cells(R: np.ndarray, describe: bool = False) -> Tuple[Cells, Dict[str, Any]]:
    """Weekly-horizon cells on DAILY log returns, rebalanced every 7 days.
    Positive = momentum earned: long the past winners (and short the losers,
    for the cross-section), or follow the sign of the past return. With
    `describe`, also each cell's sample and turnover (the real call only)."""
    out: Cells = {}
    extra: Dict[str, Any] = {}
    M = market(R)
    for L in WEEK_FORMATIONS:
        rows = np.arange(L - 1, R.shape[0] - 7, 7)
        fwd = window_sum(R, 7, forward=True)
        past = window_sum(R, L)
        spread, w = tercile_book(past, fwd, rows, short_high=False)
        ok = np.isfinite(spread)
        out[f"xs_momentum_{L}d"] = float(np.mean(spread[ok]) * 1e4) if ok.any() else math.nan
        v = np.sign(past[rows]) * fwd[rows]
        okv = np.isfinite(v)
        out[f"own_trend_{L}d"] = float(np.mean(v[okv]) * 1e4) if okv.any() else math.nan
        pm, fm = window_sum(M[:, None], L)[:, 0], window_sum(M[:, None], 7, forward=True)[:, 0]
        vm = np.sign(pm[rows]) * fm[rows]
        okm = np.isfinite(vm)
        out[f"market_trend_{L}d"] = float(np.mean(vm[okm]) * 1e4) if okm.any() else math.nan
        if describe:
            traded = np.abs(np.diff(w, axis=0)).sum(axis=1)
            extra[f"xs_momentum_{L}d"] = {"n": int(ok.sum()), "traded_per_rebalance": float(np.mean(traded[ok[1:]]))
                                         if ok[1:].any() else math.nan,
                                         "coins_per_book_median": float(np.median((w[ok] != 0).sum(axis=1))) if ok.any() else math.nan}
            extra[f"own_trend_{L}d"] = {"n": int(okv.sum())}
            extra[f"market_trend_{L}d"] = {"n": int(okm.sum())}
    return out, extra


def scan_weekly(R: np.ndarray, draws: int = DRAWS, seed: int = SEED) -> Dict[str, Any]:
    """weekly_cells with the same sign-flip null, flipping whole DAYS."""
    real, extra = weekly_cells(R, describe=True)
    rng = np.random.default_rng(seed + 1)
    null: Dict[str, List[float]] = {k: [] for k in real}
    for _ in range(draws):
        eps = rng.choice([-1.0, 1.0], size=R.shape[0])
        for k, v in weekly_cells(R * eps[:, None])[0].items():
            null[k].append(v)
    cells: Dict[str, Any] = {}
    for k, obs in real.items():
        nv = np.asarray([x for x in null[k] if math.isfinite(x)])
        sd = float(nv.std(ddof=1)) if len(nv) > 1 else math.nan
        traded = extra.get(k, {}).get("traded_per_rebalance", math.nan)
        cells[k] = {"bps_per_week": obs, "null_sd": sd, **extra.get(k, {}), "traded": traded,
                    "break_even_one_way_bps": abs(obs) / traded if math.isfinite(obs) and traded and
                    math.isfinite(traded) and traded > 0 else math.nan,
                    "z": obs / sd if sd and math.isfinite(obs) else math.nan,
                    "p": float((1 + np.sum(np.abs(nv) >= abs(obs))) / (1 + len(nv))) if math.isfinite(obs) else math.nan}
    holm = S.holm({k: v["p"] for k, v in cells.items()})
    for k in cells:
        cells[k]["p_holm"] = holm[k]["p_holm"]
    return {"cells": cells, "draws": draws}


def daily_panel(daily_dir: Path, coins: Sequence[str], min_days: int = 120
                ) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """Daily candles fetched per coin (`<COIN>.json`, the venue's own rows):
    closed days only, as hourly ones are. Coins with too little history drop."""
    bars_by = {}
    now = int(time.time() * 1000)
    for c in coins:
        p = daily_dir / f"{c}.json"
        if p.exists():
            rows = D.normalise(json.loads(p.read_text()), now, interval_ms=24 * D.HOUR_MS)
            if len(rows) >= min_days:
                bars_by[c] = rows
    return panel(bars_by, min_bars=min_days)


def load_premium(funding_dir: Path, times: np.ndarray, coins: Sequence[str]
                 ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """[hour x coin] premium and funding rate at each funding time, aligned
    so the values for the hour ENDING at t sit on the row of the bar that
    closes at t -- known at that close, used only for what follows it. NaN
    where absent. Also the realised funding: what a short collected."""
    row = {int(t) + D.HOUR_MS: k for k, t in enumerate(times)}     # bar open t closes at t + 1h
    P = np.full((len(times), len(coins)), np.nan)
    Fr = np.full((len(times), len(coins)), np.nan)
    rates: List[float] = []
    per_coin: Dict[str, float] = {}
    for j, c in enumerate(coins):
        p = funding_dir / f"{c}.json"
        if not p.exists():
            continue
        rs = []
        for r in json.loads(p.read_text()):
            t = int(r["time"]) // D.HOUR_MS * D.HOUR_MS
            k = row.get(t)
            if k is not None:
                P[k, j] = float(r["premium"])
                Fr[k, j] = float(r["fundingRate"])
            rs.append(float(r["fundingRate"]))
        if rs:
            per_coin[c] = float(np.mean(rs))
            rates += rs
    return P, Fr, {"coins_with_funding": len(per_coin), "hours": len(rates),
               "mean_hourly_rate": float(np.mean(rates)) if rates else math.nan,
               "share_positive": float(np.mean(np.asarray(rates) > 0)) if rates else math.nan,
               "annualised_pct": float(np.mean(rates) * 24 * 365 * 100) if rates else math.nan,
               "per_coin_annualised_pct": {c: v * 24 * 365 * 100 for c, v in per_coin.items()}}


def fmt(x, d=2, sign=True):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def _cell_table(cells: Dict[str, Any], unit: str = "bps/period", costed: bool = True) -> List[str]:
    head = f"| cell | n | gross {unit} | null sd | z | p | Holm p |"
    L = [head + (" traded/period | break-even one-way fee (bps) |" if costed else ""),
         "|---|---|---|---|---|---|---|" + ("---|---|" if costed else "")]
    for k, v in cells.items():
        name = k + (" (diagnostic)" if v.get("diagnostic") else "")
        if "carry_bps" in v:
            name += f" [carry {fmt(v['carry_bps'])} bps; rank corr with past return {fmt(v['corr_with_past_return'])}]"
        val = v.get("bps", v.get("bps_per_week"))
        row = (f"| {name} | {v.get('n', 'n/a')} | {fmt(val)} | {fmt(v['null_sd'], 2, False)} | {fmt(v['z'])} | "
               f"{fmt(v['p'], 3, False)} | {fmt(v['p_holm'], 3, False)} |")
        if costed:
            row += f" {fmt(v.get('traded'), 2, False)} | {fmt(v.get('break_even_one_way_bps'), 2, False)} |"
        L.append(row)
    return L


def _regime_lines(r: Dict[str, Any], label: str) -> List[str]:
    if not r:
        return []
    return [f"- {label}, {r['blocks']} blocks of {r['block_hours']}: volatility rank-corr "
            f"{fmt(r['vol']['value'])} (shuffled {fmt(r['vol']['null_mean'])}, p {fmt(r['vol']['p'], 3, False)}); "
            f"trendiness {fmt(r['trendiness']['value'])} (shuffled {fmt(r['trendiness']['null_mean'])}, "
            f"p {fmt(r['trendiness']['p'], 3, False)}); trend-following after the trendiest third minus after the "
            f"least trendy third {fmt(r['trend_pays_gap_bps']['value'], 1)} bps (shuffled "
            f"{fmt(r['trend_pays_gap_bps']['null_mean'], 1)}, p {fmt(r['trend_pays_gap_bps']['p'], 3, False)})"]


def render(s: Dict[str, Any]) -> str:
    h = s["hourly"]
    L = ["# Predictability scan -- EXPLORATORY, decides nothing", "",
         f"{len(s['coins'])} coins, {s['hours']} hours of 1h bars, data fetched {s['data']['fetched_at_utc']}. "
         f"Null: {h['draws']} sign-flip draws (seed {h['seed']}). Positive = the rule named earned money: "
         f"market = follow the universe's own trend; relative = long laggards / short leaders (reversal); "
         f"shock = continue the move; premium = long the cheap / short the rich; xs_momentum = long past "
         f"winners / short losers; own_trend = each coin follows its own sign.", "",
         "## Hours to a week, on the 1h archive", ""] + _cell_table(h["cells"]) + [
         "", f"This account pays {MAKER_BPS} bps one way as maker and {TAKER_BPS} as taker."]
    if s.get("premium"):
        pr = s["premium"]
        L += ["", f"## Premium, on the {s['funding_window']['days']:.0f} days the funding history covers", ""] \
            + _cell_table(pr["cells"])
    if s.get("funding"):
        f = s["funding"]
        q = f["bps_per_hour_p0.1_p1_p50_p99_p99.9"]
        L += ["", "## Funding", "",
              f"- {f['coins']} coins, {f['coin_hours']} coin-hours: mean {fmt(f['mean_bps_per_hour'], 3)} bps/hour "
              f"({fmt(f['annualised_pct'], 1)}% a year, paid by longs to shorts); exactly the formula's fixed "
              f"interest component in {fmt(f['share_at_interest_baseline'] * 100, 1, False)}% of coin-hours; "
              f"negative in {fmt(f['share_negative'] * 100, 1, False)}%",
              f"- percentiles 0.1/1/50/99/99.9: " + " / ".join(fmt(x, 3) for x in q) + " bps/hour"]
    if s.get("carry_vs_drift"):
        c = s["carry_vs_drift"]
        L += [f"- a perp-only short held {c['days']:.0f} days, mean over {c['coins']} coins: funding collected "
              f"{fmt(c['funding_collected_pct_mean'], 2)}%, price move {fmt(c['price_move_pct_mean'], 2)}% (median "
              f"{fmt(c['price_move_pct_median'], 2)}%, sd across coins {fmt(c['price_move_sd_across_coins_pct'], 1, False)}%), "
              f"net to the short {fmt(c['short_net_pct_mean'], 2)}%; the short made money on "
              f"{fmt(c['share_of_coins_where_short_made_money'] * 100, 0, False)}% of coins"]
    if s.get("weekly"):
        w, d = s["weekly"], s["daily"]
        L += ["", f"## Weeks, on daily candles ({d['coins']} coins, {d['days']} days, {d['first']} to {d['last']})",
              ""] + _cell_table(w["cells"], unit="bps/week")
    L += ["", "## Do regimes persist?", ""] + _regime_lines(s.get("regime_hourly", {}), "1h archive") \
        + _regime_lines(s.get("regime_daily", {}), "daily candles")
    return "\n".join(L) + "\n"


def dir_digest(d: Path) -> Dict[str, Any]:
    """SHA-256 over every <COIN>.json in `d`, in name order, so a result can
    be tied to the exact files that produced it."""
    import hashlib
    h = hashlib.sha256()
    files = sorted(p for p in d.glob("*.json"))
    for p in files:
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return {"path": str(d), "files": len(files), "sha256": h.hexdigest()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--funding", type=Path, default=None)
    ap.add_argument("--daily", type=Path, default=None)
    ap.add_argument("--draws", type=int, default=DRAWS)
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    snap = D.Snapshot.load(SNAPSHOT)
    times, coins, R = panel(snap.bars)
    s: Dict[str, Any] = {"coins": coins, "hours": int(R.shape[0]),
                         "data": {k: snap.manifest.get(k) for k in ("source", "fetched_at_utc")}}
    s["hourly"] = scan(R, draws=args.draws)
    s["regime_hourly"] = regime_persistence(R)
    if args.funding is not None:
        premium, rate, _ = load_premium(args.funding, times, coins)
        rows = np.isfinite(premium).sum(axis=1) >= 9
        cover = np.flatnonzero(rows)
        lo, hi = int(cover[0]), int(cover[-1]) + 1
        # premium cells only on the hours the funding history covers; their own Holm family
        s["premium"] = scan(R[lo:hi], np.where(rows[:, None], premium, np.nan)[lo:hi], draws=args.draws,
                            rate=rate[lo:hi], only="premium")
        start, end = int(times[lo]), int(times[hi - 1]) + D.HOUR_MS
        s["funding_window"] = {"start_ms": start, "end_ms": end, "days": (end - start) / 864e5,
                               "data": dir_digest(args.funding),
                               "source": "https://api.hyperliquid.xyz/info fundingHistory (public, read-only)"}
        s["funding"] = funding_profile(args.funding, coins)
        s["carry_vs_drift"] = carry_vs_drift(args.funding, snap.bars, coins, start, end)
    if args.daily is not None:
        td, dcoins, Rd = daily_panel(args.daily, coins)
        day = lambda ms: time.strftime("%Y-%m-%d", time.gmtime(ms / 1000))   # noqa: E731
        s["daily"] = {"coins": len(dcoins), "days": int(Rd.shape[0]), "first": day(int(td[0])), "last": day(int(td[-1])),
                      "data": dir_digest(args.daily),
                      "source": "https://api.hyperliquid.xyz/info candleSnapshot 1d (public, read-only)"}
        s["weekly"] = scan_weekly(Rd, draws=args.draws)
        s["regime_daily"] = regime_persistence(Rd, block=7)
    s["runtime_sec"] = round(time.time() - t0, 1)
    (RESULTS / "summary.json").write_text(json.dumps(s, indent=1, default=str))
    (RESULTS / "report.md").write_text(render(s))
    print(render(s))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

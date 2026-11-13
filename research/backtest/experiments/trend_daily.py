"""H-TREND: daily trend-following, long and short, on a long history.

Pre-registered in research/backtest/PREREG_trend_daily.md; read that first.
Each coin is held in the direction of its own 14/28/56-day trend, sized
inversely to its 30-day volatility, and the book's net return is compared
with 2,000 control books whose trend signal has been rotated in time -- same
long/short mix, same persistence, same sizing and costs, same market; only
the alignment with what follows is broken. The difference is timing, which
is the only thing a trend rule can claim. Drift is in both and cancels.

    python -m research.backtest.experiments.trend_daily --build-archive SRC
    python -m research.backtest.experiments.trend_daily --calibrate
    python -m research.backtest.experiments.trend_daily
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from research.backtest import stats as S

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "trend_daily"
ARCHIVE = ROOT / "cache" / "daily_binance_spot"
SOURCE = ("github.com/finom/static-klines @ "
          "2a160c07c975d4e569bd5a264a9351073e6778db (Binance spot 1d klines)")

DAY_MS = 86_400_000
COINS = ("BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "AVAX", "LINK", "DOT")
MAJORS = ("BTC", "ETH")
LOOKBACKS = (14, 28, 56)
VOL_DAYS = 30
MIN_HISTORY = 120
VOL_TARGET = 0.50
CAP_GROSS_PER_COIN = 2.0          # |w| <= 2/N
BAND = 0.25
COST_BPS = 9.32                   # 4.32 taker (costs.MEASURED) + 5 assumed slippage
FUND_LONG_BPS = 3.0               # a day, paid on long notional
FUND_SHORT_BPS = 0.0              # a day, received on short notional
CONTROLS = 2000
MIN_OFFSET = 90
SEED = 20261112
BOOT_B = 10_000
BOOT_BLOCK = 28
EQUITY_USD = 716.0                # results/carry, 2026-11-05
HYPOTHESES: Dict[str, Dict[str, Any]] = {
    "T_all": {"coins": COINS, "start": "2020-01-01", "end": "2026-11-11"},
    "T_majors": {"coins": MAJORS, "start": "2018-01-01", "end": "2026-11-11"},
}


# ------------------------------------------------------------------ data --
def day_ms(iso: str) -> int:
    return int(datetime.strptime(iso, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


def iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%d")


def build_archive(src: Path, dst: Path = ARCHIVE) -> Dict[str, Any]:
    """Flatten the mirror's per-period kline files into one [t, close] series
    a coin, and write a manifest with a SHA-256 of each."""
    dst.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Any] = {"source": SOURCE, "coins": {}}
    for coin in COINS:
        rows: Dict[int, float] = {}
        for f in sorted((src / f"{coin}USDT" / "1d").glob("*.json")):
            for k in json.loads(f.read_text()):
                rows[int(k[0])] = float(k[4])
        series = [[t, rows[t]] for t in sorted(rows)]
        body = json.dumps(series, separators=(",", ":"))
        (dst / f"{coin}.json").write_text(body)
        manifest["coins"][coin] = {"days": len(series), "first": iso(int(series[0][0])),
                                   "last": iso(int(series[-1][0])),
                                   "sha256": hashlib.sha256(body.encode()).hexdigest()}
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def load(coins: Sequence[str], archive: Path = ARCHIVE) -> Tuple[np.ndarray, np.ndarray]:
    """(day timestamps, closes [day x coin]); NaN before a coin lists. The
    archive is verified against its manifest and must have no missing day."""
    manifest = json.loads((archive / "manifest.json").read_text())
    series = {}
    for c in coins:
        body = (archive / f"{c}.json").read_text()
        if hashlib.sha256(body.encode()).hexdigest() != manifest["coins"][c]["sha256"]:
            raise ValueError(f"{c}: archive does not match its manifest")
        series[c] = json.loads(body)
    t0 = min(s[0][0] for s in series.values())
    t1 = max(s[-1][0] for s in series.values())
    times = np.arange(t0, t1 + DAY_MS, DAY_MS, dtype=np.int64)
    close = np.full((len(times), len(coins)), np.nan)
    for j, c in enumerate(coins):
        idx = (np.asarray([r[0] for r in series[c]], dtype=np.int64) - t0) // DAY_MS
        close[idx, j] = [r[1] for r in series[c]]
        seen = np.isfinite(close[:, j])
        first = int(np.argmax(seen))
        if not seen[first:].all():
            raise ValueError(f"{c}: a day is missing inside its history")
    return times, close


# -------------------------------------------------------------- features --
def features(close: np.ndarray, lookbacks: Sequence[int] = LOOKBACKS
             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(trend s, annualised vol v, eligible) per day and coin, each from
    closes up to and including that day. s is the mean sign of the trailing
    return over the lookbacks; a coin is eligible from its MIN_HISTORY-th day
    once s and v both exist."""
    T, C = close.shape
    s = np.zeros((T, C))
    ok = np.ones((T, C), dtype=bool)
    for L in lookbacks:
        past = np.full((T, C), np.nan)
        past[L:] = close[:-L]
        ret = close / past - 1.0
        ok &= np.isfinite(ret)
        s += np.sign(np.nan_to_num(ret))
    s /= len(lookbacks)
    lr = np.full((T, C), np.nan)
    lr[1:] = np.log(close[1:] / close[:-1])
    v = np.full((T, C), np.nan)
    for t in range(VOL_DAYS, T):
        w = lr[t - VOL_DAYS + 1: t + 1]
        full = np.isfinite(w).all(axis=0)
        v[t, full] = w[:, full].std(axis=0, ddof=1) * math.sqrt(365.0)
    age = np.cumsum(np.isfinite(close), axis=0)
    elig = ok & np.isfinite(v) & (v > 0) & (age >= MIN_HISTORY)
    s[~elig] = np.nan
    return s, v, elig


def next_returns(close: np.ndarray) -> np.ndarray:
    """Simple return from each day's close to the next; NaN on the last."""
    r = np.full(close.shape, np.nan)
    r[:-1] = close[1:] / close[:-1] - 1.0
    return r


# ------------------------------------------------------------------ book --
def simulate(s: np.ndarray, v: np.ndarray, elig: np.ndarray, rnext: np.ndarray,
             days: np.ndarray, offsets: Sequence[int] | np.ndarray = (0,),
             cost_bps: float = COST_BPS, fund_long_bps: float = FUND_LONG_BPS,
             fund_short_bps: float = FUND_SHORT_BPS, lag: int = 0,
             detail: bool = False) -> Dict[str, np.ndarray]:
    """Run K books over the decision days `days` (row indices; each earns the
    return to the next close). Book k reads each coin's trend rotated by
    offsets[k] within that coin's own eligible stretch of `days` (0 = the
    real book). Everything else -- volatility, eligibility, N, the band,
    costs -- is read at the true day.

    Returns net [K x len(days)] and, with detail, gross / cost / funding /
    turnover / long and short P&L for book 0 and its per-coin P&L.
    """
    days = np.asarray(days)
    offs = np.asarray(offsets, dtype=np.int64)
    K, C, D = len(offs), s.shape[1], len(days)
    e = elig[days]                                          # [D x C]
    first = np.where(e.any(axis=0), e.argmax(axis=0), D)    # first eligible position
    if not all(e[first[c]:, c].all() for c in range(C) if first[c] < D):
        raise ValueError("eligibility must be contiguous once it starts")
    span = np.maximum(D - first, 1)
    sd = s[days]                                            # [D x C]
    held = np.zeros((K, C))
    net = np.zeros((K, D))
    out: Dict[str, np.ndarray] = {}
    if detail:
        for name in ("gross", "cost", "funding", "turnover", "long", "short"):
            out[name] = np.zeros(D)
        out["coin_pnl"] = np.zeros(C)
        out["trades"] = np.zeros(D)
    cols = np.arange(C)
    for p in range(D):
        live = e[p]
        n = int(live.sum())
        if n:
            src = first[None, :] + (p - first[None, :] + offs[:, None]) % span[None, :]
            sig = np.where(live[None, :], sd[np.minimum(src, D - 1), cols[None, :]], 0.0)
            sig = np.nan_to_num(sig)
            cap = CAP_GROSS_PER_COIN / n
            vol = np.where(live, v[days[p]], 1.0)
            target = np.clip(sig * (VOL_TARGET / vol)[None, :] / n, -cap, cap)
        else:
            target = np.zeros((K, C))
        move = ((np.sign(target) != np.sign(held)) | (target == 0.0)
                | (np.abs(target - held) > BAND * np.abs(target)))
        new = np.where(move, target, held)
        traded = np.abs(new - held)
        held = new
        q = p + lag
        if q >= D:
            r = np.zeros(C)
        else:
            r = np.nan_to_num(rnext[days[q]])
        gross = held @ r
        cost = traded.sum(axis=1) * cost_bps / 1e4
        fund = (np.clip(held, 0, None).sum(axis=1) * fund_long_bps
                - np.clip(-held, 0, None).sum(axis=1) * fund_short_bps) / 1e4
        net[:, p] = gross - cost - fund
        if detail:
            h = held[0]
            out["gross"][p], out["cost"][p], out["funding"][p] = gross[0], cost[0], fund[0]
            out["turnover"][p] = traded[0].sum()
            out["trades"][p] = int((traded[0] > 0).sum())
            out["long"][p] = np.clip(h, 0, None) @ r
            out["short"][p] = -np.clip(-h, 0, None) @ r
            out["coin_pnl"] += h * r
    out["net"] = net
    return out


def draw_offsets(D: int, k: int, rng: np.random.Generator) -> np.ndarray:
    if D <= 2 * MIN_OFFSET:
        raise ValueError("window too short to rotate by MIN_OFFSET either way")
    return rng.integers(MIN_OFFSET, D - MIN_OFFSET + 1, k)


def timing_test(s, v, elig, rnext, days, controls: int = CONTROLS, seed: int = SEED,
                **kw) -> Dict[str, Any]:
    """Real book against rotated controls on `days`: the excess mean daily
    net return and its two-sided rank p."""
    rng = np.random.default_rng(seed)
    offs = np.concatenate([[0], draw_offsets(len(days), controls, rng)])
    net = simulate(s, v, elig, rnext, days, offs, **kw)["net"]
    means = net.mean(axis=1)
    real, ctrl = float(means[0]), means[1:]
    ge = (1 + int((ctrl >= real).sum())) / (controls + 1)
    le = (1 + int((ctrl <= real).sum())) / (controls + 1)
    return {"days": int(len(days)), "real_bps_day": real * 1e4,
            "control_bps_day": float(ctrl.mean()) * 1e4,
            "control_sd_bps_day": float(ctrl.std(ddof=1)) * 1e4,
            "excess_bps_day": (real - float(ctrl.mean())) * 1e4,
            "z": (real - float(ctrl.mean())) / float(ctrl.std(ddof=1)),
            "p": min(1.0, 2 * min(ge, le)), "controls": controls}


# ----------------------------------------------------------------- stats --
def sharpe(x: np.ndarray) -> float:
    sd = x.std(ddof=1)
    return float(x.mean() / sd * math.sqrt(365.0)) if sd > 0 else float("nan")


def block_bootstrap_sharpe(x: np.ndarray, B: int = BOOT_B, block: int = BOOT_BLOCK,
                           seed: int = SEED) -> Dict[str, float]:
    """Circular block bootstrap: 95% percentile CI for the annualised Sharpe."""
    rng = np.random.default_rng(seed)
    n = len(x)
    nb = math.ceil(n / block)
    draws = np.empty(B)
    step = np.arange(block)
    for b in range(B):
        starts = rng.integers(0, n, nb)
        idx = ((starts[:, None] + step[None, :]) % n).ravel()[:n]
        draws[b] = sharpe(x[idx])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"sharpe": sharpe(x), "ci_lo": float(lo), "ci_hi": float(hi)}


def drawdown(x: np.ndarray) -> float:
    eq = np.cumprod(1.0 + x)
    return float((eq / np.maximum.accumulate(eq) - 1.0).min())


def describe(x: np.ndarray) -> Dict[str, float]:
    return {"ann_return_pct": float(x.mean() * 365 * 100),
            "ann_vol_pct": float(x.std(ddof=1) * math.sqrt(365) * 100),
            "sharpe": sharpe(x), "max_drawdown_pct": drawdown(x) * 100,
            "total_return_pct": float((np.cumprod(1 + x)[-1] - 1) * 100)}


# -------------------------------------------------------------- the test --
def window(times: np.ndarray, start: str, end: str) -> np.ndarray:
    """Decision days whose P&L day (the next) lies in [start, end]."""
    pnl_day = times + DAY_MS
    return np.where((pnl_day >= day_ms(start)) & (pnl_day <= day_ms(end)))[0]


def run_hypothesis(name: str, archive: Path = ARCHIVE, controls: int = CONTROLS
                   ) -> Dict[str, Any]:
    spec = HYPOTHESES[name]
    coins = list(spec["coins"])
    times, close = load(coins, archive)
    s, v, elig = features(close)
    rnext = next_returns(close)
    days = window(times, spec["start"], spec["end"])
    days = days[np.isfinite(rnext[days]).any(axis=1)]
    half = len(days) // 2
    out: Dict[str, Any] = {"coins": coins, "first_pnl_day": iso(int(times[days[0]]) + DAY_MS),
                           "last_pnl_day": iso(int(times[days[-1]]) + DAY_MS),
                           "split_day": iso(int(times[days[half]]) + DAY_MS)}
    out["test"] = timing_test(s, v, elig, rnext, days, controls)
    out["halves"] = [timing_test(s, v, elig, rnext, d, controls)
                     for d in (days[:half], days[half:])]

    real = simulate(s, v, elig, rnext, days, detail=True)
    net = real["net"][0]
    out["net"] = describe(net)
    out["net"].update({k: v_ for k, v_ in block_bootstrap_sharpe(net).items() if k != "sharpe"})
    out["gross"] = describe(real["gross"])
    out["halves_net"] = [describe(net[:half]), describe(net[half:])]
    years = np.asarray([int(iso(int(times[d]) + DAY_MS)[:4]) for d in days])
    out["by_year"] = {int(y): {"net_pct": float((np.cumprod(1 + net[years == y])[-1] - 1) * 100),
                               "sharpe": sharpe(net[years == y]),
                               "max_drawdown_pct": drawdown(net[years == y]) * 100}
                      for y in sorted(set(years))}
    yrs = len(days) / 365.0
    out["activity"] = {"turnover_x_per_year": float(real["turnover"].sum() / yrs),
                       "trades_per_year": float(real["trades"].sum() / yrs),
                       "cost_pct_per_year": float(real["cost"].sum() / yrs * 100),
                       "funding_pct_per_year": float(real["funding"].sum() / yrs * 100)}
    out["legs_pct_per_year"] = {"long": float(real["long"].sum() / yrs * 100),
                                "short": float(real["short"].sum() / yrs * 100)}
    out["coin_pct_per_year"] = {c: float(real["coin_pnl"][j] / yrs * 100)
                                for j, c in enumerate(coins)}

    def alt(lookbacks=LOOKBACKS, **kw) -> Dict[str, float]:
        s2, v2, e2 = features(close, lookbacks)
        d2 = days[e2[days].any(axis=1)] if lookbacks != LOOKBACKS else days
        x = simulate(s2, v2, e2, rnext, d2, **kw)["net"][0]
        return {"net_bps_day": float(x.mean() * 1e4), "sharpe": sharpe(x),
                "ann_return_pct": float(x.mean() * 365 * 100)}

    out["each_lookback"] = {str(L): alt((L,)) for L in LOOKBACKS}
    out["neighbours"] = {str(L): alt((L,)) for L in (7, 112)}
    out["sensitivity"] = {"one_day_delay": alt(lag=1), "cost_20bps": alt(cost_bps=20.0),
                          "fee_only_4.32bps": alt(cost_bps=4.32),
                          "shorts_credited_funding": alt(fund_short_bps=FUND_LONG_BPS),
                          "no_funding": alt(fund_long_bps=0.0)}
    out["capital"] = {"equity_usd": EQUITY_USD,
                      "usd_per_year": EQUITY_USD * out["net"]["ann_return_pct"] / 100}
    return out


def verdict(res: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    holm = S.holm({k: r["test"]["p"] for k, r in res.items()})
    v: Dict[str, Any] = {}
    for k, r in res.items():
        edge = (holm[k]["p_holm"] < 0.05 and r["test"]["excess_bps_day"] > 0
                and all(h["excess_bps_day"] > 0 for h in r["halves"]))
        v[k] = {"p": r["test"]["p"], "p_holm": holm[k]["p_holm"], "edge": bool(edge)}
    a = res["T_all"]
    checks = {
        "edge_on_T_all": v["T_all"]["edge"],
        "sharpe_at_least_0.7": a["net"]["sharpe"] >= 0.7,
        "sharpe_ci_above_0": a["net"]["ci_lo"] > 0,
        "both_halves_net_positive": all(h["total_return_pct"] > 0 for h in a["halves_net"]),
        "each_lookback_net_positive": all(x["net_bps_day"] > 0 for x in a["each_lookback"].values()),
        "survives_one_day_delay": a["sensitivity"]["one_day_delay"]["net_bps_day"] > 0,
        "survives_20bps_costs": a["sensitivity"]["cost_20bps"]["net_bps_day"] > 0,
    }
    v["forward_trial_checks"] = checks
    v["worth_a_forward_trial"] = bool(all(checks.values()))
    return v


# ----------------------------------------------------------- calibration --
def null_archive(close: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Drift without trend: each day's log returns, all coins together, get
    one random sign; each coin's own mean daily log return is added back."""
    lr = np.log(close[1:] / close[:-1])
    mean = np.nanmean(lr, axis=0)
    flip = rng.choice([-1.0, 1.0], size=len(lr))
    new = flip[:, None] * lr + mean[None, :]
    out = np.full(close.shape, np.nan)
    for j in range(close.shape[1]):
        seen = np.where(np.isfinite(close[:, j]))[0]
        a = int(seen[0])
        out[a, j] = close[a, j]
        out[a + 1:, j] = close[a, j] * np.exp(np.cumsum(new[a:, j]))
    return out


def calibrate(archive: Path = ARCHIVE, n: int = 20, controls: int = CONTROLS,
              seed: int = SEED) -> Dict[str, Any]:
    spec = HYPOTHESES["T_all"]
    times, close = load(list(spec["coins"]), archive)
    days0 = window(times, spec["start"], spec["end"])
    rng = np.random.default_rng(seed + 1)
    runs = []
    for i in range(n):
        fake = null_archive(close, rng)
        s, v, elig = features(fake)
        rnext = next_returns(fake)
        days = days0[np.isfinite(rnext[days0]).any(axis=1)]
        t = timing_test(s, v, elig, rnext, days, controls, seed=seed + 100 + i)
        runs.append({"dataset": i, "excess_bps_day": t["excess_bps_day"], "p": t["p"],
                     "real_bps_day": t["real_bps_day"], "control_bps_day": t["control_bps_day"]})
    ex = np.asarray([r["excess_bps_day"] for r in runs])
    se = float(ex.std(ddof=1) / math.sqrt(n))
    fired = int(sum(r["p"] < 0.05 for r in runs))
    return {"datasets": n, "fired": fired, "mean_excess_bps_day": float(ex.mean()),
            "se": se, "passed": bool(fired <= 2 and abs(ex.mean()) <= 2 * se), "runs": runs}


# ---------------------------------------------------------------- report --
def f(x: float, d: int = 2, sign: bool = True) -> str:
    if x is None or not math.isfinite(x):
        return "n/a"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def render(res: Dict[str, Dict[str, Any]], v: Dict[str, Any], cal: Optional[Dict[str, Any]]) -> str:
    L = ["# H-TREND: daily trend-following, long and short (generated)", "",
         f"Data: {SOURCE}. Costs {COST_BPS} bps per unit traded; longs pay "
         f"{FUND_LONG_BPS} bps a day funding; shorts receive {FUND_SHORT_BPS}.", ""]
    if cal:
        L += [f"Null calibration: {cal['fired']} of {cal['datasets']} fired; mean excess "
              f"{f(cal['mean_excess_bps_day'])} +/- {f(cal['se'], sign=False)} bps a day; "
              f"{'PASSED' if cal['passed'] else 'FAILED'}.", ""]
    L += ["## The pre-registered test (timing: real book minus rotated controls)", "",
          "| | P&L days | real bps/day | controls bps/day | excess | z | p | Holm p | halves' excess | edge |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for k, r in res.items():
        t = r["test"]
        L.append(f"| {k} | {t['days']} | {f(t['real_bps_day'])} | {f(t['control_bps_day'])} | "
                 f"{f(t['excess_bps_day'])} | {f(t['z'])} | {f(t['p'], 4, False)} | "
                 f"{f(v[k]['p_holm'], 4, False)} | "
                 f"{f(r['halves'][0]['excess_bps_day'])}, {f(r['halves'][1]['excess_bps_day'])} | "
                 f"{'YES' if v[k]['edge'] else 'no'} |")
    L += ["", "## Forward-trial checks (T_all)", ""]
    L += [f"- {k}: {'pass' if ok else 'FAIL'}" for k, ok in v["forward_trial_checks"].items()]
    L += ["", f"**Worth a forward trial: {'YES' if v['worth_a_forward_trial'] else 'NO'}**", ""]
    for k, r in res.items():
        n = r["net"]
        L += [f"## {k} as a strategy, net ({r['first_pnl_day']} to {r['last_pnl_day']})", "",
              f"- {f(n['ann_return_pct'], 1)}% a year at {f(n['ann_vol_pct'], 1, False)}% volatility; "
              f"Sharpe {f(n['sharpe'])} (95% CI {f(n['ci_lo'])} to {f(n['ci_hi'])}); "
              f"worst drawdown {f(n['max_drawdown_pct'], 1)}%; gross Sharpe {f(r['gross']['sharpe'])}.",
              f"- Halves (split {r['split_day']}): Sharpe {f(r['halves_net'][0]['sharpe'])} then "
              f"{f(r['halves_net'][1]['sharpe'])}; total {f(r['halves_net'][0]['total_return_pct'], 1)}% "
              f"then {f(r['halves_net'][1]['total_return_pct'], 1)}%.",
              f"- Activity: {f(r['activity']['trades_per_year'], 0, False)} trades and "
              f"{f(r['activity']['turnover_x_per_year'], 1, False)}x turnover a year; costs "
              f"{f(r['activity']['cost_pct_per_year'], 2, False)}% and funding "
              f"{f(r['activity']['funding_pct_per_year'], 2, False)}% a year.",
              f"- Legs, gross % a year: long {f(r['legs_pct_per_year']['long'], 1)}, "
              f"short {f(r['legs_pct_per_year']['short'], 1)}.",
              "- By coin, gross % a year: " + ", ".join(
                  f"{c} {f(x, 1)}" for c, x in r["coin_pct_per_year"].items()) + ".",
              "- Lookbacks alone, net Sharpe: " + ", ".join(
                  f"{L_}d {f(x['sharpe'])}" for L_, x in
                  {**r["neighbours"], **r["each_lookback"]}.items()) + ".",
              "- Sensitivity, net Sharpe: " + ", ".join(
                  f"{name} {f(x['sharpe'])}" for name, x in r["sensitivity"].items()) + ".",
              f"- Capital check: ${f(r['capital']['usd_per_year'], 0)} a year on "
              f"${r['capital']['equity_usd']:.0f}.", "",
              "| year | net % | Sharpe | worst drawdown % |", "|---|---|---|---|"]
        L += [f"| {y} | {f(x['net_pct'], 1)} | {f(x['sharpe'])} | {f(x['max_drawdown_pct'], 1)} |"
              for y, x in r["by_year"].items()]
        L.append("")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-archive", metavar="SRC")
    ap.add_argument("--calibrate", action="store_true")
    a = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    if a.build_archive:
        print(json.dumps(build_archive(Path(a.build_archive)), indent=2))
        return 0
    cal_path = RESULTS / "null_calibration.json"
    if a.calibrate:
        cal = calibrate()
        cal_path.write_text(json.dumps(cal, indent=2))
        print(json.dumps({k: v for k, v in cal.items() if k != "runs"}, indent=2))
        return 0 if cal["passed"] else 1
    if not cal_path.exists() or not json.loads(cal_path.read_text())["passed"]:
        print("null calibration has not passed; run --calibrate first")
        return 1
    t0 = time.time()
    res = {k: run_hypothesis(k) for k in HYPOTHESES}
    v = verdict(res)
    manifest = json.loads((ARCHIVE / "manifest.json").read_text())
    (RESULTS / "summary.json").write_text(json.dumps(
        {"source": SOURCE, "archive": manifest, "seed": SEED, "results": res,
         "verdict": v, "seconds": round(time.time() - t0, 1)}, indent=2))
    report = render(res, v, json.loads(cal_path.read_text()))
    (RESULTS / "report.md").write_text(report)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

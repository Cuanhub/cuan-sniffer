"""Pre-registered experiment: does a trendline breakout add edge?

Implements research/backtest/PREREG_trendline_breakout.md exactly. Every
constant below is named there; changing one is a new experiment, not a tweak.

    python -m research.backtest.experiments.trendline_breakout --null     # 1. calibrate on random walks
    python -m research.backtest.experiments.trendline_breakout --fetch    # 2. fetch real data, run
    python -m research.backtest.experiments.trendline_breakout            #    rerun on the saved snapshot

Writes research/backtest/results/trendline_breakout/. The real-data snapshot
lives in research/backtest/cache/ (not committed); its per-coin SHA-256
digests are in summary.json, so a rerun can prove it used the same bars.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from research.backtest import band
from research.backtest import data as D
from research.backtest import stats as S
from research.backtest import trendline as TL
from research.backtest.costs import MEASURED, Costs
from research.backtest.engine import (LIMIT, MARKET_CLOSE, MARKET_NEXT_OPEN,
                                      Order, Trade, atr_pct_series, simulate)

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"
RESULTS = ROOT / "results" / "trendline_breakout"
SNAPSHOT = CACHE / "snapshot_1h.json"

PRIMARY = TL.TLParams()            # k=3, lookback=120, span 6, touches 2
H1_WINDOW = 24
H2_WAIT = 48
H2_REPS = 100                      # random-event replicates per coin
H2_REPS_GRID = 30                  # ...for robustness variants
H2_MIN_RR = 2.0                    # secondary: production's rr floor re-applied
H3_STOP_ATR = 1.5
H3_RR = 2.0
CONTROL_DRAWS = 200
CONTROL_SPAN = 240
SEED = 20261027
WARMUP = band.Z.ZONE_MIN_BARS - 1  # first bar the baseline can decide at
MIN_CONTROL = 50                   # resolved control draws needed per trade
ENTRY_SLIP_SENSITIVITY = 0.0564    # median real stop slippage, as a proxy
NULL_DATASETS = 20
NULL_COINS = 12
NULL_BARS = 5000
NULL_MAX_FALSE = 2                 # of 20; more marks a framing miscalibrated

_BARS: Dict[str, List[Dict[str, Any]]] = {}


# ------------------------------------------------------------------ data --
def load_snapshot(fetch: bool) -> D.Snapshot:
    if fetch or not SNAPSHOT.exists():
        from research.levels.universe import select_universe
        from research.levels.venue_client import HyperliquidReadOnlyClient
        client = HyperliquidReadOnlyClient()
        universe = [str(r.get("symbol") or r.get("coin")).upper()
                    for r in select_universe(client)]
        print(f"[DATA] live universe: {len(universe)} coins")
        snap = D.fetch(universe, client=client)
        snap.save(SNAPSHOT)
        print(f"[DATA] saved {SNAPSHOT}")
        return snap
    snap = D.Snapshot.load(SNAPSHOT)       # verifies every digest
    print(f"[DATA] loaded {SNAPSHOT} (fetched {snap.manifest['fetched_at_utc']}, "
          f"digests verified)")
    return snap


def random_walks(seed: int, coins: int = NULL_COINS, n: int = NULL_BARS
                 ) -> Dict[str, List[Dict[str, Any]]]:
    """Driftless random walks for the null calibration -- and ONLY for it.

    Exactly a martingale in price (log-drift -sigma^2/2), 12 sub-steps per
    bar so highs and lows are path extremes rather than endpoints. These
    contain no market information; anything a framing "finds" in them is a
    false positive by construction.
    """
    rng = np.random.default_rng([SEED, seed, 99])
    out = {}
    for c in range(coins):
        sigma_bar = rng.uniform(0.006, 0.016)
        sub = 12
        s = sigma_bar / math.sqrt(sub)
        r = rng.normal(-s * s / 2, s, n * sub)
        path = 100.0 * np.exp(np.cumsum(r))
        bars, prev = [], 100.0
        for k in range(n):
            seg = path[k * sub:(k + 1) * sub]
            bars.append({"t": 1_700_006_400_000 + k * D.HOUR_MS, "o": prev,
                         "h": float(max(prev, seg.max())),
                         "l": float(min(prev, seg.min())), "c": float(seg[-1]),
                         "v": 1.0, "n": 1})
            prev = float(seg[-1])
        out[f"NULL{seed}_{c}"] = bars
    return out


def _init(bars: Dict[str, List[Dict[str, Any]]]) -> None:
    global _BARS
    _BARS = bars


def _pool(bars):
    return ProcessPoolExecutor(max_workers=8, initializer=_init, initargs=(bars,))


# -------------------------------------------------------------- baseline --
def _replay(coin: str) -> List[Trade]:
    return band.run(coin, _BARS[coin], MEASURED)


def _replay_gated(args) -> List[Trade]:
    coin, is_break, window = args

    def gate(o: Order) -> bool:
        return TL.last_break_before(is_break, o.decided_i, window) is not None
    return band.run(coin, _BARS[coin], MEASURED, gate=gate)


# ------------------------------------------------------------ trendlines --
def _breaks_1h(args) -> Tuple[str, List[bool]]:
    coin, p, tf = args
    bars = _BARS[coin]
    if tf == "1h":
        return coin, TL.breaks(bars, p)[0]
    # 4h line from complete groups of real 1h bars, mapped back onto the 1h
    # bar whose close IS the 4h close -- the moment the 4h break is known.
    four = D.resample(bars, 4)
    br4, _ = TL.breaks(four, p)
    pos = {b["t"]: i for i, b in enumerate(bars)}
    out = [False] * len(bars)
    for q, hit in enumerate(br4):
        if hit:
            i = pos.get(four[q]["t"] + 3 * D.HOUR_MS)
            if i is not None:
                out[i] = True
    return coin, out


def all_breaks(pool, coins, p: TL.TLParams, tf: str = "1h") -> Dict[str, List[bool]]:
    return dict(pool.map(_breaks_1h, [(c, p, tf) for c in coins]))


# -------------------------------------------------------------- controls --
def _rng(coin: str, i: int, salt: int) -> np.random.Generator:
    return np.random.default_rng([SEED, zlib.crc32(coin.encode()), i, salt])


def _draw_bars(n: int, centre: int, rng) -> np.ndarray:
    lo = max(WARMUP, centre - CONTROL_SPAN)
    hi = min(n - 2, centre + CONTROL_SPAN)
    return rng.integers(lo, hi + 1, CONTROL_DRAWS)


def _matched_control(args) -> Tuple[int, Optional[float], int]:
    """H1 control for one band trade: random market entries on the same coin
    with the same risk %, reward % and entry fee type."""
    idx, coin, entry_i, risk_pct, reward_pct, maker = args
    bars = _BARS[coin]
    rs = []
    for u in _draw_bars(len(bars), entry_i, _rng(coin, entry_i, 1)):
        c = bars[u]["c"]
        o = Order(coin, int(u), MARKET_CLOSE, c * (1 - risk_pct / 100),
                  c * (1 + reward_pct / 100),
                  entry_fee="maker" if maker else "taker")
        tr = simulate(bars, o, MEASURED)
        if tr is not None and tr.resolved:
            rs.append(tr.r_net)
    if len(rs) < MIN_CONTROL:
        return idx, None, len(rs)
    return idx, float(np.mean(rs)), len(rs)


def _h3_order(coin, bars, atr, t, tag="h3") -> Optional[Order]:
    c = bars[t]["c"]
    stop = c * (1 - H3_STOP_ATR * atr[t] / 100)
    if not 0 < stop < c:
        return None
    return Order(coin, t, MARKET_NEXT_OPEN, stop, c + H3_RR * (c - stop), tag=tag)


def _h3_coin(args) -> List[Dict[str, Any]]:
    """H3 breakout trades on one coin, each with its random-entry control."""
    coin, is_break, costs = args
    bars = _BARS[coin]
    atr = atr_pct_series(bars)
    out = []
    t = WARMUP
    n = len(bars)
    while t < n - 1:
        if not is_break[t]:
            t += 1
            continue
        o = _h3_order(coin, bars, atr, t)
        tr = simulate(bars, o, costs) if o else None
        if tr is None:
            t += 1
            continue
        ctrl = []
        for u in _draw_bars(n, t, _rng(coin, t, 3)):
            co = _h3_order(coin, bars, atr, int(u), tag="h3_ctrl")
            ctr = simulate(bars, co, costs) if co else None
            if ctr is not None and ctr.resolved:
                ctrl.append(ctr.r_net)
        out.append({"trade": tr, "ctrl_mean": float(np.mean(ctrl))
                    if len(ctrl) >= MIN_CONTROL else None, "ctrl_n": len(ctrl)})
        if tr.exit_i is None:
            break
        t = max(tr.exit_i, t + 1)
    return out


# -------------------------------------------------------------------- H2 --
def confirm(bars, is_break, base: Trade, wait: int, costs: Costs,
            min_rr: float = 0.0) -> Tuple[str, Optional[Trade]]:
    """The confirmation arm for one baseline setup: (status, trade)."""
    arm = base.entry_i
    n = len(bars)
    for t in range(arm, min(arm + wait + 1, n - 1)):
        # The decision close of a market setup came after its own low, so
        # that low cannot invalidate a setup armed AT the close. A bid's fill
        # bar is different: its low came after the fill.
        if bars[t]["l"] <= base.stop and not (base.kind == MARKET_CLOSE and t == arm):
            return "stopped_first", None
        if is_break[t]:
            o = Order(base.coin, t, MARKET_NEXT_OPEN, base.stop, base.tp,
                      tag="h2", features=base.features)
            tr = simulate(bars, o, costs)
            if tr is None:
                return "invalid_at_entry", None
            if min_rr and tr.planned_rr < min_rr:
                return "rr_below_floor", None
            return "entered", tr
    return "no_break", None


def _shift(series: List[bool], offset: int) -> List[bool]:
    return series[-offset:] + series[:-offset]


def _h2_coin(args) -> List[Dict[str, Any]]:
    """Every setup on one coin under the trendline arm and the random-event
    arm (the same break series circularly shifted, `reps` offsets)."""
    coin, is_break, trades, wait, costs, reps, min_rr = args
    bars = _BARS[coin]
    n = len(bars)
    rng = np.random.default_rng([SEED, zlib.crc32(coin.encode()), 7])
    shifted = [_shift(is_break, int(o)) for o in rng.integers(wait, n - wait, reps)]
    rows = []
    for b in trades:
        status, tr = confirm(bars, is_break, b, wait, costs, min_rr)
        vals, entered = [], 0
        for sh in shifted:
            _, tr2 = confirm(bars, sh, b, wait, costs, min_rr)
            if tr2 is not None and not tr2.resolved:
                continue
            vals.append(tr2.r_net if tr2 is not None else 0.0)
            entered += tr2 is not None
        rows.append({"base": b, "status": status, "conf": tr,
                     "ctrl": float(np.mean(vals)) if vals else None,
                     "ctrl_entry_rate": entered / len(vals) if vals else None})
    return rows


def h2_rows(pool, baseline, breaks_by, wait=H2_WAIT, costs=MEASURED,
            reps=H2_REPS, min_rr=0.0) -> List[Dict[str, Any]]:
    jobs = [(c, breaks_by[c], baseline[c], wait, costs, reps, min_rr)
            for c in baseline]
    return [r for rows in pool.map(_h2_coin, jobs) for r in rows]


def h2_paired(rows):
    """Per resolved setup: trendline arm, random arm, baseline (net R)."""
    out = {"d_primary": [], "d_vs_base": [], "d_random_vs_base": [],
           "cl": [], "tm": [], "dropped_open": 0}
    for r in rows:
        b, c = r["base"], r["conf"]
        if not b.resolved or (c is not None and not c.resolved) or r["ctrl"] is None:
            out["dropped_open"] += 1
            continue
        arm = c.r_net if c is not None else 0.0
        out["d_primary"].append(arm - r["ctrl"])
        out["d_vs_base"].append(arm - b.r_net)
        out["d_random_vs_base"].append(r["ctrl"] - b.r_net)
        out["cl"].append(S.week_of(b.entry_t))
        out["tm"].append(b.entry_t)
    return out


# -------------------------------------------------------------------- H1 --
def h1_split(baseline, breaks_by, window: int):
    vals, grp, cl, tm, atrx, kinds = [], [], [], [], [], []
    for coin, trades in baseline.items():
        br = breaks_by[coin]
        for b in trades:
            if not b.resolved:
                continue
            vals.append(b.r_net)
            grp.append(int(TL.last_break_before(br, b.decided_i, window) is not None))
            cl.append(S.week_of(b.entry_t))
            tm.append(b.entry_t)
            atrx.append(float(b.features.get("stop_atr_x") or 0.0))
            kinds.append(b.kind)
    return vals, grp, cl, tm, atrx, kinds


def _mean(xs):
    return float(np.mean(xs)) if len(xs) else float("nan")


def _halves(values, times):
    order = np.argsort(np.asarray(times))
    v = np.asarray(values, float)[order]
    h = len(v) // 2
    return [list(v[:h]), list(v[h:])]


# --------------------------------------------------------------- analyse --
def analyse(pool, bars_by, grid: bool = True, log=print, B: int = 10_000,
            secondaries: bool = True) -> Dict[str, Any]:
    """Every framing on one dataset. `secondaries=False` skips the analyses
    that do not feed the decision (the null calibration's 20 runs)."""
    coins = sorted(bars_by)
    t0 = time.time()
    s: Dict[str, Any] = {}

    base_lists = list(pool.map(_replay, coins))
    baseline = dict(zip(coins, base_lists))
    allb = [t for ts in base_lists for t in ts]
    res = [t for t in allb if t.resolved]
    s["baseline"] = {
        **S.describe([t.r_net for t in res]),
        "trades": len(allb), "open_at_end": len(allb) - len(res),
        "limit_entries": sum(t.kind == LIMIT for t in res),
        "market_entries": sum(t.kind == MARKET_CLOSE for t in res),
        "ambiguous_bars": sum(t.ambiguous for t in res),
        "fill_bar_tp_not_counted": sum(t.fill_bar_tp for t in res),
        "gross_mean_r": _mean([t.r_gross for t in res]),
        "mean_cost_r": _mean([t.cost_r for t in res]),
        "median_planned_rr": float(np.median([t.planned_rr for t in res])) if res else None,
        "median_risk_pct": float(np.median([t.risk_pct for t in res])) if res else None,
    }
    log(f"[BASE] {len(res)} resolved band trades, mean net R "
        f"{fmt(s['baseline'].get('mean'))}  ({time.time()-t0:.0f}s)")

    br = all_breaks(pool, coins, PRIMARY)
    s["breaks_per_1000_bars"] = float(np.mean(
        [sum(br[c]) / len(bars_by[c]) * 1000 for c in coins]))

    # ------------------------------------------------------------- H1 --
    vals, grp, cl, tm, atrx, kinds = h1_split(baseline, br, H1_WINDOW)
    g1 = [v for v, g in zip(vals, grp) if g]
    g0 = [v for v, g in zip(vals, grp) if not g]
    h1: Dict[str, Any] = {
        "with_break": S.describe(g1), "without_break": S.describe(g0),
        "welch_iid": S.welch(g1, g0),
        "cluster": S.cluster_bootstrap(vals, grp, cl, "diff", B=B),
        "split_half": S.split_half(vals, grp, tm),
        "limit_share": {"with_break": _mean([k == LIMIT for k, g in zip(kinds, grp) if g]),
                        "without_break": _mean([k == LIMIT for k, g in zip(kinds, grp) if not g])}}
    med = float(np.median(atrx)) if atrx else 0.0
    strata: Dict[str, Any] = {"median_stop_atr_x": med}
    for name, keep in (("below", lambda x: x < med), ("at_or_above", lambda x: x >= med)):
        s1 = [v for v, g, x in zip(vals, grp, atrx) if keep(x) and g]
        s0 = [v for v, g, x in zip(vals, grp, atrx) if keep(x) and not g]
        strata[name] = {"diff": _mean(s1) - _mean(s0), "n1": len(s1), "n0": len(s0)}
    h1["stop_atr_strata"] = strata

    jobs: List[Tuple[Any, ...]] = []
    keyed: List[Tuple[str, Trade]] = []
    for coin, trades in baseline.items():
        for b in trades:
            if b.resolved:
                jobs.append((len(keyed), coin, b.entry_i, b.risk_pct,
                             (b.tp - b.entry_px) / b.entry_px * 100, b.kind == LIMIT))
                keyed.append((coin, b))
    ctrl = {idx: m for idx, m, _ in pool.map(_matched_control, jobs, chunksize=8)}
    ex_v, ex_g, ex_c = [], [], []
    for idx, (coin, b) in enumerate(keyed):
        if ctrl[idx] is None:
            continue
        ex_v.append(b.r_net - ctrl[idx])
        ex_g.append(int(TL.last_break_before(br[coin], b.decided_i, H1_WINDOW) is not None))
        ex_c.append(S.week_of(b.entry_t))
    h1["matched_control"] = {
        "excess_diff": _mean([v for v, g in zip(ex_v, ex_g) if g])
        - _mean([v for v, g in zip(ex_v, ex_g) if not g]),
        "cluster": S.cluster_bootstrap(ex_v, ex_g, ex_c, "diff", B=B),
        "without_enough_control": sum(1 for m in ctrl.values() if m is None),
        "baseline_mean_excess": _mean(ex_v)}
    if secondaries:
        gated = list(pool.map(_replay_gated, [(c, br[c], H1_WINDOW) for c in coins]))
        h1["filtered_strategy"] = S.describe(
            [t.r_net for ts in gated for t in ts if t.resolved])
    s["H1"] = h1
    log(f"[H1] with {len(g1)} / without {len(g0)}: diff {fmt(h1['cluster']['estimate'])}  "
        f"p={fmt(h1['cluster']['p'], 4, False)}  ({time.time()-t0:.0f}s)")

    # ------------------------------------------------------------- H2 --
    rows = h2_rows(pool, baseline, br)
    pr = h2_paired(rows)
    status: Dict[str, int] = {}
    for r in rows:
        status[r["status"]] = status.get(r["status"], 0) + 1
    conf_tr = [r["conf"] for r in rows if r["conf"] is not None and r["conf"].resolved]
    brk = {f"base_{bw}_conf_{cw}": 0 for bw in ("win", "loss")
           for cw in ("skipped", "win", "loss")}
    for r in rows:
        b, c = r["base"], r["conf"]
        if not b.resolved or (c is not None and not c.resolved):
            continue
        bw = "win" if b.r_net > 0 else "loss"
        cw = "skipped" if c is None else ("win" if c.r_net > 0 else "loss")
        brk[f"base_{bw}_conf_{cw}"] += 1
    zeros = lambda xs: [0] * len(xs)                          # noqa: E731
    h2: Dict[str, Any] = {
        "setups": len(rows), "status": status, "dropped_open": pr["dropped_open"],
        "paired": S.describe(pr["d_primary"]),
        "cluster": S.cluster_bootstrap(pr["d_primary"], zeros(pr["d_primary"]),
                                       pr["cl"], "mean", B=B),
        "split_half": [S.describe(x) for x in _halves(pr["d_primary"], pr["tm"])],
        "vs_baseline": S.cluster_bootstrap(pr["d_vs_base"], zeros(pr["d_vs_base"]),
                                           pr["cl"], "mean", B=B),
        "random_arm_vs_baseline": S.cluster_bootstrap(
            pr["d_random_vs_base"], zeros(pr["d_random_vs_base"]), pr["cl"], "mean", B=B),
        "entry_rate": {"trendline_arm": status.get("entered", 0) / max(1, len(rows)),
                       "random_arm": _mean([r["ctrl_entry_rate"] for r in rows
                                            if r["ctrl_entry_rate"] is not None])},
        "confirmation_trades": S.describe([t.r_net for t in conf_tr]),
        "breakdown": brk,
        "median_rr_at_confirmation": float(np.median([t.planned_rr for t in conf_tr])) if conf_tr else None}
    if secondaries:
        pr_rr = h2_paired(h2_rows(pool, baseline, br, min_rr=H2_MIN_RR))
        h2["with_rr_floor"] = S.cluster_bootstrap(
            pr_rr["d_primary"], zeros(pr_rr["d_primary"]), pr_rr["cl"], "mean", B=B)
        slip = replace(MEASURED, market_entry_slip_pct=ENTRY_SLIP_SENSITIVITY)
        pr_sl = h2_paired(h2_rows(pool, baseline, br, costs=slip))
        h2["entry_slip_sensitivity"] = {"slip_pct": ENTRY_SLIP_SENSITIVITY,
                                        "primary": S.describe(pr_sl["d_primary"]),
                                        "vs_baseline": S.describe(pr_sl["d_vs_base"])}
    s["H2"] = h2
    log(f"[H2] {len(pr['d_primary'])} setups: trendline minus random-event "
        f"{fmt(h2['cluster']['estimate'])}  p={fmt(h2['cluster']['p'], 4, False)}  "
        f"({time.time()-t0:.0f}s)")

    # ------------------------------------------------------------- H3 --
    h3_all = [x for xs in pool.map(_h3_coin, [(c, br[c], MEASURED) for c in coins]) for x in xs]
    h3_res = [x for x in h3_all if x["trade"].resolved]
    good = [x for x in h3_res if x["ctrl_mean"] is not None]
    ex = [x["trade"].r_net - x["ctrl_mean"] for x in good]
    h3 = {"trades": len(h3_all), "resolved": len(h3_res),
          "absolute": S.describe([x["trade"].r_net for x in h3_res]),
          "control_mean_r": _mean([x["ctrl_mean"] for x in good]),
          "excess": S.describe(ex),
          "cluster": S.cluster_bootstrap(ex, [0] * len(ex),
                                         [S.week_of(x["trade"].entry_t) for x in good],
                                         "mean", B=B),
          "split_half": [S.describe(x) for x in _halves(ex, [x["trade"].entry_t for x in good])],
          "without_enough_control": len(h3_res) - len(good),
          "ambiguous_bars": sum(x["trade"].ambiguous for x in h3_res)}
    s["H3"] = h3
    log(f"[H3] {len(h3_res)} breakout trades: absolute {fmt(h3['absolute'].get('mean'))}, "
        f"excess over random {fmt(h3['cluster']['estimate'])}  "
        f"p={fmt(h3['cluster']['p'], 4, False)}  ({time.time()-t0:.0f}s)")

    s["holm"] = S.holm({"H1": h1["cluster"]["p"], "H2": h2["cluster"]["p"],
                        "H3": h3["cluster"]["p"]})
    s["decision"] = decide(s)
    if grid:
        s["robustness"] = robustness(pool, bars_by, coins, baseline, log)
    return s


def decide(s: Dict[str, Any], miscalibrated: Optional[Dict[str, bool]] = None
           ) -> Dict[str, Any]:
    h1, h2, h3, fam = s["H1"], s["H2"], s["H3"], s["holm"]
    pos = lambda x: x is not None and math.isfinite(x) and x > 0     # noqa: E731
    crit = {
        "H1": {"holm_rejected": fam["H1"]["reject"],
               "positive": pos(h1["cluster"]["estimate"]),
               "same_sign_both_halves": all(pos(x["diff"]) for x in h1["split_half"]),
               "matched_control_agrees": pos(h1["matched_control"]["excess_diff"]),
               "both_stop_atr_strata_agree": all(pos(h1["stop_atr_strata"][k]["diff"])
                                                 for k in ("below", "at_or_above"))},
        "H2": {"holm_rejected": fam["H2"]["reject"],
               "positive": pos(h2["cluster"]["estimate"]),
               "same_sign_both_halves": all(pos(x.get("mean")) for x in h2["split_half"])},
        "H3": {"holm_rejected": fam["H3"]["reject"],
               "positive": pos(h3["cluster"]["estimate"]),
               "same_sign_both_halves": all(pos(x.get("mean")) for x in h3["split_half"])},
    }
    if miscalibrated is not None:
        for k in crit:
            crit[k]["null_calibration_ok"] = not miscalibrated.get(k, False)
    return {k: {"criteria": c, "adds_edge": all(c.values())} for k, c in crit.items()}


def robustness(pool, bars_by, coins, baseline, log=print) -> Dict[str, Any]:
    variants: List[Tuple[str, TL.TLParams, str]] = []
    for k in (2, 3, 5):
        for lb in (72, 120, 240):
            variants.append((f"k{k}_lb{lb}", replace(PRIMARY, k=k, lookback=lb), "1h"))
    variants.append(("buffer_0.25atr", replace(PRIMARY, buffer_atr=0.25), "1h"))
    variants.append(("touches_3", replace(PRIMARY, min_touches=3), "1h"))
    variants.append(("4h_lb30", replace(PRIMARY, lookback=30), "4h"))
    out: Dict[str, Any] = {"H1": [], "H2": [], "H3": []}
    for name, p, tf in variants:
        br = all_breaks(pool, coins, p, tf)
        for w in ((12, 24, 48) if name.startswith("k") else (H1_WINDOW,)):
            vals, grp, _, _, _, _ = h1_split(baseline, br, w)
            g1 = [v for v, g in zip(vals, grp) if g]
            g0 = [v for v, g in zip(vals, grp) if not g]
            wl = S.welch(g1, g0)
            out["H1"].append({"variant": name, "window": w, "n1": len(g1),
                              "n0": len(g0), "diff": wl["diff"], "p_iid": wl["p"]})
        if tf == "1h":
            for wait in ((24, 48, 96) if name.startswith("k") else (H2_WAIT,)):
                pr = h2_paired(h2_rows(pool, baseline, br, wait=wait, reps=H2_REPS_GRID))
                ds = S.describe(pr["d_primary"])
                out["H2"].append({"variant": name, "wait": wait, "n": ds["n"],
                                  "diff": ds.get("mean"), "p_iid": ds.get("p")})
        h3l = [x for xs in pool.map(_h3_coin, [(c, br[c], MEASURED) for c in coins]) for x in xs]
        good = [x for x in h3l if x["trade"].resolved and x["ctrl_mean"] is not None]
        ds = S.describe([x["trade"].r_net - x["ctrl_mean"] for x in good])
        out["H3"].append({"variant": name, "n": ds["n"], "excess": ds.get("mean"),
                          "p_iid": ds.get("p"),
                          "absolute": _mean([x["trade"].r_net for x in good])})
        log(f"  [ROBUST] {name} done")
    for h in ("H1", "H2", "H3"):
        key = "excess" if h == "H3" else "diff"
        xs = [r for r in out[h] if r.get(key) is not None and math.isfinite(r[key])]
        out[f"{h}_summary"] = {
            "variants": len(xs),
            "positive": sum(1 for r in xs if r[key] > 0),
            "p_below_0.05": sum(1 for r in xs if r["p_iid"] is not None
                                and math.isfinite(r["p_iid"]) and r["p_iid"] < 0.05),
            "positive_and_p_below_0.05": sum(1 for r in xs if r[key] > 0 and r["p_iid"] is not None
                                             and math.isfinite(r["p_iid"]) and r["p_iid"] < 0.05),
            "expected_by_chance": round(0.05 * len(xs), 1)}
    return out


# ------------------------------------------------------------ calibration --
def null_calibration(log=print) -> Dict[str, Any]:
    """The whole decision procedure on NULL_DATASETS random-walk datasets."""
    fired = {"H1": 0, "H2": 0, "H3": 0}
    rejected = {"H1": 0, "H2": 0, "H3": 0}
    per = []
    for k in range(NULL_DATASETS):
        bars = random_walks(k)
        with _pool(bars) as pool:
            s = analyse(pool, bars, grid=False, log=lambda *_: None, B=2000,
                        secondaries=False)
        row = {}
        for h in ("H1", "H2", "H3"):
            rejected[h] += s["holm"][h]["reject"]
            fired[h] += s["decision"][h]["adds_edge"]
            row[h] = {"estimate": s[h]["cluster"]["estimate"], "p": s[h]["cluster"]["p"],
                      "adds_edge": s["decision"][h]["adds_edge"]}
        per.append(row)
        log(f"  [NULL {k+1:2}/{NULL_DATASETS}] " + "  ".join(
            f"{h} {fmt(row[h]['estimate'])} p={fmt(row[h]['p'], 3, False)}"
            f"{' FIRED' if row[h]['adds_edge'] else ''}" for h in row))
    return {"datasets": NULL_DATASETS, "coins": NULL_COINS, "bars": NULL_BARS,
            "holm_rejections": rejected, "decision_fired": fired,
            "miscalibrated": {h: fired[h] > NULL_MAX_FALSE for h in fired},
            "max_false_allowed": NULL_MAX_FALSE, "per_dataset": per}


# ---------------------------------------------------------------- report --
def fmt(x, d=3, sign=True):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def render(s: Dict[str, Any]) -> str:
    b, h1, h2, h3 = s["baseline"], s["H1"], s["H2"], s["H3"]
    L = ["# Trendline breakout -- results", "",
         f"Pre-registration: `{s['prereg']}`.  Data fetched "
         f"{s['data']['fetched_at_utc']} from {s['data']['source']}.", "",
         f"{len(s['coins'])} coins, {sum(c['bars'] for c in s['coins'].values()):,} closed 1h bars. "
         f"Excluded: {len(s['data']['excluded'])}.", ""]
    if "null" in s:
        n = s["null"]
        L += ["## Null calibration", "",
              f"{n['datasets']} random-walk datasets ({n['coins']} coins x {n['bars']} bars). "
              f"Decision fired (false positives): " + ", ".join(
                  f"{h} {n['decision_fired'][h]}/{n['datasets']}" for h in ("H1", "H2", "H3"))
              + f". Holm rejections: " + ", ".join(
                  f"{h} {n['holm_rejections'][h]}" for h in ("H1", "H2", "H3")) + ".", ""]
    L += ["## Baseline (the live strategy, replayed)", "",
          f"{b['n']} resolved trades ({b['limit_entries']} bids, {b['market_entries']} market), "
          f"mean net R {fmt(b.get('mean'))} (gross {fmt(b['gross_mean_r'])}, costs "
          f"{fmt(b['mean_cost_r'])}), win {b.get('win', 0)*100:.1f}%, total {fmt(b.get('total'), 1)}R, "
          f"median rr {fmt(b['median_planned_rr'], 2, False)}, median risk {fmt(b['median_risk_pct'], 2, False)}%. "
          f"{b['open_at_end']} still open at the data end.", "",
          "## Decisions", "", "| framing | adds edge | criteria |", "|---|---|---|"]
    for k, v in s["decision"].items():
        crit = ", ".join(f"{n}={'yes' if ok else 'NO'}" for n, ok in v["criteria"].items())
        L.append(f"| {k} | **{'YES' if v['adds_edge'] else 'no'}** | {crit} |")
    L += ["", "## H1 -- filter", "",
          f"break in the prior {s['params']['h1_window']} bars: n={h1['with_break']['n']}, mean "
          f"{fmt(h1['with_break'].get('mean'))};  none: n={h1['without_break']['n']}, mean "
          f"{fmt(h1['without_break'].get('mean'))}", "",
          f"difference {fmt(h1['cluster']['estimate'])}, week-cluster 95% CI "
          f"[{fmt(h1['cluster']['ci_lo'])}, {fmt(h1['cluster']['ci_hi'])}], p={fmt(h1['cluster']['p'], 4, False)} "
          f"(iid Welch p={fmt(h1['welch_iid']['p'], 4, False)})", "",
          f"matched-random control: excess difference {fmt(h1['matched_control']['excess_diff'])} "
          f"(p={fmt(h1['matched_control']['cluster']['p'], 4, False)}); stop/ATR strata: "
          f"{fmt(h1['stop_atr_strata']['below']['diff'])} below / "
          f"{fmt(h1['stop_atr_strata']['at_or_above']['diff'])} at-or-above the median", "",
          "halves: " + "  ".join(f"{fmt(x['diff'])} (n {x['n1']}/{x['n0']})" for x in h1["split_half"]), "",
          f"filter applied as a strategy: n={h1['filtered_strategy']['n']}, mean "
          f"{fmt(h1['filtered_strategy'].get('mean'))}, total {fmt(h1['filtered_strategy'].get('total'), 1)}R", ""]
    L += ["## H2 -- confirmation entry", "",
          f"{h2['setups']} setups: " + ", ".join(f"{k} {v}" for k, v in h2["status"].items()), "",
          f"**trendline arm minus random-event arm** per setup {fmt(h2['cluster']['estimate'])}, "
          f"CI [{fmt(h2['cluster']['ci_lo'])}, {fmt(h2['cluster']['ci_hi'])}], p={fmt(h2['cluster']['p'], 4, False)}", "",
          f"trendline arm minus baseline {fmt(h2['vs_baseline']['estimate'])} "
          f"(p={fmt(h2['vs_baseline']['p'], 4, False)}); random-event arm minus baseline "
          f"{fmt(h2['random_arm_vs_baseline']['estimate'])} (p={fmt(h2['random_arm_vs_baseline']['p'], 4, False)})"
          f" -- the second number is what abstention alone is worth", "",
          f"entry rate: trendline {fmt(h2['entry_rate']['trendline_arm'], 3, False)}, random "
          f"{fmt(h2['entry_rate']['random_arm'], 3, False)}; confirmation trades mean "
          f"{fmt(h2['confirmation_trades'].get('mean'))} (n {h2['confirmation_trades']['n']}), median rr "
          f"{fmt(h2['median_rr_at_confirmation'], 2, False)}", "",
          "breakdown: " + ", ".join(f"{k} {v}" for k, v in h2["breakdown"].items()), "",
          f"with rr >= 2 re-applied: {fmt(h2['with_rr_floor']['estimate'])} "
          f"(p={fmt(h2['with_rr_floor']['p'], 4, False)});  entry slippage "
          f"{h2['entry_slip_sensitivity']['slip_pct']}%: {fmt(h2['entry_slip_sensitivity']['primary'].get('mean'))}", ""]
    L += ["## H3 -- standalone breakout", "",
          f"{h3['resolved']} resolved trades: absolute mean {fmt(h3['absolute'].get('mean'))} "
          f"(win {h3['absolute'].get('win', 0)*100:.1f}%), random-entry control {fmt(h3['control_mean_r'])}, "
          f"excess {fmt(h3['cluster']['estimate'])}, CI [{fmt(h3['cluster']['ci_lo'])}, "
          f"{fmt(h3['cluster']['ci_hi'])}], p={fmt(h3['cluster']['p'], 4, False)}", "",
          "halves: " + "  ".join(fmt(x.get("mean")) for x in h3["split_half"]), ""]
    L += ["## Holm family", "", "| | p | Holm p | rejected |", "|---|---|---|---|"]
    for k, v in s["holm"].items():
        L.append(f"| {k} | {fmt(v['p'], 4, False)} | {fmt(v['p_holm'], 4, False)} | {v['reject']} |")
    if "robustness" in s:
        L += ["", "## Robustness (never used to choose)", ""]
        for h in ("H1", "H2", "H3"):
            r = s["robustness"][f"{h}_summary"]
            L.append(f"- {h}: {r['positive']}/{r['variants']} variants positive; "
                     f"{r['positive_and_p_below_0.05']} positive with uncorrected p<0.05, "
                     f"{r['p_below_0.05']} with p<0.05 either way (chance alone: ~{r['expected_by_chance']})")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="fetch a fresh real snapshot first")
    ap.add_argument("--null", action="store_true", help="run the null calibration only")
    ap.add_argument("--quick", action="store_true", help="skip the robustness grid")
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if args.null:
        nc = null_calibration()
        nc["runtime_sec"] = round(time.time() - t0, 1)
        (RESULTS / "null_calibration.json").write_text(json.dumps(nc, indent=1, default=str))
        print(f"[NULL] fired: {nc['decision_fired']}  miscalibrated: {nc['miscalibrated']}")
        return 0
    cal_path = RESULTS / "null_calibration.json"
    cal: Optional[Dict[str, Any]] = (json.loads(cal_path.read_text())
                                     if cal_path.exists() else None)
    if cal is None:
        print("[WARN] no null calibration on file -- run --null first; "
              "decisions will lack criterion 5")
    snap = load_snapshot(args.fetch)
    with _pool(snap.bars) as pool:
        s = analyse(pool, snap.bars, grid=not args.quick)
    s["decision"] = decide(s, cal["miscalibrated"] if cal else None)
    s.update({
        "prereg": "research/backtest/PREREG_trendline_breakout.md",
        "data": {k: snap.manifest[k] for k in ("source", "interval", "fetched_at_utc", "excluded")},
        "coins": {c: {k: snap.manifest["coins"][c][k]
                      for k in ("bars", "first_t", "last_t", "gaps", "missing_bars", "sha256")}
                  for c in sorted(snap.bars)},
        "costs": MEASURED.as_dict(),
        "params": {"trendline": asdict(PRIMARY), "h1_window": H1_WINDOW, "h2_wait": H2_WAIT,
                   "h2_reps": H2_REPS, "h3_stop_atr": H3_STOP_ATR, "h3_rr": H3_RR,
                   "control_draws": CONTROL_DRAWS, "control_span": CONTROL_SPAN, "seed": SEED},
        "runtime_sec": round(time.time() - t0, 1)})
    if cal:
        s["null"] = {k: cal[k] for k in ("datasets", "coins", "bars", "holm_rejections",
                                         "decision_fired", "miscalibrated")}
    (RESULTS / "summary.json").write_text(json.dumps(s, indent=1, default=str))
    (RESULTS / "report.md").write_text(render(s))
    print(f"[DONE] {RESULTS}  ({s['runtime_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

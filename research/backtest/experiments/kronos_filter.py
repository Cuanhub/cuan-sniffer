"""Pre-registered experiment: Kronos as a reader of the band strategy's
trades. See PREREG_kronos.md.

Runs in research/kronos/.venv, because the forecasts need PyTorch:

    research/kronos/.venv/bin/python -m research.backtest.experiments.kronos_filter --null
    research/kronos/.venv/bin/python -m research.backtest.experiments.kronos_filter --spread
    research/kronos/.venv/bin/python -m research.backtest.experiments.kronos_filter

--spread (PREREG Amendment 1) needs no model and can run while --null is
still scoring; the real run refuses to start until it covers all 20.

Replaying the strategy and drawing the controls is CPU work, done in a
process pool while the forecasts run on the GPU in this process. Workers
never import PyTorch: the forecaster is imported inside kronos_paths().
Scores are cached per dataset under cache/kronos/, keyed by the pre-registered
configuration, so a long run that stops resumes where it stopped.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import NormalDist
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from research.backtest import band
from research.backtest import common as C
from research.backtest import data as D
from research.backtest import stats as S
from research.backtest.costs import MEASURED
from research.backtest.engine import Trade
from research.kronos import inputs as I
from research.kronos import score as K

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "kronos"
SCORES = ROOT / "cache" / "kronos"
SNAPSHOT = ROOT / "cache" / "snapshot_1h_fresh.json"

MIN_RR = 2.0
LOOKBACK, HORIZON, SAMPLES, BATCH = 464, 48, 16, 8
DIR_STEP = 24
MIN_VALID = 12
MIN_GROUP = 50
NULL_DATASETS, NULL_COINS, NULL_BARS, NULL_SALT = 20, 44, 5000, 31
SEED = 20261103
HYPOTHESES = ("K1_direction", "K2_expected_r")
KEY = {"K1_direction": "k1", "K2_expected_r": "k2"}

PathFn = Callable[[Dict[str, List[Dict[str, Any]]], List[Tuple[str, int]], int], np.ndarray]


def config() -> Dict[str, Any]:
    from research.kronos import forecast as Fk      # PyTorch: main process only
    import torch
    return {"lookback": LOOKBACK, "horizon": HORIZON, "samples": SAMPLES, "batch": BATCH,
            "dir_step": DIR_STEP, "seed": SEED, "min_rr": MIN_RR, "revisions": Fk.REVISIONS,
            "sampler": "fast.sample", "T": 1.0, "top_p": 0.9, "top_k": 0,
            "torch": torch.__version__}


# ------------------------------------------------------------- the trades --
def _run(coin: str) -> List[Trade]:
    return [t for t in band.run(coin, C.BARS[coin], MEASURED, min_rr=MIN_RR) if t.resolved]


def _ctrl(args) -> Tuple[str, Optional[float]]:
    idx, t = args
    m = C.same_order_control(C.BARS[t.coin], t)
    return idx, (t.r_net - m) if m is not None else None


_NULL: Dict[Tuple[int, int, int], Dict[str, List[Dict[str, Any]]]] = {}


def null_bars(seed: int, coins: int, n: int) -> Dict[str, List[Dict[str, Any]]]:
    """A driftless random-walk dataset, with volume that follows each bar's
    own range -- as real volume does -- and carries no direction. Null
    calibration ONLY. Kept per process, two at a time. The sizes are
    arguments, never module constants: workers build their own copy, and a
    constant changed in the parent would not reach them."""
    key = (seed, coins, n)
    if key not in _NULL:
        bars = C.random_walks(seed, coins=coins, n=n, salt=NULL_SALT)
        rng = np.random.default_rng([C.SEED, seed, NULL_SALT])
        for coin in sorted(bars):
            for b in bars[coin]:
                b["v"] = float(1000.0 * math.exp(0.5 * rng.normal())
                               * (1.0 + 50.0 * (b["h"] - b["l"]) / b["c"]))
        if len(_NULL) >= 2:
            _NULL.pop(next(iter(_NULL)))
        _NULL[key] = bars
    return _NULL[key]


def _null_run(args) -> List[Trade]:
    seed, coins, n, coin = args
    return [t for t in band.run(coin, null_bars(seed, coins, n)[coin], MEASURED,
                                min_rr=MIN_RR) if t.resolved]


def _null_ctrl(args) -> Tuple[str, Optional[float]]:
    seed, coins, n, idx, t = args
    m = C.same_order_control(null_bars(seed, coins, n)[t.coin], t)
    return idx, (t.r_net - m) if m is not None else None


def ordered(trades: Sequence[Trade]) -> List[Trade]:
    return sorted(trades, key=lambda t: (t.coin, t.decided_i))


def key_of(t: Trade) -> str:
    return f"{t.coin}:{t.decided_i}"


# ------------------------------------------------------------ the reading --
def kronos_paths(model) -> PathFn:
    """The real forecaster: S sampled paths per decision, batch-seeded."""
    from research.kronos import forecast as Fk
    tok, mdl, dev = model

    def fn(bars_by, decisions, seed):
        wins = [I.window(bars_by[c], i, LOOKBACK) for c, i in decisions]
        return Fk.sample_paths(tok, mdl, dev, wins, HORIZON, SAMPLES, seed)
    return fn


def score_one(t: Trade, bars: List[Dict[str, Any]], paths: np.ndarray) -> Dict[str, Any]:
    valid = K.valid_paths(paths)
    dec = bars[t.decided_i]
    out: Dict[str, Any] = {"valid": len(valid), "k1": None, "k2": None}
    if len(valid) < MIN_VALID:
        return out
    out["k1"] = K.direction(valid, dec["c"], DIR_STEP)
    out["k2"] = K.expected_r(t, dec, valid)
    j = t.decided_i + DIR_STEP
    if j < len(bars):
        fc = float(np.mean([p[DIR_STEP - 1, 3] for p in valid]))
        out["err_pct"] = abs(fc - bars[j]["c"]) / dec["c"] * 100
        out["naive_err_pct"] = abs(dec["c"] - bars[j]["c"]) / dec["c"] * 100
    return out


def score_trades(bars_by, trades: Sequence[Trade], path_fn: PathFn, seed_base: int,
                 cache: Optional[Path] = None, cfg: Optional[Dict[str, Any]] = None,
                 log=print) -> Dict[str, Dict[str, Any]]:
    """{trade key: scores}. Batches are fixed by the sorted trade list, so a
    resumed run draws exactly what an uninterrupted one would have."""
    have: Dict[str, Dict[str, Any]] = {}
    if cache is not None and cache.exists():
        saved = json.loads(cache.read_text())
        if saved.get("config") == cfg:
            have = saved["scores"]
    ts = ordered(trades)
    batches = [ts[b:b + BATCH] for b in range(0, len(ts), BATCH)]
    t0, done = time.time(), 0
    for b, chunk in enumerate(batches):
        if all(key_of(t) in have for t in chunk):
            continue
        paths = path_fn(bars_by, [(t.coin, t.decided_i) for t in chunk], seed_base + b)
        for t, p in zip(chunk, paths):
            have[key_of(t)] = score_one(t, bars_by[t.coin], p)
        done += 1
        if cache is not None and (done % 10 == 0 or b == len(batches) - 1):
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps({"config": cfg, "scores": have}))
        if done % 25 == 0:
            rate = (time.time() - t0) / done
            log(f"    batch {b + 1}/{len(batches)}  {rate:.1f}s/batch  "
                f"~{rate * (len(batches) - b - 1) / 60:.0f} min left")
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"config": cfg, "scores": have}))
    return {key_of(t): have[key_of(t)] for t in ts}


# ------------------------------------------------------------ the analysis --
def assign(scores: Sequence[float]) -> Tuple[List[int], str]:
    """1 = approved. The rule is score > 0; if either side would hold fewer
    than MIN_GROUP trades, the median split instead (information only)."""
    g = [1 if s > 0 else 0 for s in scores]
    if min(sum(g), len(g) - sum(g)) >= MIN_GROUP:
        return g, "sign"
    med = float(np.median(scores))
    return [1 if s > med else 0 for s in scores], "median"


def two_way(values: Sequence[float], groups: Sequence[int], weeks: Sequence[Any],
            coins: Sequence[str], se_week: float, B: int) -> Dict[str, float]:
    """PREREG Amendment 1: the bootstrap SE clustered by coin and by
    coin-week as well as by week, combined as Cameron, Gelbach & Miller
    (2011) combine one-way variances, and never below either one-way SE."""
    se_coin = S.cluster_bootstrap(values, groups, list(coins), "diff", B=B)["se"]
    se_cell = S.cluster_bootstrap(values, groups, list(zip(coins, weeks)), "diff", B=B)["se"]
    var = max(se_week ** 2 + se_coin ** 2 - se_cell ** 2, se_week ** 2, se_coin ** 2)
    return {"se_coin": se_coin, "se_coin_week": se_cell, "se_2way": math.sqrt(var)}


def diff_stats(values: Sequence[float], groups: Sequence[int], times: Sequence[int],
               B: int, coins: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    g = np.asarray(groups)
    v = np.asarray(values, float)
    if len(v) < 4 or g.sum() < 2 or (1 - g).sum() < 2:
        return {"n": len(v), "estimate": float("nan"), "p": float("nan"),
                "se_2way": float("nan"), "halves": [float("nan")] * 2}
    weeks = [S.week_of(t) for t in times]
    boot = S.cluster_bootstrap(values, groups, weeks, "diff", B=B)
    order = np.argsort(np.asarray(times))
    h = len(order) // 2
    halves = []
    for part in (order[:h], order[h:]):
        a, r = v[part][g[part] == 1], v[part][g[part] == 0]
        halves.append(float(a.mean() - r.mean()) if len(a) and len(r) else float("nan"))
    out = {"n": len(v), "n_approved": int(g.sum()), "halves": halves,
           **{k: boot[k] for k in ("estimate", "se", "ci_lo", "ci_hi", "p", "clusters")}}
    if coins is not None:
        out.update(two_way(values, groups, weeks, coins, boot["se"], B))
    return out


def p_two_way(st: Dict[str, Any], widen: float = 1.0) -> float:
    """Two-sided normal p of D on its two-way SE times max(1, widen); 1.0
    when that SE is missing, so a statistic without one can never pass."""
    se = st.get("se_2way", float("nan")) * max(1.0, widen)
    est = st["estimate"]
    if not (math.isfinite(se) and se > 0 and math.isfinite(est)):
        return 1.0
    return S.normal_two_sided_p(est / se)


def hypothesis(trades: Sequence[Trade], scores: Dict[str, Dict[str, Any]],
               excess: Dict[str, Optional[float]], name: str, B: int) -> Dict[str, Any]:
    k = KEY[name]
    rows = [(t, scores[key_of(t)][k]) for t in ordered(trades)
            if scores[key_of(t)][k] is not None]
    if len(rows) < 4:
        return {"scored": len(rows), "raw": diff_stats([], [], [], B),
                "excess": diff_stats([], [], [], B)}
    g, mode = assign([s for _, s in rows])
    raw = diff_stats([t.r_net for t, _ in rows], g, [t.entry_t for t, _ in rows], B,
                     [t.coin for t, _ in rows])
    ex = [(t, gi, x) for (t, _), gi in zip(rows, g)
          if (x := excess.get(key_of(t))) is not None]
    exs = diff_stats([x for _, _, x in ex], [gi for _, gi, _ in ex],
                     [t.entry_t for t, _, _ in ex], B, [t.coin for t, _, _ in ex])
    app = [t.r_net for (t, _), gi in zip(rows, g) if gi]
    rej = [t.r_net for (t, _), gi in zip(rows, g) if not gi]
    return {"scored": len(rows), "split": mode, "raw": raw, "excess": exs,
            "approved_mean_net_r": float(np.mean(app)) if app else float("nan"),
            "rejected_mean_net_r": float(np.mean(rej)) if rej else float("nan"),
            "approved_total_net_r": float(np.sum(app)),
            "spearman": K.spearman([s for _, s in rows], [t.r_net for t, _ in rows])}


def fires(h: Dict[str, Dict[str, Any]], stat: str, two_way_se: bool = False) -> Dict[str, bool]:
    """The decision rule's "carries information", ignoring calibration: Holm
    across the hypotheses, positive, positive in both halves. two_way_se:
    Holm on the two-way SE's p instead (Amendment 1, not widened)."""
    holm = S.holm({n: p_two_way(h[n][stat]) if two_way_se else
                   h[n][stat]["p"] if math.isfinite(h[n][stat]["p"]) else 1.0 for n in h})
    pos = lambda x: isinstance(x, float) and math.isfinite(x) and x > 0   # noqa: E731
    return {n: bool(holm[n]["reject"]) and pos(h[n][stat]["estimate"])
            and all(pos(x) for x in h[n][stat]["halves"]) for n in h}


def analyse(trades, scores, excess, B: int) -> Dict[str, Any]:
    h = {n: hypothesis(trades, scores, excess, n, B) for n in HYPOTHESES}
    return {"hypotheses": h, "fires": {s: fires(h, s) for s in ("raw", "excess")},
            "fires_2way": {s: fires(h, s, two_way_se=True) for s in ("raw", "excess")}}


def decide(a: Dict[str, Any], miscal: Dict[str, bool],
           widen: Dict[str, float]) -> Dict[str, Any]:
    """PREREG decision rules. The statistic is D on raw net R if the null
    calibrated it, else D on excess over controls if that calibrated, else
    the hypothesis is untestable by this design.

    Amendment 1: Holm must also reject on D's two-way SE, widened by
    widen["<hypothesis>:<statistic>"] (the null's spread ratio) when that is
    above 1. The verdict without it is kept as "as_first_registered"."""
    h = a["hypotheses"]
    used = {n: ("raw" if not miscal[f"{n}:raw"] else
                "excess" if not miscal[f"{n}:excess"] else None) for n in HYPOTHESES}
    testable = [n for n in HYPOTHESES if used[n]]
    holm = S.holm({n: (h[n][used[n]]["p"] if math.isfinite(h[n][used[n]]["p"]) else 1.0)
                   for n in testable}) if testable else {}
    by = {n: max(1.0, widen[f"{n}:{used[n]}"]) for n in testable}
    holm2 = S.holm({n: p_two_way(h[n][used[n]], by[n]) for n in testable}) if testable else {}
    pos = lambda x: isinstance(x, float) and math.isfinite(x) and x > 0   # noqa: E731
    no = {"information": False, "material": False, "deployable": False}
    out: Dict[str, Any] = {}
    for n in HYPOTHESES:
        if not used[n]:
            out[n] = {"statistic": None, **no, "as_first_registered": dict(no),
                      "note": "null calibration failed both statistics"}
            continue
        st = h[n][used[n]]
        crit = {"calibrated": True, "holm_rejected": bool(holm[n]["reject"]),
                "positive": pos(st["estimate"]),
                "both_halves_positive": all(pos(x) for x in st["halves"])}
        verdict = {}
        for label, c in (("as_first_registered", crit),
                         ("amended", {**crit, "holm_rejected_two_way": bool(holm2[n]["reject"])})):
            info = all(c.values())
            material = info and st["estimate"] >= 0.10
            verdict[label] = {"information": info, "material": material,
                              "deployable": material and pos(h[n]["approved_mean_net_r"])}
        out[n] = {"statistic": used[n],
                  "criteria": {**crit, "holm_rejected_two_way": bool(holm2[n]["reject"])},
                  "p_holm": holm[n]["p_holm"], "p_holm_two_way": holm2[n]["p_holm"],
                  "widened_by": by[n], **verdict["amended"],
                  "as_first_registered": verdict["as_first_registered"]}
    return out


# ---------------------------------------------------------------- runs --
def null_calibration(model, B: int = 2000, log=print) -> Dict[str, Any]:
    cfg = config()
    path_fn = kronos_paths(model)
    per: List[Dict[str, Dict[str, Any]]] = []
    t0 = time.time()
    nc, nb = NULL_COINS, NULL_BARS
    with ProcessPoolExecutor(max_workers=8) as pool:
        runs = {k: [pool.submit(_null_run, (k, nc, nb, c)) for c in sorted(null_bars(k, nc, nb))]
                for k in range(NULL_DATASETS)}
        for k in range(NULL_DATASETS):
            trades = ordered([t for f in runs[k] for t in f.result()])
            ctrl = pool.map(_null_ctrl, [(k, nc, nb, key_of(t), t) for t in trades],
                            chunksize=8)
            sc = score_trades(null_bars(k, nc, nb), trades, path_fn, SEED + 1000 * (k + 1),
                              SCORES / f"null_{k:02}.json", {**cfg, "coins": nc, "bars": nb},
                              log=log)
            excess = dict(ctrl)
            a = analyse(trades, sc, excess, B)
            row = {}
            for n in HYPOTHESES:
                for s in ("raw", "excess"):
                    row[f"{n}:{s}"] = {"estimate": a["hypotheses"][n][s]["estimate"],
                                       "adds_edge": a["fires"][s][n]}
            per.append(row)
            log(f"  [NULL {k + 1:2}/{NULL_DATASETS}] {len(trades)} trades  " + "  ".join(
                f"{n} {r['estimate']:+.3f}{' FIRED' if r['adds_edge'] else ''}"
                for n, r in row.items()) + f"  ({(time.time() - t0) / 60:.0f} min)")
    summ = C.null_summary(per)
    return {"datasets": NULL_DATASETS, "coins": NULL_COINS, "bars": NULL_BARS,
            "framings": summ, "miscalibrated": {k: v["miscalibrated"] for k, v in summ.items()},
            "per_dataset": per, "config": cfg}


def chance_band(n: int) -> Tuple[float, float]:
    """95% band of (SD of n draws) / sigma when the SE is right:
    sqrt(chi-square(n-1) / (n-1)), Wilson-Hilferty."""
    k = n - 1
    z = NormalDist().inv_cdf(0.975)
    q = lambda zz: k * (1 - 2 / (9 * k) + zz * math.sqrt(2 / (9 * k))) ** 3   # noqa: E731
    return math.sqrt(q(-z) / k), math.sqrt(q(z) / k)


def spread_summary(per: Sequence[Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
    """Per statistic: how far D spreads across null datasets, against the
    root-mean-square of the SE the test reported for it on each."""
    out: Dict[str, Any] = {}
    for name in (per[0] if per else {}):
        rows = [r[name] for r in per if math.isfinite(r[name]["estimate"])]
        if len(rows) < 3:
            continue
        sd = float(np.std([r["estimate"] for r in rows], ddof=1))
        rms = {k: float(math.sqrt(np.mean([r[k] ** 2 for r in rows]))) for k in ("se", "se_2way")}
        out[name] = {"datasets": len(rows), "spread": sd, "rms_se": rms["se"],
                     "rms_se_2way": rms["se_2way"], "ratio_week": sd / rms["se"],
                     "ratio_2way": sd / rms["se_2way"], "chance_band": list(chance_band(len(rows))),
                     "fired": sum(bool(r["fires"]) for r in rows),
                     "fired_2way": sum(bool(r["fires_2way"]) for r in rows)}
    return out


def null_spread(cfg: Dict[str, Any], B: int = 2000, path: Optional[Path] = None,
                workers: int = 6, log=print) -> Dict[str, Any]:
    """PREREG Amendment 1's measurement on the null. Re-analyses each null
    dataset whose scores are all cached -- no model -- and keeps each one it
    finishes in `path`, so it can run while the null is still being scored."""
    path = path or RESULTS / "null_spread.json"
    nc, nb = NULL_COINS, NULL_BARS
    want = {**cfg, "coins": nc, "bars": nb}
    done: Dict[str, Any] = {}
    if path.exists():
        saved = json.loads(path.read_text())
        if saved.get("config") == want and saved.get("B") == B:
            done = saved["per_dataset"]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for k in range(NULL_DATASETS):
            cache = SCORES / f"null_{k:02}.json"
            if str(k) in done or not cache.exists():
                continue
            try:
                cached = json.loads(cache.read_text())
            except json.JSONDecodeError:            # --null is writing it right now
                continue
            if cached.get("config") != want:
                continue
            futs = [pool.submit(_null_run, (k, nc, nb, c)) for c in sorted(null_bars(k, nc, nb))]
            trades = ordered([t for f in futs for t in f.result()])
            if any(key_of(t) not in cached["scores"] for t in trades):
                log(f"  [SPREAD {k + 1:2}] still being scored")
                continue
            sc = {key_of(t): cached["scores"][key_of(t)] for t in trades}
            excess = dict(pool.map(_null_ctrl, [(k, nc, nb, key_of(t), t) for t in trades],
                                   chunksize=8))
            a = analyse(trades, sc, excess, B)
            done[str(k)] = {f"{n}:{s}": {"estimate": a["hypotheses"][n][s]["estimate"],
                                         "se": a["hypotheses"][n][s].get("se", float("nan")),
                                         "se_2way": a["hypotheses"][n][s]["se_2way"],
                                         "fires": a["fires"][s][n],
                                         "fires_2way": a["fires_2way"][s][n]}
                            for n in HYPOTHESES for s in ("raw", "excess")}
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"config": want, "B": B, "per_dataset": done}, indent=1))
            log(f"  [SPREAD {k + 1:2}] {len(trades)} trades  " + "  ".join(
                f"{f} D {r['estimate']:+.3f} se {r['se']:.3f} two-way {r['se_2way']:.3f}"
                for f, r in done[str(k)].items()))
    out = {"config": want, "B": B, "per_dataset": done, "datasets": len(done),
           "complete": len(done) == NULL_DATASETS,
           "framings": spread_summary([done[k] for k in sorted(done, key=int)])}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, default=str))
    return out


def real_run(model, miscal: Dict[str, bool], widen: Dict[str, float], B: int = 10_000,
             log=print) -> Dict[str, Any]:
    cfg = config()
    snap = D.Snapshot.load(SNAPSHOT)
    with C.pool(snap.bars) as pool:
        trades = ordered([t for ts in pool.map(_run, sorted(snap.bars)) for t in ts])
        log(f"[TRADES] {len(trades)} resolved trades at MIN_RR {MIN_RR}")
        ctrl = pool.map(_ctrl, [(key_of(t), t) for t in trades], chunksize=8)
        sc = score_trades(snap.bars, trades, kronos_paths(model), SEED,
                          SCORES / "real.json", cfg, log=log)
        excess = dict(ctrl)
    a = analyse(trades, sc, excess, B)
    a["decision"] = decide(a, miscal, widen)
    a["descriptive"] = describe(trades, sc, excess)
    a.update({"data": {k: snap.manifest.get(k) for k in ("source", "fetched_at_utc", "excluded")},
              "coins": {c: m["sha256"] for c, m in snap.manifest["coins"].items()},
              "costs": MEASURED.as_dict(), "config": cfg, "trades": len(trades)})
    rows = [{"coin": t.coin, "decided_i": t.decided_i, "entry_t": t.entry_t, "kind": t.kind,
             "outcome": t.outcome, "r_net": round(t.r_net, 4), **sc[key_of(t)],
             "excess": excess.get(key_of(t))} for t in trades]
    return {"summary": a, "rows": rows}


def describe(trades, scores, excess) -> Dict[str, Any]:
    """Secondary, descriptive only (PREREG)."""
    sc = [scores[key_of(t)] for t in trades]
    errs = [(s["err_pct"], s["naive_err_pct"]) for s in sc if "err_pct" in s]
    out: Dict[str, Any] = {
        "unscored": sum(1 for s in sc if s["k1"] is None),
        "mean_valid_paths": float(np.mean([s["valid"] for s in sc])) if sc else float("nan"),
        "close_24h_abs_error_pct": {"kronos": float(np.mean([e for e, _ in errs])),
                                    "no_change": float(np.mean([n for _, n in errs])),
                                    "n": len(errs)} if errs else None}
    for n in HYPOTHESES:
        k = KEY[n]
        rows = [(t, s[k]) for t, s in zip(trades, sc) if s[k] is not None]
        by_kind = {}
        for kind in ("limit", "market_close"):
            sub = [(t, v) for t, v in rows if t.kind == kind]
            if len(sub) >= 4:
                g, mode = assign([v for _, v in sub])
                a = [t.r_net for (t, _), gi in zip(sub, g) if gi]
                r = [t.r_net for (t, _), gi in zip(sub, g) if not gi]
                by_kind[kind] = {"n": len(sub), "split": mode,
                                 "approved": float(np.mean(a)) if a else None,
                                 "rejected": float(np.mean(r)) if r else None}
        q = np.quantile([v for _, v in rows], [0.2, 0.4, 0.6, 0.8]) if rows else []
        quint = []
        for lo, hi in zip([-math.inf, *q], [*q, math.inf]):
            sub = [(t, v) for t, v in rows if lo < v <= hi]
            if sub:
                quint.append({"mean_score": float(np.mean([v for _, v in sub])),
                              "mean_net_r": float(np.mean([t.r_net for t, _ in sub])),
                              "n": len(sub)})
        out[n] = {"approval_rate": float(np.mean([v > 0 for _, v in rows])) if rows else None,
                  "by_kind": by_kind, "quintiles": quint}
    return out


def fmt(x, d=3, sign=True):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def verdict_of(d: Dict[str, Any]) -> str:
    return ("deployable" if d["deployable"] else "material" if d["material"] else
            "information" if d["information"] else "no")


def render(s: Dict[str, Any], null: Dict[str, Any], spread: Dict[str, Any]) -> str:
    L = ["# Kronos as a reader of the band strategy's trades -- results", "",
         f"Pre-registration: `research/backtest/PREREG_kronos.md`. Data fetched "
         f"{s['data']['fetched_at_utc']}; {len(s['coins'])} coins; {s['trades']} trades.", "",
         "## Null calibration (20 random-walk datasets, real size)", "",
         "| statistic | fired | mean D | SE | verdict |", "|---|---|---|---|---|"]
    for k, v in null["framings"].items():
        L.append(f"| {k} | {v['fired']}/{null['datasets']} | {fmt(v['mean'], 4)} | "
                 f"{fmt(v['se'], 4, False)} | {'MISCALIBRATED' if v['miscalibrated'] else 'ok'} |")
    L += ["", "## Null spread (Amendment 1)", "",
          "How far D spreads across the null datasets, against the SE the test reports. "
          "A ratio inside the chance band means the SE is the right size.", "",
          "| statistic | datasets | spread of D | week SE | ratio | two-way SE | ratio | "
          "chance band | fired (week) | fired (two-way) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for k, v in spread["framings"].items():
        lo, hi = v["chance_band"]
        L.append(f"| {k} | {v['datasets']} | {fmt(v['spread'], 3, False)} | "
                 f"{fmt(v['rms_se'], 3, False)} | {fmt(v['ratio_week'], 2, False)} | "
                 f"{fmt(v['rms_se_2way'], 3, False)} | {fmt(v['ratio_2way'], 2, False)} | "
                 f"{lo:.2f}-{hi:.2f} | {v['fired']} | {v['fired_2way']} |")
    L += ["", "## Approved minus rejected trades (net R, real outcomes)", "",
          "| hypothesis | statistic | split | scored | approved | D | 95% CI | p | Holm p | "
          "two-way SE x widened | Holm p, two-way | halves | approved mean R | rejected mean R | "
          "as first registered | verdict (amended) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n, d in s["decision"].items():
        h = s["hypotheses"][n]
        st = h[d["statistic"]] if d["statistic"] else h["raw"]
        L.append(f"| {n} | {d['statistic']} | {h.get('split')} | {h['scored']} | "
                 f"{st.get('n_approved', 'n/a')} | {fmt(st['estimate'])} | "
                 f"[{fmt(st.get('ci_lo'))}, {fmt(st.get('ci_hi'))}] | {fmt(st['p'], 4, False)} | "
                 f"{fmt(d.get('p_holm'), 4, False)} | {fmt(st.get('se_2way'), 3, False)} x "
                 f"{fmt(d.get('widened_by'), 2, False)} | {fmt(d.get('p_holm_two_way'), 4, False)} | "
                 f"{', '.join(fmt(x) for x in st['halves'])} | "
                 f"{fmt(h.get('approved_mean_net_r'))} | {fmt(h.get('rejected_mean_net_r'))} | "
                 f"{verdict_of(d['as_first_registered'])} | **{verdict_of(d)}** |")
    de = s["descriptive"]
    L += ["", "## Descriptive (no decisions)", "",
          f"- unscored decisions: {de['unscored']}; mean valid paths {fmt(de['mean_valid_paths'], 1, False)}"]
    e = de.get("close_24h_abs_error_pct")
    if e:
        L.append(f"- 24h close, mean absolute error: Kronos {fmt(e['kronos'], 2, False)}% vs "
                 f"no-change {fmt(e['no_change'], 2, False)}% (n {e['n']})")
    for n in HYPOTHESES:
        h, d = s["hypotheses"][n], de[n]
        L.append(f"- {n}: approval rate {fmt(d['approval_rate'], 2, False)}, Spearman(score, net R) "
                 f"{fmt(h.get('spearman'))}; excess-based D {fmt(h['excess']['estimate'])} "
                 f"(p {fmt(h['excess']['p'], 4, False)}, n {h['excess']['n']})")
        for kind, v in d["by_kind"].items():
            L.append(f"  - {kind}: n {v['n']} ({v['split']}), approved {fmt(v['approved'])}, "
                     f"rejected {fmt(v['rejected'])}")
        L.append("  - by score quintile (mean score -> mean net R): " + "; ".join(
            f"{fmt(q['mean_score'], 4)} -> {fmt(q['mean_net_r'])} (n {q['n']})" for q in d["quintiles"]))
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--null", action="store_true")
    ap.add_argument("--spread", action="store_true")
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if args.spread:
        sp = null_spread(config())
        print(f"[SPREAD] {sp['datasets']}/{NULL_DATASETS} null datasets")
        for k, v in sp["framings"].items():
            lo, hi = v["chance_band"]
            print(f"[SPREAD] {k}: D spreads {v['spread']:.3f}; week SE {v['rms_se']:.3f} "
                  f"(x{v['ratio_week']:.2f}), two-way SE {v['rms_se_2way']:.3f} "
                  f"(x{v['ratio_2way']:.2f}); chance band {lo:.2f}-{hi:.2f}")
        return 0
    if not args.null:
        null = json.loads((RESULTS / "null_calibration.json").read_text())
        p = RESULTS / "null_spread.json"
        spread = json.loads(p.read_text()) if p.exists() else {}
        if not spread.get("complete") or spread.get("config") != {
                **config(), "coins": NULL_COINS, "bars": NULL_BARS}:
            raise SystemExit("PREREG Amendment 1: run --spread over all "
                             f"{NULL_DATASETS} null datasets before the real run")
    from research.kronos import forecast as Fk
    model = Fk.load()
    print(f"[MODEL] Kronos-small on {model[2]}")
    if args.null:
        cal = null_calibration(model)
        cal["runtime_sec"] = round(time.time() - t0, 1)
        (RESULTS / "null_calibration.json").write_text(json.dumps(cal, indent=1, default=str))
        for k, v in cal["framings"].items():
            print(f"[NULL] {k}: fired {v['fired']}/{NULL_DATASETS}, mean {v['mean']:+.4f} "
                  f"+/- {v['se']:.4f} -> {'MISCALIBRATED' if v['miscalibrated'] else 'ok'}")
        return 0
    widen = {k: v["ratio_2way"] for k, v in spread["framings"].items()}
    out = real_run(model, null["miscalibrated"], widen)
    s = out["summary"]
    s["runtime_sec"] = round(time.time() - t0, 1)
    s["widen"] = widen
    (RESULTS / "summary.json").write_text(json.dumps(s, indent=1, default=str))
    (RESULTS / "trades.json").write_text(json.dumps(out["rows"], indent=0, default=str))
    (RESULTS / "report.md").write_text(render(s, null, spread))
    print(f"[DONE] {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

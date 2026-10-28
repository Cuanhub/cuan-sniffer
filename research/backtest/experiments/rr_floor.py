"""Pre-registered experiment: the reward:risk floor. See PREREG_rr_floor.md.

    python -m research.backtest.experiments.rr_floor --null    # calibrate first
    python -m research.backtest.experiments.rr_floor           # on the fresh snapshot
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from research.backtest import band
from research.backtest import common as C
from research.backtest import data as D
from research.backtest import stats as S
from research.backtest.costs import MEASURED
from research.backtest.engine import LIMIT, Trade

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"
RESULTS = ROOT / "results" / "rr_floor"
SNAPSHOT = CACHE / "snapshot_1h_fresh.json"

CONFIGS = (0.3, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
LIVE = 2.0
# framing -> (the run it comes from, rr low inclusive, rr high exclusive)
BUCKETS: Dict[str, Tuple[float, float, float]] = {
    "E_live": (2.0, 2.0, math.inf),
    "E_mid": (1.0, 1.0, 2.0),
    "E_low": (0.3, 0.3, 1.0),
}
NULL_DATASETS = 20


def _run(args) -> List[Trade]:
    coin, min_rr = args
    return [t for t in band.run(coin, C.BARS[coin], MEASURED, min_rr=min_rr) if t.resolved]


def _excess(args) -> Tuple[int, Optional[float]]:
    idx, coin, t = args
    m = C.matched_control(C.BARS[coin], coin, t.entry_i, t.risk_pct,
                          (t.tp - t.entry_px) / t.entry_px * 100, t.kind == LIMIT)
    return idx, (t.r_net - m) if m is not None else None


def runs_for(pool, coins, configs) -> Dict[float, Dict[str, List[Trade]]]:
    out = {}
    for T in configs:
        out[T] = dict(zip(coins, pool.map(_run, [(c, T) for c in coins])))
    return out


def bucket_trades(runs, name) -> List[Trade]:
    src, lo, hi = BUCKETS[name]
    return [t for ts in runs[src].values() for t in ts
            if lo <= float(t.features.get("rr") or 0.0) < hi]


def excess_rows(pool, trades: List[Trade]) -> List[Tuple[Trade, float]]:
    jobs = [(i, t.coin, t) for i, t in enumerate(trades)]
    got = dict(pool.map(_excess, jobs, chunksize=4))
    return [(trades[i], v) for i, v in got.items() if v is not None]


def framing_stats(rows: List[Tuple[Trade, float]], B: int) -> Dict[str, Any]:
    vals = [v for _, v in rows]
    if len(vals) < 2:
        return {"n": len(vals), "estimate": float("nan"), "p": float("nan"),
                "halves": [float("nan"), float("nan")]}
    boot = S.cluster_bootstrap(vals, [0] * len(vals), [S.week_of(t.entry_t) for t, _ in rows],
                               "mean", B=B)
    hv = C.halves(vals, [t.entry_t for t, _ in rows])
    return {"n": len(vals), **{k: boot[k] for k in ("estimate", "se", "ci_lo", "ci_hi", "p",
                                                     "clusters")},
            "halves": [float(np.mean(h)) if h else float("nan") for h in hv],
            "p_iid": S.describe(vals)["p"],
            "absolute_mean_net_r": float(np.mean([t.r_net for t, _ in rows])),
            "control_mean_r": float(np.mean([t.r_net - v for t, v in rows]))}


def decide(fr: Dict[str, Dict[str, Any]], holm: Dict[str, Any],
           miscal: Optional[Dict[str, bool]] = None) -> Dict[str, Any]:
    out = {}
    pos = lambda x: isinstance(x, (int, float)) and math.isfinite(x) and x > 0   # noqa: E731
    for k, v in fr.items():
        crit = {"holm_rejected": bool(holm[k]["reject"]), "positive": pos(v["estimate"]),
                "both_halves_positive": all(pos(h) for h in v["halves"])}
        if miscal is not None:
            crit["null_calibration_ok"] = not miscal.get(k, False)
        out[k] = {"criteria": crit, "has_edge": all(crit.values()),
                  "material": pos(v["estimate"]) and v["estimate"] >= 0.10}
    return out


def analyse(pool, bars_by, B: int = 10_000, secondaries: bool = True,
            log=print) -> Dict[str, Any]:
    coins = sorted(bars_by)
    t0 = time.time()
    configs = CONFIGS if secondaries else tuple(sorted({b[0] for b in BUCKETS.values()}))
    runs = runs_for(pool, coins, configs)
    log(f"[RUNS] {', '.join(f'{T}: {sum(len(v) for v in runs[T].values())}' for T in configs)} "
        f"trades ({time.time() - t0:.0f}s)")
    fr: Dict[str, Dict[str, Any]] = {}
    rows_by: Dict[str, List[Tuple[Trade, float]]] = {}
    for name in BUCKETS:
        rows_by[name] = excess_rows(pool, bucket_trades(runs, name))
        fr[name] = framing_stats(rows_by[name], B)
        log(f"[{name}] n={fr[name]['n']} excess {fr[name]['estimate']:+.3f} "
            f"p={fr[name]['p']:.4f} ({time.time() - t0:.0f}s)")
    holm = S.holm({k: v["p"] for k, v in fr.items()})
    s: Dict[str, Any] = {"framings": fr, "holm": holm, "decision": decide(fr, holm)}
    if not secondaries:
        return s

    # -------------------------------------------------- secondary --
    weekly: Dict[float, Dict[Any, float]] = {}
    per_config: Dict[str, Dict[str, Any]] = {}
    for T in configs:
        ts = [t for v in runs[T].values() for t in v]
        d = S.describe([t.r_net for t in ts])
        wk: Dict[Any, float] = defaultdict(float)
        for t in ts:
            wk[S.week_of(t.entry_t)] += t.r_net
        weekly[T] = wk
        per_config[str(T)] = {**{k: d.get(k) for k in ("n", "mean", "total", "win")},
                              "weeks": len(wk)}
    allw = sorted({w for T in configs for w in weekly[T]})
    for T in configs:
        diffs = [weekly[T].get(w, 0.0) - weekly[LIVE].get(w, 0.0) for w in allw]
        boot = S.cluster_bootstrap(diffs, [0] * len(diffs), list(range(len(diffs))), "mean", B=B)
        per_config[str(T)]["weekly_diff_vs_live"] = {k: boot[k] for k in
                                                     ("estimate", "ci_lo", "ci_hi", "p")}
    s["per_config"] = per_config
    by_kind = {}
    for name, rows in rows_by.items():
        for kind in ("limit", "market_close"):
            vals = [v for t, v in rows if t.kind == kind]
            by_kind[f"{name}:{kind}"] = S.describe(vals)
    s["by_entry_kind"] = by_kind
    hi_rows = [(t, v) for t, v in rows_by["E_live"] if float(t.features.get("rr") or 0) >= 3.0]
    s["rr_3_plus_within_live"] = framing_stats(hi_rows, B)
    strata = {}
    for name, rows in rows_by.items():
        ax = [float(t.features.get("stop_atr_x") or 0) for t, _ in rows]
        med = float(np.median(ax)) if ax else 0.0
        lo = [v for (t, v), a in zip(rows, ax) if a < med]
        hi = [v for (t, v), a in zip(rows, ax) if a >= med]
        strata[name] = {"median_stop_atr_x": med, "below": S.describe(lo), "at_or_above": S.describe(hi)}
    s["stop_atr_strata"] = strata
    return s


def null_calibration(log=print) -> Dict[str, Any]:
    per = []
    for k in range(NULL_DATASETS):
        bars = C.random_walks(k)
        with C.pool(bars) as pool:
            s = analyse(pool, bars, B=2000, secondaries=False, log=lambda *_: None)
        per.append({n: {"estimate": v["estimate"], "adds_edge": s["decision"][n]["has_edge"]}
                    for n, v in s["framings"].items()})
        log(f"  [NULL {k + 1:2}/{NULL_DATASETS}] " + "  ".join(
            f"{n} {r['estimate']:+.3f}{' FIRED' if r['adds_edge'] else ''}" for n, r in per[-1].items()))
    summ = C.null_summary(per)
    return {"datasets": NULL_DATASETS, "framings": summ,
            "miscalibrated": {k: v["miscalibrated"] for k, v in summ.items()}, "per_dataset": per}


def fmt(x, d=3, sign=True):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def render(s: Dict[str, Any]) -> str:
    L = ["# Reward:risk floor -- results", "",
         f"Pre-registration: `research/backtest/PREREG_rr_floor.md`. Data fetched "
         f"{s['data']['fetched_at_utc']}; {len(s['coins'])} coins.", ""]
    if "null" in s:
        L += ["Null calibration: " + ", ".join(
            f"{k} fired {v['fired']}/20, mean {fmt(v['mean'], 4)} +/- {fmt(v['se'], 4, False)}"
            for k, v in s["null"].items()), ""]
    L += ["## Edge by reward:risk bucket (excess over random entries with the same geometry)", "",
          "| bucket | n | excess R | 95% CI | p (week-cluster) | Holm p | halves | absolute net R | control R | edge |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for k, v in s["framings"].items():
        d = s["decision"][k]
        L.append(f"| {k} | {v['n']} | {fmt(v['estimate'])} | [{fmt(v.get('ci_lo'))}, {fmt(v.get('ci_hi'))}] | "
                 f"{fmt(v['p'], 4, False)} | {fmt(s['holm'][k]['p_holm'], 4, False)} | "
                 f"{', '.join(fmt(h) for h in v['halves'])} | {fmt(v.get('absolute_mean_net_r'))} | "
                 f"{fmt(v.get('control_mean_r'))} | **{'YES' if d['has_edge'] else 'no'}** |")
    L += ["", "## Each floor as a strategy (descriptive; portfolio rails not modelled)", "",
          "| MIN_RR | trades | win | mean net R | total net R | weekly vs live (CI) |", "|---|---|---|---|---|---|"]
    for T, v in s["per_config"].items():
        w = v["weekly_diff_vs_live"]
        L.append(f"| {T} | {v['n']} | {(v['win'] or 0) * 100:.1f}% | {fmt(v['mean'])} | {fmt(v['total'], 1)} | "
                 f"{fmt(w['estimate'], 2)} [{fmt(w['ci_lo'], 2)}, {fmt(w['ci_hi'], 2)}] |")
    h = s["rr_3_plus_within_live"]
    L += ["", f"rr >= 3.0 within the live trades: n={h['n']}, excess {fmt(h['estimate'])} "
          f"(p={fmt(h['p'], 4, False)})", "", "## Stop/ATR strata (excess R)", ""]
    for k, v in s["stop_atr_strata"].items():
        L.append(f"- {k}: below {fmt(v['below'].get('mean'))} (n {v['below']['n']}), at-or-above "
                 f"{fmt(v['at_or_above'].get('mean'))} (n {v['at_or_above']['n']}), median stop "
                 f"{v['median_stop_atr_x']:.2f}x ATR")
    L += ["", "## By entry kind (excess R)", ""]
    for k, v in s["by_entry_kind"].items():
        if v["n"]:
            L.append(f"- {k}: n {v['n']}, {fmt(v.get('mean'))} (iid p {fmt(v.get('p'), 4, False)})")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--null", action="store_true")
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if args.null:
        cal = null_calibration()
        cal["runtime_sec"] = round(time.time() - t0, 1)
        (RESULTS / "null_calibration.json").write_text(json.dumps(cal, indent=1, default=str))
        for k, v in cal["framings"].items():
            print(f"[NULL] {k}: fired {v['fired']}/20, mean {v['mean']:+.4f} +/- {v['se']:.4f} "
                  f"-> {'MISCALIBRATED' if v['miscalibrated'] else 'ok'}")
        return 0
    cal = json.loads((RESULTS / "null_calibration.json").read_text())
    snap = D.Snapshot.load(SNAPSHOT)
    with C.pool(snap.bars) as pool:
        s = analyse(pool, snap.bars)
    s["decision"] = decide(s["framings"], s["holm"], cal["miscalibrated"])
    s.update({"data": {k: snap.manifest.get(k) for k in ("source", "fetched_at_utc", "excluded")},
              "coins": {c: m["sha256"] for c, m in snap.manifest["coins"].items()},
              "null": cal["framings"], "costs": MEASURED.as_dict(),
              "runtime_sec": round(time.time() - t0, 1)})
    (RESULTS / "summary.json").write_text(json.dumps(s, indent=1, default=str))
    (RESULTS / "report.md").write_text(render(s))
    print(f"[DONE] {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

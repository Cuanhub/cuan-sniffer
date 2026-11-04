"""Pre-registered experiment: the band strategy's short-side mirror. See PREREG_short_mirror.md.

    python -m research.backtest.experiments.short_mirror --null    # calibrate first
    python -m research.backtest.experiments.short_mirror           # on the archive
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from research.backtest import band
from research.backtest import common as C
from research.backtest import data as D
from research.backtest import stats as S
from research.backtest.costs import MEASURED
from research.backtest.engine import LIMIT, LONG, MARKET_CLOSE, SHORT, Trade

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "short_mirror"
SNAPSHOT = ROOT / "cache" / "snapshot_1h_fresh.json"
MIN_RR = 2.0
ALPHA = 0.05
NULL_DATASETS = 20
# The funding formula's interest component, 0.01% per 8h, paid by longs to
# shorts whenever the premium is zero. A sensitivity only (PREREG secondary).
FUNDING_PCT_PER_HOUR = 0.01 / 8

Rows = List[Tuple[Trade, float]]


def _run(args) -> List[Trade]:
    coin, side = args
    return [t for t in band.run(coin, C.BARS[coin], MEASURED, min_rr=MIN_RR, side=side)
            if t.resolved]


def _excess(args) -> Tuple[int, Optional[float]]:
    """Net R minus same-order-type random controls of the trade's own side."""
    idx, coin, t, funding = args
    m = C.same_order_control(C.BARS[coin], t, funding_pct_per_hour=funding)
    if m is None:
        return idx, None
    return idx, t.r_net + C.funding_r(t, funding) - m


def trades_for(pool, coins: Sequence[str], side: str) -> List[Trade]:
    out: List[Trade] = []
    for ts in pool.map(_run, [(c, side) for c in coins]):
        out += ts
    return out


def excess_rows(pool, trades: Sequence[Trade], funding: float = 0.0) -> Rows:
    jobs = [(i, t.coin, t, funding) for i, t in enumerate(trades)]
    got = dict(pool.map(_excess, jobs, chunksize=4))
    return [(trades[i], v) for i, v in sorted(got.items()) if v is not None]


def two_way(vals: Sequence[float], weeks: Sequence[Any], coins: Sequence[str],
            se_week: float, B: int) -> Dict[str, float]:
    """Kronos Amendment 1, for a mean: the bootstrap SE clustered by coin and
    by coin-week as well as by week, combined as Cameron, Gelbach & Miller
    (2011) combine one-way variances, and never below either one-way SE."""
    zeros = [0] * len(vals)
    se_coin = S.cluster_bootstrap(vals, zeros, list(coins), "mean", B=B)["se"]
    se_cell = S.cluster_bootstrap(vals, zeros, list(zip(coins, weeks)), "mean", B=B)["se"]
    var = max(se_week ** 2 + se_coin ** 2 - se_cell ** 2, se_week ** 2, se_coin ** 2)
    return {"se_coin": se_coin, "se_coin_week": se_cell, "se_2way": math.sqrt(var)}


def framing_stats(rows: Rows, B: int, two_way_se: bool = True) -> Dict[str, Any]:
    vals = [v for _, v in rows]
    if len(vals) < 2:
        return {"n": len(vals), "estimate": float("nan"), "p": float("nan"),
                "halves": [float("nan"), float("nan")]}
    weeks = [S.week_of(t.entry_t) for t, _ in rows]
    boot = S.cluster_bootstrap(vals, [0] * len(vals), weeks, "mean", B=B)
    hv = C.halves(vals, [t.entry_t for t, _ in rows])
    out: Dict[str, Any] = {
        "n": len(vals), **{k: boot[k] for k in ("estimate", "se", "ci_lo", "ci_hi", "p", "clusters")},
        "halves": [float(np.mean(h)) if h else float("nan") for h in hv],
        "p_iid": S.describe(vals)["p"],
        "absolute_mean_net_r": float(np.mean([t.r_net for t, _ in rows])),
        "control_mean_r": float(np.mean([t.r_net - v for t, v in rows]))}
    if two_way_se:
        out.update(two_way(vals, weeks, [t.coin for t, _ in rows], boot["se"], B))
        se2 = out["se_2way"]
        out["p_2way"] = (S.normal_two_sided_p(out["estimate"] / se2)
                         if math.isfinite(se2) and se2 > 0 else float("nan"))
    return out


def decide(fr: Dict[str, Dict[str, Any]], miscal: Optional[Dict[str, bool]] = None
           ) -> Dict[str, Any]:
    """PREREG decision rule. One hypothesis, so no Holm step."""
    pos = lambda x: isinstance(x, (int, float)) and math.isfinite(x) and x > 0   # noqa: E731
    out = {}
    for k, v in fr.items():
        p = v.get("p", float("nan"))
        crit = {"p_below_alpha": bool(math.isfinite(p) and p < ALPHA),
                "positive": pos(v["estimate"]),
                "both_halves_positive": all(pos(h) for h in v["halves"])}
        if miscal is not None:
            crit["null_calibration_ok"] = not miscal.get(k, False)
        out[k] = {"criteria": crit, "has_edge": all(crit.values()),
                  "material": pos(v["estimate"]) and v["estimate"] >= 0.10}
    return out


def market_weeks(bars_by: Dict[str, List[Dict[str, Any]]]) -> Dict[Any, float]:
    """Equal-weighted universe return per ISO week: the mean over coins of
    (last close in the week / first open in the week - 1)."""
    per: Dict[Any, List[float]] = defaultdict(list)
    for bars in bars_by.values():
        first: Dict[Any, float] = {}
        last: Dict[Any, float] = {}
        for b in bars:
            w = S.week_of(b["t"])
            first.setdefault(w, b["o"])
            last[w] = b["c"]
        for w in first:
            if first[w] > 0:
                per[w].append(last[w] / first[w] - 1)
    return {w: float(np.mean(v)) for w, v in per.items() if v}


def hold_bars(t: Trade) -> int:
    """Bars from fill to exit; 0 for a trade still open (none here are)."""
    return (t.exit_i - t.entry_i) if t.exit_i is not None else 0


def weekly(trades: Sequence[Trade]) -> Dict[Any, float]:
    out: Dict[Any, float] = defaultdict(float)
    for t in trades:
        out[S.week_of(t.entry_t)] += t.r_net
    return out


def combined_book(longs: Sequence[Trade], shorts: Sequence[Trade], B: int) -> Dict[str, Any]:
    """Weekly net R of long + short against long alone, paired by week."""
    wl, ws = weekly(longs), weekly(shorts)
    allw = sorted(set(wl) | set(ws))
    if len(allw) < 2:
        return {"weeks": len(allw)}
    L = np.array([wl.get(w, 0.0) for w in allw])
    Sh = np.array([ws.get(w, 0.0) for w in allw])
    boot = S.cluster_bootstrap(list(Sh), [0] * len(allw), list(range(len(allw))), "mean", B=B)
    corr = float(np.corrcoef(L, Sh)[0, 1]) if len(allw) > 2 and L.std() > 0 and Sh.std() > 0 else float("nan")
    return {"weeks": len(allw), "long_weekly_mean": float(L.mean()), "long_weekly_sd": float(L.std(ddof=1)),
            "combined_weekly_mean": float((L + Sh).mean()),
            "combined_weekly_sd": float((L + Sh).std(ddof=1)),
            "added_by_shorts": {k: boot[k] for k in ("estimate", "ci_lo", "ci_hi", "p")},
            "corr_long_short_weekly": corr}


def analyse(pool, bars_by: Dict[str, List[Dict[str, Any]]], B: int = 10_000,
            secondaries: bool = True, log=print) -> Dict[str, Any]:
    coins = sorted(bars_by)
    t0 = time.time()
    shorts = trades_for(pool, coins, SHORT)
    rows = excess_rows(pool, shorts)
    fr = {"S_live": framing_stats(rows, B, two_way_se=secondaries)}
    log(f"[S_live] {len(shorts)} resolved shorts, {fr['S_live']['n']} with controls, excess "
        f"{fr['S_live']['estimate']:+.3f} p={fr['S_live']['p']:.4f} ({time.time() - t0:.0f}s)")
    s: Dict[str, Any] = {"framings": fr, "decision": decide(fr)}
    if not secondaries:
        return s

    # ---------------------------------------------------- secondary --
    d = S.describe([t.r_net for t in shorts])
    s["absolute"] = {**{k: d.get(k) for k in ("n", "mean", "total", "win", "se")},
                     "ambiguous": sum(t.ambiguous for t in shorts),
                     "fill_bar_tp": sum(t.fill_bar_tp for t in shorts),
                     "by_outcome": {o: sum(t.outcome == o for t in shorts) for o in ("tp", "stop")},
                     "median_hold_bars": float(np.median([hold_bars(t) for t in shorts]))
                     if shorts else float("nan"),
                     "median_risk_pct": float(np.median([t.risk_pct for t in shorts])) if shorts else float("nan"),
                     "planner_rr_median": float(np.median([float(t.features.get("rr") or 0) for t in shorts]))
                     if shorts else float("nan"),
                     "native_rr_median": float(np.median([t.planned_rr for t in shorts])) if shorts else float("nan")}
    s["by_entry_kind"] = {kind: S.describe([v for t, v in rows if t.kind == kind])
                          for kind in (LIMIT, MARKET_CLOSE)}
    ax = [float(t.features.get("stop_atr_x") or 0) for t, _ in rows]
    med = float(np.median(ax)) if ax else 0.0
    s["stop_atr_strata"] = {"median_stop_atr_x": med,
                            "below": S.describe([v for (t, v), a in zip(rows, ax) if a < med]),
                            "at_or_above": S.describe([v for (t, v), a in zip(rows, ax) if a >= med])}
    mk = market_weeks(bars_by)
    s["by_market_week"] = {
        "universe_fell": S.describe([v for t, v in rows if mk.get(S.week_of(t.entry_t), 0.0) < 0]),
        "universe_rose": S.describe([v for t, v in rows if mk.get(S.week_of(t.entry_t), 0.0) >= 0])}
    log(f"[secondary] shorts done ({time.time() - t0:.0f}s)")

    longs = trades_for(pool, coins, LONG)
    long_rows = excess_rows(pool, longs)
    dl = S.describe([t.r_net for t in longs])
    s["long_side_same_code"] = {**framing_stats(long_rows, B, two_way_se=False),
                                "trades": dl.get("n"), "mean_net_r": dl.get("mean"),
                                "total_net_r": dl.get("total"), "win": dl.get("win")}
    s["combined_book"] = combined_book(longs, shorts, B)
    log(f"[secondary] long side and combined book done ({time.time() - t0:.0f}s)")

    f_rows = excess_rows(pool, shorts, funding=FUNDING_PCT_PER_HOUR)
    s["funding_sensitivity"] = {
        "pct_per_hour": FUNDING_PCT_PER_HOUR,
        "absolute_mean_net_r": float(np.mean([t.r_net + C.funding_r(t, FUNDING_PCT_PER_HOUR)
                                              for t in shorts])) if shorts else float("nan"),
        "excess": framing_stats(f_rows, B, two_way_se=False)}
    log(f"[secondary] funding sensitivity done ({time.time() - t0:.0f}s)")
    s["trades"] = [{"coin": t.coin, "entry_t": t.entry_t, "kind": t.kind, "outcome": t.outcome,
                    "r_net": round(t.r_net, 4), "excess": round(v, 4),
                    "risk_pct": round(t.risk_pct, 3), "hold_bars": hold_bars(t)}
                   for t, v in rows]
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
            f"{n} {r['estimate']:+.3f} (n {s['framings'][n]['n']}){' FIRED' if r['adds_edge'] else ''}"
            for n, r in per[-1].items()))
    summ = C.null_summary(per)
    return {"datasets": NULL_DATASETS, "framings": summ,
            "miscalibrated": {k: v["miscalibrated"] for k, v in summ.items()}, "per_dataset": per}


def fmt(x, d=3, sign=True):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def render(s: Dict[str, Any]) -> str:
    v, dec = s["framings"]["S_live"], s["decision"]["S_live"]
    a, cb, ls = s["absolute"], s["combined_book"], s["long_side_same_code"]
    L = ["# Short-side mirror -- results", "",
         f"Pre-registration: `research/backtest/PREREG_short_mirror.md`. Data fetched "
         f"{s['data']['fetched_at_utc']}; {len(s['coins'])} coins.", ""]
    if "null" in s:
        L += ["Null calibration: " + ", ".join(
            f"{k} fired {n['fired']}/{NULL_DATASETS}, mean {fmt(n['mean'], 4)} +/- {fmt(n['se'], 4, False)}"
            for k, n in s["null"].items()), ""]
    L += ["## The test (excess over random shorts with the same geometry)", "",
          "| | n | excess R | 95% CI | p (week-cluster) | p (two-way) | halves | absolute net R | controls' net R | edge |",
          "|---|---|---|---|---|---|---|---|---|---|",
          f"| S_live | {v['n']} | {fmt(v['estimate'])} | [{fmt(v.get('ci_lo'))}, {fmt(v.get('ci_hi'))}] | "
          f"{fmt(v['p'], 4, False)} | {fmt(v.get('p_2way'), 4, False)} | {', '.join(fmt(h) for h in v['halves'])} | "
          f"{fmt(v.get('absolute_mean_net_r'))} | {fmt(v.get('control_mean_r'))} | "
          f"**{'YES' if dec['has_edge'] else 'no'}** |", "",
          "Criteria: " + ", ".join(f"{k} {'yes' if ok else 'NO'}" for k, ok in dec["criteria"].items()), "",
          "## The mirror as a strategy (descriptive; portfolio rails not modelled)", "",
          f"- {a['n']} resolved shorts, win rate {fmt((a['win'] or 0) * 100, 1, False)}%, mean net R "
          f"{fmt(a['mean'])}, total {fmt(a['total'], 1)}R; {a['by_outcome']['tp']} targets, "
          f"{a['by_outcome']['stop']} stops; median hold {fmt(a['median_hold_bars'], 0, False)} bars, "
          f"median stop {fmt(a['median_risk_pct'], 2, False)}%",
          f"- planner rr median {fmt(a['planner_rr_median'], 2, False)} (inverted prices) vs "
          f"{fmt(a['native_rr_median'], 2, False)} as a real short",
          f"- long side through the same code: {ls['trades']} trades, mean net R {fmt(ls['mean_net_r'])}; "
          f"excess {fmt(ls['estimate'])} (p {fmt(ls['p'], 4, False)}, n {ls['n']})",
          f"- combined book over {cb['weeks']} weeks: long-only {fmt(cb['long_weekly_mean'], 2)}R/week "
          f"(sd {fmt(cb['long_weekly_sd'], 2, False)}), long + short {fmt(cb['combined_weekly_mean'], 2)}R/week "
          f"(sd {fmt(cb['combined_weekly_sd'], 2, False)}); shorts add {fmt(cb['added_by_shorts']['estimate'], 2)}"
          f" [{fmt(cb['added_by_shorts']['ci_lo'], 2)}, {fmt(cb['added_by_shorts']['ci_hi'], 2)}]; "
          f"weekly correlation {fmt(cb['corr_long_short_weekly'], 2)}", "",
          "## Secondary splits of the excess (descriptive)", ""]
    for k, d in s["by_entry_kind"].items():
        if d["n"]:
            L.append(f"- entry {k}: n {d['n']}, {fmt(d.get('mean'))} (iid p {fmt(d.get('p'), 4, False)})")
    st = s["stop_atr_strata"]
    L.append(f"- stop below {st['median_stop_atr_x']:.2f}x ATR: {fmt(st['below'].get('mean'))} (n {st['below']['n']}); "
             f"at or above: {fmt(st['at_or_above'].get('mean'))} (n {st['at_or_above']['n']})")
    for k, d in s["by_market_week"].items():
        L.append(f"- weeks the {k.replace('_', ' ')}: n {d['n']}, {fmt(d.get('mean'))}")
    fs = s["funding_sensitivity"]
    L.append(f"- with {fs['pct_per_hour']:.5f}%/h funding to shorts: absolute {fmt(fs['absolute_mean_net_r'])}, "
             f"excess {fmt(fs['excess']['estimate'])} (p {fmt(fs['excess']['p'], 4, False)})")
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
            print(f"[NULL] {k}: fired {v['fired']}/{NULL_DATASETS}, mean {v['mean']:+.4f} +/- {v['se']:.4f} "
                  f"-> {'MISCALIBRATED' if v['miscalibrated'] else 'ok'}")
        return 0
    cal = json.loads((RESULTS / "null_calibration.json").read_text())
    snap = D.Snapshot.load(SNAPSHOT)
    with C.pool(snap.bars) as pool:
        s = analyse(pool, snap.bars)
    s["decision"] = decide(s["framings"], cal["miscalibrated"])
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

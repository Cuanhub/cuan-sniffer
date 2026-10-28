"""Pre-registered experiment: does a 15m read help? See PREREG_ltf_15m.md.

    python -m research.backtest.experiments.ltf_15m --null    # calibrate first
    python -m research.backtest.experiments.ltf_15m           # on the fresh snapshots
"""

from __future__ import annotations

import argparse
import json
import math
import time
import zlib
from bisect import bisect_right
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from research.backtest import band
from research.backtest import common as C
from research.backtest import data as D
from research.backtest import ltf
from research.backtest import stats as S
from research.backtest.costs import MEASURED, Costs
from research.backtest.engine import (LIMIT, MARKET_CLOSE, MARKET_NEXT_OPEN, Order, Trade,
                                      atr_pct_series, simulate)

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"
RESULTS = ROOT / "results" / "ltf_15m"
SNAPSHOT_1H = CACHE / "snapshot_1h_fresh.json"
SNAPSHOT_15M = CACHE / "snapshot_15m.json"

WAIT = 192                     # 15m bars: 48 hours
REPS = 100
MIN_CTRL_ENTERED = 10          # random-arm replicates that must have entered
BID_TTL_15M = 48               # the live 12h bid life, in 15m bars
ENTRY_SLIP_SENSITIVITY = 0.0564
NULL_DATASETS = 20
NULL_WINDOW_15M = 5000
FRAMINGS = ("B1_band_stop", "B2_ltf_stop")
CENSOR_GUARD = 480             # 15m bars (5 days) every arm must leave for resolution
CONTROL_SPAN_15M = 240


def after_entry_control(bars15, tr: Trade, costs: Costs = MEASURED) -> Optional[float]:
    """PRIMARY control (PREREG Amendment 2): random 15m market entries in the
    240 15m bars AFTER this trade's entry, with its risk %, reward % and taker
    fee. None if fewer than common.MIN_CONTROL resolved."""
    risk = (tr.entry_px - tr.stop) / tr.entry_px
    reward = (tr.tp - tr.entry_px) / tr.entry_px
    rs = []
    for u in C.draws_after(len(bars15), tr.entry_i, C.rng_for(tr.coin, tr.entry_i, 4),
                           span=CONTROL_SPAN_15M, first=0):
        c = bars15[int(u)]["c"]
        t2 = simulate(bars15, Order(tr.coin, int(u), MARKET_CLOSE, c * (1 - risk),
                                    c * (1 + reward)), costs)
        if t2 is not None and t2.resolved:
            rs.append(t2.r_net)
    return float(np.mean(rs)) if len(rs) >= C.MIN_CONTROL else None


def confirm(bars15, br, lows, atr, arm: int, base: Trade, rule: str,
            costs: Costs) -> Tuple[str, Optional[Trade]]:
    """From the arm, wait up to WAIT 15m bars for a break (PREREG B1/B2)."""
    n = len(bars15)
    for t in range(arm, min(arm + WAIT + 1, n - 1)):
        # On 15m the arm bar is already after the arming moment (a bid's fill
        # quarter, or the first quarter after a 1h close), so its low counts.
        if bars15[t]["l"] <= base.stop:
            return "stopped_first", None
        if br[t]:
            stop = base.stop if rule == "band" else ltf.ltf_stop(bars15, lows, atr, t, base.stop)
            tr = simulate(bars15, Order(base.coin, t, MARKET_NEXT_OPEN, stop, base.tp), costs)
            return ("entered", tr) if tr is not None else ("invalid_at_entry", None)
    return "no_break", None


def measure_on_15m(base: Trade, bars_1h, t15, bars15) -> Optional[Trade]:
    """The SAME order the 1h engine simulated, simulated on 15m bars."""
    close_t = bars_1h[base.decided_i]["t"] + D.HOUR_MS        # the 1h decision close
    i = bisect_right(t15, close_t - 1) - 1                     # last 15m bar before it
    if i < 0 or bars15[i]["t"] + ltf.QUARTER_MS != close_t:
        return None
    if base.kind == LIMIT:
        o = Order(base.coin, i, LIMIT, base.stop, base.tp, limit=base.entry_px, ttl=BID_TTL_15M)
    elif base.kind == MARKET_CLOSE:
        o = Order(base.coin, i, MARKET_CLOSE, base.stop, base.tp)
    else:
        return None
    tr = simulate(bars15, o, MEASURED)
    return tr if tr is not None and tr.resolved else None


def _coin(args) -> List[Dict[str, Any]]:
    coin, trades, costs = args
    bars_1h, bars15 = C.BARS[coin], C.EXTRA["b15"].get(coin)
    if not bars15:
        return []
    t15 = [b["t"] for b in bars15]
    br = ltf.bos_breaks(bars15)
    lows = ltf.swing_lows(bars15)
    atr = atr_pct_series(bars15)
    atr1h = atr_pct_series(bars_1h)
    n = len(bars15)
    rng = np.random.default_rng([C.SEED, zlib.crc32(coin.encode()), 15])
    shifted = [br[-int(o):] + br[:-int(o)] for o in rng.integers(WAIT, n - WAIT, REPS)]
    rows = []
    for base in trades:
        row: Dict[str, Any] = {"base": base}
        m15 = measure_on_15m(base, bars_1h, t15, bars15)
        row["measure_15m"] = m15
        arm = ltf.arm_index(base, bars_1h, t15, bars15)
        if arm is None or arm + WAIT + CENSOR_GUARD >= n - 1:
            rows.append(row)
            continue
        row["arm"] = arm
        for rule, name in (("band", "B1_band_stop"), ("ltf", "B2_ltf_stop")):
            status, tr = confirm(bars15, br, lows, atr, arm, base, rule, costs)
            vals, entered_r = [], []
            for sh in shifted:
                _, t2 = confirm(bars15, sh, lows, atr, arm, base, rule, costs)
                if t2 is not None and not t2.resolved:
                    continue
                vals.append(t2.r_net if t2 is not None else 0.0)
                if t2 is not None:
                    entered_r.append(t2.r_net)
            row[name] = {"status": status, "trade": tr,
                         "excess": (tr.r_net - m) if tr is not None and tr.resolved
                         and (m := after_entry_control(bars15, tr, costs)) is not None else None,
                         "ctrl": float(np.mean(vals)) if vals else None,
                         "ctrl_entered_mean": float(np.mean(entered_r))
                         if len(entered_r) >= MIN_CTRL_ENTERED else None,
                         "ctrl_entry_rate": len(entered_r) / len(vals) if vals else None}
            if tr is not None and name == "B2_ltf_stop":
                risk_pct = (tr.entry_px - tr.stop) / tr.entry_px * 100
                hour = min(base.entry_i, len(atr1h) - 1)
                row["b2_stop_atr1h"] = risk_pct / atr1h[hour] if atr1h[hour] > 0 else None
        rows.append(row)
    return rows


def excess_rows(rows, name: str):
    """PRIMARY (PREREG Amendment 2): per confirmed trade, its net R minus
    after-entry geometry-matched random entries."""
    d, cl, tm = [], [], []
    for r in rows:
        x = r.get(name)
        if x and x.get("excess") is not None:
            d.append(x["excess"])
            cl.append(S.week_of(r["base"].entry_t))
            tm.append(r["base"].entry_t)
    return d, cl, tm


def quality(rows, name: str):
    """PRIMARY (PREREG Amendment 1): two per-trade means, UNPAIRED.

    Group 1: the arm's trades, one per setup where it entered. Group 0: for
    EVERY armed setup, the mean net R of the random arm's entered trades.
    Pairing them on the same setups selected setups where price rallied after
    arming, and the random arm's early entries on those rode the rally -- a
    -0.244R bias on random walks. Neither group here conditions on the other,
    and per-trade means cannot reward entering less often.
    """
    vals, grp, cl, tm = [], [], [], []
    for r in rows:
        x = r.get(name)
        if not x:
            continue
        week, t = S.week_of(r["base"].entry_t), r["base"].entry_t
        if x["trade"] is not None and x["trade"].resolved:
            vals.append(x["trade"].r_net)
            grp.append(1)
            cl.append(week)
            tm.append(t)
        if x["ctrl_entered_mean"] is not None:
            vals.append(x["ctrl_entered_mean"])
            grp.append(0)
            cl.append(week)
            tm.append(t)
    return vals, grp, cl, tm


def paired(rows, name: str):
    """SECONDARY: per setup (arm or 0) minus (random arm or 0)."""
    d, d_base, cl, tm = [], [], [], []
    for r in rows:
        x = r.get(name)
        if not x or x["ctrl"] is None:
            continue
        tr = x["trade"]
        if tr is not None and not tr.resolved:
            continue
        arm_r = tr.r_net if tr is not None else 0.0
        d.append(arm_r - x["ctrl"])
        d_base.append(arm_r - r["base"].r_net)
        cl.append(S.week_of(r["base"].entry_t))
        tm.append(r["base"].entry_t)
    return d, d_base, cl, tm


def analyse(pool, bars_1h, bars_15m, B: int = 10_000, secondaries: bool = True,
            log=print, costs: Costs = MEASURED) -> Dict[str, Any]:
    coins = sorted(c for c in bars_1h if c in bars_15m)
    t0 = time.time()
    base = dict(zip(coins, pool.map(_baseline, coins)))
    rows = [r for rs in pool.map(_coin, [(c, base[c], costs) for c in coins]) for r in rs]
    armed = [r for r in rows if "arm" in r]
    log(f"[LTF] {len(rows)} baseline trades, {len(armed)} armed inside the 15m window "
        f"({time.time() - t0:.0f}s)")
    fr: Dict[str, Dict[str, Any]] = {}
    extra: Dict[str, Any] = {}
    for name in FRAMINGS:
        d, cl, tm = excess_rows(rows, name)
        _, d_base, _, _ = paired(rows, name)
        if len(d) >= 2:
            boot = S.cluster_bootstrap(d, [0] * len(d), cl, "mean", B=B)
            fr[name] = {"n": len(d),
                        **{k: boot[k] for k in ("estimate", "se", "ci_lo", "ci_hi", "p", "clusters")},
                        "halves": [float(np.mean(h)) if h else float("nan") for h in C.halves(d, tm)],
                        "p_iid": S.describe(d)["p"]}
        else:
            fr[name] = {"n": len(d), "estimate": float("nan"), "p": float("nan"),
                        "halves": [float("nan")] * 2}
        if secondaries:
            st: Dict[str, int] = {}
            for r in armed:
                st[r[name]["status"]] = st.get(r[name]["status"], 0) + 1
            conf = [r[name]["trade"] for r in armed
                    if r[name]["trade"] is not None and r[name]["trade"].resolved]
            ps, _, pcl, _ = paired(rows, name)
            uv, ug, ucl, _ = quality(rows, name)
            extra[name] = {"status": st, "vs_baseline": S.describe(d_base),
                           "unpaired_vs_random_arm": S.cluster_bootstrap(uv, ug, ucl, "diff", B=B)
                           if sum(ug) > 1 and len(ug) - sum(ug) > 1 else None,
                           "per_setup_vs_random": S.cluster_bootstrap(
                               ps, [0] * len(ps), pcl, "mean", B=B) if len(ps) > 1 else None,
                           "entry_rate": st.get("entered", 0) / max(1, len(armed)),
                           "ctrl_entry_rate": float(np.mean([r[name]["ctrl_entry_rate"]
                                                             for r in armed
                                                             if r[name]["ctrl_entry_rate"] is not None]))
                           if armed else None,
                           "confirmed_trades": S.describe([t.r_net for t in conf]),
                           "median_rr_at_entry": float(np.median([t.planned_rr for t in conf]))
                           if conf else None}
    holm = S.holm({k: v["p"] for k, v in fr.items()})
    s: Dict[str, Any] = {"framings": fr, "holm": holm, "decision": decide(fr, holm)}
    if not secondaries:
        return s
    s["secondary"] = extra
    b2x = [r["b2_stop_atr1h"] for r in armed if r.get("b2_stop_atr1h") is not None]
    s["b2_stop_in_1h_atr"] = {"median": float(np.median(b2x)) if b2x else None, "n": len(b2x),
                              "baseline_median": float(np.median(
                                  [float(r["base"].features.get("stop_atr_x") or 0) for r in armed]))
                              if armed else None}
    # A -- measurement: the same orders seen at 15m
    pairs = [(r["base"], r["measure_15m"]) for r in rows if r.get("measure_15m") is not None]
    diffs = [m.r_net - b.r_net for b, m in pairs]
    s["measurement"] = {
        "trades": len(pairs),
        "mean_r_1h": float(np.mean([b.r_net for b, _ in pairs])) if pairs else None,
        "mean_r_15m": float(np.mean([m.r_net for _, m in pairs])) if pairs else None,
        "mean_diff_15m_minus_1h": float(np.mean(diffs)) if diffs else None,
        "outcome_changed": sum(1 for b, m in pairs if b.outcome != m.outcome),
        "1h_ambiguous": sum(1 for b, _ in pairs if b.ambiguous),
        "1h_ambiguous_won_at_15m": sum(1 for b, m in pairs if b.ambiguous and m.outcome == "tp"),
        "1h_fill_hour_target": sum(1 for b, _ in pairs if b.fill_bar_tp),
        "1h_fill_hour_target_won_at_15m": sum(1 for b, m in pairs if b.fill_bar_tp and m.outcome == "tp"),
    }
    return s


def _baseline(coin: str) -> List[Trade]:
    return [t for t in band.run(coin, C.BARS[coin], MEASURED) if t.resolved]


def decide(fr, holm, miscal: Optional[Dict[str, bool]] = None) -> Dict[str, Any]:
    pos = lambda x: isinstance(x, (int, float)) and math.isfinite(x) and x > 0   # noqa: E731
    out = {}
    for k, v in fr.items():
        crit = {"holm_rejected": bool(holm[k]["reject"]), "positive": pos(v["estimate"]),
                "both_halves_positive": all(pos(h) for h in v["halves"])}
        if miscal is not None:
            crit["null_calibration_ok"] = not miscal.get(k, False)
        out[k] = {"criteria": crit, "adds_edge": all(crit.values()),
                  "material": pos(v["estimate"]) and v["estimate"] >= 0.10}
    return out


def null_calibration(log=print) -> Dict[str, Any]:
    per = []
    for k in range(NULL_DATASETS):
        hourly, fifteen = C.random_walks_15m(k)
        fifteen = {c: b[-NULL_WINDOW_15M:] for c, b in fifteen.items()}
        with C.pool(hourly, {"b15": fifteen}) as pool:
            s = analyse(pool, hourly, fifteen, B=2000, secondaries=False, log=lambda *_: None)
        per.append({n: {"estimate": v["estimate"], "adds_edge": s["decision"][n]["adds_edge"]}
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
    L = ["# 15-minute read -- results", "",
         f"Pre-registration: `research/backtest/PREREG_ltf_15m.md`. 1h data fetched "
         f"{s['data']['fetched_1h']}, 15m {s['data']['fetched_15m']} "
         f"({s['data']['days_15m']:.1f} days of 15m bars, {len(s['coins'])} coins).", ""]
    if "null" in s:
        L += ["Null calibration: " + ", ".join(
            f"{k} fired {v['fired']}/20, mean {fmt(v['mean'], 4)} +/- {fmt(v['se'], 4, False)}"
            for k, v in s["null"].items()), ""]
    L += ["## B -- 15m confirmation: each confirmed trade vs random entries after it, same geometry", "",
          "| framing | confirmed trades | excess R over later random entries | 95% CI | p (week-cluster) | Holm p | halves | adds edge |",
          "|---|---|---|---|---|---|---|---|"]
    for k, v in s["framings"].items():
        L.append(f"| {k} | {v['n']} | {fmt(v['estimate'])} | [{fmt(v.get('ci_lo'))}, {fmt(v.get('ci_hi'))}] | "
                 f"{fmt(v['p'], 4, False)} | {fmt(s['holm'][k]['p_holm'], 4, False)} | "
                 f"{', '.join(fmt(h) for h in v['halves'])} | **{'YES' if s['decision'][k]['adds_edge'] else 'no'}** |")
    for k, v in s.get("secondary", {}).items():
        L += ["", f"{k}: " + ", ".join(f"{a} {b}" for a, b in v["status"].items())
              + f"; entry rate {fmt(v['entry_rate'], 3, False)} (random arm {fmt(v['ctrl_entry_rate'], 3, False)}); "
              f"confirmed trades {fmt(v['confirmed_trades'].get('mean'))} net R (n {v['confirmed_trades']['n']}), "
              f"median rr at entry {fmt(v['median_rr_at_entry'], 2, False)}; vs baseline "
              f"{fmt(v['vs_baseline'].get('mean'))}; per setup vs random (secondary) "
              f"{fmt((v.get('per_setup_vs_random') or {}).get('estimate'))}"]
    b2 = s.get("b2_stop_in_1h_atr", {})
    L += ["", f"B2 stop distance: median {fmt(b2.get('median'), 2, False)}x 1h ATR "
          f"(baseline band stop {fmt(b2.get('baseline_median'), 2, False)}x)", "",
          "## A -- measurement: the same 1h-engine trades replayed on 15m bars", ""]
    m = s["measurement"]
    L += [f"{m['trades']} trades: mean net R {fmt(m['mean_r_1h'])} on 1h vs {fmt(m['mean_r_15m'])} on 15m "
          f"(difference {fmt(m['mean_diff_15m_minus_1h'])}); outcome changed for {m['outcome_changed']}; "
          f"1h-ambiguous bars {m['1h_ambiguous']} ({m['1h_ambiguous_won_at_15m']} were wins at 15m); "
          f"fill-hour targets {m['1h_fill_hour_target']} ({m['1h_fill_hour_target_won_at_15m']} wins at 15m)."]
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
    s1, s15 = D.Snapshot.load(SNAPSHOT_1H), D.Snapshot.load(SNAPSHOT_15M)
    with C.pool(s1.bars, {"b15": s15.bars}) as pool:
        s = analyse(pool, s1.bars, s15.bars)
        from dataclasses import replace
        slip = analyse(pool, s1.bars, s15.bars, secondaries=False, log=lambda *_: None,
                       costs=replace(MEASURED, market_entry_slip_pct=ENTRY_SLIP_SENSITIVITY))
    s["entry_slip_sensitivity"] = {k: v["estimate"] for k, v in slip["framings"].items()}
    s["decision"] = decide(s["framings"], s["holm"], cal["miscalibrated"])
    first = min(b[0]["t"] for b in s15.bars.values())
    last = max(b[-1]["t"] for b in s15.bars.values())
    s.update({"data": {"fetched_1h": s1.manifest.get("fetched_at_utc"),
                       "fetched_15m": s15.manifest.get("fetched_at_utc"),
                       "days_15m": (last - first) / 86_400_000},
              "coins": {c: {"1h": s1.manifest["coins"][c]["sha256"],
                            "15m": s15.manifest["coins"][c]["sha256"]}
                        for c in s15.bars if c in s1.manifest["coins"]},
              "null": cal["framings"], "runtime_sec": round(time.time() - t0, 1)})
    (RESULTS / "summary.json").write_text(json.dumps(s, indent=1, default=str))
    (RESULTS / "report.md").write_text(render(s))
    print(f"[DONE] {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

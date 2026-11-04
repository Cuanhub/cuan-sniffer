"""Post-hoc descriptives behind results/kronos/INTERPRETATION.md: written
after the pre-registered run (PREREG_kronos.md), and they decide nothing.

    python3 -m research.backtest.experiments.kronos_posthoc

No model: it reads the scores the run cached in cache/kronos/ (real.json and
the 20 null datasets), the snapshot and the null's own bars, and refuses any
cache that is not the run summary.json records.

- Forecast error: the mean path's 24h close against no change, on real bars
  and on the random walks.
- Direction: P(up 24h | K1 > 0) - P(up 24h | K1 <= 0). A bid becomes a trade
  only if price comes down to it, so most decisions are followed by a fall,
  on random walks too; the null shows what this statistic reads in noise.
- Why Spearman(K1, net R) is negative while D is positive: the ranks within
  stop-outs and within targets, and the win rate on each side of K1's sign.
"""

from __future__ import annotations

import json
import math
from statistics import mean, stdev
from typing import Any, Callable, Dict, List

import numpy as np

from research.backtest import data as D
from research.backtest import stats as S
from research.backtest.experiments import kronos_filter as KF
from research.kronos import score as K


def readings(scores: Dict[str, Dict[str, Any]], bars_by) -> List[Dict[str, Any]]:
    """One per scored decision with a 24h close: K1, the real move over the
    same 24 bars, both errors, week and coin. Each move must reproduce the
    naive error the run stored, so this reads the bar the run read."""
    out = []
    for key, s in scores.items():
        if s.get("k1") is None or "err_pct" not in s:
            continue
        coin, i = key.rsplit(":", 1)
        bars = bars_by[coin]
        dec = bars[int(i)]
        move = bars[int(i) + KF.DIR_STEP]["c"] / dec["c"] - 1.0
        if not math.isclose(abs(move) * 100, s["naive_err_pct"], rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(f"{key}: the 24h move is not the one the run scored")
        out.append({"k1": s["k1"], "move": move, "err": s["err_pct"],
                    "naive": s["naive_err_pct"], "week": S.week_of(dec["t"]), "coin": coin})
    return out


def forecast_error(rs: List[Dict[str, Any]]) -> Dict[str, float]:
    err = np.array([r["err"] for r in rs])
    naive = np.array([r["naive"] for r in rs])
    return {"n": len(rs), "kronos_pct": float(err.mean()), "no_change_pct": float(naive.mean()),
            "ratio": float(err.mean() / naive.mean()), "closer": float(np.mean(err < naive))}


def direction(rs: List[Dict[str, Any]], B: int) -> Dict[str, float]:
    up = [float(r["move"] > 0) for r in rs]
    g = [int(r["k1"] > 0) for r in rs]
    weeks = [r["week"] for r in rs]
    b = S.cluster_bootstrap(up, g, weeks, "diff", B=B)
    tw = KF.two_way(up, g, weeks, [r["coin"] for r in rs], b["se"], B)
    u, a = np.array(up), np.array(g)
    return {"estimate": b["estimate"], "se": b["se"], "se_2way": tw["se_2way"],
            "up_if_k1_up": float(u[a == 1].mean()), "up_if_k1_down": float(u[a == 0].mean()),
            "up_share": float(u.mean()), "k1_up_share": float(a.mean())}


def ranks(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """rows: the run's trades.json."""
    rows = [r for r in rows if r["k1"] is not None]
    out: Dict[str, Any] = {"all": K.spearman([r["k1"] for r in rows], [r["r_net"] for r in rows])}
    for o in ("stop", "tp"):
        sub = [r for r in rows if r["outcome"] == o]
        out[o] = {"n": len(sub), "spearman": K.spearman([r["k1"] for r in sub],
                                                       [r["r_net"] for r in sub])}
    for side, keep in (("k1_up", lambda r: r["k1"] > 0), ("k1_down", lambda r: r["k1"] <= 0)):
        sub = [r for r in rows if keep(r)]
        out[f"win_rate_{side}"] = float(np.mean([r["outcome"] == "tp" for r in sub]))
        out[f"n_{side}"] = len(sub)
    return out


def across(per: List[Dict[str, Any]], get: Callable[[Dict[str, Any]], float]) -> Dict[str, float]:
    v = [get(d) for d in per]
    return {"mean": mean(v), "sd": stdev(v), "min": min(v), "max": max(v),
            "positive": sum(x > 0 for x in v)}


def main() -> int:
    cfg = json.loads((KF.RESULTS / "summary.json").read_text())["config"]
    real = json.loads((KF.SCORES / "real.json").read_text())
    if real["config"] != cfg:
        raise SystemExit("cache/kronos/real.json is not the run summary.json records")
    rs = readings(real["scores"], D.Snapshot.load(KF.SNAPSHOT).bars)
    trades = json.loads((KF.RESULTS / "trades.json").read_text())
    out: Dict[str, Any] = {"real": {"forecast_error": forecast_error(rs),
                                    "direction": direction(rs, 10_000), "ranks": ranks(trades)}}
    want = {**cfg, "coins": KF.NULL_COINS, "bars": KF.NULL_BARS}
    per = []
    for k in range(KF.NULL_DATASETS):
        cached = json.loads((KF.SCORES / f"null_{k:02}.json").read_text())
        if cached["config"] != want:
            raise SystemExit(f"cache/kronos/null_{k:02}.json is not the registered null")
        nr = readings(cached["scores"], KF.null_bars(k, KF.NULL_COINS, KF.NULL_BARS))
        per.append({"forecast_error": forecast_error(nr), "direction": direction(nr, 2000)})
        print(f"  [NULL {k + 1:2}] {len(nr)} readings  error ratio "
              f"{per[-1]['forecast_error']['ratio']:.3f}  direction "
              f"{per[-1]['direction']['estimate']:+.3f}", flush=True)
    out["null"] = {"per_dataset": per,
                   "ratio": across(per, lambda d: d["forecast_error"]["ratio"]),
                   "closer": across(per, lambda d: d["forecast_error"]["closer"]),
                   "direction": across(per, lambda d: d["direction"]["estimate"]),
                   "direction_se_2way": across(per, lambda d: d["direction"]["se_2way"]),
                   "up_share": across(per, lambda d: d["direction"]["up_share"])}
    (KF.RESULTS / "posthoc.json").write_text(json.dumps(out, indent=1))
    fe, di, rk, nu = out["real"]["forecast_error"], out["real"]["direction"], out["real"]["ranks"], out["null"]
    print(f"[ERROR] 24h close: Kronos {fe['kronos_pct']:.2f}% vs no change {fe['no_change_pct']:.2f}% "
          f"(x{fe['ratio']:.3f}; closer in {fe['closer']:.1%} of {fe['n']}); null x"
          f"{nu['ratio']['mean']:.3f} +/- {nu['ratio']['sd']:.3f}, closer "
          f"{nu['closer']['mean']:.1%} +/- {nu['closer']['sd']:.1%}")
    print(f"[DIRECTION] up 24h: {di['up_if_k1_up']:.3f} if K1 > 0, {di['up_if_k1_down']:.3f} if not "
          f"(share up {di['up_share']:.3f}): {di['estimate']:+.4f}, two-way SE {di['se_2way']:.4f}; "
          f"null {nu['direction']['mean']:+.4f} +/- {nu['direction']['sd']:.4f}, positive in "
          f"{nu['direction']['positive']}/{KF.NULL_DATASETS}")
    print(f"[RANKS] Spearman(K1, net R) {rk['all']:+.3f}: within stop-outs {rk['stop']['spearman']:+.3f} "
          f"(n {rk['stop']['n']}), within targets {rk['tp']['spearman']:+.3f} (n {rk['tp']['n']}); "
          f"win rate {rk['win_rate_k1_up']:.3f} if K1 > 0 vs {rk['win_rate_k1_down']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

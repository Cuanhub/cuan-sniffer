"""Null calibration: does the experiment find edge in data that has none?

Random walks are exactly a martingale in price, so every "adds edge" on
them is a false positive and every non-zero mean is a bias. An experiment
is miscalibrated for a framing if, across the null datasets, EITHER
  - its decision fired more than MAX_FALSE times, OR
  - its mean estimate is further than MAX_BIAS_SE standard errors from 0.
The second check exists because a bias pointing AWAY from "adds edge"
never fires: the trendline experiment's first control design was biased by
-0.027R and passed the false-positive count 0 of 20.

These bars contain no market information and are never used as evidence
about the market -- only about whether the test itself can be trusted.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from research.backtest import data as D

DATASETS = 20
COINS = 12
BARS = 5000
MAX_FALSE = 2
MAX_BIAS_SE = 2.5


def random_walks(seed: int, coins: Optional[int] = None, n: Optional[int] = None
                 ) -> Dict[str, List[Dict[str, Any]]]:
    # Module values read HERE, not bound as defaults: a default argument is
    # captured at import, and then changing COINS changes nothing -- the bug
    # that once made a nine-config sweep test one config nine times.
    coins = COINS if coins is None else coins
    n = BARS if n is None else n
    rng = np.random.default_rng([20261027, seed, 99])
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
                         "h": float(max(prev, seg.max())), "l": float(min(prev, seg.min())),
                         "c": float(seg[-1]), "v": 1.0, "n": 1})
            prev = float(seg[-1])
        out[f"NULL{seed}_{c}"] = bars
    return out


def calibrate(run_once: Callable[[Dict[str, List[Dict[str, Any]]]], Dict[str, Any]],
              datasets: Optional[int] = None, log=print) -> Dict[str, Any]:
    """run_once(bars_by) -> evaluate() output. Returns the calibration."""
    datasets = DATASETS if datasets is None else datasets
    per = []
    for k in range(datasets):
        res = run_once(random_walks(k))
        row = {name: {"estimate": d_est(res, name), "adds_edge": d["adds_edge"]}
               for name, d in res["decisions"].items()}
        per.append(row)
        log(f"  [NULL {k + 1:2}/{datasets}] " + "  ".join(
            f"{n} {r['estimate']:+.3f}{' FIRED' if r['adds_edge'] else ''}"
            for n, r in row.items()))
    names = list(per[0]) if per else []
    out: Dict[str, Any] = {"datasets": datasets, "framings": {}}
    for name in names:
        est = [r[name]["estimate"] for r in per if math.isfinite(r[name]["estimate"])]
        fired = sum(r[name]["adds_edge"] for r in per)
        m = float(np.mean(est)) if est else float("nan")
        se = float(np.std(est, ddof=1) / math.sqrt(len(est))) if len(est) > 1 else float("nan")
        centred = bool(math.isfinite(m) and math.isfinite(se) and abs(m) <= MAX_BIAS_SE * se)
        out["framings"][name] = {"fired": fired, "mean": m, "se": se, "centred": centred,
                                 "miscalibrated": fired > MAX_FALSE or not centred}
    out["miscalibrated"] = {k: v["miscalibrated"] for k, v in out["framings"].items()}
    return out


def d_est(res: Dict[str, Any], name: str) -> float:
    v = res["framings"][name].get("estimate")
    return float(v) if isinstance(v, (int, float)) else float("nan")

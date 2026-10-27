"""Statistics for backtest comparisons, with the dependence taken seriously.

Crypto perps move together. Forty coins stopping out in the same week are not
forty independent observations, and an iid t-test on them reports a precision
the data does not have. So the decision statistic here is a CLUSTER
bootstrap that resamples whole calendar weeks -- every trade from a drawn
week comes along -- which keeps the cross-coin correlation intact. The iid
Welch test is reported beside it for comparison, never instead of it.

No scipy in this environment, so the t distribution is computed here from
the regularized incomplete beta function (Numerical Recipes' continued
fraction) and pinned in test_backtest_stats to published table values.
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------- t tails --
def _betacf(a: float, b: float, x: float) -> float:
    MAXIT, EPS, FPMIN = 300, 3e-16, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > FPMIN else FPMIN)
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > FPMIN else FPMIN)
        c = 1.0 + aa / c
        c = c if abs(c) > FPMIN else FPMIN
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > FPMIN else FPMIN)
        c = 1.0 + aa / c
        c = c if abs(c) > FPMIN else FPMIN
        de = d * c
        h *= de
        if abs(de - 1.0) < EPS:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbt = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
           + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lbt) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbt) * _betacf(b, a, 1.0 - x) / b


def t_two_sided_p(t: float, df: float) -> float:
    if not math.isfinite(t):
        return 0.0
    if df <= 0:
        return float("nan")
    return betainc(df / 2.0, 0.5, df / (df + t * t))


def normal_two_sided_p(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2.0))


# ---------------------------------------------------------- descriptives --
def describe(xs: Sequence[float]) -> Dict[str, float]:
    a = np.asarray(xs, dtype=float)
    n = len(a)
    if n == 0:
        return {"n": 0}
    mean = float(a.mean())
    sd = float(a.std(ddof=1)) if n > 1 else float("nan")
    se = sd / math.sqrt(n) if n > 1 else float("nan")
    t = mean / se if n > 1 and se > 0 else float("nan")
    return {"n": n, "mean": mean, "sd": sd, "se": se, "t": t,
            "p": t_two_sided_p(t, n - 1) if n > 1 and math.isfinite(t) else float("nan"),
            "total": float(a.sum()), "win": float((a > 0).mean())}


def welch(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """mean(a) - mean(b), iid. Reported, never decided on alone."""
    x, y = np.asarray(a, float), np.asarray(b, float)
    if len(x) < 2 or len(y) < 2:
        return {"diff": float("nan"), "t": float("nan"), "df": float("nan"),
                "p": float("nan")}
    vx, vy = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
    se = math.sqrt(vx + vy)
    diff = float(x.mean() - y.mean())
    t = diff / se if se > 0 else float("nan")
    df = (vx + vy) ** 2 / (vx ** 2 / (len(x) - 1) + vy ** 2 / (len(y) - 1)) \
        if vx + vy > 0 else float("nan")
    return {"diff": diff, "se": se, "t": t, "df": df,
            "p": t_two_sided_p(t, df) if math.isfinite(t) else float("nan")}


# ----------------------------------------------------- cluster bootstrap --
def week_of(ms: int) -> Tuple[int, int]:
    d = datetime.fromtimestamp(ms / 1000, timezone.utc).isocalendar()
    return (d[0], d[1])


def cluster_bootstrap(values: Sequence[float], groups: Sequence[int],
                      clusters: Sequence[Any], stat: str = "diff",
                      B: int = 10_000, seed: int = 20261027
                      ) -> Dict[str, float]:
    """Resample whole clusters (weeks) with replacement.

    stat="mean": mean of all values (groups ignored).
    stat="diff": mean(group 1) - mean(group 0); a draw with either group
                 empty is redrawn's equivalent -- it is skipped and counted.
    Returns the point estimate, the bootstrap SE, a 95% percentile CI and a
    normal-approximation p-value from the bootstrap SE.
    """
    v = np.asarray(values, float)
    g = np.asarray(groups, int)
    by: Dict[Any, List[int]] = defaultdict(list)
    for idx, c in enumerate(clusters):
        by[c].append(idx)
    keys = list(by)
    members = [np.asarray(by[k]) for k in keys]
    sums = np.array([[v[m][g[m] == 0].sum(), v[m][g[m] == 1].sum()] for m in members])
    cnts = np.array([[(g[m] == 0).sum(), (g[m] == 1).sum()] for m in members])

    def point(s, c):
        if stat == "mean":
            tot = c.sum()
            return s.sum() / tot if tot else float("nan")
        if c[:, 0].sum() == 0 or c[:, 1].sum() == 0:
            return float("nan")
        return s[:, 1].sum() / c[:, 1].sum() - s[:, 0].sum() / c[:, 0].sum()

    est = point(sums, cnts)
    rng = np.random.default_rng(seed)
    K = len(keys)
    draws = np.empty(B)
    skipped = 0
    for bi in range(B):
        pick = rng.integers(0, K, K)
        val = point(sums[pick], cnts[pick])
        if not math.isfinite(val):
            skipped += 1
        draws[bi] = val
    draws = draws[np.isfinite(draws)]
    se = float(draws.std(ddof=1)) if len(draws) > 1 else float("nan")
    lo, hi = (np.percentile(draws, [2.5, 97.5]) if len(draws)
              else (float("nan"), float("nan")))
    z = est / se if se and se > 0 and math.isfinite(est) else float("nan")
    return {"estimate": float(est), "se": se, "ci_lo": float(lo),
            "ci_hi": float(hi), "z": z,
            "p": normal_two_sided_p(z) if math.isfinite(z) else float("nan"),
            "clusters": K, "draws": len(draws), "skipped": skipped}


def holm(pvals: Dict[str, float], alpha: float = 0.05) -> Dict[str, Dict[str, Any]]:
    """Holm-Bonferroni. Adjusted p-values are monotone, as they must be."""
    items = sorted(pvals.items(), key=lambda kv: (math.isnan(kv[1]), kv[1]))
    m = len(items)
    out: Dict[str, Dict[str, Any]] = {}
    running = 0.0
    still = True
    for rank, (name, p) in enumerate(items):
        adj = min(1.0, (m - rank) * p) if math.isfinite(p) else float("nan")
        running = max(running, adj) if math.isfinite(adj) else running
        reject = still and math.isfinite(p) and p <= alpha / (m - rank)
        still = still and reject
        out[name] = {"p": p, "p_holm": running if math.isfinite(adj) else float("nan"),
                     "reject": bool(reject)}
    return out


def split_half(values: Sequence[float], groups: Sequence[int],
               times: Sequence[int]) -> List[Dict[str, float]]:
    """The group difference in the first and second half of the period,
    split at the median time. An effect real enough to trade has the same
    sign in both."""
    order = np.argsort(np.asarray(times))
    half = len(order) // 2
    out = []
    for part in (order[:half], order[half:]):
        v = np.asarray(values, float)[part]
        g = np.asarray(groups, int)[part]
        a, b = v[g == 1], v[g == 0]
        out.append({"n1": int(len(a)), "n0": int(len(b)),
                    "mean1": float(a.mean()) if len(a) else float("nan"),
                    "mean0": float(b.mean()) if len(b) else float("nan"),
                    "diff": float(a.mean() - b.mean()) if len(a) and len(b) else float("nan")})
    return out

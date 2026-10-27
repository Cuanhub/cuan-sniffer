"""Records in, verdicts out. The only place a verdict is computed.

An experiment returns one row per unit of analysis:

    {"framing": str, "t": int (decision time, ms), "coin": str,
     "value": float, "group": 0 | 1}        # group only for stat "diff"

and declares its framings:

    FRAMINGS = {
      "filter":      {"role": "primary", "stat": "diff"},
      "filter_ctrl": {"role": "must_agree", "with": "filter", "stat": "diff"},
    }

"diff" estimates mean(group 1) - mean(group 0); "mean" estimates the mean
value (an excess over a control, or a paired difference). Positive must
always mean "better for the strategy" -- the pre-registration says how.

The decision rule for a primary framing, ALL required:
  1. Holm-rejected among the experiment's primary framings (alpha 0.05 in
     development; 0.05 / openings in the holdout)
  2. estimate > 0
  3. positive in both halves of the period (split at the median time)
  4. every must_agree framing linked to it is positive
  5. at least MIN_N units (per group, for "diff")
  6. the null calibration did not mark it miscalibrated
Material, reported separately: estimate >= MIN_EFFECT (0.10R).

Inference is a cluster bootstrap over ISO weeks (research.backtest.stats):
crypto perps move together, and trades in one week are not independent.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from research.backtest import stats as S

MIN_N = 30
MIN_EFFECT = 0.10
ALPHA = 0.05
ROLES = ("primary", "must_agree", "secondary")
STATS = ("diff", "mean")


def check_framings(framings: Dict[str, Dict[str, Any]]) -> None:
    if not framings:
        raise ValueError("FRAMINGS is empty")
    prim = [k for k, v in framings.items() if v.get("role") == "primary"]
    if not prim:
        raise ValueError("at least one framing must have role 'primary'")
    for k, v in framings.items():
        if v.get("role") not in ROLES:
            raise ValueError(f"{k}: role must be one of {ROLES}")
        if v.get("stat") not in STATS:
            raise ValueError(f"{k}: stat must be one of {STATS}")
        if v["role"] == "must_agree" and framings.get(v.get("with", ""), {}).get("role") != "primary":
            raise ValueError(f"{k}: must_agree needs 'with' naming a primary framing")


def check_rows(rows: List[Dict[str, Any]], framings: Dict[str, Dict[str, Any]]) -> None:
    for i, r in enumerate(rows):
        if r.get("framing") not in framings:
            raise ValueError(f"row {i}: framing {r.get('framing')!r} not declared")
        v = r.get("value")
        if not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError(f"row {i}: value must be a finite number, got {v!r}")
        if not isinstance(r.get("t"), int) or r["t"] < 1_000_000_000_000:
            raise ValueError(f"row {i}: t must be a millisecond timestamp")
        if framings[r["framing"]]["stat"] == "diff" and r.get("group") not in (0, 1):
            raise ValueError(f"row {i}: 'diff' rows need group 0 or 1")


def _select(rows, name, window: Optional[Tuple[int, int]]):
    out = [r for r in rows if r["framing"] == name]
    if window is not None:
        lo, hi = window
        out = [r for r in out if lo <= r["t"] < hi]
    return out


def framing_stats(rows: List[Dict[str, Any]], stat: str, B: int) -> Dict[str, Any]:
    vals = [float(r["value"]) for r in rows]
    times = [int(r["t"]) for r in rows]
    cl = [S.week_of(t) for t in times]
    if stat == "diff":
        grp = [int(r["group"]) for r in rows]
        n1, n0 = sum(grp), len(grp) - sum(grp)
        if n1 < 2 or n0 < 2:
            return {"n": len(vals), "n1": n1, "n0": n0, "estimate": float("nan"),
                    "p": float("nan"), "halves": [float("nan")] * 2}
        boot = S.cluster_bootstrap(vals, grp, cl, "diff", B=B)
        halves = [h["diff"] for h in S.split_half(vals, grp, times)]
        iid = S.welch([v for v, g in zip(vals, grp) if g],
                      [v for v, g in zip(vals, grp) if not g])
        return {"n": len(vals), "n1": n1, "n0": n0, **_pick(boot), "halves": halves,
                "p_iid": iid["p"]}
    if len(vals) < 2:
        return {"n": len(vals), "estimate": float("nan"), "p": float("nan"),
                "halves": [float("nan")] * 2}
    boot = S.cluster_bootstrap(vals, [0] * len(vals), cl, "mean", B=B)
    order = np.argsort(times)
    v = np.asarray(vals)[order]
    h = len(v) // 2
    halves = [float(v[:h].mean()) if h else float("nan"),
              float(v[h:].mean()) if len(v) - h else float("nan")]
    return {"n": len(vals), **_pick(boot), "halves": halves,
            "p_iid": S.describe(vals)["p"]}


def _pick(boot):
    return {k: boot[k] for k in ("estimate", "se", "ci_lo", "ci_hi", "p", "clusters")}


def _pos(x) -> bool:
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x) and x > 0


def evaluate(rows: List[Dict[str, Any]], framings: Dict[str, Dict[str, Any]],
             window: Optional[Tuple[int, int]] = None, alpha: Optional[float] = None,
             miscalibrated: Optional[Dict[str, bool]] = None, B: int = 10_000,
             min_n: Optional[int] = None) -> Dict[str, Any]:
    # Module values read here, not bound as defaults (captured at import).
    alpha = ALPHA if alpha is None else alpha
    min_n = MIN_N if min_n is None else min_n
    check_framings(framings)
    check_rows(rows, framings)
    out: Dict[str, Any] = {"framings": {}, "alpha": alpha, "window": window}
    for name, spec in framings.items():
        out["framings"][name] = {"role": spec["role"], "stat": spec["stat"],
                                 **framing_stats(_select(rows, name, window), spec["stat"], B)}
    prim = {k: v for k, v in out["framings"].items() if v["role"] == "primary"}
    holm = S.holm({k: v["p"] for k, v in prim.items()}, alpha=alpha)
    decisions = {}
    for k, v in prim.items():
        enough = (v.get("n1", v["n"]) >= min_n and v.get("n0", v["n"]) >= min_n)
        agree = [f for f, s in framings.items()
                 if s["role"] == "must_agree" and s.get("with") == k]
        crit = {
            "holm_rejected": bool(holm[k]["reject"]),
            "positive": _pos(v["estimate"]),
            "both_halves_positive": all(_pos(h) for h in v["halves"]),
            "must_agree_positive": all(_pos(out["framings"][f]["estimate"]) for f in agree),
            "enough_units": enough,
            "null_calibration_ok": not (miscalibrated or {}).get(k, False),
        }
        decisions[k] = {"criteria": crit, "adds_edge": all(crit.values()),
                        "material": _pos(v["estimate"]) and v["estimate"] >= MIN_EFFECT,
                        "p_holm": holm[k]["p_holm"]}
    out["holm"] = holm
    out["decisions"] = decisions
    return out


def render(res: Dict[str, Any], title: str) -> str:
    L = [f"# {title}", ""]
    if res.get("window"):
        L.append(f"window: {res['window']}  alpha: {res['alpha']}")
        L.append("")
    L += ["| framing | role | n | estimate | 95% CI | p (week-cluster) | p (iid) | halves |",
          "|---|---|---|---|---|---|---|---|"]
    for k, v in res["framings"].items():
        n = f"{v.get('n1', '')}/{v.get('n0', '')}" if v["stat"] == "diff" else str(v["n"])
        L.append(f"| {k} | {v['role']} | {n} | {_f(v.get('estimate'))} | "
                 f"[{_f(v.get('ci_lo'))}, {_f(v.get('ci_hi'))}] | {_f(v.get('p'), 4, False)} | "
                 f"{_f(v.get('p_iid'), 4, False)} | "
                 f"{', '.join(_f(h) for h in v.get('halves', []))} |")
    L += ["", "| primary | adds edge | material | failed criteria |", "|---|---|---|---|"]
    for k, d in res["decisions"].items():
        failed = [c for c, ok in d["criteria"].items() if not ok]
        L.append(f"| {k} | **{'YES' if d['adds_edge'] else 'no'}** | "
                 f"{'yes' if d['material'] else 'no'} | {', '.join(failed) or '-'} |")
    return "\n".join(L) + "\n"


def _f(x, d=3, sign=True):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"

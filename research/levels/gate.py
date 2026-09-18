"""
The "is this level live right now?" gate, plus full feature capture.

The user's own account showed why this gate is the whole system: a limit at
0.1428 on CASHCAT left resting blind from 08-20 fills immediately and rides a
-73.67% drawdown, while the identical order placed on 09-08 fills at the same
price on 09-10 and never goes more than -0.48% underwater. Same level, same
price, same exit -- a 150x difference in the drawdown you must survive, from
nothing but when the order became live.

Two conditions are gated on, both in sigma:
  1. a run of >= RUN_SIGMA_MIN followed by a pullback of >= PULL_SIGMA_MIN
  2. the level retested at least MIN_RETESTS times within RETEST_LOOKBACK bars

VOLUME IS DEDUCTED FROM THE GATE BUT STILL RECORDED. The user recalled volume
contracting into the level; at the actual fill bar every window measured
0.90-1.25x its reference, i.e. normal to elevated, so the remembered feature
is not the one computed here. Rather than guess which of "low absolute
volume", "declining volume trend" or "down-volume vs up-volume" was meant,
all three are captured on every episode so the question can be settled from
data later instead of argued. See DECISIONS.md 2026-09-17.

THRESHOLDS ARE PLACEHOLDERS. They were fitted to two levels the user named
after the fact, on charts used to generate the hypotheses. They are expected
to move once labelled episodes exist, and no result should be read from them
before then.
"""

import math
import statistics
from typing import Any, Dict, List, Optional

from research.levels.levels import log_sigma, sigma_distance

RUN_LOOKBACK = 84          # 4h bars (~14d) to measure the run over
RUN_SIGMA_MIN = 6.0
PULL_SIGMA_MIN = 5.0
RETEST_LOOKBACK = 180      # ~30d
MIN_RETESTS = 2
RETEST_BAND_SIGMA = 1.0
NEAR_SIGMA_MAX = 5.0       # how close price must already be to the level
VOL_RECENT = 12
VOL_REF = 60


def _vol_features(bars: List[Dict[str, Any]], i: int) -> Dict[str, float]:
    """Three competing readings of 'volume contracted into the level'. None is
    gated on; all are recorded so the real one can be identified later."""
    rec = [x.get("v", 0.0) for x in bars[max(0, i - VOL_RECENT):i + 1]]
    ref = [x.get("v", 0.0) for x in bars[max(0, i - VOL_REF):i + 1]]
    out = {"vol_ratio_recent_ref": 0.0, "vol_trend_slope": 0.0, "vol_down_up_ratio": 0.0}
    if ref and statistics.mean(ref) > 0:
        out["vol_ratio_recent_ref"] = statistics.mean(rec) / statistics.mean(ref)
    # declining-trend reading: OLS slope of recent volume, normalised by its mean
    if len(rec) > 3 and statistics.mean(rec) > 0:
        n = len(rec); xs = list(range(n)); mx = (n - 1) / 2; my = statistics.mean(rec)
        den = sum((x - mx) ** 2 for x in xs)
        if den > 0:
            out["vol_trend_slope"] = (sum((xs[k] - mx) * (rec[k] - my) for k in range(n)) / den) / my
    # down-volume vs up-volume over the recent window
    seg = bars[max(0, i - VOL_RECENT):i + 1]
    dn = sum(x.get("v", 0.0) for x in seg if x["c"] < x["o"])
    up = sum(x.get("v", 0.0) for x in seg if x["c"] >= x["o"])
    if up > 0:
        out["vol_down_up_ratio"] = dn / up
    return out


def evaluate(bars: List[Dict[str, Any]], i: int, level_price: float) -> Optional[Dict[str, Any]]:
    """Every feature for this (bar, level) pair, plus whether the gate fires.

    Returns None only when sigma is undefined. A non-firing evaluation is
    still returned in full: near-misses are exactly what a labelling pass
    needs in order to find where the gate's boundary should actually sit.
    """
    sigma = log_sigma(bars, i)
    if sigma <= 0 or level_price <= 0:
        return None
    w = bars[max(0, i - RUN_LOOKBACK):i + 1]
    if len(w) < 10:
        return None
    lo = min(x["l"] for x in w); hi = max(x["h"] for x in w); px = bars[i]["c"]
    if lo <= 0 or px <= 0:
        return None

    run_sigma = math.log(hi / lo) / sigma
    pull_sigma = math.log(hi / px) / sigma
    hi_idx = max(range(len(w)), key=lambda k: w[k]["h"])
    bars_since_high = len(w) - 1 - hi_idx

    prior = bars[max(0, i - RETEST_LOOKBACK):max(0, i - 3)]
    retests, last_retest_idx = 0, -1
    in_t = False
    for k, x in enumerate(prior, start=max(0, i - RETEST_LOOKBACK)):
        near = abs(sigma_distance(x["l"], level_price, sigma)) <= RETEST_BAND_SIGMA
        if near and not in_t:
            retests += 1; last_retest_idx = k; in_t = True
        elif not near:
            in_t = False

    near_sigma = sigma_distance(px, level_price, sigma)
    c_run = run_sigma >= RUN_SIGMA_MIN and pull_sigma >= PULL_SIGMA_MIN
    c_retest = retests >= MIN_RETESTS
    c_near = near_sigma <= NEAR_SIGMA_MAX

    feats = {
        "sigma_4h": round(sigma, 6),
        "run_sigma": round(run_sigma, 3),
        "pull_sigma": round(pull_sigma, 3),
        "run_pct": round((hi - lo) / lo * 100, 3),
        "pull_pct": round((hi - px) / hi * 100, 3),
        "bars_since_high": bars_since_high,
        "retests": retests,
        "bars_since_last_retest": (i - last_retest_idx) if last_retest_idx >= 0 else -1,
        "near_sigma": round(near_sigma, 3),
        "dist_to_level_pct": round((px - level_price) / level_price * 100, 3),
        "price": px,
        "cond_run_pullback": c_run,
        "cond_retest": c_retest,
        "cond_near": c_near,
        "gate_fires": bool(c_run and c_retest and c_near),
    }
    feats.update({k: round(v, 5) for k, v in _vol_features(bars, i).items()})
    return feats

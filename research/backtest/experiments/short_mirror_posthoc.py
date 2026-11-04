"""POST HOC for the short mirror (PREREG_short_mirror.md). Decides nothing.

Written after the registered run, to read it, not to rescue it:

  - native rr: the mirror's reward:risk gate acts on inverted prices, where
    a target below reads further than it is (median 2.40 there, 2.19 as a
    real short). Does the subset a native rr >= 2.0 gate would keep look
    different from what the leniency added?
  - entry kind on the week-clustered SE (the registered report gives iid).

    python -m research.backtest.experiments.short_mirror_posthoc
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from research.backtest import common as C
from research.backtest import data as D
from research.backtest import stats as S
from research.backtest.engine import LIMIT, MARKET_CLOSE, SHORT, Trade
from research.backtest.experiments import short_mirror as M


def split(rows: List[Tuple[Trade, float]], B: int = 10_000) -> Dict[str, Any]:
    if len(rows) < 10:
        return {"n": len(rows)}
    b = S.cluster_bootstrap([v for _, v in rows], [0] * len(rows),
                            [S.week_of(t.entry_t) for t, _ in rows], "mean", B=B)
    return {"n": len(rows), **{k: b[k] for k in ("estimate", "ci_lo", "ci_hi", "p")},
            "absolute_mean_net_r": float(np.mean([t.r_net for t, _ in rows]))}


def main() -> int:
    snap = D.Snapshot.load(M.SNAPSHOT)
    with C.pool(snap.bars) as pool:
        rows = M.excess_rows(pool, M.trades_for(pool, sorted(snap.bars), SHORT))
    out = {"all": split(rows),
           "native_rr_at_least_2": split([(t, v) for t, v in rows if t.planned_rr >= 2.0]),
           "native_rr_below_2": split([(t, v) for t, v in rows if t.planned_rr < 2.0]),
           "resting_sells": split([(t, v) for t, v in rows if t.kind == LIMIT]),
           "in_band_market_shorts": split([(t, v) for t, v in rows if t.kind == MARKET_CLOSE])}
    (M.RESULTS / "posthoc.json").write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        print(f"{k:<24} n {v['n']:4}  excess {v.get('estimate', float('nan')):+.3f}  "
              f"CI [{v.get('ci_lo', float('nan')):+.3f}, {v.get('ci_hi', float('nan')):+.3f}]  "
              f"p {v.get('p', float('nan')):.3f}  absolute {v.get('absolute_mean_net_r', float('nan')):+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""{{ID}}: EXAMPLE -- replace with your experiment. Design: PREREG.md.

THE PATTERN (keep it; replace the example inside it)

  TITLE      one line.
  FRAMINGS   name -> {"role": primary|must_agree|secondary, "stat": diff|mean}
             (+ "with": <primary> for must_agree). Positive = better.
  FEATURES   every per-bar series this experiment reads, as fn(bars) -> list
             with one value per bar. `lab register` proves each is causal on
             real bars by recomputing on a cut series. Read features at the
             DECISION bar, never later.
  records()  one row per unit per framing:
               {"framing", "t" (ms), "coin", "value" (net R), "group" (0/1, diff only)}
             NO statistics in here. The lab computes every estimate, interval,
             p-value and verdict from these rows (labcore/evaluate.py).

The example: does the band strategy do better in high-volatility regimes?
Group 1 = trades whose decision bar has ATR above its own 30-day median.
"""

from __future__ import annotations

from typing import Any, Dict, List

from labcore import pool
from research.backtest.common import same_order_control
from research.backtest.engine import atr_pct_series

TITLE = "EXAMPLE: band trades in high- vs low-volatility regimes"

FRAMINGS = {
    "high_vol_minus_low_vol": {"role": "primary", "stat": "diff"},
    "same_vs_random_entries": {"role": "must_agree", "with": "high_vol_minus_low_vol",
                               "stat": "diff"},
}

LOOKBACK = 720        # 30 days of 1h bars


def atr_regime(bars: List[Dict[str, Any]]) -> List[int]:
    """1 if ATR at bar t is above the median of the previous LOOKBACK ATRs.
    Uses bars 0..t only."""
    a = atr_pct_series(bars)
    out = []
    for t in range(len(bars)):
        prev = sorted(a[max(1, t - LOOKBACK):t])
        out.append(int(bool(prev) and a[t] > prev[len(prev) // 2]))
    return out


FEATURES = {"atr_regime": atr_regime}


def _coin(args) -> List[Dict[str, Any]]:
    """Runs in a worker: rows for one coin's baseline trades."""
    coin, trades = args
    bars = pool.bars(coin)
    regime = atr_regime(bars)
    rows = []
    for tr in trades:
        if not tr.resolved:
            continue                        # open at the data's end: no outcome yet
        g = regime[tr.decided_i]            # the feature AT the decision, never later
        rows.append({"framing": "high_vol_minus_low_vol", "t": tr.entry_t,
                     "coin": coin, "value": tr.r_net, "group": g})
        # Must-agree: the same comparison after subtracting what random orders
        # OF THE TRADE'S OWN TYPE earned after it, with its geometry -- a bid's
        # control is a bid, at the same offset, with the same life. Market
        # controls for bids were biased by -0.04R on random walks (the bid's
        # fill hour is judged conservatively; a market entry has none).
        cm = same_order_control(bars, tr)
        if cm is not None:
            rows.append({"framing": "same_vs_random_entries", "t": tr.entry_t,
                         "coin": coin, "value": tr.r_net - cm, "group": g})
    return rows


def records(ctx) -> List[Dict[str, Any]]:
    base = ctx.baseline()
    return [r for rows in ctx.map(_coin, [(c, base[c]) for c in ctx.coins()]) for r in rows]

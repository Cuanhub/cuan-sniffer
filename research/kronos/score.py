"""What a set of sampled futures says about one trade. Pure numpy and the
backtest engine: no PyTorch, so the main suite tests every line of it.

Two readings, both fixed by PREREG_kronos.md before any real forecast:

  direction()   K1: the mean forecast move from the decision close to the
                close `step` bars later, across the sampled paths;
  expected_r()  K2: the trade's OWN order -- same kind, stop, target, bid
                price and bid life -- run through the conservative engine on
                each sampled path, and averaged. A bid that never fills on a
                path scores 0 there; a trade still open at the end of a path
                is marked to its last close, paying the taker exit fee.

A sampled candle is not guaranteed to be a candle: the model emits open,
high, low and close as separate numbers. repair() makes each one
self-consistent (high the largest, low the smallest) and drops a path with a
non-finite or non-positive price, rather than letting the engine read an
impossible bar.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from research.backtest.costs import Costs, MEASURED
from research.backtest.engine import LIMIT, MARKET_CLOSE, Order, Trade, simulate

HOUR_MS = 3_600_000
BID_TTL_BARS = 12          # band.BID_TTL_BARS; the live bid life


def repair(path: np.ndarray) -> Optional[np.ndarray]:
    """(H, 6) sampled candles -> self-consistent candles, or None if any price
    is non-finite or not positive."""
    p = np.array(path[:, :4], dtype=np.float64)
    if not np.isfinite(p).all() or (p <= 0).any():
        return None
    o, h, l, c = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
    return np.column_stack([o, np.maximum.reduce([o, h, c]), np.minimum.reduce([o, l, c]), c])


def path_bars(path: np.ndarray, decision_t: int) -> List[Dict[str, Any]]:
    """Repaired (H, 4) candles as engine bars, timed after the decision bar."""
    return [{"t": decision_t + HOUR_MS * (k + 1), "o": float(r[0]), "h": float(r[1]),
             "l": float(r[2]), "c": float(r[3])} for k, r in enumerate(path)]


def valid_paths(paths: np.ndarray) -> List[np.ndarray]:
    out = []
    for p in paths:
        r = repair(p)
        if r is not None:
            out.append(r)
    return out


def direction(paths: Sequence[np.ndarray], decision_close: float, step: int) -> float:
    """K1: mean over paths of close[step] / decision close - 1 (step is
    1-based: step 24 is the close 24 bars after the decision bar)."""
    return float(np.mean([p[step - 1, 3] / decision_close - 1.0 for p in paths]))


def order_for(trade: Trade) -> Order:
    """The trade's own order, decided at index 0 of a [decision bar] + path
    sequence."""
    if trade.kind == LIMIT:
        return Order(trade.coin, 0, LIMIT, trade.stop, trade.tp, limit=trade.entry_px,
                     ttl=BID_TTL_BARS)
    if trade.kind == MARKET_CLOSE:
        return Order(trade.coin, 0, MARKET_CLOSE, trade.stop, trade.tp)
    raise ValueError(f"no path reading for entry kind {trade.kind!r}")


def path_r(order: Order, decision_bar: Dict[str, Any], path: List[Dict[str, Any]],
           costs: Costs = MEASURED) -> float:
    """Net R of the order on one path: resolved -> its net R; filled but open
    at the path's end -> marked to the last close, taker exit fee paid;
    never filled (a bid the path never reached) -> 0."""
    tr = simulate([decision_bar] + path, order, costs)
    if tr is None:
        return 0.0
    if tr.resolved:
        return tr.r_net
    risk = tr.entry_px - tr.stop
    return tr.r_net - costs.taker_fee_pct / 100 * tr.exit_px / risk


def expected_r(trade: Trade, decision_bar: Dict[str, Any], paths: Sequence[np.ndarray],
               costs: Costs = MEASURED) -> float:
    """K2: mean path_r over the paths."""
    o = order_for(trade)
    t0 = int(decision_bar["t"])
    return float(np.mean([path_r(o, decision_bar, path_bars(p, t0), costs) for p in paths]))


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Rank correlation, ties given their mean rank."""
    def ranks(v):
        v = np.asarray(v, float)
        order = np.argsort(v, kind="mergesort")
        r = np.empty(len(v))
        r[order] = np.arange(len(v), dtype=float)
        for val in np.unique(v):
            m = v == val
            if m.sum() > 1:
                r[m] = r[m].mean()
        return r
    if len(x) < 3:
        return float("nan")
    rx, ry = ranks(x), ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])

"""What Kronos is shown for a decision bar. Pure numpy: the main test suite
imports this without PyTorch, so the causality of the input is pinned there.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

HOUR_MS = 3_600_000
CLIP = 5.0
COLUMNS = ("o", "h", "l", "c", "v")


def window(bars: Sequence[Dict[str, Any]], i: int, lookback: int) -> Sequence[Dict[str, Any]]:
    """The `lookback` bars ending at bar i, inclusive. Bar i is the decision
    bar: closed when the decision is made, and the last thing the model sees."""
    if i + 1 < lookback:
        raise ValueError(f"bar {i} has only {i + 1} bars of history, need {lookback}")
    if i >= len(bars):
        raise IndexError(f"bar {i} is past the end of {len(bars)} bars")
    return bars[i + 1 - lookback:i + 1]


def features(win: Sequence[Dict[str, Any]]) -> np.ndarray:
    """(L, 6): open, high, low, close, volume, amount. Hyperliquid candles carry
    no quote volume, so amount is upstream's own fill-in for a missing column
    (KronosPredictor.predict): volume times the mean of open, high, low, close."""
    x = np.array([[b[k] for k in COLUMNS] for b in win], dtype=np.float64)
    amount = x[:, 4] * x[:, :4].mean(axis=1)
    return np.column_stack([x, amount])


def normalise(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Upstream's: per-window mean and std per column, then clip at 5.
    Returns (normalised, mean, std)."""
    m, s = x.mean(axis=0), x.std(axis=0)
    return np.clip((x - m) / (s + 1e-5), -CLIP, CLIP), m, s


def stamps(ts_ms: Sequence[int]) -> np.ndarray:
    """(N, 5) minute, hour, weekday, day, month in UTC: upstream's
    calc_time_stamps, without pandas. Bar times are candle OPEN times."""
    out = np.empty((len(ts_ms), 5), dtype=np.float32)
    for r, t in enumerate(ts_ms):
        d = np.datetime64(int(t), "ms").astype(object)
        out[r] = (d.minute, d.hour, d.weekday(), d.day, d.month)
    return out


def future_times(last_t: int, horizon: int) -> List[int]:
    return [last_t + HOUR_MS * (k + 1) for k in range(horizon)]

"""Random controls that cannot see the signal.

The lab's hardest-won rule (trendline PREREG Amendment 1, 2026-10-27): a
control entry must not overlap whatever DEFINES the signal. A breakout at
bar t exists because price rose into t, so random entries drawn from a
window AROUND t ride that same rise and make the control look better than
random. On random walks that bias was +0.021R -- enough to hide or invent a
small edge. These helpers only ever draw AFTER the signal.

Use them instead of writing your own. If an experiment needs a different
control, its pre-registration must say why and its null calibration must
show the control is centred on zero.
"""

from __future__ import annotations

import zlib
from typing import Callable, List, Optional

import numpy as np

SEED = 20261027


def rng_for(coin: str, i: int, salt: int = 0) -> np.random.Generator:
    """Deterministic per (coin, bar, salt): reruns reproduce exactly."""
    return np.random.default_rng([SEED, zlib.crc32(coin.encode()), int(i), int(salt)])


def draws_after(n_bars: int, signal_i: int, k: int = 200, span: int = 240,
                first_allowed: int = 0, rng: Optional[np.random.Generator] = None,
                coin: str = "") -> np.ndarray:
    """k random bar indices in (signal_i, signal_i + span], within the data."""
    lo = max(first_allowed, signal_i + 1)
    hi = min(n_bars - 2, signal_i + span)
    if hi < lo:
        return np.empty(0, dtype=np.int64)
    r = rng if rng is not None else rng_for(coin, signal_i, 1)
    return r.integers(lo, hi + 1, k)


def control_mean(outcome: Callable[[int], Optional[float]], draws: np.ndarray,
                 min_resolved: int = 50) -> Optional[float]:
    """Mean of outcome(u) over draws that resolved; None if too few did.

    `outcome` returns the control trade's net R at entry bar u, or None if
    that trade never filled or is still open at the end of the data. None
    is dropped, never counted as zero.
    """
    vals = [v for v in (outcome(int(u)) for u in draws) if v is not None]
    return float(np.mean(vals)) if len(vals) >= min_resolved else None


def circular_shift(series: List[bool], offset: int) -> List[bool]:
    """The same events at unrelated times: count and clustering kept,
    any relation to price destroyed. For "wait for an event" designs."""
    return series[-offset:] + series[:-offset]

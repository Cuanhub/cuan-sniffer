"""Worker processes that already hold the bars.

Experiment functions passed to `ctx.map` run in these workers and read bars
with `bars(coin)`. They must be module-level functions in the experiment's
module (that is how a process pool finds them).
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List

_BARS: Dict[str, List[Dict[str, Any]]] = {}


def _init(b: Dict[str, List[Dict[str, Any]]]) -> None:
    global _BARS
    _BARS = b


def bars(coin: str) -> List[Dict[str, Any]]:
    return _BARS[coin]


def make(bars_by: Dict[str, List[Dict[str, Any]]], workers: int = 8) -> ProcessPoolExecutor:
    _init(bars_by)                      # the parent can call worker functions too
    return ProcessPoolExecutor(max_workers=workers, initializer=_init,
                               initargs=(bars_by,))


def replay_band(coin: str):
    """The live strategy's trades on one coin (research.backtest.band)."""
    from research.backtest import band
    from research.backtest.costs import MEASURED
    return band.run(coin, _BARS[coin], MEASURED)

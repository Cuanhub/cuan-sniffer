"""Which book owns which coin, when two books share one account.

The venue holds one net position per coin per account, so two strategies on
the same account cannot share a coin: a band long and a trend short in BTC net
to whatever is left, and neither book holds what it decided.

While the trend book is armed (TREND_LIVE_ARMED=true) its ten coins are its
own. The band trader neither opens them nor counts them against its slots, its
loss-streak pause and close alerts ignore their fills, and protect leaves their
stops to the trend pass (tools/trend_live.py), which keeps a disaster stop on
each. A coin that still carries a band plan stays the band's until that trade
is over, so arming never strips a band position of its stop.

Unarmed, nothing anywhere changes.
"""

from __future__ import annotations

import os
from typing import FrozenSet, Optional

from core.meta import TradeMeta

# The trend book's universe, research/backtest/experiments/trend_daily.COINS.
# Spelled out so core does not import research code; a test holds them equal.
TREND_COINS: FrozenSet[str] = frozenset(
    ("BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "AVAX", "LINK", "DOT"))


def trend_armed() -> bool:
    """Read when asked, not at import, so no process depends on import order."""
    return os.getenv("TREND_LIVE_ARMED", "").strip().lower() == "true"


def band_excludes() -> FrozenSet[str]:
    """Coins the band book must not open, count or account for."""
    return TREND_COINS if trend_armed() else frozenset()


def trend_owns(coin: str, band_plan: Optional[TradeMeta]) -> bool:
    """Whether a position in `coin` is the trend book's to manage."""
    return trend_armed() and coin.upper() in TREND_COINS and band_plan is None

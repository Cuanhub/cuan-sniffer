# execution_backend.py
"""
Thin execution interface + fill result type.

Backend responsibilities:
  - entry fill simulation (paper) or real exchange fill (live)
  - exit fill simulation (paper) or real exchange close (live)
  - current price retrieval

Backend does NOT own:
  - balance, position state machine, risk logic, trade persistence
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class FillResult:
    """
    Returned by execute_entry / execute_exit.

    fill_ratio semantics:
        None  = fill was not attempted (rejected pre-flight)
        0.0   = fill attempted, got nothing (empty book / timeout)
        0.6   = partial fill, 60% of requested size
        1.0   = full fill
    """
    filled: bool
    fill_price: float
    fill_size_usd: float
    slippage_bps: float
    reject_reason: str = ""
    requested_price: float = 0.0
    requested_size_usd: float = 0.0
    fill_ratio: Optional[float] = None
    reason: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)


class ExecutionBackend(ABC):
    """
    Thin execution interface.

    BOTH entries AND exits go through this backend so that:
      - Paper: slippage + gap simulation applied to ALL fills
      - Live:  real exchange orders for ALL fills
      - No exit path ever bypasses the backend
    """

    @abstractmethod
    def execute_entry(
        self,
        coin: str,
        side: str,
        price: float,
        size_usd: float,
        confidence: float,
    ) -> FillResult:
        ...

    @abstractmethod
    def execute_exit(
        self,
        coin: str,
        side: str,
        price: float,
        size_usd: float,
        reason: str = "",
    ) -> FillResult:
        """
        reason describes exit type for slippage profiling:
          "stop_full", "stop_runner", "tp_full", "partial", "manual"
        Stop exits get worse slippage than TP exits.
        """
        ...

    @abstractmethod
    def get_mid_price(self, coin: str) -> Optional[float]:
        ...

    @abstractmethod
    def shutdown(self):
        ...
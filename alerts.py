# alerts.py
"""
AlertManager — context alerts for large flow events and funding extremes.

Called by agent.py on every cycle:
    alert_mgr.maybe_alert_large_flow(flow_snapshot)
    alert_mgr.maybe_alert_funding_extreme(funding_rate)

Separate from signal alerts: these fire on raw on-chain/funding conditions
regardless of whether a trade signal is generated.
"""

import time
from typing import Dict, Any

from notifier import send_telegram_message


class AlertManager:
    """
    Fires Telegram alerts for:
    1. Large flow events   — whale_pressure or 30m imbalance above threshold
    2. Funding extremes    — funding rate magnitude above threshold

    cooldown_seconds prevents the same alert type from spamming.
    """

    def __init__(
        self,
        min_flow_strength: float = 0.7,
        min_imbalance_30m: float = 0.7,
        min_funding_mag: float = 0.01,
        cooldown_seconds: int = 900,
    ):
        self.min_flow_strength  = min_flow_strength
        self.min_imbalance_30m  = min_imbalance_30m
        self.min_funding_mag    = min_funding_mag
        self.cooldown_seconds   = cooldown_seconds

        # Last fire timestamps per alert type
        self._last_fired: Dict[str, float] = {
            "large_flow":      0.0,
            "funding_extreme": 0.0,
        }

    # ----------------------------------------------------------------

    def _on_cooldown(self, key: str) -> bool:
        return (time.time() - self._last_fired.get(key, 0.0)) < self.cooldown_seconds

    def _mark_fired(self, key: str):
        self._last_fired[key] = time.time()

    # ----------------------------------------------------------------

    def maybe_alert_large_flow(self, flow_snapshot: Dict[str, Any]) -> bool:
        """
        Fire if whale_pressure or 30m imbalance exceeds threshold.
        Returns True if an alert was sent.
        """
        if not flow_snapshot:
            return False

        if self._on_cooldown("large_flow"):
            return False

        whale     = float(flow_snapshot.get("whale_pressure", 0.0))
        imbal_30m = 0.0
        w30 = flow_snapshot.get("30m", {})
        if isinstance(w30, dict):
            imbal_30m = float(w30.get("imbalance", 0.0))

        net_30m   = float(w30.get("net_flow", 0.0)) if isinstance(w30, dict) else 0.0
        inflow    = float(w30.get("inflow",   0.0)) if isinstance(w30, dict) else 0.0
        outflow   = float(w30.get("outflow",  0.0)) if isinstance(w30, dict) else 0.0

        triggered = (
            abs(whale)     >= self.min_flow_strength or
            abs(imbal_30m) >= self.min_imbalance_30m
        )

        if not triggered:
            return False

        direction = "inflow" if net_30m >= 0 else "outflow"
        emoji     = "🟢" if net_30m >= 0 else "🔴"

        msg = (
            f"{emoji} *Large SOL flow detected*\n"
            f"Whale pressure: `{whale:.3f}`\n"
            f"30m imbalance: `{imbal_30m:.3f}`\n"
            f"30m net flow: `{net_30m:.1f} SOL` ({direction})\n"
            f"30m inflow: `{inflow:.1f}` | outflow: `{outflow:.1f}`"
        )

        send_telegram_message(msg)
        self._mark_fired("large_flow")
        print(f"[ALERT] Large flow: whale={whale:.3f} imbal={imbal_30m:.3f}")
        return True

    # ----------------------------------------------------------------

    def maybe_alert_funding_extreme(self, funding_rate: float) -> bool:
        """
        Fire if |funding_rate| exceeds threshold.
        Returns True if an alert was sent.
        """
        if self._on_cooldown("funding_extreme"):
            return False

        if abs(funding_rate) < self.min_funding_mag:
            return False

        if funding_rate > 0:
            emoji  = "🔥"
            label  = "EXTREME positive — longs overloaded"
            signal = "contrarian SHORT bias"
        else:
            emoji  = "🧊"
            label  = "EXTREME negative — shorts overloaded"
            signal = "contrarian LONG bias"

        msg = (
            f"{emoji} *Funding rate extreme*\n"
            f"Funding: `{funding_rate:.6f}`\n"
            f"Condition: {label}\n"
            f"Context: {signal}"
        )

        send_telegram_message(msg)
        self._mark_fired("funding_extreme")
        print(f"[ALERT] Funding extreme: {funding_rate:.6f}")
        return True
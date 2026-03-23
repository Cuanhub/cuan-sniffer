from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from typing import Dict, List

from db import FlowEvent


class FlowContext:
    """
    Produces flow-based features for the strategy engine.

    - net inflow / outflow across time windows
    - whale cluster activity (distinct wallets)
    - flow momentum (increasing? decreasing?)
    - flow imbalance (normalized pressure)
    """

    def __init__(self, session_factory, windows=None):
        """
        windows: dict of {label: timedelta}
        Example:
            {
                "5m": timedelta(minutes=5),
                "30m": timedelta(minutes=30),
                "2h": timedelta(hours=2),
            }
        """
        self.session_factory = session_factory

        self.windows = windows or {
            "5m": timedelta(minutes=5),
            "30m": timedelta(minutes=30),
            "2h": timedelta(hours=2),
        }

    # -----------------------------------------------------------

    def _fetch_events(self, session: Session, since: datetime) -> List[FlowEvent]:
        """
        Fetch all flow events from DB since a given time.
        """
        return (
            session.query(FlowEvent)
            .filter(FlowEvent.created_at >= since)
            .order_by(FlowEvent.created_at.desc())
            .all()
        )

    # -----------------------------------------------------------

    def _compute_window_features(self, events: List[FlowEvent]) -> Dict:
        """
        Compute aggregated stats for a time window.
        """
        inflow = sum(ev.sol_amount for ev in events if ev.direction == "IN")
        outflow = sum(ev.sol_amount for ev in events if ev.direction == "OUT")

        total = inflow + outflow

        # Flow imbalance = normalized pressure value
        imbalance = 0.0
        if total > 0:
            imbalance = (inflow - outflow) / total  # range: -1 (full outflow) to +1 (full inflow)

        wallets = {ev.address for ev in events}   # distinct wallet count
        whale_count = len(wallets)

        return {
            "inflow": inflow,
            "outflow": outflow,
            "net_flow": inflow - outflow,
            "imbalance": imbalance,
            "whale_count": whale_count,
            "event_count": len(events),
        }

    # -----------------------------------------------------------

    def compute_flow_snapshot(self) -> Dict[str, Dict]:
        """
        Compute flow features across all configured time windows.
        Structure:
            {
                "5m": { ...features },
                "30m": { ... },
                "2h": { ... },
                "flow_momentum": value,
                "whale_pressure": value,
            }
        """
        session = self.session_factory()
        now = datetime.utcnow()

        snapshot = {}

        window_stats = []

        # compute per-window stats
        for label, delta in self.windows.items():
            events = self._fetch_events(session, now - delta)
            stats = self._compute_window_features(events)
            snapshot[label] = stats
            window_stats.append((label, stats["net_flow"]))

        session.close()

        # -------------------------------------------------------
        # Flow momentum
        # example: net_flow_5m > net_flow_30m -> accelerating
        # -------------------------------------------------------

        try:
            nf_5m = snapshot["5m"]["net_flow"]
            nf_30m = snapshot["30m"]["net_flow"]

            # positive = accelerating inflow
            # negative = accelerating outflow
            flow_momentum = nf_5m - nf_30m
        except KeyError:
            flow_momentum = 0.0

        # -------------------------------------------------------
        # Whale pressure score
        # Weighted combination of imbalance + momentum + whale count
        # -------------------------------------------------------

        try:
            imbalance = snapshot["30m"]["imbalance"]
            whales = snapshot["30m"]["whale_count"]
            whale_pressure = imbalance * (1 + whales / 5)
        except KeyError:
            whale_pressure = 0.0

        snapshot["flow_momentum"] = flow_momentum
        snapshot["whale_pressure"] = whale_pressure

        return snapshot

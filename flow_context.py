import os
import time
from datetime import datetime, timedelta, timezone
from sqlalchemy.orm import Session
from typing import Dict, List, Optional

from db import FlowEvent
from known_entities import is_known_entity
from token_config import MIN_TOKEN_FLOW_USD

# How long a computed snapshot is considered fresh before the next DB query.
# Set to 0 to disable caching (always recompute).  Default: 30s — short enough
# to track fast-moving flow bursts, long enough to avoid hammering SQLite on
# every 5-second agent poll cycle.
FLOW_SNAPSHOT_TTL_SEC = int(os.getenv("FLOW_SNAPSHOT_TTL_SEC", "30"))

# Max age a snapshot is allowed to be *served* even from cache.  If the cache
# is older than this, return an empty snapshot and log a warning rather than
# serving data that is too stale to be meaningful.
FLOW_SNAPSHOT_MAX_AGE_SEC = int(os.getenv("FLOW_SNAPSHOT_MAX_AGE_SEC", "300"))


def _utc_now_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class FlowContext:
    """
    Produces flow-based features for the strategy engine.

    - net inflow / outflow across time windows
    - whale cluster activity (distinct wallets)
    - flow momentum (increasing? decreasing?)
    - flow imbalance (normalized pressure)

    One instance per tracked coin.  SOL events store flow in SOL units;
    token events (JTO/WIF/FARTCOIN) store USD-equivalent value in sol_amount
    so the imbalance ratio and whale_pressure formula stay consistent.

    Caching: snapshots are cached for FLOW_SNAPSHOT_TTL_SEC seconds to avoid
    unnecessary DB round-trips on every agent poll cycle.  The cache is
    invalidated and re-queried on expiry.  If the cache is older than
    FLOW_SNAPSHOT_MAX_AGE_SEC (e.g. DB is unreachable), an empty snapshot
    with snapshot_age_sec set is returned so callers can detect staleness.
    """

    def __init__(self, session_factory, coin: str = "SOL", windows=None):
        """
        coin:    which coin's FlowEvents to query (default "SOL").
        windows: dict of {label: timedelta}
        """
        self.session_factory = session_factory
        self.coin = coin.upper()

        self.windows = windows or {
            "5m": timedelta(minutes=5),
            "30m": timedelta(minutes=30),
            "2h": timedelta(hours=2),
        }

        # Snapshot cache
        self._cached_snapshot: Optional[Dict] = None
        self._cache_computed_at: float = 0.0  # epoch seconds

    # -----------------------------------------------------------

    def _fetch_events(self, session: Session, since: datetime) -> List[FlowEvent]:
        """
        Fetch flow events for self.coin from DB since a given time, excluding
        known exchange/program wallets whose moves are routine operations.
        Token coins also apply a USD minimum retroactively (SOL already pre-filtered
        at write time by MIN_SOL_ALERT which converts to well above MIN_TOKEN_FLOW_USD).
        """
        query = session.query(FlowEvent).filter(
            FlowEvent.created_at >= since,
            FlowEvent.coin == self.coin,
        )
        if self.coin != "SOL":
            query = query.filter(FlowEvent.usd_value >= MIN_TOKEN_FLOW_USD)
        events = query.order_by(FlowEvent.created_at.desc()).all()
        return [ev for ev in events if not is_known_entity(ev.address)]

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

    def _build_snapshot(self) -> Dict:
        """Query the DB and compute a fresh snapshot."""
        session = self.session_factory()
        now = _utc_now_naive()
        snapshot = {}

        try:
            for label, delta in self.windows.items():
                events = self._fetch_events(session, now - delta)
                stats = self._compute_window_features(events)
                snapshot[label] = stats
        finally:
            session.close()

        # Flow momentum: rate-based acceleration (per-minute), not absolute subtraction.
        # Old formula subtracted absolute net_flow values — 30m window always dominated
        # because it accumulates more volume. Rate normalisation makes windows comparable.
        try:
            rate_5m = snapshot["5m"]["net_flow"] / 5.0
            rate_30m = snapshot["30m"]["net_flow"] / 30.0
            flow_momentum = rate_5m - rate_30m  # positive = accelerating inflow
        except KeyError:
            flow_momentum = 0.0

        # Whale pressure: volume-intensity weighted imbalance (self-calibrating).
        # Old formula used wallet count as the amplifier — that boosted RETAIL signals
        # (many small wallets) and underweighted institutional (few large wallets).
        # New formula: compare 30m volume vs the average 30m-equivalent rate from the 2h
        # window. If the current 30m is busier than the 2h baseline, intensity > 1.
        try:
            imbalance = snapshot["30m"]["imbalance"]
            total_30m = snapshot["30m"]["inflow"] + snapshot["30m"]["outflow"]
            total_2h = (
                snapshot.get("2h", {}).get("inflow", 0.0)
                + snapshot.get("2h", {}).get("outflow", 0.0)
            )
            if total_2h > 0:
                avg_rate_2h_per_30m = total_2h / 4.0
                vol_intensity = min(total_30m / avg_rate_2h_per_30m, 5.0)
            elif total_30m > 0:
                vol_intensity = 1.0
            else:
                vol_intensity = 0.0
            whale_pressure = imbalance * vol_intensity
        except KeyError:
            whale_pressure = 0.0

        snapshot["flow_momentum"] = flow_momentum
        snapshot["whale_pressure"] = whale_pressure
        return snapshot

    # -----------------------------------------------------------

    def compute_flow_snapshot(self) -> Dict:
        """
        Return a flow snapshot, served from the in-memory cache if fresh enough.

        The returned dict always contains a 'snapshot_age_sec' key so callers
        can decide how much weight to give the data.  A value of 0 means just
        computed; a large value signals staleness.

        If the cache is older than FLOW_SNAPSHOT_MAX_AGE_SEC, an empty snapshot
        is returned rather than serving data that could actively mislead signals.
        """
        now_ts = time.time()
        cache_age = now_ts - self._cache_computed_at

        # Serve from cache if still within TTL
        if self._cached_snapshot is not None and cache_age < FLOW_SNAPSHOT_TTL_SEC:
            snapshot = dict(self._cached_snapshot)
            snapshot["snapshot_age_sec"] = cache_age
            return snapshot

        # Cache is stale — attempt a fresh DB query
        try:
            fresh = self._build_snapshot()
            self._cached_snapshot = fresh
            self._cache_computed_at = now_ts
            snapshot = dict(fresh)
            snapshot["snapshot_age_sec"] = 0.0
            return snapshot

        except Exception as e:
            print(f"[FLOW_CONTEXT/{self.coin}] Failed to build snapshot: {e}")

            # Return the stale cache if it exists and isn't dangerously old
            if self._cached_snapshot is not None and cache_age < FLOW_SNAPSHOT_MAX_AGE_SEC:
                snapshot = dict(self._cached_snapshot)
                snapshot["snapshot_age_sec"] = cache_age
                print(f"[FLOW_CONTEXT/{self.coin}] Serving stale snapshot ({cache_age:.0f}s old)")
                return snapshot

            # Cache is gone or too old — return empty to avoid misleading signals
            print(f"[FLOW_CONTEXT/{self.coin}] No usable snapshot available — returning empty")
            return {"snapshot_age_sec": cache_age}

    def invalidate_cache(self):
        """Force next call to recompute from DB (e.g. after a new FlowEvent is written)."""
        self._cache_computed_at = 0.0

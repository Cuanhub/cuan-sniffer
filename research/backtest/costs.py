"""Execution costs, MEASURED from this account's own fills.

The defaults below are not assumptions. Each was read from the venue on
2026-10-27 and `measure()` re-derives them the same way, so they can be
refreshed rather than trusted:

  maker fee   0.0144%   28 maker fills, every one at exactly this rate
  taker fee   0.0432%   333 taker fills, min 0.0430% max 0.0432%
              (the schedule says 0.015% / 0.045% with a 4% referral
              discount; the fills show the discount is applied)
  stop slip   0.1193%   MEAN adverse distance between trigger price and the
                        volume-weighted fill, over 171 real Stop Market
                        exits. Median 0.0564%, p90 0.2925%, worst 1.7922%.
                        The mean, not the median, because expected R is a
                        mean and the tail is where stops actually cost.
  tp slip     0.0       Measured mean is -0.0413% over 42 Take Profit Market
                        exits -- they fill slightly BETTER than trigger.
                        Counted as zero rather than as a gain.

One thing that could NOT be measured: slippage on market ENTRIES. No band
trade has entered at market yet (every one came through a resting bid), and
the older strategy's log recorded a planned price rather than the price at
send time, so it cannot be used. `market_entry_slip_pct` defaults to zero and
experiments that enter at market must report a sensitivity with it set.
"""

from __future__ import annotations

import os
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Any, Dict


@dataclass(frozen=True)
class Costs:
    maker_fee_pct: float = 0.0144
    taker_fee_pct: float = 0.0432
    stop_slip_pct: float = 0.1193
    tp_slip_pct: float = 0.0
    market_entry_slip_pct: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


MEASURED = Costs()
FRICTIONLESS = Costs(0.0, 0.0, 0.0, 0.0, 0.0)


def measure(client=None, address: str = "", days: int = 180) -> Dict[str, Any]:
    """Re-derive every number above from the venue. Read-only info calls."""
    if client is None:
        from research.levels.venue_client import HyperliquidReadOnlyClient
        client = HyperliquidReadOnlyClient()
    addr = (address or os.getenv("HL_ACCOUNT_ADDRESS", "")).strip().lower()
    if not addr:
        raise ValueError("HL_ACCOUNT_ADDRESS is needed to read this account's fills")
    since = int(time.time() * 1000) - days * 86_400_000
    fills = client._post({"type": "userFillsByTime", "user": addr,
                          "startTime": since}, "userFillsByTime")
    hist = client._post({"type": "historicalOrders", "user": addr},
                        "historicalOrders")

    maker: list = []
    taker: list = []
    by_oid = defaultdict(list)
    for f in fills or []:
        notional = float(f["px"]) * float(f["sz"])
        if notional > 0:
            (taker if f.get("crossed") else maker).append(float(f["fee"]) / notional)
        by_oid[f["oid"]].append(f)

    trig: Dict[int, Any] = {}
    filled = set()
    for h in hist or []:
        o = h.get("order", {})
        if o.get("orderType") in ("Stop Market", "Take Profit Market"):
            if float(o.get("triggerPx") or 0) > 0:
                trig.setdefault(o["oid"], (o["orderType"], float(o["triggerPx"])))
            if h.get("status") == "filled":
                filled.add(o["oid"])
    slip = defaultdict(list)
    for oid in filled:
        if oid not in trig or oid not in by_oid:
            continue
        otype, tpx = trig[oid]
        fs = by_oid[oid]
        sz = sum(float(f["sz"]) for f in fs)
        vwap = sum(float(f["px"]) * float(f["sz"]) for f in fs) / sz
        adverse = ((tpx - vwap) if fs[0]["side"] == "A" else (vwap - tpx)) / tpx * 100
        slip[otype].append(adverse)

    def summ(xs, scale=1.0):
        if not xs:
            return {"n": 0}
        s = sorted(xs)
        return {"n": len(s), "mean": statistics.mean(s) * scale,
                "median": statistics.median(s) * scale,
                "p90": s[int(0.9 * (len(s) - 1))] * scale, "worst": s[-1] * scale}

    return {"maker_fee_pct": summ(maker, 100), "taker_fee_pct": summ(taker, 100),
            "stop_slip_pct": summ(slip["Stop Market"]),
            "tp_slip_pct": summ(slip["Take Profit Market"]),
            "fills": len(fills or []), "days": days}

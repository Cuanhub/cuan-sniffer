"""The band strategy's own record: what it saw, what it did, how it turned out.

Why this exists
---------------
The 2026-10-26 analysis found the largest effect in this project -- a stop
inside the coin's own noise -- and it could only do so because the SCANNER
records level episodes with features and resolves their outcomes: 2,256
resolved rows with 40 columns. The band strategy had no equivalent, so the
same questions had to be answered by borrowing level episodes and
hand-rolling a backtest. A finding about bands rested on a different
strategy's data, and two of the candidate features could not be tested at
all because nothing stored them.

This closes that. One row per BAND the pass considered -- taken OR refused,
because "why are we losing" needs the counterfactual -- and one row per
resolved outcome.

What it must never do
---------------------
Affect a trade. Every entry point here is called for its side effect only and
swallows its own failures: a recorder that can break the pass is worse than
no recorder.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from research.levels import storage
from research.levels.zones import Plan


def episode_id(plan: Plan) -> str:
    """One id per BAND, so a band re-planned every pass is recorded once.

    Not per pass: at 450s a single band would otherwise write ~190 identical
    rows a day. The band's base bar and low identify it, and both are stable
    for as long as it exists.
    """
    z = plan.zone
    return f"zone:{z.coin}:{z.base_i}:{z.lo:.10g}"


def _f(v: Any) -> Any:
    """A float rounded for the file, or "" so a missing value stays missing
    rather than becoming a zero somebody later averages."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return ""
    return round(x, 10) if math.isfinite(x) else ""


def episode_row(plan: Plan, bars: Sequence[Dict[str, Any]], px: float,
                taken: bool, refused: str = "",
                universe_rank: Optional[int] = None,
                day_ntl_vlm: Optional[float] = None,
                max_leverage: Optional[int] = None,
                ctx: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """One band, as the pass saw it."""
    z = plan.zone
    now = datetime.now(timezone.utc)
    ts = int(bars[-1].get("t") or 0) if bars else 0
    mark = (ctx or {}).get("mark_px")
    oracle = (ctx or {}).get("oracle_px")
    basis: Any = ""
    try:
        if mark and oracle and float(oracle) > 0:
            basis = round((float(mark) / float(oracle) - 1) * 10_000, 3)
    except (TypeError, ValueError):
        basis = ""
    return {
        "zone_episode_id": episode_id(plan),
        "coin": z.coin,
        "detected_at_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "bar_ts_ms": ts,
        "zone_lo": _f(z.lo), "zone_hi": _f(z.hi),
        "zone_width_pct": _f(z.width_pct),
        "zone_base_i": z.base_i, "zone_bos_i": z.bos_i,
        "bars_since_base": len(bars) - z.base_i if bars else "",
        "touches": sum(1 for b in bars[z.bos_i:] if b.get("l", 0) <= z.hi)
                   if bars else "",
        "entry": _f(plan.entry), "stop": _f(plan.stop), "tp": _f(plan.tp),
        "rr": _f(plan.rr), "risk_pct": _f(plan.risk_pct),
        "immediate": bool(plan.immediate),
        "atr_pct": _f(plan.atr_pct), "stop_atr_mult": _f(plan.stop_atr_mult),
        "target_lo": _f(plan.target.lo), "target_hi": _f(plan.target.hi),
        "target_dist_pct": _f((plan.tp - plan.entry) / plan.entry * 100
                              if plan.entry > 0 else ""),
        "price_at_detect": _f(px),
        "entry_hour_utc": now.hour,
        "max_leverage": max_leverage if max_leverage is not None else "",
        "day_ntl_vlm": _f(day_ntl_vlm) if day_ntl_vlm is not None else "",
        "universe_rank": universe_rank if universe_rank is not None else "",
        "mark_px": _f(mark) if mark is not None else "",
        "oracle_px": _f(oracle) if oracle is not None else "",
        "basis_bps": basis,
        "taken": bool(taken),
        "refused_reason": refused or "",
    }


def record_pass(found: Dict[str, List[Plan]],
                min_rr: float,
                bars_by_coin: Dict[str, List[Dict[str, Any]]],
                mids: Optional[Dict[str, float]] = None,
                universe: Optional[Sequence[Dict[str, Any]]] = None,
                leverage: Optional[Dict[str, int]] = None,
                ctx: Optional[Dict[str, Any]] = None) -> int:
    """Write every band this pass considered. Returns rows written.

    Never raises. A recorder that can break the trading pass is worse than no
    recorder, so the whole body is guarded and a failure is counted as zero
    rows rather than propagated.
    """
    written = 0
    try:
        rank = {}
        vol = {}
        for i, row in enumerate(universe or [], start=1):
            sym = str(row.get("symbol") or row.get("coin") or "").upper()
            if sym:
                rank[sym] = i
                try:
                    vol[sym] = float(row.get("day_ntl_vlm") or 0.0)
                except (TypeError, ValueError):
                    pass
        # `taken` here means CLEARED THE GATE, not "an order went out". The
        # rails that stop an order -- concurrency, entry spacing, the daily
        # cap -- are not judgements about the setup, and folding them in
        # would make a full book look like a bad band.
        from research.levels.zone_pass import rejection
        for coin, plans in (found or {}).items():
            bars = (bars_by_coin or {}).get(coin) or []
            px = float((mids or {}).get(coin) or
                       (bars[-1]["c"] if bars else 0.0) or 0.0)
            for plan in plans:
                why = rejection(plan, min_rr)
                row = episode_row(
                    plan, bars, px,
                    taken=why is None,
                    refused=why or "",
                    universe_rank=rank.get(coin),
                    day_ntl_vlm=vol.get(coin),
                    max_leverage=(leverage or {}).get(coin),
                    ctx=(ctx or {}).get(coin))
                if storage.append_zone_episode(row):
                    written += 1
    except Exception as exc:                                   # noqa: BLE001
        print(f"[ZONE_LOG] recording failed ({str(exc)[:120]}) -- "
              f"the pass is unaffected")
    return written


# ── resolving what happened ────────────────────────────────────────────────

MIN_BARS_TO_RESOLVE = 24     # a winner's median time-to-target was 5 bars


def resolve_one(ep: Dict[str, Any], fwd: Sequence[Dict[str, Any]]
                ) -> Optional[Dict[str, Any]]:
    """One episode's outcome from the bars that followed it, or None if it
    cannot be resolved yet.

    `fwd` must START at the bar after detection. The entry fills when a low
    reaches it -- the same mechanic as the resting bid -- and only then can
    the stop or target be touched.

    A bar that touches BOTH the stop and the target counts as a LOSS. The
    data cannot say which came first inside a bar, and assuming the good one
    is how a backtest flatters itself; the scanner's own resolver makes the
    same choice.
    """
    def g(k: str) -> Optional[float]:
        try:
            v = float(ep.get(k, ""))
            return v if math.isfinite(v) else None
        except (TypeError, ValueError):
            return None

    entry, stop, tp = g("entry"), g("stop"), g("tp")
    if entry is None or stop is None or tp is None:
        return None
    if not (entry > stop > 0) or tp <= entry:
        return None
    if len(fwd) < MIN_BARS_TO_RESOLVE:
        return None

    fill_i = next((k for k, b in enumerate(fwd) if b["l"] <= entry), None)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    base = {"zone_episode_id": ep["zone_episode_id"], "coin": ep.get("coin", ""),
            "resolved_at_utc": now, "bars_elapsed": len(fwd)}
    if fill_i is None:
        # Never filled. Recorded rather than dropped: a band price never
        # reached is a different outcome from one that filled and lost, and
        # collapsing them hides how often the strategy simply waits.
        return {**base, "filled": False, "bars_to_fill": -1,
                "hit_stop": "", "hit_tp": "", "bars_to_stop": -1,
                "bars_to_tp": -1, "won": "", "outcome_r": "",
                "mae_pct": "", "mfe_pct": ""}

    after = fwd[fill_i:]
    s_i = next((k for k, b in enumerate(after) if b["l"] <= stop), None)
    t_i = next((k for k, b in enumerate(after) if b["h"] >= tp), None)
    if s_i is None and t_i is None:
        return None                      # still open: resolve it later
    won = t_i is not None and (s_i is None or t_i < s_i)
    rr = (tp - entry) / (entry - stop)
    lo = min(b["l"] for b in after)
    hi = max(b["h"] for b in after)
    return {**base, "filled": True, "bars_to_fill": fill_i,
            "hit_stop": s_i is not None, "hit_tp": t_i is not None,
            "bars_to_stop": s_i if s_i is not None else -1,
            "bars_to_tp": t_i if t_i is not None else -1,
            "won": bool(won),
            "outcome_r": round(rr if won else -1.0, 4),
            "mae_pct": round((lo - entry) / entry * 100, 4),
            "mfe_pct": round((hi - entry) / entry * 100, 4)}


def resolve(client, hours: int = 1200) -> int:
    """Resolve every unresolved band episode. Returns rows written.

    Reads only, apart from appending outcomes, and one candle fetch per coin
    that has something unresolved.
    """
    eps = [e for e in storage.load_zone_episodes()
           if e.get("zone_episode_id") not in storage.zone_resolved_ids()]
    if not eps:
        return 0
    import time as _t
    now_ms = int(_t.time() * 1000)
    by_coin: Dict[str, List[Dict[str, Any]]] = {}
    for e in eps:
        by_coin.setdefault(str(e.get("coin", "")).upper(), []).append(e)
    written = 0
    for coin, rows in by_coin.items():
        if not coin:
            continue
        try:
            raw = client.get_candles(coin, "1h",
                                     now_ms - hours * 3600_000, now_ms)
        except Exception as exc:                               # noqa: BLE001
            print(f"[ZONE_LOG] {coin}: bars unavailable "
                  f"({str(exc)[:70]}) -- leaving its episodes unresolved")
            continue
        bars = []
        for b in raw or []:
            try:
                bars.append({"t": int(b["t"]), "h": float(b["h"]),
                             "l": float(b["l"]), "c": float(b["c"])})
            except (KeyError, TypeError, ValueError):
                continue
        for e in rows:
            try:
                ts = int(float(e.get("bar_ts_ms") or 0))
            except (TypeError, ValueError):
                continue
            fwd = [b for b in bars if b["t"] > ts]
            out = resolve_one(e, fwd)
            if out and storage.append_zone_outcome(out):
                written += 1
    return written

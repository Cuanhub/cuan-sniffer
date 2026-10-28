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


HOUR_MS = 3_600_000


def band_id(coin: str, base_t: int, lo: float) -> str:
    """A band's identity that survives the window sliding: coin, the base
    candle's TIMESTAMP, the band low. (episode_id above uses the base
    candle's INDEX, which moves by one each hour as the 1200-bar window
    slides, so it named the same band afresh every hour.)"""
    return f"band:{str(coin).upper()}:{int(base_t)}:{float(lo):.10g}"


def band_key(row: Dict[str, Any]) -> Optional[str]:
    """band_id for a recorded row -- from its own column, or, for rows
    written before that column existed, from fields every row has: the
    detection bar's timestamp and how many bars back the base sat."""
    if row.get("band_id"):
        return str(row["band_id"])
    try:
        ts = int(float(row["bar_ts_ms"]))
        since = int(float(row["bars_since_base"]))
        lo = float(row["zone_lo"])
    except (KeyError, TypeError, ValueError):
        return None
    return band_id(row.get("coin", ""), ts - (since - 1) * HOUR_MS, lo)


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
        "band_id": (band_id(z.coin, int(bars[z.base_i].get("t") or 0), z.lo)
                    if bars and 0 <= z.base_i < len(bars) else ""),
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
#
# Delegated to research.backtest.engine, so a recorded outcome and a
# backtested one are produced by ONE implementation and cannot drift apart.
# What that changed, from the rules this file used to apply itself
# (2026-10-27):
#   - a target touched in the same hour a bid filled no longer counts as a
#     win: the high may have printed before the dip that filled it. 6 of the
#     first 23 recorded "wins" were that;
#   - a bid fills only within its 12h life, as the live one does, instead of
#     at any later bar;
#   - an in-band market entry starts at its recorded price and is judged
#     from the NEXT hour, not by the hour it happened in;
#   - stops pay the measured slippage, gaps through a stop exit at the
#     open, and r_net is after this account's measured fees.
# And each BAND resolves once, at its first sighting -- not once per hourly
# re-sighting.

RESOLVER = "engine-v1"
BID_TTL_BARS = 12        # = core.protect.BID_TTL_HOURS; test_backtest_live_parity pins it


def resolve_one(ep: Dict[str, Any], bars: Sequence[Dict[str, Any]]
                ) -> Optional[Dict[str, Any]]:
    """One episode's outcome, or None if it cannot be resolved yet.

    `bars` are CLOSED 1h bars beginning AT the detection bar (the bar whose
    open time is the episode's bar_ts_ms). That bar is where the decision
    was made; nothing about the exit is read from it.
    """
    from research.backtest.costs import MEASURED
    from research.backtest.engine import LIMIT, MARKET_AT, Order, simulate

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
    try:
        ts = int(float(ep.get("bar_ts_ms") or 0))
    except (TypeError, ValueError):
        return None
    if not bars or int(bars[0].get("t", -1)) != ts:
        return None                      # the detection bar is not closed yet
    immediate = str(ep.get("immediate", "")).strip().lower() in ("true", "1")
    o = Order(str(ep.get("coin", "")), 0, MARKET_AT if immediate else LIMIT,
              stop, tp, limit=entry, ttl=BID_TTL_BARS)
    tr = simulate(list(bars), o, MEASURED)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    base = {"zone_episode_id": ep.get("zone_episode_id", ""),
            "coin": ep.get("coin", ""), "resolved_at_utc": now,
            "bars_elapsed": len(bars) - 1, "band_id": band_key(ep) or "",
            "resolver": RESOLVER}
    if tr is None:
        if immediate or len(bars) - 1 < BID_TTL_BARS:
            return None                  # a bid still inside its life
        # Never filled within the bid's life. Recorded, not dropped: a band
        # price never reached is a different outcome from one that filled
        # and lost, and collapsing them hides how often the strategy waits.
        return {**base, "filled": False, "bars_to_fill": -1, "hit_stop": "",
                "hit_tp": "", "bars_to_stop": -1, "bars_to_tp": -1, "won": "",
                "outcome_r": "", "mae_pct": "", "mfe_pct": "", "r_gross": "",
                "r_net": "", "cost_r": "", "exit": "expired",
                "fill_bar_tp": "", "ambiguous": ""}
    if not tr.resolved or tr.exit_i is None:
        return None                      # still open: resolve it later
    exit_i = tr.exit_i
    held = bars[tr.entry_i + (1 if immediate else 0): exit_i + 1]
    lo = min(b["l"] for b in held) if held else entry
    hi = max(b["h"] for b in held) if held else entry
    won = tr.outcome == "tp"
    k = exit_i - tr.entry_i
    return {**base, "filled": True, "bars_to_fill": tr.entry_i,
            "hit_stop": tr.outcome == "stop", "hit_tp": won or tr.ambiguous,
            "bars_to_stop": k if not won else -1, "bars_to_tp": k if won else -1,
            "won": won, "outcome_r": round(tr.r_gross, 4),
            "mae_pct": round((lo - entry) / entry * 100, 4),
            "mfe_pct": round((hi - entry) / entry * 100, 4),
            "r_gross": round(tr.r_gross, 4), "r_net": round(tr.r_net, 4),
            "cost_r": round(tr.cost_r, 4), "exit": tr.outcome,
            "fill_bar_tp": bool(tr.fill_bar_tp), "ambiguous": bool(tr.ambiguous)}


def first_sightings(eps: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """band_id -> the band's FIRST recorded episode. Later re-sightings of a
    band are the same opportunity seen again, not new trades."""
    out: Dict[str, Dict[str, Any]] = {}
    order = sorted(eps, key=lambda e: (str(e.get("detected_at_utc", "")),
                                       str(e.get("bar_ts_ms", ""))))
    for e in order:
        k = band_key(e)
        if k and k not in out:
            out[k] = e
    return out


def resolve(client, hours: int = 1200) -> int:
    """Resolve every band whose first sighting has no outcome yet. Returns
    rows written. Reads only, apart from appending outcomes, and one candle
    fetch per coin with something pending."""
    from research.backtest import data as D
    done = storage.zone_resolved_bands()
    pending = [e for k, e in first_sightings(storage.load_zone_episodes()).items()
               if k not in done]
    if not pending:
        return 0
    import time as _t
    now_ms = int(_t.time() * 1000)
    by_coin: Dict[str, List[Dict[str, Any]]] = {}
    for e in pending:
        by_coin.setdefault(str(e.get("coin", "")).upper(), []).append(e)
    written = 0
    for coin, rows in by_coin.items():
        if not coin:
            continue
        try:
            raw = client.get_candles(coin, "1h", now_ms - hours * HOUR_MS, now_ms)
        except Exception as exc:                               # noqa: BLE001
            print(f"[ZONE_LOG] {coin}: bars unavailable "
                  f"({str(exc)[:70]}) -- leaving its episodes unresolved")
            continue
        bars = D.normalise(raw, now_ms)          # closed bars only
        for e in rows:
            try:
                ts = int(float(e.get("bar_ts_ms") or 0))
            except (TypeError, ValueError):
                continue
            out = resolve_one(e, [b for b in bars if b["t"] >= ts])
            if out and storage.append_zone_outcome(out):
                written += 1
    return written


# ── which bands are worth a human verdict ──────────────────────────────────

# How far from the gate's decision line still counts as "near it".
#
# Sampling at random wastes the scarce resource. A band at rr 0.02 teaches
# nothing: the gate refuses it and so would anyone. A band at rr 1.90, just
# under a 2.0 floor, is where a human verdict carries information the gate
# does not have -- it is the only place a label can tell you the line is in
# the wrong place.
#
# 0.4 of the floor is wide enough to catch both sides of it. At MIN_RR 2.0
# that is rr 1.6 to 2.4, which on a representative pass was 6 of 47 bands --
# a sitting's worth, not a mailing list.
BOUNDARY_FRAC = 0.2


def boundary_cases(rows: Sequence[Dict[str, Any]], min_rr: float,
                   labelled: Optional[Sequence[str]] = None,
                   limit: int = 10) -> List[Dict[str, Any]]:
    """Recorded band episodes sitting closest to the gate's decision line.

    `rows` are zone_episode dicts as written to disk. Already-labelled ids
    are dropped, and the rest are ordered by how close their reward/risk sits
    to `min_rr` -- nearest first, because that is where a verdict is worth
    most.

    Deliberately NOT ordered by how good the setup looks. The point is to
    find out where the line belongs, and a band everybody agrees on cannot
    say anything about that.
    """
    done = set(labelled or ())
    band = max(min_rr * BOUNDARY_FRAC, 0.05)
    out = []
    for r in rows:
        if str(r.get("zone_episode_id") or "") in done:
            continue
        try:
            rr = float(r.get("rr", ""))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(rr):
            continue
        gap = abs(rr - min_rr)
        if gap <= band:
            out.append({**r, "_gap": gap, "_side":
                        "would TAKE" if rr >= min_rr else "would REFUSE"})
    out.sort(key=lambda r: r["_gap"])
    return out[:limit]

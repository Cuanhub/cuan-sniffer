"""Pieces every experiment needs, written once.

The trendline experiment carries its own copies of these (it was pre-
registered before this module existed and is left exactly as it ran).
Everything here encodes a lesson that experiment paid for:

  - controls draw entries AFTER the signal (trendline PREREG Amendment 1: a
    window around the signal let the control ride the move that defines it);
  - the null calibration checks the MEAN as well as false positives;
  - module values are read inside functions, never bound as defaults.
"""

from __future__ import annotations

import math
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from research.backtest import data as D
from research.backtest.costs import MEASURED, Costs
from research.backtest.engine import LIMIT, MARKET_CLOSE, Order, Trade, simulate

SEED = 20261027
WARMUP = 799                     # zones.ZONE_MIN_BARS - 1
CONTROL_DRAWS = 200
CONTROL_SPAN = 240
MIN_CONTROL = 50
NULL_MAX_FALSE = 2
NULL_MAX_BIAS_SE = 2.5

BARS: Dict[str, List[Dict[str, Any]]] = {}
EXTRA: Dict[str, Any] = {}       # e.g. 15m bars, for experiments that need two timeframes


def _init(bars: Dict[str, List[Dict[str, Any]]], extra: Dict[str, Any]) -> None:
    global BARS, EXTRA
    BARS, EXTRA = bars, extra


def pool(bars: Dict[str, List[Dict[str, Any]]], extra: Optional[Dict[str, Any]] = None,
         workers: int = 8) -> ProcessPoolExecutor:
    _init(bars, extra or {})
    return ProcessPoolExecutor(max_workers=workers, initializer=_init,
                               initargs=(bars, extra or {}))


def rng_for(coin: str, i: int, salt: int) -> np.random.Generator:
    return np.random.default_rng([SEED, zlib.crc32(coin.encode()), int(i), int(salt)])


def draws_after(n: int, centre: int, rng: np.random.Generator,
                span: Optional[int] = None, k: Optional[int] = None,
                first: Optional[int] = None) -> np.ndarray:
    """k random bar indices in (centre, centre + span], inside the data."""
    span = CONTROL_SPAN if span is None else span
    k = CONTROL_DRAWS if k is None else k
    first = WARMUP if first is None else first
    lo, hi = max(first, centre + 1), min(n - 2, centre + span)
    if hi < lo:
        return np.empty(0, dtype=np.int64)
    return rng.integers(lo, hi + 1, k)


def matched_control(bars: List[Dict[str, Any]], coin: str, entry_i: int,
                    risk_pct: float, reward_pct: float, maker: bool,
                    costs: Costs = MEASURED) -> Optional[float]:
    """Mean net R of random market entries AFTER `entry_i` on the same coin,
    with this trade's risk %, reward % and entry fee type. None if fewer than
    MIN_CONTROL of them resolved."""
    rs = []
    for u in draws_after(len(bars), entry_i, rng_for(coin, entry_i, 1)):
        c = bars[int(u)]["c"]
        o = Order(coin, int(u), MARKET_CLOSE, c * (1 - risk_pct / 100),
                  c * (1 + reward_pct / 100), entry_fee="maker" if maker else "taker")
        tr = simulate(bars, o, costs)
        if tr is not None and tr.resolved:
            rs.append(tr.r_net)
    return float(np.mean(rs)) if len(rs) >= MIN_CONTROL else None


BID_TTL_BARS = 12        # band.BID_TTL_BARS; kept here to avoid an import cycle


def same_order_control(bars: List[Dict[str, Any]], trade: Trade,
                       costs: Costs = MEASURED) -> Optional[float]:
    """Mean net R of random orders of the trade's OWN type, placed AFTER its
    entry with its geometry (rr_floor PREREG Amendment 1).

    A bid gets bids: at a random later decision bar, a limit the same
    distance below that bar's close as the trade's bid sat below its own
    decision close, resting BID_TTL_BARS, with the trade's stop and target
    distances from the bid -- only the ones that fill count. That way the
    engine's conservative fill-hour rule (stop counted, target not) applies
    to trade and control alike. A market entry gets market entries.
    """
    coin = trade.coin
    risk = (trade.entry_px - trade.stop) / trade.entry_px
    reward = (trade.tp - trade.entry_px) / trade.entry_px
    if trade.kind == LIMIT:
        dec_c = bars[trade.decided_i]["c"]
        offset = (dec_c - trade.entry_px) / dec_c if dec_c > 0 else 0.0
    rs = []
    for u in draws_after(len(bars), trade.entry_i, rng_for(coin, trade.entry_i, 2)):
        u = int(u)
        c = bars[u]["c"]
        if trade.kind == LIMIT:
            lim = c * (1 - offset)
            o = Order(coin, u, LIMIT, lim * (1 - risk), lim * (1 + reward), limit=lim,
                      ttl=BID_TTL_BARS)
        else:
            o = Order(coin, u, MARKET_CLOSE, c * (1 - risk), c * (1 + reward))
        tr = simulate(bars, o, costs)
        if tr is not None and tr.resolved:
            rs.append(tr.r_net)
    return float(np.mean(rs)) if len(rs) >= MIN_CONTROL else None


def halves(values: Sequence[float], times: Sequence[int]) -> List[List[float]]:
    order = np.argsort(np.asarray(times))
    v = np.asarray(values, float)[order]
    h = len(v) // 2
    return [list(v[:h]), list(v[h:])]


# ------------------------------------------------------------- null data --
def random_walks(seed: int, coins: int = 12, n: int = 5000, sub: int = 12,
                 interval_ms: int = D.HOUR_MS, salt: int = 99
                 ) -> Dict[str, List[Dict[str, Any]]]:
    """Driftless random walks -- exactly a martingale in price -- for null
    calibration ONLY. `sub` sub-steps per bar make highs and lows path
    extremes rather than endpoints. No market information is in them."""
    rng = np.random.default_rng([SEED, seed, salt])
    out = {}
    for c in range(coins):
        sigma_bar = rng.uniform(0.006, 0.016) * math.sqrt(interval_ms / D.HOUR_MS)
        s = sigma_bar / math.sqrt(sub)
        path = 100.0 * np.exp(np.cumsum(rng.normal(-s * s / 2, s, n * sub)))
        bars, prev = [], 100.0
        for k in range(n):
            seg = path[k * sub:(k + 1) * sub]
            bars.append({"t": 1_700_006_400_000 + k * interval_ms, "o": prev,
                         "h": float(max(prev, seg.max())), "l": float(min(prev, seg.min())),
                         "c": float(seg[-1]), "v": 1.0, "n": 1})
            prev = float(seg[-1])
        out[f"NULL{seed}_{c}"] = bars
    return out


def random_walks_15m(seed: int, coins: int = 12, n_1h: int = 2400
                     ) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, List[Dict[str, Any]]]]:
    """(1h bars, 15m bars) from ONE random walk: the 1h bars are exact
    aggregates of the 15m bars, as real candles are."""
    fifteen = random_walks(seed, coins, n_1h * 4, sub=24, interval_ms=D.HOUR_MS // 4, salt=15)
    hourly = {c: D.resample(b, 4, D.HOUR_MS // 4) for c, b in fifteen.items()}
    return hourly, fifteen


def null_summary(per: List[Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
    """Fires and centring per framing across null datasets."""
    out: Dict[str, Any] = {}
    for name in (per[0] if per else {}):
        est = [r[name]["estimate"] for r in per if math.isfinite(r[name]["estimate"])]
        fired = sum(bool(r[name]["adds_edge"]) for r in per)
        m = float(np.mean(est)) if est else float("nan")
        se = float(np.std(est, ddof=1) / math.sqrt(len(est))) if len(est) > 1 else float("nan")
        ok = math.isfinite(m) and math.isfinite(se) and abs(m) <= NULL_MAX_BIAS_SE * se
        out[name] = {"fired": fired, "mean": m, "se": se, "centred": bool(ok),
                     "miscalibrated": fired > NULL_MAX_FALSE or not ok}
    return out


# ----------------------------------------------------------------- fetch --
def fetch_paced(coins: List[str], interval: str, gap_sec: float = 10.0,
                client=None, log=print) -> D.Snapshot:
    """data.fetch, one coin at a time with a pause between requests.

    A 5000-candle request costs about 100 weight against Hyperliquid's
    per-IP budget of roughly 1200 a minute, and the live bot shares that IP.
    Fetching at full speed is what produced the 2026-10-27 burst of 429s on
    the live trader. Slow is fine; starving the live bot is not.
    """
    if client is None:
        from research.levels.venue_client import HyperliquidReadOnlyClient
        client = HyperliquidReadOnlyClient()
    parts: List[D.Snapshot] = []
    for i, coin in enumerate(coins):
        if i:
            time.sleep(gap_sec)
        parts.append(D.fetch([coin], client=client, interval=interval, log=log))
    bars: Dict[str, List[Dict[str, Any]]] = {}
    coins_meta: Dict[str, Any] = {}
    excluded: Dict[str, str] = {}
    for p in parts:
        bars.update(p.bars)
        coins_meta.update(p.manifest["coins"])
        excluded.update(p.manifest["excluded"])
    m = dict(parts[0].manifest) if parts else {}
    m.update({"requested": list(coins), "coins": coins_meta, "excluded": excluded,
              "paced_gap_sec": gap_sec})
    return D.Snapshot(bars=bars, manifest=m)

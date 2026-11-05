"""MEASUREMENT, not a hypothesis test: what a cash-and-carry -- hold the coin
as spot, short the same amount of its perp -- collected on this venue, coin
by coin, over each perp's whole funding history.

Why it can earn at all: the funding formula pays shorts its fixed interest
component (0.01% per 8h, ~11% a year) whenever the premium sits inside the
clamp. Why it can still lose: funding turns negative when the perp trades
below spot; a squeeze can liquidate the short leg before spot is sold to top
it up; the spot leg may be a wrapped token whose issuer fails. This reads the
venue's record of each. In-sample and descriptive: no p-values.

Fixed on 2026-11-05, before the full histories were read:
  funding   per coin: annualised mean to a short since listing, by calendar
            year and over the last 12 months; share of hours at exactly the
            interest constant; share negative; complete months (>= 600
            funding hours): share negative and the worst; the longest run of
            negative weeks.
  rule      always on, and on only while the trailing 7-day mean rate is
            above zero, decided at each 00:00 UTC; each entry or exit costs
            one spot and one perp fill (0.1132% at taker, 0.0544% at maker).
            Both over the same window (from the first decision), net and
            annualised, with time in market.
  capital   return on capital with the perp leg at L x and spot bought with
            the rest: f * L / (L + 1), L in 1, 1.5, 2, 3; beside the rise that
            liquidates a short at L, (1/L - m) / (1 + m) with maintenance
            m = 1 / (2 * max leverage), and the worst 1-day and 7-day rises in
            the daily candles.
  basis     spot against perp on hourly closes over the ~208 days the venue
            serves: mean, sd, 1st/99th percentiles, worst, hours beyond 1%.
  HLP       the venue's market-making vault -- the other structural premium
            here -- by calendar year and over the trailing 12, 6 and 3 months
            (modified Dietz on its own record), for comparison.

    python -m research.backtest.experiments.carry_measure --funding DIR
        [--daily DIR] [--basis DIR] [--meta FILE] [--spot-meta FILE] [--hlp FILE]
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from research.backtest import data as D
from research.backtest.experiments.predictability_scan import INTEREST_HOURLY, dir_digest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "carry"
H = D.HOUR_MS
DAY = 24 * H
YEAR_H = 24 * 365
LOOKBACK_H = 168
COMPLETE_MONTH_H = 600
SPOT_TAKER, SPOT_MAKER = 0.070, 0.040        # % one way, base tier
PERP_TAKER, PERP_MAKER = 0.0432, 0.0144      # % one way, this account's measured fees
SWITCH_PCT = {"taker": SPOT_TAKER + PERP_TAKER, "maker": SPOT_MAKER + PERP_MAKER}
LEVERAGES = (1.0, 1.5, 2.0, 3.0)
# perp -> its USDC spot token on this venue, matched 2026-11-05 by price (within 1.5%) and name
SPOT = {"BTC": "UBTC", "ETH": "UETH", "SOL": "USOL", "HYPE": "HYPE", "XMR": "XMR1", "ZEC": "UZEC",
        "NEAR": "ONEAR", "FARTCOIN": "UFART", "PUMP": "UPUMP"}


def load_funding(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """(funding hours in ms, rates) from the venue's fundingHistory rows,
    sorted, one per hour (a repeated hour keeps its last row)."""
    by: Dict[int, float] = {}
    for r in json.loads(path.read_text()):
        by[int(r["time"]) // H * H] = float(r["fundingRate"])
    t = np.asarray(sorted(by), dtype=np.int64)
    return t, np.asarray([by[int(x)] for x in t])


def annualised_pct(rates: np.ndarray) -> float:
    return float(np.mean(rates) * YEAR_H * 100) if len(rates) else math.nan


def _ym(ms: int) -> str:
    return time.strftime("%Y-%m", time.gmtime(ms / 1000))


def monthly(t: np.ndarray, r: np.ndarray) -> "collections.OrderedDict[str, Tuple[float, int]]":
    """Month -> (funding collected by a short, as a fraction of notional; hours)."""
    out: "collections.OrderedDict[str, Tuple[float, int]]" = collections.OrderedDict()
    for ms, x in zip(t, r):
        k = _ym(int(ms))
        s, n = out.get(k, (0.0, 0))
        out[k] = (s + float(x), n + 1)
    return out


def longest_negative_weeks(t: np.ndarray, r: np.ndarray) -> int:
    """Longest run of consecutive 7-day blocks (from the first hour) whose
    funding summed below zero."""
    if not len(t):
        return 0
    wk = (t - t[0]) // (7 * DAY)
    sums = collections.OrderedDict()   # type: collections.OrderedDict[int, float]
    for w, x in zip(wk, r):
        sums[int(w)] = sums.get(int(w), 0.0) + float(x)
    best = run = 0
    prev = None
    for w, s in sums.items():
        if s < 0:
            run = run + 1 if run and prev is not None and w == prev + 1 else 1
        else:
            run = 0
        prev = w
        best = max(best, run)
    return best


def funding_record(t: np.ndarray, r: np.ndarray, now_ms: Optional[int] = None) -> Dict[str, Any]:
    now_ms = int(t[-1]) + H if now_ms is None else now_ms
    years = collections.OrderedDict()   # type: collections.OrderedDict[str, Dict[str, Any]]
    for y in sorted({time.gmtime(int(ms) / 1000).tm_year for ms in t}):
        m = np.asarray([time.gmtime(int(ms) / 1000).tm_year == y for ms in t])
        years[str(y)] = {"pct_per_year": annualised_pct(r[m]), "hours": int(m.sum())}
    months = monthly(t, r)
    full = {k: s for k, (s, n) in months.items() if n >= COMPLETE_MONTH_H}
    worst = min(full.items(), key=lambda kv: kv[1]) if full else None
    last12 = t >= now_ms - YEAR_H * H
    return {"first": time.strftime("%Y-%m-%d", time.gmtime(int(t[0]) / 1000)), "hours": int(len(r)),
            "span_hours": int((t[-1] - t[0]) // H + 1),
            "pct_per_year": annualised_pct(r), "last_12m_pct_per_year": annualised_pct(r[last12]),
            "by_year": years,
            "share_at_interest_constant": float(np.mean(np.abs(r - INTEREST_HOURLY) < 1e-12)),
            "share_negative": float(np.mean(r < 0)),
            "bps_per_hour_p1": float(np.percentile(r, 1) * 1e4),
            "complete_months": len(full),
            "share_negative_months": float(np.mean([s < 0 for s in full.values()])) if full else math.nan,
            "worst_month": {"month": worst[0], "pct_of_notional": worst[1] * 100} if worst else None,
            "longest_negative_weeks": longest_negative_weeks(t, r)}


def rule_record(t: np.ndarray, r: np.ndarray, lookback_h: int = LOOKBACK_H) -> Dict[str, Any]:
    """Always on against on-only-while-the-trailing-week-paid, over the same
    window: from the first 00:00 UTC with `lookback_h` hours behind it. Each
    day's funding hours are those after its 00:00 up to and including the
    next 00:00 (a rate stamped at T is for the hour ending at T)."""
    by = {int(ms): float(x) for ms, x in zip(t, r)}
    d0 = (int(t[0]) + (lookback_h - 1) * H + DAY - 1) // DAY * DAY
    days = list(range(d0, int(t[-1]), DAY))
    on_sum = all_sum = 0.0
    hours = on_hours = switches = 0
    on = False
    for d in days:
        past = [by[k] for k in range(d - (lookback_h - 1) * H, d + H, H) if k in by]
        want = bool(past) and float(np.mean(past)) > 0
        if want != on:
            switches += 1
            on = want
        day = [by[k] for k in range(d + H, d + DAY + H, H) if k in by]
        all_sum += sum(day)
        hours += len(day)
        if on:
            on_sum += sum(day)
            on_hours += len(day)
    switches += 1 if on else 0                     # the exit at the end
    yrs = hours / YEAR_H if hours else math.nan
    out: Dict[str, Any] = {"window_from": time.strftime("%Y-%m-%d", time.gmtime(d0 / 1000)) if days else None,
                           "years": yrs, "time_in_market": on_hours / hours if hours else math.nan,
                           "switches": switches, "gross_always_pct_per_year": all_sum / yrs * 100 if hours else math.nan,
                           "gross_filtered_pct_per_year": on_sum / yrs * 100 if hours else math.nan}
    for side, c in SWITCH_PCT.items():
        out[f"net_always_{side}_pct_per_year"] = (all_sum * 100 - 2 * c) / yrs if hours else math.nan
        out[f"net_filtered_{side}_pct_per_year"] = (on_sum * 100 - switches * c) / yrs if hours else math.nan
    return out


def return_on_capital(f_pct: float, lev: float) -> float:
    """Funding on the notional, over spot plus the perp leg's margin."""
    return f_pct * lev / (lev + 1)


def liquidation_rise(lev: float, max_lev: float) -> float:
    """The rise that takes a short at `lev` to maintenance, m = 1/(2 max_lev):
    margin 1/L minus the loss x equals m(1 + x)."""
    m = 1 / (2 * max_lev)
    return (1 / lev - m) / (1 + m)


def worst_rises(bars: Sequence[Dict[str, Any]], spans: Sequence[int] = (1, 7)) -> Dict[str, Any]:
    """Largest close-to-later-high rise within each span of days, and when."""
    c = [float(b["c"]) for b in bars]
    h = [float(b["h"]) for b in bars]
    out: Dict[str, Any] = {}
    for n in spans:
        best, at = -math.inf, None
        for i in range(len(c) - 1):
            x = max(h[i + 1:i + 1 + n]) / c[i] - 1
            if x > best:
                best, at = x, bars[i]["t"]
        out[f"{n}d"] = {"rise": best, "from": time.strftime("%Y-%m-%d", time.gmtime(int(at) / 1000)) if at else None}
    return out


def capital_table(f_pct: float, max_lev: Optional[float], rises: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for L in LEVERAGES:
        row: Dict[str, Any] = {"leverage": L, "return_on_capital_pct": return_on_capital(f_pct, L)}
        if max_lev:
            liq = liquidation_rise(L, max_lev)
            row["liquidation_rise"] = liq
            if rises:
                row["survives_worst_1d"] = liq > rises["1d"]["rise"]
                row["survives_worst_7d"] = liq > rises["7d"]["rise"]
        rows.append(row)
    return rows


def basis_record(spot: Sequence[Dict[str, Any]], perp: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Spot close over perp close, in bps, on the hours both have."""
    pc = {int(b["t"]): float(b["c"]) for b in perp}
    pairs = [(int(b["t"]), float(b["c"]) / pc[int(b["t"])] - 1) for b in spot if int(b["t"]) in pc and pc[int(b["t"])] > 0]
    if not pairs:
        return {}
    g = np.asarray([x for _, x in pairs]) * 1e4
    k = int(np.argmax(np.abs(g)))
    return {"hours": int(len(g)), "first": time.strftime("%Y-%m-%d", time.gmtime(pairs[0][0] / 1000)),
            "mean_bps": float(g.mean()), "sd_bps": float(g.std(ddof=1)) if len(g) > 1 else math.nan,
            "p1_bps": float(np.percentile(g, 1)), "p99_bps": float(np.percentile(g, 99)),
            "worst_bps": float(g[k]), "worst_at": time.strftime("%Y-%m-%d %H:00", time.gmtime(pairs[k][0] / 1000)),
            "share_beyond_100bps": float(np.mean(np.abs(g) > 100))}


def hlp_record(det: Dict[str, Any]) -> Dict[str, Any]:
    """Time-weighted return of a vault from its own record: allTime, refined
    by the month and week windows where they overlap; each interval's flow
    (change in value not explained by P&L) at mid-interval, plus the range
    with every flow at the start or at the end of its interval."""
    P = dict(det["portfolio"])

    def pts(w: str) -> List[Tuple[int, float, float]]:
        av = {int(t): float(v) for t, v in P[w]["accountValueHistory"]}
        pn = {int(t): float(v) for t, v in P[w]["pnlHistory"]}
        return [(t, av[t], pn[t]) for t in sorted(av) if t in pn]

    def splice(base: List[Tuple[int, float, float]], nxt: List[Tuple[int, float, float]]
               ) -> List[Tuple[int, float, float]]:
        if not nxt:
            return base
        t0 = nxt[0][0]
        keep = [p for p in base if p[0] < t0]
        prev = max((p for p in base if p[0] <= t0), key=lambda p: p[0], default=base[0])
        later = [p for p in base if p[0] >= t0]
        if later and later[0][0] > prev[0]:
            q = later[0]
            x = prev[2] + (q[2] - prev[2]) * (t0 - prev[0]) / (q[0] - prev[0])
        else:
            x = prev[2]
        return keep + [(t, a, p + x) for t, a, p in nxt]

    S = splice(splice(pts("allTime"), pts("month")), pts("week"))
    g, lo, hi = 1.0, 1.0, 1.0
    curve = [(S[0][0], 1.0)]
    years: "collections.OrderedDict[str, float]" = collections.OrderedDict()
    for (t0, a0, p0), (t1, a1, p1) in zip(S, S[1:]):
        dp = p1 - p0
        flow = (a1 - a0) - dp
        base = a0 + 0.5 * flow
        r = dp / base if base > 1.0 else 0.0
        rs = [dp / b if b > 1.0 else 0.0 for b in (a0, a0 + flow)]
        g *= 1 + r
        lo *= 1 + min(rs)
        hi *= 1 + max(rs)
        curve.append((t1, g))
        y = time.strftime("%Y", time.gmtime(t1 / 1000))
        years[y] = years.get(y, 1.0) * (1 + r)

    def g_at(ms: float) -> float:
        return max((c for c in curve if c[0] <= ms), key=lambda c: c[0], default=curve[0])[1]

    now = S[-1][0]
    life_d = (now - S[0][0]) / 864e5
    return {"first": time.strftime("%Y-%m-%d", time.gmtime(S[0][0] / 1000)),
            "last": time.strftime("%Y-%m-%d", time.gmtime(now / 1000)), "points": len(S),
            "twr_pct": (g - 1) * 100, "twr_range_pct": [(lo - 1) * 100, (hi - 1) * 100],
            "annualised_pct": (g ** (365 / life_d) - 1) * 100 if life_d > 0 and g > 0 else math.nan,
            "by_year_pct": {y: (v - 1) * 100 for y, v in years.items()},
            "trailing_pct": {f"{d}d": (g / g_at(now - d * 864e5) - 1) * 100 for d in (365, 182, 91)},
            "account_value": S[-1][1]}


def fmt(x: Any, d: int = 1, sign: bool = True) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def render(s: Dict[str, Any]) -> str:
    L = ["# Cash-and-carry on this venue -- MEASUREMENT, in-sample, no p-values", "",
         "Hold the coin as spot, short the same amount of its perp; the short collects funding. "
         f"Costs: an entry or an exit is one spot and one perp fill ({fmt(SWITCH_PCT['taker'], 4, False)}% "
         f"taker, {fmt(SWITCH_PCT['maker'], 4, False)}% maker). Funding in % of notional a year.", "",
         "## Funding to a short", "",
         "| coin | since | % a year | last 12m | by year | at constant | negative hours | negative months | "
         "worst month | longest negative weeks |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for c, v in s["coins"].items():
        f = v["funding"]
        yrs = ", ".join(f"{y} {fmt(x['pct_per_year'])}" for y, x in f["by_year"].items())
        wm = f["worst_month"]
        L.append(f"| {c} | {f['first']} | {fmt(f['pct_per_year'])} | {fmt(f['last_12m_pct_per_year'])} | {yrs} | "
                 f"{fmt(f['share_at_interest_constant'] * 100, 0, False)}% | {fmt(f['share_negative'] * 100, 0, False)}% | "
                 f"{fmt(f['share_negative_months'] * 100, 0, False)}% of {f['complete_months']} | "
                 f"{(wm['month'] + ' ' + fmt(wm['pct_of_notional'], 2)) if wm else 'n/a'}% | {f['longest_negative_weeks']} |")
    L += ["", "## Always on, or only while the trailing week paid (same window, net)", "",
          "| coin | from | years | always, taker | always, maker | filtered, taker | filtered, maker | in market | switches |",
          "|---|---|---|---|---|---|---|---|---|"]
    for c, v in s["coins"].items():
        r = v["rule"]
        L.append(f"| {c} | {r['window_from']} | {fmt(r['years'], 2, False)} | {fmt(r['net_always_taker_pct_per_year'])} | "
                 f"{fmt(r['net_always_maker_pct_per_year'])} | {fmt(r['net_filtered_taker_pct_per_year'])} | "
                 f"{fmt(r['net_filtered_maker_pct_per_year'])} | {fmt(r['time_in_market'] * 100, 0, False)}% | {r['switches']} |")
    L += ["", "## On capital, and what liquidates the short leg", "",
          "Return on capital at the last 12 months' funding, perp leg at L x; the rise that liquidates "
          "that short; the worst rises in the daily candles.", "",
          "| coin | max lev | worst 1d rise | worst 7d rise | " + " | ".join(f"L {x:g}" for x in LEVERAGES) + " |",
          "|---|---|---|---|" + "---|" * len(LEVERAGES)]
    for c, v in s["coins"].items():
        rises = v.get("rises")
        cells = []
        for row in v["capital"]:
            cell = f"{fmt(row['return_on_capital_pct'])}%"
            if "liquidation_rise" in row:
                cell += f", liq +{row['liquidation_rise'] * 100:.0f}%"
                if "survives_worst_1d" in row:
                    cell += " ok" if row["survives_worst_1d"] else " LIQ 1d"
            cells.append(cell)
        r1 = f"+{rises['1d']['rise'] * 100:.0f}% ({rises['1d']['from']})" if rises else "n/a"
        r7 = f"+{rises['7d']['rise'] * 100:.0f}% ({rises['7d']['from']})" if rises else "n/a"
        L.append(f"| {c} | {v.get('max_leverage', 'n/a')} | {r1} | {r7} | " + " | ".join(cells) + " |")
    if any(v.get("basis") for v in s["coins"].values()):
        L += ["", "## Spot against perp (hourly closes)", "",
              "| coin | spot token | hours | mean bps | sd | 1st pct | 99th pct | worst | beyond 1% |",
              "|---|---|---|---|---|---|---|---|---|"]
        for c, v in s["coins"].items():
            b = v.get("basis")
            if b:
                L.append(f"| {c} | {v.get('spot_token', 'n/a')} | {b['hours']} | {fmt(b['mean_bps'])} | {fmt(b['sd_bps'], 1, False)} | "
                         f"{fmt(b['p1_bps'])} | {fmt(b['p99_bps'])} | {fmt(b['worst_bps'])} ({b['worst_at']}) | "
                         f"{fmt(b['share_beyond_100bps'] * 100, 2, False)}% |")
    if s.get("spot_tokens"):
        L += ["", "## Spot tokens (the issuer is the operator's claim, as the venue lists it)", "",
              "| coin | token | listed as | 24h spot volume |", "|---|---|---|---|"]
        for c, t in s["spot_tokens"].items():
            L.append(f"| {c} | {t['token']} | {t['full_name']} | ${t['day_volume']:,.0f} |")
    if s.get("hlp"):
        h = s["hlp"]
        L += ["", "## HLP, the venue's market-making vault, for comparison", "",
              f"- {h['first']} to {h['last']}: time-weighted {fmt(h['twr_pct'])}% ({fmt(h['annualised_pct'])}% a year); "
              f"{fmt(h['twr_range_pct'][0])}% to {fmt(h['twr_range_pct'][1])}% depending on when deposits landed "
              "within each sampling interval",
              "- by year: " + ", ".join(f"{y} {fmt(x)}%" for y, x in h["by_year_pct"].items()),
              "- trailing: " + ", ".join(f"{k} {fmt(x)}%" for k, x in h["trailing_pct"].items())]
    return "\n".join(L) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--funding", type=Path, required=True)
    ap.add_argument("--daily", type=Path, default=None)
    ap.add_argument("--basis", type=Path, default=None)
    ap.add_argument("--meta", type=Path, default=None)
    ap.add_argument("--spot-meta", type=Path, default=None)
    ap.add_argument("--hlp", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=RESULTS)
    args = ap.parse_args(argv)
    max_lev: Dict[str, float] = {}
    if args.meta is not None:
        meta = json.loads(args.meta.read_text())[0]
        max_lev = {m["name"]: float(m["maxLeverage"]) for m in meta["universe"]}
    s: Dict[str, Any] = {"coins": {}, "data": {"funding": dir_digest(args.funding)},
                         "source": "https://api.hyperliquid.xyz/info (public, read-only)"}
    for p in sorted(args.funding.glob("*.json")):
        c = p.stem
        t, r = load_funding(p)
        if len(r) < LOOKBACK_H + 2 * 24:
            continue
        f = funding_record(t, r)
        v: Dict[str, Any] = {"funding": f, "rule": rule_record(t, r), "spot_token": SPOT.get(c)}
        rises = None
        if args.daily is not None and (args.daily / f"{c}.json").exists():
            bars = D.normalise(json.loads((args.daily / f"{c}.json").read_text()), int(time.time() * 1000),
                               interval_ms=DAY)
            rises = worst_rises(bars)
            v["rises"] = rises
        if c in max_lev:
            v["max_leverage"] = max_lev[c]
        v["capital"] = capital_table(f["last_12m_pct_per_year"], max_lev.get(c), rises)
        if args.basis is not None:
            sp, pp = args.basis / f"{c}_spot.json", args.basis / f"{c}_perp.json"
            if sp.exists() and pp.exists():
                now = int(time.time() * 1000)
                v["basis"] = basis_record(D.normalise(json.loads(sp.read_text()), now),
                                          D.normalise(json.loads(pp.read_text()), now))
        s["coins"][c] = v
    if args.daily is not None:
        s["data"]["daily"] = dir_digest(args.daily)
    if args.basis is not None:
        s["data"]["basis"] = dir_digest(args.basis)
    if args.spot_meta is not None:
        sm, sc = json.loads(args.spot_meta.read_text())
        tok = {t["name"]: t for t in sm["tokens"]}
        idx = {t["index"]: t["name"] for t in sm["tokens"]}
        usdc = next(t["index"] for t in sm["tokens"] if t["name"] == "USDC")
        ctx = {x["coin"]: x for x in sc}                       # contexts are keyed by coin, not universe order
        vol: Dict[str, float] = {}
        for u in sm["universe"]:
            if u["tokens"][1] == usdc and u["name"] in ctx:
                n = idx[u["tokens"][0]]
                vol[n] = max(vol.get(n, 0.0), float(ctx[u["name"]].get("dayNtlVlm") or 0))
        s["spot_tokens"] = {c: {"token": SPOT[c], "full_name": tok[SPOT[c]].get("fullName"),
                                "day_volume": vol.get(SPOT[c], 0.0)}
                            for c in s["coins"] if c in SPOT and SPOT[c] in tok}
    if args.hlp is not None:
        s["hlp"] = hlp_record(json.loads(args.hlp.read_text()))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(s, indent=1, default=str))
    (args.out / "report.md").write_text(render(s))
    print(render(s))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

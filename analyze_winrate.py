# analyze_winrate.py
"""
Signal performance analyzer for SOL Flow Radar (Cuan Sniffer).

Features:
  - R-multiple tracking (actual PnL in units of risk)
  - Signal Sharpe & Sortino ratios
  - Regime filter analysis (htf, macro, local, session, coin, side, score)
  - Auto charts: equity curve, R distribution, win rate by regime
  - Max drawdown, win streak, loss streak
"""

import csv
import time
import argparse
import os
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Tuple

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

LOG_FILE                = "signals.csv"
RESULTS_FILE            = "signals_evaluated.csv"
HYPERLIQUID_INFO_URL    = "https://api.hyperliquid.xyz/info"
API_DELAY_SECONDS       = 1.2
MAX_RETRIES             = 3
RETRY_BACKOFF           = 5
MIN_SIGNAL_AGE_MINUTES  = 60
EVALUATION_WINDOW_MIN   = 360   # 6 hours


# ── Data loading ───────────────────────────────────────────────────────────────

def load_signals() -> List[Dict]:
    rows = []
    try:
        with open(LOG_FILE, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    except FileNotFoundError:
        print(f"[WARN] {LOG_FILE} not found. Run agent.py first.")
    return rows


def load_results() -> Dict[str, Dict]:
    """Load previously evaluated results to avoid re-fetching."""
    results = {}
    if not os.path.exists(RESULTS_FILE):
        return results
    try:
        with open(RESULTS_FILE, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                results[row["signal_id"]] = row
    except Exception:
        pass
    return results


def save_results(results: Dict[str, Dict]):
    if not results:
        return
    fieldnames = list(next(iter(results.values())).keys())
    with open(RESULTS_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results.values())


# ── API fetch ──────────────────────────────────────────────────────────────────

def fetch_candles(coin: str, timestamp_iso: str, minutes: int = EVALUATION_WINDOW_MIN) -> pd.DataFrame:
    start_dt = datetime.fromisoformat(timestamp_iso)
    if start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=timezone.utc)
    end_dt = start_dt + timedelta(minutes=minutes)

    payload = {
        "type": "candleSnapshot",
        "req": {
            "coin":      coin,
            "interval":  "1m",
            "startTime": int(start_dt.timestamp() * 1000),
            "endTime":   int(end_dt.timestamp() * 1000),
        },
    }

    data = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(HYPERLIQUID_INFO_URL, json=payload, timeout=10)
            if resp.status_code == 429:
                wait = RETRY_BACKOFF * attempt
                print(f"  [429] rate limited — waiting {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            break
        except Exception as e:
            if attempt == MAX_RETRIES:
                print(f"  [FETCH ERROR] {coin}: {e}")
                return pd.DataFrame()
            time.sleep(RETRY_BACKOFF * attempt)

    if not data:
        return pd.DataFrame()

    candles = []
    for c in data:
        ts = datetime.fromtimestamp((c.get("T") or c.get("t")) / 1000, tz=timezone.utc)
        candles.append({
            "time":  ts,
            "open":  float(c["o"]),
            "high":  float(c["h"]),
            "low":   float(c["l"]),
            "close": float(c["c"]),
        })
    return pd.DataFrame(candles)


# ── Signal evaluation ──────────────────────────────────────────────────────────

def evaluate_signal(row: Dict) -> Tuple[str, float]:
    """
    Returns (outcome, r_multiple).

    outcome: "win" | "loss" | "timeout"
    r_multiple:
      win     → +rr_planned  (full TP hit)
      loss    → -1.0          (SL hit)
      timeout → unrealized R at end of window
                (final_close - entry) / stop_dist  [LONG]
                (entry - final_close) / stop_dist  [SHORT]
    """
    coin = row.get("coin", "SOL")
    side = row.get("side", "LONG")
    ts   = row.get("timestamp_utc", "")

    try:
        entry     = float(row["entry"])
        stop      = float(row["stop"])
        tp        = float(row["tp"])
        stop_dist = float(row["stop_dist"])
        rr        = float(row["rr_planned"])
    except (KeyError, ValueError):
        return "timeout", 0.0

    if stop_dist <= 0:
        return "timeout", 0.0

    df = fetch_candles(coin, ts)
    if df.empty:
        return "timeout", 0.0

    for _, bar in df.iterrows():
        high = bar["high"]
        low  = bar["low"]

        if side == "LONG":
            if low <= stop and high >= tp:
                return "loss", -1.0   # same-candle: conservative
            if low <= stop:
                return "loss", -1.0
            if high >= tp:
                return "win", +rr
        else:
            if high >= stop and low <= tp:
                return "loss", -1.0
            if high >= stop:
                return "loss", -1.0
            if low <= tp:
                return "win", +rr

    # Timeout: compute unrealized R from final close
    final_close = float(df.iloc[-1]["close"])
    if side == "LONG":
        unrealized_r = (final_close - entry) / stop_dist
    else:
        unrealized_r = (entry - final_close) / stop_dist

    return "timeout", round(unrealized_r, 3)


def is_old_enough(timestamp_iso: str) -> bool:
    try:
        ts = datetime.fromisoformat(timestamp_iso)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        age = datetime.now(timezone.utc) - ts
        return age.total_seconds() >= MIN_SIGNAL_AGE_MINUTES * 60
    except Exception:
        return False


# ── Stats ──────────────────────────────────────────────────────────────────────

def compute_stats(df: pd.DataFrame) -> Dict:
    """Compute all performance metrics on a (filtered) results dataframe."""
    resolved = df[df["outcome"].isin(["win", "loss"])]
    total    = len(df)
    n_res    = len(resolved)
    n_wins   = (resolved["outcome"] == "win").sum()
    n_loss   = (resolved["outcome"] == "loss").sum()
    n_timeout = (df["outcome"] == "timeout").sum()

    win_rate = n_wins / n_res if n_res > 0 else 0.0

    r_series  = df["r_multiple"].values
    mean_r    = np.mean(r_series) if len(r_series) > 0 else 0.0
    std_r     = np.std(r_series)  if len(r_series) > 1 else 0.0
    neg_r     = r_series[r_series < 0]
    sortino_d = np.std(neg_r) if len(neg_r) > 1 else 1.0

    sharpe  = mean_r / std_r     if std_r    > 0 else 0.0
    sortino = mean_r / sortino_d if sortino_d > 0 else 0.0

    # Equity curve & drawdown
    equity = np.cumsum(r_series)
    peak   = np.maximum.accumulate(equity) if len(equity) > 0 else np.array([0])
    dd     = equity - peak
    max_dd = float(np.min(dd)) if len(dd) > 0 else 0.0

    # Streaks
    outcomes_list = df["outcome"].tolist()
    best_streak = worst_streak = cur_w = cur_l = 0
    for o in outcomes_list:
        if o == "win":
            cur_w += 1
            cur_l  = 0
        elif o == "loss":
            cur_l += 1
            cur_w  = 0
        else:
            cur_w = cur_l = 0
        best_streak  = max(best_streak,  cur_w)
        worst_streak = max(worst_streak, cur_l)

    return {
        "total": total, "resolved": n_res, "wins": n_wins,
        "losses": n_loss, "timeouts": n_timeout,
        "win_rate": win_rate, "mean_r": mean_r, "std_r": std_r,
        "sharpe": sharpe, "sortino": sortino,
        "total_r": float(np.sum(r_series)),
        "max_dd": max_dd,
        "best_streak": best_streak, "worst_streak": worst_streak,
        "equity": equity.tolist() if len(equity) > 0 else [],
        "r_series": r_series.tolist(),
    }


def regime_breakdown(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Win rate + mean R + count per value of a regime column."""
    rows = []
    for val in sorted(df[col].dropna().unique()):
        sub = df[df[col] == val]
        res = sub[sub["outcome"].isin(["win", "loss"])]
        wr  = (res["outcome"] == "win").sum() / len(res) if len(res) > 0 else 0.0
        rows.append({
            "value":    val,
            "n":        len(sub),
            "resolved": len(res),
            "win_rate": round(wr * 100, 1),
            "mean_r":   round(sub["r_multiple"].mean(), 3),
            "total_r":  round(sub["r_multiple"].sum(), 3),
        })
    return pd.DataFrame(rows).sort_values("mean_r", ascending=False)


# ── Charts ─────────────────────────────────────────────────────────────────────

def plot_dashboard(df: pd.DataFrame, stats: Dict, output_path: str = "winrate_report.png"):
    """Generate a 6-panel performance dashboard."""

    fig = plt.figure(figsize=(18, 14), facecolor="#0e1117")
    fig.suptitle(
        "SOL Flow Radar — Signal Performance Dashboard",
        fontsize=16, color="white", fontweight="bold", y=0.98
    )

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    text_color  = "#e0e0e0"
    grid_color  = "#2a2a3a"
    accent_green = "#00c896"
    accent_red   = "#ff4d6d"
    accent_blue  = "#4da6ff"
    accent_amber = "#ffb347"

    def style_ax(ax, title):
        ax.set_facecolor("#161b22")
        ax.tick_params(colors=text_color, labelsize=8)
        ax.set_title(title, color=text_color, fontsize=10, fontweight="bold", pad=8)
        for spine in ax.spines.values():
            spine.set_color(grid_color)
        ax.yaxis.label.set_color(text_color)
        ax.xaxis.label.set_color(text_color)
        ax.grid(color=grid_color, linewidth=0.5, alpha=0.6)

    # ── 1. Equity curve ───────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    equity = np.array(stats["equity"])
    if len(equity) > 0:
        x = np.arange(len(equity))
        colors = [accent_green if v >= 0 else accent_red for v in equity]
        ax1.fill_between(x, 0, equity, where=equity >= 0, color=accent_green, alpha=0.25)
        ax1.fill_between(x, 0, equity, where=equity < 0,  color=accent_red,   alpha=0.25)
        ax1.plot(x, equity, color=accent_blue, linewidth=1.5)
        ax1.axhline(0, color=grid_color, linewidth=0.8)
        ax1.set_ylabel("Cumulative R")
    style_ax(ax1, "Equity Curve (Cumulative R-multiples)")

    # ── 2. Summary stats box ──────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor("#161b22")
    ax2.axis("off")
    ax2.set_title("Summary", color=text_color, fontsize=10, fontweight="bold", pad=8)

    wr_color = accent_green if stats["win_rate"] >= 0.5 else accent_red
    lines = [
        ("Signals",      f"{stats['total']}"),
        ("Resolved",     f"{stats['resolved']}"),
        ("Win Rate",     f"{stats['win_rate']*100:.1f}%"),
        ("Total R",      f"{stats['total_r']:+.2f}R"),
        ("Mean R",       f"{stats['mean_r']:+.3f}R"),
        ("Sharpe",       f"{stats['sharpe']:.2f}"),
        ("Sortino",      f"{stats['sortino']:.2f}"),
        ("Max DD",       f"{stats['max_dd']:.2f}R"),
        ("Best streak",  f"{stats['best_streak']} wins"),
        ("Worst streak", f"{stats['worst_streak']} losses"),
    ]
    for j, (label, value) in enumerate(lines):
        y = 0.92 - j * 0.089
        ax2.text(0.05, y, label + ":", transform=ax2.transAxes,
                 fontsize=9, color="#888", va="top")
        color = wr_color if label == "Win Rate" else (
            accent_green if label == "Total R" and stats["total_r"] > 0 else
            accent_red   if label == "Total R" and stats["total_r"] < 0 else text_color
        )
        ax2.text(0.62, y, value, transform=ax2.transAxes,
                 fontsize=9, color=color, va="top", fontweight="bold")

    # ── 3. R distribution ─────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    r_vals = np.array(stats["r_series"])
    if len(r_vals) > 0:
        bins = np.linspace(r_vals.min() - 0.1, r_vals.max() + 0.1, 30)
        ax3.hist(r_vals[r_vals >= 0], bins=bins, color=accent_green, alpha=0.8, label="Positive R")
        ax3.hist(r_vals[r_vals < 0],  bins=bins, color=accent_red,   alpha=0.8, label="Negative R")
        ax3.axvline(0, color="white", linewidth=0.8, linestyle="--")
        ax3.axvline(float(np.mean(r_vals)), color=accent_amber, linewidth=1.2,
                    linestyle="--", label=f"Mean {np.mean(r_vals):+.2f}R")
        ax3.legend(fontsize=7, facecolor="#0e1117", edgecolor=grid_color, labelcolor=text_color)
        ax3.set_xlabel("R-multiple")
    style_ax(ax3, "R Distribution")

    # ── 4. Win rate by coin ───────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    coin_data = regime_breakdown(df, "coin")
    if not coin_data.empty:
        bar_colors = [accent_green if w >= 50 else accent_red for w in coin_data["win_rate"]]
        bars = ax4.bar(coin_data["value"], coin_data["win_rate"], color=bar_colors, alpha=0.85)
        ax4.axhline(50, color="white", linewidth=0.8, linestyle="--", alpha=0.5)
        ax4.set_ylabel("Win Rate %")
        for bar, n in zip(bars, coin_data["n"]):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f"n={n}", ha="center", fontsize=7, color=text_color)
    style_ax(ax4, "Win Rate by Coin")

    # ── 5. Win rate by session ────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    sess_data = regime_breakdown(df, "session")
    if not sess_data.empty:
        bar_colors = [accent_green if w >= 50 else accent_red for w in sess_data["win_rate"]]
        bars = ax5.barh(sess_data["value"], sess_data["win_rate"], color=bar_colors, alpha=0.85)
        ax5.axvline(50, color="white", linewidth=0.8, linestyle="--", alpha=0.5)
        ax5.set_xlabel("Win Rate %")
        for bar, r, n in zip(bars, sess_data["mean_r"], sess_data["n"]):
            ax5.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                     f"{r:+.2f}R  n={n}", va="center", fontsize=7, color=text_color)
    style_ax(ax5, "Win Rate by Session")

    # ── 6. HTF regime breakdown ───────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 0])
    htf_data = regime_breakdown(df, "regime_htf_1h")
    if not htf_data.empty:
        bar_colors = [accent_green if w >= 50 else accent_red for w in htf_data["win_rate"]]
        bars = ax6.bar(htf_data["value"], htf_data["win_rate"], color=bar_colors, alpha=0.85)
        ax6.axhline(50, color="white", linewidth=0.8, linestyle="--", alpha=0.5)
        ax6.set_ylabel("Win Rate %")
        for bar, r, n in zip(bars, htf_data["mean_r"], htf_data["n"]):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f"{r:+.2f}R\nn={n}", ha="center", fontsize=7, color=text_color)
    style_ax(ax6, "Win Rate by HTF Regime (1h)")

    # ── 7. Macro regime breakdown ─────────────────────────────────────────────
    ax7 = fig.add_subplot(gs[2, 1])
    macro_data = regime_breakdown(df, "regime_macro_4h")
    if not macro_data.empty:
        bar_colors = [accent_green if w >= 50 else accent_red for w in macro_data["win_rate"]]
        bars = ax7.bar(macro_data["value"], macro_data["win_rate"], color=bar_colors, alpha=0.85)
        ax7.axhline(50, color="white", linewidth=0.8, linestyle="--", alpha=0.5)
        ax7.set_ylabel("Win Rate %")
        for bar, r, n in zip(bars, macro_data["mean_r"], macro_data["n"]):
            ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f"{r:+.2f}R\nn={n}", ha="center", fontsize=7, color=text_color)
    style_ax(ax7, "Win Rate by Macro Regime (4h)")

    # ── 8. Mean R by score bucket ─────────────────────────────────────────────
    ax8 = fig.add_subplot(gs[2, 2])
    df2 = df.copy()
    df2["score_bucket"] = pd.cut(
        df2["total_score"].astype(float),
        bins=[0, 0.70, 0.80, 0.90, 1.01],
        labels=["<0.70", "0.70-0.80", "0.80-0.90", "0.90+"]
    )
    score_data = df2.groupby("score_bucket", observed=True)["r_multiple"].agg(
        mean_r="mean", count="count"
    ).reset_index()
    if not score_data.empty:
        bar_colors = [accent_green if r >= 0 else accent_red for r in score_data["mean_r"]]
        bars = ax8.bar(score_data["score_bucket"].astype(str), score_data["mean_r"],
                       color=bar_colors, alpha=0.85)
        ax8.axhline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.5)
        ax8.set_ylabel("Mean R")
        for bar, n in zip(bars, score_data["count"]):
            ax8.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + (0.01 if bar.get_height() >= 0 else -0.05),
                     f"n={n}", ha="center", fontsize=7, color=text_color)
    style_ax(ax8, "Mean R by Score Bucket")

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\nChart saved → {output_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Signal win rate analyzer")
    parser.add_argument("--htf",    help="Filter by HTF regime: up/down/chop")
    parser.add_argument("--macro",  help="Filter by macro regime: up/down/chop")
    parser.add_argument("--regime", help="Filter by setup family: continuation/reversal")
    parser.add_argument("--session",help="Filter by session: london_open/asia_open/etc")
    parser.add_argument("--coin",   help="Filter by coin: SOL/BTC/JUP/etc")
    parser.add_argument("--side",   help="Filter by side: LONG/SHORT")
    parser.add_argument("--no-fetch", action="store_true",
                        help="Skip API fetch — analyze cached results only")
    parser.add_argument("--chart",  default="winrate_report.png",
                        help="Output chart filename (default: winrate_report.png)")
    args = parser.parse_args()

    signals = load_signals()
    if not signals:
        return

    cached = load_results()

    results = {}
    skipped = 0
    fetched = 0

    if not args.no_fetch:
        print(f"Evaluating {len(signals)} signals...\n")
        for i, row in enumerate(signals, 1):
            sig_id = str(row.get("signal_id", i))
            coin   = row.get("coin", "SOL")
            side   = row.get("side", "?")
            ts     = row.get("timestamp_utc", "")[:16]

            # Use cached result if available
            if sig_id in cached:
                results[sig_id] = cached[sig_id]
                skipped += 1
                continue

            if not is_old_enough(row.get("timestamp_utc", "")):
                skipped += 1
                continue

            print(f"[{i}/{len(signals)}] {coin} {side} @ {ts}", end=" ... ", flush=True)
            outcome, r_mult = evaluate_signal(row)
            print(f"{outcome}  ({r_mult:+.2f}R)")

            result_row = {**row, "outcome": outcome, "r_multiple": r_mult}
            results[sig_id] = result_row
            fetched += 1
            time.sleep(API_DELAY_SECONDS)

        # Merge cached with fresh
        for sig_id, cached_row in cached.items():
            if sig_id not in results:
                results[sig_id] = cached_row

        save_results(results)
        print(f"\nFetched: {fetched}  |  From cache: {skipped}")
    else:
        results = cached
        print(f"Using {len(results)} cached results (--no-fetch mode)")

    if not results:
        print("No evaluated results yet. Run without --no-fetch first.")
        return

    # Build dataframe
    df = pd.DataFrame(list(results.values()))
    df["r_multiple"]  = pd.to_numeric(df["r_multiple"],  errors="coerce").fillna(0)
    df["total_score"] = pd.to_numeric(df["total_score"], errors="coerce").fillna(0)

    # Apply filters
    active_filters = []
    if args.htf:
        df = df[df["regime_htf_1h"] == args.htf]
        active_filters.append(f"htf={args.htf}")
    if args.macro:
        df = df[df["regime_macro_4h"] == args.macro]
        active_filters.append(f"macro={args.macro}")
    if args.regime:
        df = df[df["regime_local"] == args.regime]
        active_filters.append(f"regime={args.regime}")
    if args.session:
        df = df[df["session"] == args.session]
        active_filters.append(f"session={args.session}")
    if args.coin:
        df = df[df["coin"].str.upper() == args.coin.upper()]
        active_filters.append(f"coin={args.coin.upper()}")
    if args.side:
        df = df[df["side"].str.upper() == args.side.upper()]
        active_filters.append(f"side={args.side.upper()}")

    if df.empty:
        print("No signals match the applied filters.")
        return

    filter_label = "  |  ".join(active_filters) if active_filters else "All signals"
    print(f"\n{'='*60}")
    print(f"FILTER: {filter_label}")
    print(f"{'='*60}")

    stats = compute_stats(df)

    print(f"\n{'─'*40}")
    print(f"  Signals     : {stats['total']}")
    print(f"  Resolved    : {stats['resolved']}  (wins={stats['wins']} losses={stats['losses']})")
    print(f"  Timeouts    : {stats['timeouts']}")
    print(f"{'─'*40}")
    print(f"  Win rate    : {stats['win_rate']*100:.1f}%")
    print(f"  Total R     : {stats['total_r']:+.2f}R")
    print(f"  Mean R      : {stats['mean_r']:+.3f}R  (std={stats['std_r']:.3f})")
    print(f"  Sharpe      : {stats['sharpe']:.2f}")
    print(f"  Sortino     : {stats['sortino']:.2f}")
    print(f"  Max DD      : {stats['max_dd']:.2f}R")
    print(f"  Best streak : {stats['best_streak']} wins")
    print(f"  Worst streak: {stats['worst_streak']} losses")

    # Regime breakdowns (only show when not already filtered)
    dims = [
        ("coin",           "By Coin",            args.coin),
        ("session",        "By Session",          args.session),
        ("regime_htf_1h",  "By HTF Regime (1h)",  args.htf),
        ("regime_macro_4h","By Macro Regime (4h)", args.macro),
        ("regime_local",   "By Setup Family",     args.regime),
        ("side",           "By Side",             args.side),
    ]
    for col, label, already_filtered in dims:
        if already_filtered or col not in df.columns:
            continue
        bk = regime_breakdown(df, col)
        if bk.empty:
            continue
        print(f"\n{'─'*40}")
        print(f"  {label}")
        print(f"{'─'*40}")
        for _, r in bk.iterrows():
            bar = "█" * int(min(r["win_rate"], 100) / 5)
            print(f"  {r['value']:15s}  {r['win_rate']:5.1f}%  {r['mean_r']:+.3f}R  "
                  f"n={r['n']:3d}  {bar}")

    # Auto chart
    plot_dashboard(df, stats, output_path=args.chart)


if __name__ == "__main__":
    main()
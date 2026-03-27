# trades_recap.py
"""
Trade P&L recap — reads trades.csv and generates a Telegram-formatted
daily and all-time performance statement.

Called from agent.py alongside the existing signal recap.
Can also be run standalone:
    python trades_recap.py             # all-time
    python trades_recap.py --today     # today only
    python trades_recap.py --dry-run   # print instead of send
"""

import csv
import os
import argparse
from datetime import datetime, timezone, date
from typing import List, Dict, Optional

TRADES_FILE = os.getenv("TRADES_FILE", "trades.csv")


# ── Data loading ───────────────────────────────────────────────────────────────

def load_closed_trades(since_date: Optional[date] = None) -> List[Dict]:
    """
    Load all CLOSED trades from trades.csv.
    If since_date is set, only return trades closed on or after that date.
    """
    if not os.path.exists(TRADES_FILE):
        return []

    rows = []
    try:
        with open(TRADES_FILE, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("state", "") != "closed":
                    continue

                closed_at_raw = row.get("closed_at", "")
                if not closed_at_raw:
                    continue

                try:
                    closed_at = datetime.fromisoformat(closed_at_raw)
                    if closed_at.tzinfo is None:
                        closed_at = closed_at.replace(tzinfo=timezone.utc)
                except ValueError:
                    continue

                if since_date and closed_at.date() < since_date:
                    continue

                row["_closed_at"] = closed_at
                rows.append(row)

    except Exception as e:
        print(f"[TRADES_RECAP] Error reading {TRADES_FILE}: {e}")

    return rows


# ── Stats computation ──────────────────────────────────────────────────────────

def compute_trade_stats(trades: List[Dict]) -> Dict:
    """
    Compute full P&L stats from a list of closed trade rows.
    """
    if not trades:
        return {}

    total_trades = len(trades)
    wins   = [t for t in trades if float(t.get("realized_r", 0)) > 0]
    losses = [t for t in trades if float(t.get("realized_r", 0)) <= 0]

    total_r   = sum(float(t.get("realized_r", 0)) for t in trades)
    total_pnl = sum(float(t.get("pnl_usd",    0)) for t in trades)
    win_rate  = len(wins) / total_trades * 100 if total_trades > 0 else 0.0

    r_vals = [float(t.get("realized_r", 0)) for t in trades]
    best  = max(r_vals) if r_vals else 0.0
    worst = min(r_vals) if r_vals else 0.0

    # Drawdown on R-series
    equity = 0.0
    peak   = 0.0
    max_dd = 0.0
    for r in r_vals:
        equity += r
        if equity > peak:
            peak = equity
        dd = equity - peak
        if dd < max_dd:
            max_dd = dd

    # Per-coin breakdown
    coin_stats: Dict[str, Dict] = {}
    for t in trades:
        coin = t.get("coin", "?")
        r    = float(t.get("realized_r", 0))
        pnl  = float(t.get("pnl_usd",   0))
        if coin not in coin_stats:
            coin_stats[coin] = {"n": 0, "wins": 0, "total_r": 0.0, "total_pnl": 0.0}
        coin_stats[coin]["n"]         += 1
        coin_stats[coin]["total_r"]   += r
        coin_stats[coin]["total_pnl"] += pnl
        if r > 0:
            coin_stats[coin]["wins"] += 1

    # Best and worst trade
    best_trade  = max(trades, key=lambda t: float(t.get("realized_r", 0)))
    worst_trade = min(trades, key=lambda t: float(t.get("realized_r", 0)))

    # Avg trade duration
    durations = []
    for t in trades:
        try:
            opened = datetime.fromisoformat(t.get("opened_at", ""))
            closed = t["_closed_at"]
            if opened.tzinfo is None:
                opened = opened.replace(tzinfo=timezone.utc)
            durations.append((closed - opened).total_seconds() / 60)
        except Exception:
            pass
    avg_duration_min = sum(durations) / len(durations) if durations else 0.0

    return {
        "total_trades":     total_trades,
        "wins":             len(wins),
        "losses":           len(losses),
        "win_rate":         win_rate,
        "total_r":          total_r,
        "total_pnl":        total_pnl,
        "best_r":           best,
        "worst_r":          worst,
        "max_dd_r":         max_dd,
        "avg_duration_min": avg_duration_min,
        "coin_stats":       coin_stats,
        "best_trade":       best_trade,
        "worst_trade":      worst_trade,
    }


# ── Message builder ────────────────────────────────────────────────────────────

def _utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _dur(minutes: float) -> str:
    h = int(minutes // 60)
    m = int(minutes % 60)
    return f"{h}h {m}m" if h > 0 else f"{m}m"


def build_daily_recap(stats: Dict, trade_date: date) -> str:
    if not stats:
        return (
            f"📋 *[PAPER] Daily Trade Recap — {trade_date}*\n\n"
            "_No closed trades today._"
        )

    total_r   = stats["total_r"]
    total_pnl = stats["total_pnl"]
    pnl_emoji = "✅" if total_pnl >= 0 else "❌"
    r_emoji   = "🟢" if total_r   >= 0 else "🔴"

    lines = [
        f"📋 *[PAPER] Daily Trade Recap — {trade_date}*",
        "",
        f"🕒 `{_utc_now_str()}`",
        "",
        "```",
        f"  Trades      : {stats['total_trades']}  "
        f"(W:{stats['wins']} / L:{stats['losses']})",
        f"  Win rate    : {stats['win_rate']:.1f}%",
        f"  Total R     : {total_r:+.2f}R",
        f"  P&L (USD)   : ${total_pnl:+.2f}",
        f"  Max DD      : {stats['max_dd_r']:.2f}R",
        f"  Avg duration: {_dur(stats['avg_duration_min'])}",
        "",
        f"  Best trade  : {float(stats['best_trade'].get('realized_r',0)):+.2f}R "
        f"({stats['best_trade'].get('coin','?')} {stats['best_trade'].get('side','?')})",
        f"  Worst trade : {float(stats['worst_trade'].get('realized_r',0)):+.2f}R "
        f"({stats['worst_trade'].get('coin','?')} {stats['worst_trade'].get('side','?')})",
    ]

    # Per-coin rows (sorted by total_r)
    if stats["coin_stats"]:
        lines.append("")
        lines.append("  By coin:")
        sorted_coins = sorted(
            stats["coin_stats"].items(),
            key=lambda x: x[1]["total_r"],
            reverse=True
        )
        for coin, cs in sorted_coins:
            wr = cs["wins"] / cs["n"] * 100 if cs["n"] > 0 else 0
            lines.append(
                f"    {coin:<6}  {cs['total_r']:+.2f}R  "
                f"${cs['total_pnl']:+.2f}  "
                f"WR:{wr:.0f}%  n={cs['n']}"
            )

    lines.append("```")
    lines.append(f"{pnl_emoji} {r_emoji} Net: `{total_r:+.2f}R`  `${total_pnl:+.2f}`")

    return "\n".join(lines)


def build_alltime_recap(
    stats_today: Dict,
    stats_all:   Dict,
    trade_date:  date,
    starting_balance: float,
) -> str:
    if not stats_all:
        return (
            "📊 *[PAPER] All-Time P&L Statement*\n\n"
            "_No closed trades yet._"
        )

    total_r   = stats_all["total_r"]
    total_pnl = stats_all["total_pnl"]
    balance   = starting_balance + total_pnl
    pct_gain  = (total_pnl / starting_balance * 100) if starting_balance > 0 else 0.0
    pnl_emoji = "✅" if total_pnl >= 0 else "❌"

    today_r   = stats_today.get("total_r",   0.0) if stats_today else 0.0
    today_pnl = stats_today.get("total_pnl", 0.0) if stats_today else 0.0

    lines = [
        "📊 *[PAPER] All-Time P&L Statement*",
        "",
        f"🕒 `{_utc_now_str()}`",
        "",
        "```",
        f"  Balance     : ${balance:.2f}  (start: ${starting_balance:.2f})",
        f"  Total P&L   : ${total_pnl:+.2f}  ({pct_gain:+.1f}%)",
        f"  Total R     : {total_r:+.2f}R",
        "",
        f"  Trades      : {stats_all['total_trades']}  "
        f"(W:{stats_all['wins']} / L:{stats_all['losses']})",
        f"  Win rate    : {stats_all['win_rate']:.1f}%",
        f"  Max DD      : {stats_all['max_dd_r']:.2f}R",
        f"  Avg duration: {_dur(stats_all['avg_duration_min'])}",
        "",
        f"  Today ({trade_date}):",
        f"    R   : {today_r:+.2f}R",
        f"    P&L : ${today_pnl:+.2f}",
    ]

    # All-time coin breakdown
    if stats_all["coin_stats"]:
        lines.append("")
        lines.append("  All-time by coin:")
        sorted_coins = sorted(
            stats_all["coin_stats"].items(),
            key=lambda x: x[1]["total_r"],
            reverse=True
        )
        for coin, cs in sorted_coins:
            wr = cs["wins"] / cs["n"] * 100 if cs["n"] > 0 else 0
            lines.append(
                f"    {coin:<6}  {cs['total_r']:+.2f}R  "
                f"${cs['total_pnl']:+.2f}  "
                f"WR:{wr:.0f}%  n={cs['n']}"
            )

    lines.append("```")
    lines.append(f"{pnl_emoji} Balance: `${balance:.2f}`  Return: `{pct_gain:+.1f}%`")

    return "\n".join(lines)


# ── Main entry points ──────────────────────────────────────────────────────────

def run_trades_recap(notify_fn=None, starting_balance: float = 1000.0) -> str:
    """
    Called from agent.py recap worker.
    Returns the combined message (daily + all-time).
    notify_fn: if provided, sends the message via Telegram.
    """
    today          = datetime.now(timezone.utc).date()
    trades_today   = load_closed_trades(since_date=today)
    trades_all     = load_closed_trades()

    stats_today = compute_trade_stats(trades_today)
    stats_all   = compute_trade_stats(trades_all)

    daily_msg   = build_daily_recap(stats_today, today)
    alltime_msg = build_alltime_recap(stats_today, stats_all, today, starting_balance)

    if notify_fn:
        notify_fn(daily_msg)
        notify_fn(alltime_msg)

    return f"{daily_msg}\n\n{alltime_msg}"


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Trade P&L recap from trades.csv")
    parser.add_argument("--today",    action="store_true", help="Daily recap only")
    parser.add_argument("--alltime",  action="store_true", help="All-time recap only")
    parser.add_argument("--dry-run",  action="store_true", help="Print instead of sending")
    parser.add_argument("--balance",  type=float,
                        default=float(os.getenv("STARTING_BALANCE", "1000")),
                        help="Starting paper balance (default: $1000)")
    args = parser.parse_args()

    today          = datetime.now(timezone.utc).date()
    trades_today   = load_closed_trades(since_date=today)
    trades_all     = load_closed_trades()
    stats_today    = compute_trade_stats(trades_today)
    stats_all      = compute_trade_stats(trades_all)

    messages = []
    if args.today or not args.alltime:
        messages.append(build_daily_recap(stats_today, today))
    if args.alltime or not args.today:
        messages.append(build_alltime_recap(stats_today, stats_all, today, args.balance))

    for msg in messages:
        if args.dry_run:
            print(msg)
            print()
        else:
            from notifier import send_telegram_message
            send_telegram_message(msg)
            print("[TRADES_RECAP] Sent.")


if __name__ == "__main__":
    main()
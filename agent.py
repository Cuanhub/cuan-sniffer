# agent.py
"""
Multi-coin signal agent.

Scans all coins in TRACKED_COINS (.env) every cycle.
SOL gets full context (on-chain flow + perp sentiment).
Non-SOL coins get perp sentiment only (no on-chain flow).

Features:
  - Startup / stop / error Telegram notifications
  - Per-coin error isolation
  - Per-coin error cooldown — no Telegram spam during outages
  - Env validation before feeds start
  - V1 product mode: 15m signals only
  - Automatic daily recap trigger (no cron needed)
  - Recap runs in background thread — signal loop never pauses
  - Daily recap runs once per UTC day
  - Recap subprocess timeout protection
  - Paper trading execution layer wired in
  - Daily and all-time trade P&L recap via Telegram
"""

import os
import sys
import time
import subprocess
import threading
from math import fabs
from typing import Dict, Any, Optional
from datetime import datetime, timezone

from config import POLL_INTERVAL_SECONDS
from db import init_db, SessionLocal
from perp_data import PerpDataFeed
from flow_context import FlowContext
from perp_sentiment import PerpSentimentFeed
from signal_engine import AdaptiveSignalEngine, Signal
from notifier import send_telegram_message
from signal_log import init_signal_log, append_signal
from alerts import AlertManager
from executor import Executor
from trades_recap import run_trades_recap


# ── Coin list ──────────────────────────────────────────────────────────────────
_raw = os.getenv("TRACKED_COINS", "SOL")
TRACKED_COINS: list[str] = list(dict.fromkeys(
    c.strip().upper() for c in _raw.split(",") if c.strip()
))
if "SOL" not in TRACKED_COINS:
    TRACKED_COINS.insert(0, "SOL")

WARMUP_BARS: Dict[str, int] = {
    "SOL": 40, "ETH": 40, "BTC": 40,
    "JUP": 30, "JTO": 30,
    "WIF": 25, "PYTH": 30,
}
DEFAULT_WARMUP = 30

# Notify on first error, then every Nth — prevents Telegram spam
ERROR_NOTIFY_EVERY = int(os.getenv("ERROR_NOTIFY_EVERY", "10"))

# Daily recap controls
RECAP_TRACK_FILE = os.getenv("RECAP_TRACK_FILE", "last_recap.txt")
RECAP_NO_FETCH = os.getenv("RECAP_NO_FETCH", "1") == "1"
RECAP_CHART = os.getenv("RECAP_CHART", "winrate_report.png")
RECAP_TIMEOUT_SECONDS = int(os.getenv("RECAP_TIMEOUT_SECONDS", "300"))
RECAP_STARTING_BALANCE = float(os.getenv("STARTING_BALANCE", "1000.0"))

# Guard: only one recap thread at a time
_recap_running = threading.Event()


# ── Helpers ────────────────────────────────────────────────────────────────────

def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def notify(text: str):
    """Fire-and-forget Telegram — never raises."""
    try:
        send_telegram_message(text)
    except Exception as e:
        print(f"[NOTIFY ERROR] {e}")


def validate_env() -> bool:
    """Check critical env vars before starting feeds."""
    ok = True

    if not os.getenv("TELEGRAM_BOT_TOKEN"):
        print("[ENV] WARNING: TELEGRAM_BOT_TOKEN not set — alerts will not send")
        ok = False

    if not os.getenv("TELEGRAM_CHAT_ID"):
        print("[ENV] WARNING: TELEGRAM_CHAT_ID not set — alerts will not send")
        ok = False

    if not TRACKED_COINS:
        print("[ENV] ERROR: TRACKED_COINS is empty — nothing to scan")
        ok = False

    return ok


# ── Recap scheduling ───────────────────────────────────────────────────────────

def read_last_recap_time() -> Optional[datetime]:
    if not os.path.exists(RECAP_TRACK_FILE):
        return None

    try:
        with open(RECAP_TRACK_FILE, "r", encoding="utf-8") as f:
            raw = f.read().strip()

        if not raw:
            return None

        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt

    except Exception as e:
        print(f"[RECAP] Could not read {RECAP_TRACK_FILE}: {e}")
        return None


def write_last_recap_time(dt: datetime):
    try:
        with open(RECAP_TRACK_FILE, "w", encoding="utf-8") as f:
            f.write(dt.isoformat())
    except Exception as e:
        print(f"[RECAP] Could not write {RECAP_TRACK_FILE}: {e}")


def should_run_recap() -> bool:
    """
    Run once per UTC day.
    Also avoid firing when a recap thread is already running.
    """
    if _recap_running.is_set():
        return False

    now = datetime.now(timezone.utc)
    last = read_last_recap_time()

    if last is None:
        return os.path.exists("signals.csv") and os.path.getsize("signals.csv") > 0

    return now.date() > last.date()


def extract_recap_summary(stdout: str) -> str:
    """Pull the most useful lines from analyze_winrate.py output."""
    if not stdout:
        return "No analyzer output."

    lines = [line.rstrip() for line in stdout.splitlines()]
    picked: list[str] = []

    metric_prefixes = (
        "Fetched:", "Using ", "FILTER:",
        "  Signals", "  Resolved", "  Timeouts",
        "  Win rate", "  Total R", "  Mean R",
        "  Sharpe", "  Sortino", "  Max DD",
        "  Best streak", "  Worst streak",
    )

    for line in lines:
        if line.strip().startswith(metric_prefixes):
            picked.append(line)

    sections_to_keep = {"By Coin", "By Setup Family", "By Side"}

    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped in sections_to_keep:
            picked.append("")
            picked.append(stripped)

            j = i + 1
            kept_rows = 0
            while j < len(lines):
                s = lines[j].strip()
                if not s:
                    j += 1
                    continue
                if s.startswith("By ") and s not in sections_to_keep:
                    break
                if s.startswith("─"):
                    j += 1
                    continue

                picked.append(lines[j])
                kept_rows += 1
                if kept_rows >= 5:
                    break
                j += 1

            i = j
            continue
        i += 1

    if not picked:
        picked = lines[:30]

    seen: set[str] = set()
    final_lines = []
    for line in picked:
        key = line.rstrip()
        if key not in seen:
            seen.add(key)
            final_lines.append(line)

    return "\n".join(final_lines).strip()[:3000]


def build_trader_grade_recap(summary_text: str) -> str:
    return (
        "📊 *Cuan Sniffer Daily Recap*\n\n"
        f"🕒 Time: `{utc_now()}`\n"
        f"📈 Chart: `{RECAP_CHART}`\n\n"
        "```text\n"
        f"{summary_text}\n"
        "```"
    )


def _run_recap_worker():
    """
    Worker that runs in a daemon thread.

    Notes:
      - Writes timestamp BEFORE running to prevent retry spam on failure.
      - Runs in background thread so signal loop never pauses.
      - Includes timeout so a hung analyzer does not block future recaps forever.
    """
    _recap_running.set()
    try:
        print("[RECAP] Running automatic daily recap...")
        notify("📊 *Running daily performance recap...*")

        # Mark immediately so failures do not retry every cycle
        write_last_recap_time(datetime.now(timezone.utc))

        analyzer_args = ["analyze_winrate.py", "--chart", RECAP_CHART]
        if RECAP_NO_FETCH:
            analyzer_args.append("--no-fetch")

        proc = subprocess.run(
            [sys.executable, *analyzer_args],
            capture_output=True,
            text=True,
            cwd=os.getcwd(),
            timeout=RECAP_TIMEOUT_SECONDS,
        )

        if proc.returncode != 0:
            err_blob = (proc.stderr or proc.stdout or "Unknown error")[:3000]
            notify(
                "❌ *Cuan Sniffer Daily Recap Failed*\n\n"
                f"🕒 Time: `{utc_now()}`\n"
                f"Exit code: `{proc.returncode}`\n\n"
                "```text\n"
                f"{err_blob}\n"
                "```"
            )
            print(f"[RECAP ERROR] analyzer exited with code {proc.returncode}")
            return

        summary = extract_recap_summary(proc.stdout)
        msg = build_trader_grade_recap(summary)
        notify(msg)
        print("[RECAP] Daily recap sent.")

        # ── Trades P&L recap (daily + all-time) ──────────────────────────
        try:
            run_trades_recap(
                notify_fn=notify,
                starting_balance=RECAP_STARTING_BALANCE,
            )
            print("[RECAP] Trade P&L recap sent.")
        except Exception as te:
            print(f"[RECAP] Trade P&L recap failed: {te}")

    except subprocess.TimeoutExpired:
        print("[RECAP ERROR] analyzer timed out")
        notify(
            "❌ *Cuan Sniffer Daily Recap Timed Out*\n\n"
            f"🕒 Time: `{utc_now()}`\n"
            f"Timeout: `{RECAP_TIMEOUT_SECONDS}s`"
        )
    except Exception as e:
        print(f"[RECAP ERROR] {e}")
        notify(
            "❌ *Cuan Sniffer Recap Exception*\n\n"
            f"🕒 Time: `{utc_now()}`\n"
            f"`{str(e)[:250]}`"
        )
    finally:
        _recap_running.clear()


def trigger_recap():
    """Launch recap in a daemon thread — never blocks the signal loop."""
    if _recap_running.is_set():
        return

    t = threading.Thread(target=_run_recap_worker, daemon=True)
    t.start()


# ── Signal message builder ─────────────────────────────────────────────────────

def format_signal_message(
    coin: str,
    signal: Signal,
    flow_snapshot: Dict[str, Any],
    sentiment,
    tf_label: str = "15m",
) -> str:
    side = signal.side
    emoji = "🟢" if side == "LONG" else "🔴"
    arrow = "📈" if side == "LONG" else "📉"

    price = float(signal.entry_price)
    sl = float(signal.stop_price)
    tp = float(signal.tp_price)
    rr = fabs((tp - price) / (price - sl)) if price != sl else 0.0

    whale_pressure = float(flow_snapshot.get("whale_pressure", 0.0)) if flow_snapshot else 0.0

    if whale_pressure > 0.4:
        flow_bias = "Aggressive buyers"
    elif whale_pressure > 0.15:
        flow_bias = "Buyers stepping in"
    elif whale_pressure < -0.4:
        flow_bias = "Heavy selling"
    elif whale_pressure < -0.15:
        flow_bias = "Sellers active"
    else:
        flow_bias = "Neutral flow"

    funding = float(getattr(sentiment, "funding_rate", 0.0))
    oi = int(getattr(sentiment, "open_interest", 0) or 0)

    if funding > 0:
        funding_txt = "Longs crowded"
    elif funding < 0:
        funding_txt = "Shorts crowded"
    else:
        funding_txt = "Balanced"

    reasons = (signal.reason or "").lower()
    if "sweep" in reasons:
        setup = "Liquidity Sweep"
    elif "choch" in reasons:
        setup = "Structure Flip"
    elif "bos" in reasons:
        setup = "Breakout"
    else:
        setup = "Flow Setup"

    strength = int(min(100, max(50, signal.confidence * 100)))
    regime = signal.regime.replace("_", " ")
    flow_section = f"\n🐋 Flow: {flow_bias}" if coin == "SOL" else ""

    return (
        f"{emoji} *{coin} {side} SIGNAL* {arrow}\n\n"
        f"🎯 Setup: *{setup}*\n"
        f"🧭 Regime: `{regime}`\n"
        f"⏱ TF: `{tf_label}`\n\n"
        f"💰 Entry: `{price:.4f}`\n"
        f"🛑 SL: `{sl:.4f}`\n"
        f"🎯 TP: `{tp:.4f}`\n"
        f"📊 R/R: `{rr:.2f}R`\n\n"
        f"⚡ Strength: `{strength}%`\n"
        f"📈 Funding: {funding_txt}\n"
        f"📦 OI: `{oi}`"
        f"{flow_section}\n\n"
        f"🧠 _{setup} detected with confluence_"
    )


# ── Per-coin state ─────────────────────────────────────────────────────────────

class CoinState:
    def __init__(self, coin: str, flow_ctx: Optional[FlowContext] = None):
        self.coin = coin
        self.flow_ctx = flow_ctx

        self.perp_feed = PerpDataFeed(coin=coin, interval="15m", max_candles=400)
        self.sent_feed = PerpSentimentFeed(coin=coin)

        self.engine = AdaptiveSignalEngine(
            score_threshold=0.80,
            atr_stop_mult=1.3,
            atr_tp_mult=3.0,
            min_stop_pct=0.004,
            min_tp_pct=0.010,
            swing_threshold_1h=0.55,
            swing_threshold_4h=0.60,
            debug=True,
        )

        self.last_signal: Dict[str, Dict] = {
            "15m": {"side": None, "price": None},
        }

        self.alert_mgr = AlertManager(
            min_flow_strength=0.7,
            min_imbalance_30m=0.7,
            min_funding_mag=0.01,
            cooldown_seconds=900,
        )

        self._error_count: int = 0

    def start_feeds(self):
        self.perp_feed.start(interval_sec=5)
        self.sent_feed.start(interval_sec=30)
        print(f"[{self.coin}] feeds started")

    def get_flow_snapshot(self) -> Dict[str, Any]:
        return self.flow_ctx.compute_flow_snapshot() if self.flow_ctx else {}

    def is_duplicate(self, tf: str, side: str, price: float, threshold: float) -> bool:
        last = self.last_signal[tf]
        if last["side"] != side or last["price"] is None:
            return False
        return abs(price - last["price"]) / last["price"] < threshold

    def mark_signal(self, tf: str, side: str, price: float):
        self.last_signal[tf] = {"side": side, "price": price}

    def on_error(self) -> bool:
        self._error_count += 1
        return (
            self._error_count == 1 or
            self._error_count % ERROR_NOTIFY_EVERY == 0
        )

    def on_success(self):
        if self._error_count > 0:
            print(f"[{self.coin}] recovered after {self._error_count} consecutive errors")
        self._error_count = 0


# ── Per-coin processing ────────────────────────────────────────────────────────

def process_coin(state: CoinState, executor: Executor) -> bool:
    coin = state.coin
    warmup = WARMUP_BARS.get(coin, DEFAULT_WARMUP)

    df = state.perp_feed.get_ohlcv_df()
    if df is None or len(df) < warmup:
        print(f"[{coin}] warming up... {0 if df is None else len(df)}/{warmup}")
        return False

    flow_snapshot = state.get_flow_snapshot()
    sent = state.sent_feed.get_snapshot()
    funding_rate = float(getattr(sent, "funding_rate", 0.0))
    open_interest = int(getattr(sent, "open_interest", 0) or 0)
    bias = float(getattr(sent, "bias", 0.0))

    if coin == "SOL" and flow_snapshot:
        state.alert_mgr.maybe_alert_large_flow(flow_snapshot)
    state.alert_mgr.maybe_alert_funding_extreme(funding_rate)

    sig = state.engine.generate_signal(df, flow_snapshot, sent, coin=coin)
    if sig is None:
        return False

    price = float(sig.entry_price)
    if state.is_duplicate("15m", sig.side, price, 0.010):
        return False

    sig_id = int(time.time() * 1000)
    sig.meta["signal_id"] = sig_id

    append_signal(
        coin=coin,
        sig_id=sig_id,
        signal=sig,
        flow_snapshot=flow_snapshot,
        funding_rate=funding_rate,
        open_interest=open_interest,
        long_short_bias=bias,
    )

    msg = format_signal_message(coin, sig, flow_snapshot, sent, "15m")
    print(f"[{coin}] 15m signal fired")
    notify(msg)

    # Send signal into execution layer
    try:
        opened = executor.on_signal(sig, sig_id=sig_id)
        if opened:
            print(f"[{coin}] paper trade opened")
    except Exception as e:
        print(f"[EXECUTOR ERROR] {coin}: {e}")

    state.mark_signal("15m", sig.side, price)
    return True


# ── State construction ─────────────────────────────────────────────────────────

def build_states(flow_ctx: FlowContext) -> Dict[str, CoinState]:
    states: Dict[str, CoinState] = {}
    for coin in TRACKED_COINS:
        ctx = flow_ctx if coin == "SOL" else None
        state = CoinState(coin, ctx)
        state.start_feeds()
        states[coin] = state
    return states


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    if not validate_env():
        print("[AGENT] Invalid environment. Exiting.")
        return

    init_db()
    init_signal_log()

    flow_ctx = FlowContext(SessionLocal)
    states = build_states(flow_ctx)
    executor = Executor(notify_fn=notify)

    # Boot status — reconstructed from trades.csv, includes open positions
    notify(executor.boot_status_message())
    notify(
        f"🎯 Scanning: `{', '.join(TRACKED_COINS)}`\n"
        f"🕒 `{utc_now()}`"
    )

    try:
        while True:
            any_signal = False

            for coin in TRACKED_COINS:
                try:
                    if process_coin(states[coin], executor):
                        any_signal = True
                    states[coin].on_success()

                except Exception as e:
                    err_msg = str(e)[:180]
                    count = states[coin]._error_count + 1
                    print(f"[{coin} ERROR #{count}] {err_msg}")

                    if states[coin].on_error():
                        notify(
                            f"❌ *{coin} error* (x{states[coin]._error_count})\n"
                            f"{utc_now()}\n"
                            f"`{err_msg}`"
                        )

            # Update paper trader / open positions every cycle
            try:
                executor.update()
            except Exception as e:
                print(f"[EXECUTOR UPDATE ERROR] {e}")
                notify(
                    f"❌ *Executor update error*\n"
                    f"{utc_now()}\n"
                    f"`{str(e)[:180]}`"
                )

            # Automatic daily recap — runs in background, never blocks signals
            if should_run_recap():
                trigger_recap()

            if not any_signal:
                print(f"[AGENT] No signals this cycle ({utc_now()})")

            time.sleep(POLL_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        notify(
            f"🛑 *Bot stopped cleanly*\n"
            f"{utc_now()}\n"
            f"{executor.paper.session_summary()}"
        )
        print("[AGENT] Stopped.")

    except Exception as e:
        err_msg = str(e)[:200]
        print(f"[AGENT ERROR] {err_msg}")
        notify(
            f"🚨 *Agent fatal error*\n"
            f"{utc_now()}\n"
            f"`{err_msg}`"
        )
        time.sleep(5)


if __name__ == "__main__":
    main()
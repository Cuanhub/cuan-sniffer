"""
Multi-coin signal agent — Cuan Sniffer (LIVE-ONLY).

Notification system:
  - Entry fill notifications
  - Live exit notifications handled through executor/update path
  - Daily recap support retained

Live-only cleanup:
  - Removed PAPER_MODE branching
  - Removed any dependency on executor.paper
"""

import os
import sys
import time
import subprocess
import threading
from typing import Dict, Any, Optional
from datetime import datetime, timezone

from config import POLL_INTERVAL_SECONDS as CONFIG_POLL_INTERVAL_SECONDS
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


# ── Runtime config ─────────────────────────────────────────────────────────────

ENGINE_SCORE_THRESHOLD = float(os.getenv("MIN_SIGNAL_SCORE", "0.66"))
ENGINE_ATR_STOP_MULT = float(os.getenv("ENGINE_ATR_STOP_MULT", "1.3"))
ENGINE_ATR_TP_MULT = float(os.getenv("ENGINE_ATR_TP_MULT", "4.0"))
ENGINE_MIN_STOP_PCT = float(os.getenv("ENGINE_MIN_STOP_PCT", "0.004"))
ENGINE_MIN_TP_PCT = float(os.getenv("ENGINE_MIN_TP_PCT", "0.010"))
ENGINE_SWING_THRESHOLD_1H = float(os.getenv("ENGINE_SWING_THRESHOLD_1H", "0.60"))
ENGINE_SWING_THRESHOLD_4H = float(os.getenv("ENGINE_SWING_THRESHOLD_4H", "0.65"))
ENGINE_DEBUG = os.getenv("ENGINE_DEBUG", "true").lower() == "true"

AGENT_POLL_INTERVAL_SECONDS = int(os.getenv(
    "AGENT_POLL_INTERVAL_SECONDS",
    str(max(5, int(os.getenv("POLL_INTERVAL_SECONDS", str(CONFIG_POLL_INTERVAL_SECONDS)))))
))

PERP_FEED_INTERVAL_SECONDS = int(os.getenv("PERP_FEED_INTERVAL_SECONDS", "60"))
SENTIMENT_FEED_INTERVAL_SECONDS = int(os.getenv("SENTIMENT_FEED_INTERVAL_SECONDS", "45"))

# ── Coin list ──────────────────────────────────────────────────────────────────

_raw = os.getenv("TRACKED_COINS", "SOL")
TRACKED_COINS: list[str] = list(dict.fromkeys(
    c.strip().upper() for c in _raw.split(",") if c.strip()
))
if "SOL" not in TRACKED_COINS:
    TRACKED_COINS.insert(0, "SOL")

WARMUP_BARS: Dict[str, int] = {
    "SOL": 40, "ETH": 40, "BTC": 40,
    "JUP": 30, "JTO": 30, "WIF": 25, "PYTH": 30,
}
DEFAULT_WARMUP = 30

ERROR_NOTIFY_EVERY = int(os.getenv("ERROR_NOTIFY_EVERY", "3"))
SIGNAL_DETECTION_ALERTS = os.getenv("SIGNAL_DETECTION_ALERTS", "0") == "1"
REJECTED_SIGNAL_COOLDOWN_SEC = int(os.getenv("REJECTED_SIGNAL_COOLDOWN_SEC", "60"))
INTRADAY_DUPLICATE_PCT = float(os.getenv("INTRADAY_DUPLICATE_PCT", "0.010"))

SWING_ENABLED = os.getenv("SWING_ENABLED", "1") == "1"
_raw_swing_tfs = os.getenv("SWING_TIMEFRAMES", "1h,4h")
SWING_TIMEFRAMES: list[str] = [
    tf.strip().lower() for tf in _raw_swing_tfs.split(",")
    if tf.strip().lower() in {"1h", "4h"}
]
if SWING_ENABLED and not SWING_TIMEFRAMES:
    SWING_TIMEFRAMES = ["1h", "4h"]
SWING_SIGNAL_DETECTION_ALERTS = os.getenv("SWING_SIGNAL_DETECTION_ALERTS", "0") == "1"
SWING_REJECTED_SIGNAL_COOLDOWN_SEC = int(
    os.getenv("SWING_REJECTED_SIGNAL_COOLDOWN_SEC", str(REJECTED_SIGNAL_COOLDOWN_SEC))
)
SWING_DUPLICATE_PCT = float(os.getenv("SWING_DUPLICATE_PCT", "0.006"))

RECAP_TRACK_FILE = os.getenv("RECAP_TRACK_FILE", "last_recap.txt")
RECAP_NO_FETCH = os.getenv("RECAP_NO_FETCH", "1") == "1"
RECAP_CHART = os.getenv("RECAP_CHART", "winrate_report.png")
RECAP_TIMEOUT_SECONDS = int(os.getenv("RECAP_TIMEOUT_SECONDS", "300"))
RECAP_STARTING_BALANCE = float(os.getenv("STARTING_BALANCE", "1000.0"))

_recap_running = threading.Event()


# ── Helpers ────────────────────────────────────────────────────────────────────

def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def notify(text: str):
    try:
        send_telegram_message(text)
    except Exception as e:
        print(f"[NOTIFY ERROR] {e}")


def validate_env() -> bool:
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


def _mode_tag() -> str:
    return "🔴 LIVE"


def _r_color(r: float) -> str:
    if r >= 2.0:
        return "🟢🟢"
    if r >= 1.0:
        return "🟢"
    if r >= 0:
        return "🟡"
    if r >= -0.5:
        return "🟠"
    return "🔴"


# ── Notification formatters ────────────────────────────────────────────────────

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
    rr = abs((tp - price) / (price - sl)) if price != sl else 0.0

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
    funding_txt = "Longs crowded" if funding > 0 else "Shorts crowded" if funding < 0 else "Balanced"

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
        f"{emoji} *{coin} {side} SIGNAL DETECTED* {arrow}\n\n"
        f"🎯 Setup: *{setup}*\n"
        f"🧭 Regime: `{regime}`\n"
        f"⏱ TF: `{tf_label}`\n\n"
        f"💰 Entry: `{price:.5g}`\n"
        f"🛑 SL: `{sl:.5g}`\n"
        f"🎯 TP: `{tp:.5g}`\n"
        f"📊 R/R: `{rr:.2f}R`\n\n"
        f"⚡ Strength: `{strength}%`\n"
        f"📈 Funding: {funding_txt}\n"
        f"📦 OI: `{oi}`"
        f"{flow_section}\n\n"
        f"🧠 _Signal detected — execution pending_"
    )


def format_fill_message(
    coin: str,
    signal: Signal,
    fill_price: float,
    fill_slippage_bps: float,
    fill_ratio: float,
    position_id: str,
) -> str:
    side = signal.side
    emoji = "🟢" if side == "LONG" else "🔴"
    arrow = "📈" if side == "LONG" else "📉"

    sl = float(signal.stop_price)
    tp = float(signal.tp_price)
    rr = abs((tp - fill_price) / (fill_price - sl)) if fill_price != sl else 0.0
    stop_dist_pct = abs(fill_price - sl) / fill_price * 100 if fill_price > 0 else 0.0

    meta = signal.meta or {}
    session = meta.get("session", "unknown")
    score = float(meta.get("total_score", signal.confidence))
    setup = meta.get("setup_family", meta.get("regime_local", "setup")).replace("_", " ")
    htf = meta.get("regime_htf_1h", "")
    macro = meta.get("regime_macro_4h", "")
    vol = meta.get("vol_state", "")

    partial_line = f"\n⚠️ Partial fill: `{fill_ratio * 100:.0f}%`" if fill_ratio < 0.99 else ""

    return (
        f"{emoji} *{coin} {side} FILLED* {arrow}  {_mode_tag()}\n\n"
        f"💰 Fill price: `{fill_price:.5g}`{partial_line}\n"
        f"🛑 Stop: `{sl:.5g}`  `({stop_dist_pct:.2f}% risk)`\n"
        f"🎯 TP: `{tp:.5g}`\n"
        f"📊 R/R: `{rr:.2f}R`\n\n"
        f"```\n"
        f"  Setup   : {setup}\n"
        f"  HTF     : {htf}  |  Macro: {macro}\n"
        f"  Vol     : {vol}\n"
        f"  Score   : {score:.2f}\n"
        f"  Session : {session}\n"
        f"  Slip    : {fill_slippage_bps:.1f}bps\n"
        f"```\n"
        f"🆔 `{position_id}`\n"
        f"🕒 `{utc_now()}`"
    )


def format_partial_tp_message(
    coin: str,
    side: str,
    entry_price: float,
    exit_price: float,
    size_closed_usd: float,
    size_remaining_usd: float,
    r_captured: float,
    pnl_usd: float,
    new_stop: float,
    position_id: str,
) -> str:
    emoji = "🟢" if side == "LONG" else "🔴"
    r_icon = _r_color(r_captured)

    return (
        f"✂️ *{coin} {side} PARTIAL TP* {emoji}  {_mode_tag()}\n\n"
        f"💰 Exit price: `{exit_price:.5g}`\n"
        f"📥 Entry: `{entry_price:.5g}`\n\n"
        f"```\n"
        f"  R captured  : {r_captured:+.2f}R  {r_icon}\n"
        f"  P&L         : ${pnl_usd:+.2f}\n"
        f"  Closed      : ${size_closed_usd:.0f}\n"
        f"  Remaining   : ${size_remaining_usd:.0f}\n"
        f"  New stop    : {new_stop:.5g}\n"
        f"```\n"
        f"🆔 `{position_id}`\n"
        f"🕒 `{utc_now()}`"
    )


def format_stop_message(
    coin: str,
    side: str,
    entry_price: float,
    exit_price: float,
    size_usd: float,
    r_final: float,
    pnl_usd: float,
    slip_bps: float,
    position_id: str,
    reason: str = "stop_hit",
) -> str:
    emoji = "🔴" if side == "LONG" else "🟢"
    r_icon = _r_color(r_final)
    reason_label = "Stop hit" if "full" in reason else "Trailing stop"

    return (
        f"🛑 *{coin} {side} STOPPED* {emoji}  {_mode_tag()}\n\n"
        f"💰 Exit: `{exit_price:.5g}`\n"
        f"📥 Entry: `{entry_price:.5g}`\n\n"
        f"```\n"
        f"  Result  : {reason_label}\n"
        f"  R       : {r_final:+.2f}R  {r_icon}\n"
        f"  P&L     : ${pnl_usd:+.2f}\n"
        f"  Size    : ${size_usd:.0f}\n"
        f"  Slip    : {slip_bps:.1f}bps\n"
        f"```\n"
        f"🆔 `{position_id}`\n"
        f"🕒 `{utc_now()}`"
    )


def format_tp_message(
    coin: str,
    side: str,
    entry_price: float,
    exit_price: float,
    size_usd: float,
    r_final: float,
    pnl_usd: float,
    duration_min: float,
    position_id: str,
) -> str:
    emoji = "🟢" if side == "LONG" else "🔴"
    r_icon = _r_color(r_final)

    hrs = int(duration_min // 60)
    mins = int(duration_min % 60)
    duration_str = f"{hrs}h {mins}m" if hrs > 0 else f"{mins}m"

    return (
        f"🎯 *{coin} {side} TARGET HIT* {emoji}  {_mode_tag()}\n\n"
        f"💰 Exit: `{exit_price:.5g}`\n"
        f"📥 Entry: `{entry_price:.5g}`\n\n"
        f"```\n"
        f"  R       : {r_final:+.2f}R  {r_icon}\n"
        f"  P&L     : ${pnl_usd:+.2f}\n"
        f"  Size    : ${size_usd:.0f}\n"
        f"  Hold    : {duration_str}\n"
        f"```\n"
        f"🆔 `{position_id}`\n"
        f"🕒 `{utc_now()}`"
    )


def format_stale_close_message(
    coin: str,
    side: str,
    entry_price: float,
    exit_price: float,
    size_usd: float,
    r_final: float,
    pnl_usd: float,
    position_id: str,
    reason: str = "stale_exit",
) -> str:
    r_icon = _r_color(r_final)
    return (
        f"⏹ *{coin} {side} CLOSED* `{reason}`  {_mode_tag()}\n\n"
        f"💰 Exit: `{exit_price:.5g}`\n"
        f"📥 Entry: `{entry_price:.5g}`\n\n"
        f"```\n"
        f"  R       : {r_final:+.2f}R  {r_icon}\n"
        f"  P&L     : ${pnl_usd:+.2f}\n"
        f"  Size    : ${size_usd:.0f}\n"
        f"```\n"
        f"🆔 `{position_id}`\n"
        f"🕒 `{utc_now()}`"
    )


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
    if _recap_running.is_set():
        return False
    now = datetime.now(timezone.utc)
    last = read_last_recap_time()
    if last is None:
        return os.path.exists("signals.csv") and os.path.getsize("signals.csv") > 0
    return now.date() > last.date()


def extract_recap_summary(stdout: str) -> str:
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
    _recap_running.set()
    try:
        print("[RECAP] Running automatic daily recap...")
        notify("📊 *Running daily performance recap...*")
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
            return
        notify(build_trader_grade_recap(extract_recap_summary(proc.stdout)))
        print("[RECAP] Daily recap sent.")
        try:
            run_trades_recap(notify_fn=notify, starting_balance=RECAP_STARTING_BALANCE)
            print("[RECAP] Trade P&L recap sent.")
        except Exception as te:
            print(f"[RECAP] Trade P&L recap failed: {te}")
    except subprocess.TimeoutExpired:
        notify(
            "❌ *Cuan Sniffer Daily Recap Timed Out*\n\n"
            f"🕒 Time: `{utc_now()}`\n"
            f"Timeout: `{RECAP_TIMEOUT_SECONDS}s`"
        )
    except Exception as e:
        notify(
            "❌ *Cuan Sniffer Recap Exception*\n\n"
            f"🕒 Time: `{utc_now()}`\n"
            f"`{str(e)[:250]}`"
        )
    finally:
        _recap_running.clear()


def trigger_recap():
    if _recap_running.is_set():
        return
    t = threading.Thread(target=_run_recap_worker, daemon=True)
    t.start()


# ── Coin state ─────────────────────────────────────────────────────────────────

def _build_signal_engine() -> AdaptiveSignalEngine:
    return AdaptiveSignalEngine(
        score_threshold=ENGINE_SCORE_THRESHOLD,
        atr_stop_mult=ENGINE_ATR_STOP_MULT,
        atr_tp_mult=ENGINE_ATR_TP_MULT,
        min_stop_pct=ENGINE_MIN_STOP_PCT,
        min_tp_pct=ENGINE_MIN_TP_PCT,
        swing_threshold_1h=ENGINE_SWING_THRESHOLD_1H,
        swing_threshold_4h=ENGINE_SWING_THRESHOLD_4H,
        debug=ENGINE_DEBUG,
    )


class CoinState:
    def __init__(
        self,
        coin: str,
        flow_ctx: Optional[FlowContext] = None,
        engine: Optional[AdaptiveSignalEngine] = None,
    ):
        self.coin = coin
        self.flow_ctx = flow_ctx
        self.perp_feed = PerpDataFeed(coin=coin, interval="15m", max_candles=400)
        self.sent_feed = PerpSentimentFeed(coin=coin)
        self.engine = engine or _build_signal_engine()
        self.last_rejected_setup: Dict[str, float] = {}
        self.alert_mgr = AlertManager(
            min_flow_strength=0.7,
            min_imbalance_30m=0.7,
            min_funding_mag=0.01,
            cooldown_seconds=900,
        )
        tracked_tfs = ["15m"] + SWING_TIMEFRAMES
        self.last_signal: Dict[str, Dict[str, Optional[float]]] = {
            tf: {"side": None, "price": None} for tf in tracked_tfs
        }
        self._error_count: int = 0

    def start_feeds(self):
        self.perp_feed.start(interval_sec=PERP_FEED_INTERVAL_SECONDS)
        self.sent_feed.start(interval_sec=SENTIMENT_FEED_INTERVAL_SECONDS)
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

    def make_setup_fingerprint(self, signal: Signal) -> str:
        meta = signal.meta or {}
        return "|".join([
            str(meta.get("timeframe", "15m")),
            signal.coin,
            str(signal.side),
            f"{float(signal.entry_price):.8f}",
            f"{float(signal.stop_price):.8f}",
            f"{float(signal.tp_price):.8f}",
            str(meta.get("setup_family", meta.get("regime_local", ""))),
            str(meta.get("session", "")),
            str(meta.get("regime_htf_1h", "")),
            str(meta.get("regime_macro_4h", "")),
        ])

    def recently_rejected_setup(self, fingerprint: str, cooldown_sec: int) -> bool:
        return (time.time() - self.last_rejected_setup.get(fingerprint, 0.0)) < cooldown_sec

    def mark_rejected_setup(self, fingerprint: str):
        self.last_rejected_setup[fingerprint] = time.time()

    def on_error(self) -> bool:
        self._error_count += 1
        return self._error_count == 1 or self._error_count % ERROR_NOTIFY_EVERY == 0

    def on_success(self):
        if self._error_count > 0:
            print(f"[{self.coin}] recovered after {self._error_count} consecutive errors")
        self._error_count = 0


# ── Core scan loop ─────────────────────────────────────────────────────────────

def process_coin(
    state: CoinState,
    executor: Executor,
) -> tuple[bool, bool]:
    coin = state.coin
    warmup = WARMUP_BARS.get(coin, DEFAULT_WARMUP)

    df = state.perp_feed.get_ohlcv_df()
    if df is None or len(df) < warmup:
        fetch_source, fetch_sec = state.perp_feed.get_last_fetch_status()
        if fetch_source == "failed":
            print(
                f"[{coin}] skipped — candle fetch failed (no cache)"
                f" | fetch_time={fetch_sec:.2f}s"
            )
            return False, True
        print(f"[{coin}] warming up... {0 if df is None else len(df)}/{warmup}")
        return False, False

    flow_snapshot = state.get_flow_snapshot()
    sent = state.sent_feed.get_snapshot()
    funding_rate = float(getattr(sent, "funding_rate", 0.0))
    open_interest = int(getattr(sent, "open_interest", 0) or 0)
    bias = float(getattr(sent, "bias", 0.0))

    if coin == "SOL" and flow_snapshot:
        state.alert_mgr.maybe_alert_large_flow(flow_snapshot)
    state.alert_mgr.maybe_alert_funding_extreme(funding_rate)

    any_traded = False

    sig_15m = state.engine.generate_signal(df, flow_snapshot, sent, coin=coin)
    if sig_15m is not None:
        any_traded = _execute_signal(
            state=state,
            executor=executor,
            signal=sig_15m,
            flow_snapshot=flow_snapshot,
            sentiment=sent,
            funding_rate=funding_rate,
            open_interest=open_interest,
            long_short_bias=bias,
            tf_label="15m",
            duplicate_threshold=INTRADAY_DUPLICATE_PCT,
            rejected_cooldown_sec=REJECTED_SIGNAL_COOLDOWN_SEC,
            detection_alerts=SIGNAL_DETECTION_ALERTS,
        ) or any_traded

    if SWING_ENABLED:
        for swing_tf in SWING_TIMEFRAMES:
            sig_swing = state.engine.generate_swing_signal(
                df,
                flow_snapshot,
                sent,
                swing_tf=swing_tf,
                coin=coin,
            )
            if sig_swing is None:
                continue

            any_traded = _execute_signal(
                state=state,
                executor=executor,
                signal=sig_swing,
                flow_snapshot=flow_snapshot,
                sentiment=sent,
                funding_rate=funding_rate,
                open_interest=open_interest,
                long_short_bias=bias,
                tf_label=swing_tf,
                duplicate_threshold=SWING_DUPLICATE_PCT,
                rejected_cooldown_sec=SWING_REJECTED_SIGNAL_COOLDOWN_SEC,
                detection_alerts=SWING_SIGNAL_DETECTION_ALERTS,
            ) or any_traded

    return any_traded, False


def _execute_signal(
    state: CoinState,
    executor: Executor,
    signal: Signal,
    flow_snapshot: Dict[str, Any],
    sentiment,
    funding_rate: float,
    open_interest: int,
    long_short_bias: float,
    tf_label: str,
    duplicate_threshold: float,
    rejected_cooldown_sec: int,
    detection_alerts: bool,
) -> bool:
    coin = state.coin
    meta = signal.meta if signal.meta is not None else {}
    signal.meta = meta
    meta["timeframe"] = tf_label
    meta["execution_track"] = "swing" if tf_label in {"1h", "4h"} else "intraday"

    theoretical_price = float(signal.entry_price)
    if state.is_duplicate(tf_label, signal.side, theoretical_price, duplicate_threshold):
        return False

    # Short-circuit before executor if coin already has an open position.
    # Avoids polluting missed_signals with repeated pending_or_open_position entries
    # on the same symbol — the executor would reject anyway, this just skips the round-trip.
    if executor._has_open_position_for_coin(coin):
        return False

    fingerprint = state.make_setup_fingerprint(signal)
    if state.recently_rejected_setup(fingerprint, cooldown_sec=rejected_cooldown_sec):
        print(f"[{coin}] {tf_label} identical rejected setup still cooling down")
        return False

    sig_id = int(time.time_ns() // 1_000_000)
    signal.meta["signal_id"] = sig_id

    if detection_alerts:
        notify(format_signal_message(coin, signal, flow_snapshot, sentiment, tf_label))

    print(f"[{coin}] {tf_label} signal detected")

    executor_result_label = "rejected"
    reject_reason = ""
    position_id = ""
    fill_price = 0.0
    fill_slippage_bps = 0.0
    fill_ratio = 0.0

    try:
        exec_result = executor.on_signal(signal, sig_id=sig_id)

        if exec_result.traded:
            executor_result_label = "traded"
            position_id = exec_result.position_id
            fill_price = exec_result.fill_price
            fill_slippage_bps = exec_result.fill_slippage_bps
            fill_ratio = exec_result.fill_ratio

            notify(format_fill_message(
                coin=coin,
                signal=signal,
                fill_price=fill_price,
                fill_slippage_bps=fill_slippage_bps,
                fill_ratio=fill_ratio,
                position_id=position_id,
            ))

            print(f"[{coin}] {tf_label} live trade opened — {position_id}")
            state.mark_signal(tf_label, signal.side, float(fill_price))
        else:
            executor_result_label = "rejected"
            reject_reason = exec_result.reason or "unknown_rejection"
            print(f"[{coin}] {tf_label} signal rejected — {reject_reason}")
            state.mark_rejected_setup(fingerprint)

    except Exception as e:
        err_text = f"executor_exception: {str(e)[:180]}"
        print(f"[EXECUTOR ERROR] {coin} {tf_label}: {err_text}")
        executor_result_label = "error"
        reject_reason = err_text
        state.mark_rejected_setup(fingerprint)

    append_signal(
        coin=coin,
        sig_id=sig_id,
        signal=signal,
        flow_snapshot=flow_snapshot,
        funding_rate=funding_rate,
        open_interest=open_interest,
        long_short_bias=long_short_bias,
        executor_result=executor_result_label,
        reject_reason=reject_reason,
        position_id=position_id,
        fill_price=fill_price,
        fill_slippage_bps=fill_slippage_bps,
        fill_ratio=fill_ratio,
    )

    return executor_result_label == "traded"


def build_states(flow_ctx: FlowContext) -> tuple[Dict[str, CoinState], AdaptiveSignalEngine]:
    states: Dict[str, CoinState] = {}
    shared_engine = _build_signal_engine()
    for coin in TRACKED_COINS:
        ctx = flow_ctx if coin == "SOL" else None
        state = CoinState(coin, ctx, engine=shared_engine)
        state.start_feeds()
        states[coin] = state
    return states, shared_engine


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    if not validate_env():
        print("[AGENT] Invalid environment. Exiting.")
        return

    init_db()
    init_signal_log()

    flow_ctx = FlowContext(SessionLocal)
    states, shared_engine = build_states(flow_ctx)
    executor = Executor(notify_fn=notify, signal_engine=shared_engine)

    print("[AGENT] Live position monitor active (shared with executor)")

    notify(executor.boot_status_message())
    notify(
        f"🎯 Scanning: `{', '.join(TRACKED_COINS)}`\n"
        f"🕒 `{utc_now()}`\n"
        f"⚙️ Engine threshold: `{ENGINE_SCORE_THRESHOLD:.2f}`\n"
        f"📈 Swing: `{'on' if SWING_ENABLED else 'off'}`"
        f"{' (' + ','.join(SWING_TIMEFRAMES) + ')' if SWING_ENABLED else ''}\n"
        f"⏱ Agent loop: `{AGENT_POLL_INTERVAL_SECONDS}s`"
    )

    _last_all_skipped_notify_ts: float = 0.0
    _ALL_SKIPPED_NOTIFY_COOLDOWN_SEC = 600

    try:
        while True:
            cycle_started = time.time()
            any_signal = False
            skipped_symbols = 0

            for coin in TRACKED_COINS:
                try:
                    traded, skipped = process_coin(states[coin], executor)
                    if traded:
                        any_signal = True
                    if skipped:
                        skipped_symbols += 1
                    states[coin].on_success()
                except Exception as e:
                    err_msg = str(e)[:180]
                    count = states[coin]._error_count + 1
                    skipped_symbols += 1
                    print(f"[{coin} ERROR #{count}] {err_msg}")
                    if states[coin].on_error():
                        notify(
                            f"❌ *{coin} error* (x{states[coin]._error_count})\n"
                            f"{utc_now()}\n"
                            f"`{err_msg}`"
                        )

            if skipped_symbols == len(TRACKED_COINS):
                _now = time.time()
                if _now - _last_all_skipped_notify_ts >= _ALL_SKIPPED_NOTIFY_COOLDOWN_SEC:
                    _last_all_skipped_notify_ts = _now
                    notify(
                        f"⚠️ All {len(TRACKED_COINS)} symbols skipped this cycle — possible data outage\n{utc_now()}"
                    )

            try:
                executor.update()
            except Exception as e:
                print(f"[EXECUTOR UPDATE ERROR] {e}")
                notify(
                    f"❌ *Executor update error*\n"
                    f"{utc_now()}\n"
                    f"`{str(e)[:180]}`"
                )

            if should_run_recap():
                trigger_recap()

            cycle_sec = time.time() - cycle_started
            print(
                f"[AGENT] Cycle duration={cycle_sec:.2f}s"
                f" | skipped_symbols={skipped_symbols}/{len(TRACKED_COINS)}"
            )
            if not any_signal:
                print(f"[AGENT] No signals this cycle ({utc_now()})")

            time.sleep(AGENT_POLL_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        executor.shutdown()
        session_line = f"Live session | {executor.slippage_summary()}"

        notify(
            f"🛑 *Bot stopped cleanly*\n"
            f"{utc_now()}\n"
            f"{session_line}\n"
            f"📊 {executor.slippage_summary()}"
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

"""
Multi-coin signal agent — Cuan Sniffer (LIVE-ONLY).

Notification system:
  - Entry fill notifications
  - Live exit notifications handled through executor/update path
  - Daily recap support retained

Live-only cleanup:
  - Live backend only
  - Removed any dependency on executor.paper
"""

import os
import sys
import time
import fcntl
import subprocess
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timezone

from config import POLL_INTERVAL_SECONDS as CONFIG_POLL_INTERVAL_SECONDS
from db import init_db, SessionLocal
from perp_data import PerpDataFeed
from flow_context import FlowContext
from perp_sentiment import PerpSentimentFeed
from signal_engine import AdaptiveSignalEngine, Signal, log_gate_reject
from notifier import send_telegram_message
from signal_log import init_signal_log, append_signal
from alerts import AlertManager
from executor import Executor
from trades_recap import run_trades_recap
from smc_live_log import init_smc_live_log
from live_data_guard import (
    LIVE_MAX_CANDLE_CACHE_AGE_SECONDS,
    LIVE_MAX_SENTIMENT_CACHE_AGE_SECONDS,
    api_backoff_status,
)

PROJECT_ROOT = Path(__file__).resolve().parent
ANALYZE_WINRATE_SCRIPT = PROJECT_ROOT / "tools" / "research" / "analyze_winrate.py"


# ── Runtime config ─────────────────────────────────────────────────────────────

ENGINE_SCORE_THRESHOLD = float(os.getenv("MIN_SIGNAL_SCORE", "0.72"))
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
MIN_1H_CANDLE_POLL_SECONDS = int(os.getenv("MIN_1H_CANDLE_POLL_SECONDS", "60"))
DATA_HEALTH_HEARTBEAT_SECONDS = int(os.getenv("DATA_HEALTH_HEARTBEAT_SECONDS", "60"))
DATA_HEALTH_STALE_SYMBOL_BLOCK_PCT = float(os.getenv("DATA_HEALTH_STALE_SYMBOL_BLOCK_PCT", "0.30"))
DATA_HEALTH_MIN_STALE_SYMBOLS = int(os.getenv("DATA_HEALTH_MIN_STALE_SYMBOLS", "3"))
DATA_HEALTH_BREAKER_LOG_SECONDS = int(os.getenv("DATA_HEALTH_BREAKER_LOG_SECONDS", "60"))
AGENT_LOCK_FILE = os.getenv("AGENT_LOCK_FILE", "agent.py.lock")
PAUSE_NEW_SIGNALS = os.getenv("PAUSE_NEW_SIGNALS", "false").lower() == "true"

# ── Coin list ──────────────────────────────────────────────────────────────────

_raw = os.getenv("TRACKED_COINS", "SOL")
TRACKED_COINS: list[str] = list(dict.fromkeys(
    c.strip().upper() for c in _raw.split(",") if c.strip()
))
if "SOL" not in TRACKED_COINS:
    TRACKED_COINS.insert(0, "SOL")

WARMUP_BARS: Dict[str, int] = {
    "SOL": 40, "ETH": 40, "BTC": 40, "BNB": 40,
    "JUP": 30, "JTO": 30, "WIF": 25, "PYTH": 30,
    "PENGU": 25, "SUI": 30, "NEAR": 30, "TAO": 30, "ZEC": 30,
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

class AgentLockUnavailable(RuntimeError):
    pass


@contextmanager
def agent_process_lock(lock_file: str = AGENT_LOCK_FILE):
    """
    Prevent two trading-agent processes from running at the same time.
    """
    lock_path = Path(lock_file)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.touch(exist_ok=True)

    lock_f = open(lock_path, "r+")
    locked = False
    try:
        try:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock_f.seek(0)
            owner = lock_f.read().strip()
            raise AgentLockUnavailable(
                f"another_agent_instance_running lock_file={lock_path}"
                f"{' owner=' + owner if owner else ''}"
            ) from None

        locked = True
        lock_f.seek(0)
        lock_f.truncate()
        lock_f.write(
            f"pid={os.getpid()} started_at={datetime.now(timezone.utc).isoformat()}\n"
        )
        lock_f.flush()
        os.fsync(lock_f.fileno())
        print(f"[AGENT_LOCK] acquired lock_file={lock_path} pid={os.getpid()}")
        yield
    finally:
        try:
            if locked:
                lock_f.seek(0)
                lock_f.truncate()
                lock_f.flush()
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
                print(f"[AGENT_LOCK] released lock_file={lock_path} pid={os.getpid()}")
        finally:
            lock_f.close()


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def notify(text: str) -> bool:
    try:
        return bool(send_telegram_message(text))
    except Exception as e:
        print(f"[NOTIFY ERROR] {e}")
        return False


def notify_async(text: str) -> None:
    try:
        threading.Thread(target=notify, args=(text,), daemon=True).start()
    except Exception as e:
        print(f"[NOTIFY_ASYNC] failed to start thread: {e}")


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


def validate_thresholds() -> bool:
    """
    Print a unified threshold summary at startup and warn if any confidence gate
    diverges from UNIVERSAL_MIN_CONFIDENCE. Returns True if all gates are aligned.
    """
    universal = float(os.getenv("UNIVERSAL_MIN_CONFIDENCE", "0.90"))

    # Confidence gates — all should equal UNIVERSAL_MIN_CONFIDENCE unless intentionally overridden
    conf_gates = {
        "SWING_MIN_CONFIDENCE":        float(os.getenv("SWING_MIN_CONFIDENCE", str(universal))),
        "SMC_4H_MIN_CONFIDENCE":       float(os.getenv("SMC_4H_MIN_CONFIDENCE", str(universal))),
        "WEAK_TREND_MIN_CONFIDENCE":   float(os.getenv("WEAK_TREND_MIN_CONFIDENCE", str(universal))),
        "MIN_SIGNAL_CONFIDENCE":       float(os.getenv("MIN_SIGNAL_CONFIDENCE", str(universal))),
    }

    score_thresholds = {
        "REGIME_SCORE_THRESHOLD_STRONG": float(os.getenv("REGIME_SCORE_THRESHOLD_STRONG", "0.64")),
        "REGIME_SCORE_THRESHOLD_WEAK":   float(os.getenv("REGIME_SCORE_THRESHOLD_WEAK",   "0.64")),
        "REGIME_SCORE_THRESHOLD_CHOP":   float(os.getenv("REGIME_SCORE_THRESHOLD_CHOP",   "0.64")),
    }

    rr_gates = {
        "MIN_STOP_REDESIGN_RR":       float(os.getenv("MIN_STOP_REDESIGN_RR", "1.60")),
        "MIN_EXECUTION_EFFECTIVE_RR": float(os.getenv("MIN_EXECUTION_EFFECTIVE_RR", "1.55")),
    }
    regime_tp_cap_r = float(os.getenv("REGIME_TP_CAP_R", "1.75"))

    feature_flags = {
        "SMC_ENABLE_4H_LIVE":      os.getenv("SMC_ENABLE_4H_LIVE",      "true"),
        "HARD_BLOCK_CONTINUATION": os.getenv("HARD_BLOCK_CONTINUATION", "false"),
        "HARD_BLOCK_CHOP":         os.getenv("HARD_BLOCK_CHOP",         "false"),
        "BLOCK_CONTINUATION_IN_WEAK_TREND": os.getenv("BLOCK_CONTINUATION_IN_WEAK_TREND", "true"),
        "BLOCK_REVERSAL_IN_WEAK_TREND":     os.getenv("BLOCK_REVERSAL_IN_WEAK_TREND",     "true"),
        "LIVE_ELIGIBILITY_MODEL":  os.getenv("LIVE_ELIGIBILITY_MODEL",  "v3"),
    }

    sep = "─" * 58
    print(f"\n[THRESHOLD SUMMARY] {sep}")
    print(f"  UNIVERSAL_MIN_CONFIDENCE  = {universal:.2f}  ← master confidence floor")

    print(f"  {sep}")
    print("  CONFIDENCE GATES (should all match UNIVERSAL_MIN_CONFIDENCE):")
    all_aligned = True
    for name, val in conf_gates.items():
        marker = ""
        if abs(val - universal) > 1e-9:
            marker = "  ← [THRESHOLD WARNING] diverges from universal"
            all_aligned = False
        print(f"    {name:<36} = {val:.2f}{marker}")

    print(f"  {sep}")
    print("  SCORE PRE-FILTERS (permissive — should all be ~0.64):")
    for name, val in score_thresholds.items():
        marker = ""
        if abs(val - 0.64) > 0.001:
            marker = "  ← [THRESHOLD WARNING] not unified at 0.64"
            all_aligned = False
        print(f"    {name:<36} = {val:.2f}{marker}")

    print(f"  {sep}")
    print("  RR GATES (simplified — should be 1.60 / 1.55):")
    expected_rr = {"MIN_STOP_REDESIGN_RR": 1.60, "MIN_EXECUTION_EFFECTIVE_RR": 1.55}
    for name, val in rr_gates.items():
        marker = ""
        if abs(val - expected_rr[name]) > 0.001:
            marker = f"  ← [THRESHOLD WARNING] expected {expected_rr[name]:.2f}"
            all_aligned = False
        print(f"    {name:<36} = {val:.2f}{marker}")

    print(f"  {sep}")
    print("  TP CAPS:")
    cap_marker = ""
    if regime_tp_cap_r <= rr_gates["MIN_EXECUTION_EFFECTIVE_RR"]:
        cap_marker = (
            "  ← [THRESHOLD WARNING] must be > "
            f"MIN_EXECUTION_EFFECTIVE_RR ({rr_gates['MIN_EXECUTION_EFFECTIVE_RR']:.2f})"
        )
        all_aligned = False
    print(f"    {'REGIME_TP_CAP_R':<36} = {regime_tp_cap_r:.2f}{cap_marker}")

    print(f"  {sep}")
    print("  FEATURE FLAGS:")
    for name, val in feature_flags.items():
        marker = ""
        if name == "LIVE_ELIGIBILITY_MODEL" and str(val).strip().lower() not in {"v1", "v3", "v3c"}:
            marker = "  ← [THRESHOLD WARNING] supported values: v1, v3, v3c"
            all_aligned = False
        print(f"    {name:<36} = {val}{marker}")

    print(f"[THRESHOLD SUMMARY] {sep}\n")

    if not all_aligned:
        print(
            "[THRESHOLD WARNING] One or more thresholds diverge from the unified framework.\n"
            "  Review .env and ensure intentional overrides are documented.\n"
        )

    return all_aligned


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


def _fmt_px(value: float) -> str:
    return f"{float(value):.5g}"


def _fmt_money(value: float) -> str:
    return f"${float(value):+.2f}"


def _fmt_duration(minutes: float) -> str:
    hrs = int(minutes // 60)
    mins = int(minutes % 60)
    return f"{hrs}h {mins}m" if hrs > 0 else f"{mins}m"


def _signal_context(signal: Signal) -> tuple[str, str, str, str, float]:
    meta = signal.meta or {}
    setup = str(meta.get("setup_family", meta.get("regime_local", "setup"))).replace("_", " ")
    session = str(meta.get("session", "unknown") or "unknown")
    tf = str(meta.get("timeframe", "") or "")
    market = str(meta.get("market_regime", "") or "")
    htf = str(meta.get("regime_htf_1h", "") or "")
    macro = str(meta.get("regime_macro_4h", "") or "")
    regime = " / ".join(x for x in (market, htf, macro) if x) or str(signal.regime)
    score = float(meta.get("total_score", signal.confidence) or 0.0)
    return setup, session, tf, regime, score


# ── Notification formatters ────────────────────────────────────────────────────

def format_signal_message(
    coin: str,
    signal: Signal,
    flow_snapshot: Dict[str, Any],
    sentiment,
    tf_label: str = "1h",
) -> str:
    side = signal.side
    emoji = "🟢" if side == "LONG" else "🔴"

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

    funding = float(getattr(sentiment, "funding_rate", 0.0) or 0.0)
    funding_txt = "balanced"
    if funding > 0:
        funding_txt = "longs pay"
    elif funding < 0:
        funding_txt = "shorts pay"
    setup, session, meta_tf, regime, score = _signal_context(signal)
    tf = meta_tf or tf_label

    return (
        f"{emoji} *Signal* `{coin} {side}`  `{tf}`  {_mode_tag()}\n"
        f"`{setup}` | conf `{signal.confidence:.2f}` | score `{score:.2f}` | RR `{rr:.2f}`\n\n"
        f"Entry `{_fmt_px(price)}`  Stop `{_fmt_px(sl)}`  TP `{_fmt_px(tp)}`\n"
        f"Context `{session}` | `{regime}`\n"
        f"Flow `{flow_bias}` | Funding `{funding_txt}`\n\n"
        "_Execution pending_"
    )


def format_fill_message(
    coin: str,
    signal: Signal,
    fill_price: float,
    fill_slippage_bps: float,
    fill_ratio: float,
    position_id: str,
    size_usd: float = 0.0,
    risk_usd: float = 0.0,
    entry_fee_usd: float = 0.0,
    protection_status: str = "",
    stop_order_id: str = "",
    tp_order_id: str = "",
) -> str:
    side = signal.side
    emoji = "🟢" if side == "LONG" else "🔴"
    arrow = "📈" if side == "LONG" else "📉"

    sl = float(signal.stop_price)
    tp = float(signal.tp_price)
    rr = abs((tp - fill_price) / (fill_price - sl)) if fill_price != sl else 0.0
    stop_dist_pct = abs(fill_price - sl) / fill_price * 100 if fill_price > 0 else 0.0
    tp_dist_pct = abs(tp - fill_price) / fill_price * 100 if fill_price > 0 else 0.0

    meta = signal.meta or {}
    session = meta.get("session", "unknown")
    score = float(meta.get("total_score", signal.confidence))
    setup = meta.get("setup_family", meta.get("regime_local", "setup")).replace("_", " ")
    tf = meta.get("timeframe", "")

    partial_line = f"\n⚠️ Partial fill: `{fill_ratio * 100:.0f}%`" if fill_ratio < 0.99 else ""

    return (
        f"{emoji} *Opened* `{coin} {side}`  `{tf or '-'}`  {_mode_tag()}\n\n"
        f"Fill `{_fmt_px(fill_price)}`{partial_line}\n"
        f"Stop `{_fmt_px(sl)}` ({stop_dist_pct:.2f}%)  TP `{_fmt_px(tp)}` ({tp_dist_pct:.2f}%)\n"
        f"RR `{rr:.2f}` | Risk `${risk_usd:.2f}` | Size `${size_usd:.0f}`\n\n"
        f"```\n"
        f"setup   {setup}\n"
        f"session {session}\n"
        f"score   {score:.2f}\n"
        f"slip    {fill_slippage_bps:.1f}bps\n"
        f"fee     ${entry_fee_usd:.2f}\n"
        f"protect {protection_status or 'unknown'}\n"
        f"```\n"
        f"`{position_id}` | `{utc_now()}`"
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
        f"✂️ *Partial TP* `{coin} {side}`  {_mode_tag()}\n\n"
        f"Exit `{_fmt_px(exit_price)}` | Entry `{_fmt_px(entry_price)}`\n"
        f"Locked `{r_captured:+.2f}R` {_r_color(r_captured)} | P&L `{_fmt_money(pnl_usd)}`\n"
        f"Closed `${size_closed_usd:.0f}` | Runner `${size_remaining_usd:.0f}` | Stop `{_fmt_px(new_stop)}`\n\n"
        f"```\n"
        f"id {position_id}\n"
        f"time {utc_now()}\n"
        f"```\n"
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
        f"🛑 *Closed* `{coin} {side}`  {_mode_tag()}\n\n"
        f"{reason_label} | R `{r_final:+.2f}` {_r_color(r_final)} | P&L `{_fmt_money(pnl_usd)}`\n"
        f"Exit `{_fmt_px(exit_price)}` | Entry `{_fmt_px(entry_price)}` | Slip `{slip_bps:.1f}bps`\n\n"
        f"```\n"
        f"size ${size_usd:.0f}\n"
        f"id   {position_id}\n"
        f"time {utc_now()}\n"
        f"```\n"
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
        f"🎯 *Target Hit* `{coin} {side}`  {_mode_tag()}\n\n"
        f"R `{r_final:+.2f}` {_r_color(r_final)} | P&L `{_fmt_money(pnl_usd)}` | Hold `{duration_str}`\n"
        f"Exit `{_fmt_px(exit_price)}` | Entry `{_fmt_px(entry_price)}`\n\n"
        f"```\n"
        f"size ${size_usd:.0f}\n"
        f"id   {position_id}\n"
        f"time {utc_now()}\n"
        f"```\n"
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
        f"⏹ *Closed* `{coin} {side}`  `{reason}`  {_mode_tag()}\n\n"
        f"R `{r_final:+.2f}` {r_icon} | P&L `{_fmt_money(pnl_usd)}`\n"
        f"Exit `{_fmt_px(exit_price)}` | Entry `{_fmt_px(entry_price)}`\n\n"
        f"```\n"
        f"size ${size_usd:.0f}\n"
        f"id   {position_id}\n"
        f"time {utc_now()}\n"
        f"```\n"
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


def build_trader_grade_recap(summary_text: str, trade_text: str = "") -> str:
    summary = (summary_text or "No analyzer summary available.").strip()[:1300]
    trade = (trade_text or "").strip()[:2400]
    parts = [
        "📊 *Daily Recap*",
        f"`{utc_now()}` | chart `{RECAP_CHART}`",
        "",
        "*System*",
        "```text",
        summary,
        "```",
    ]
    if trade:
        parts.extend(["", trade])
    return "\n".join(parts)


def _run_recap_worker():
    _recap_running.set()
    try:
        print("[RECAP] Running automatic daily recap...")
        write_last_recap_time(datetime.now(timezone.utc))
        analyzer_args = [str(ANALYZE_WINRATE_SCRIPT), "--chart", RECAP_CHART]
        if RECAP_NO_FETCH:
            analyzer_args.append("--no-fetch")
        proc = subprocess.run(
            [sys.executable, *analyzer_args],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
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
        summary_text = extract_recap_summary(proc.stdout)
        trade_text = ""
        try:
            trade_text = run_trades_recap(notify_fn=None, starting_balance=RECAP_STARTING_BALANCE)
        except Exception as te:
            print(f"[RECAP] Trade P&L recap failed: {te}")
            trade_text = f"⚠️ Trade recap unavailable: `{str(te)[:160]}`"
        notify(build_trader_grade_recap(summary_text, trade_text))
        print("[RECAP] Daily recap sent.")
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

def _stable_jitter_seconds(value: str, max_jitter: int) -> float:
    max_jitter = max(0, int(max_jitter))
    if max_jitter <= 0:
        return 0.0
    return float(sum(ord(ch) for ch in value) % max_jitter)


def _log_presignal_reject(
    coin: str,
    reason: str,
    *,
    timeframe: str = "1h",
    metadata: str = "",
) -> None:
    log_gate_reject(
        symbol=coin,
        timeframe=timeframe,
        reject_reason=reason,
        metadata=metadata,
    )


def _age_label(age: Any) -> str:
    try:
        if age is None:
            return "unknown"
        return f"{float(age):.1f}s"
    except (TypeError, ValueError):
        return "unknown"


def summarize_data_health(states: Dict[str, "CoinState"]) -> Dict[str, Any]:
    total = len(states)
    stale_symbols: list[str] = []
    dead_feeds: list[str] = []
    ages: list[float] = []

    for coin, state in states.items():
        status = state.perp_feed.get_market_data_status(
            max_age_seconds=LIVE_MAX_CANDLE_CACHE_AGE_SECONDS
        )
        age = status.get("cache_age")
        try:
            if age is not None:
                ages.append(float(age))
        except (TypeError, ValueError):
            pass

        if not status.get("fresh", False):
            stale_symbols.append(coin)
        if status.get("poll_thread_alive") is False:
            dead_feeds.append(coin)

    stale_count = len(stale_symbols)
    stale_pct = (stale_count / total) if total else 0.0
    return {
        "total": total,
        "stale_count": stale_count,
        "stale_pct": stale_pct,
        "stale_symbols": stale_symbols,
        "dead_feeds": dead_feeds,
        "max_cache_age": max(ages) if ages else None,
    }


def data_health_breaker_active(summary: Dict[str, Any]) -> bool:
    total = int(summary.get("total", 0) or 0)
    if total <= 0:
        return False
    stale_count = int(summary.get("stale_count", 0) or 0)
    stale_pct = float(summary.get("stale_pct", 0.0) or 0.0)
    block_pct = max(0.0, min(1.0, DATA_HEALTH_STALE_SYMBOL_BLOCK_PCT))
    return (
        stale_count >= max(1, DATA_HEALTH_MIN_STALE_SYMBOLS)
        and stale_pct >= block_pct
    )


def format_data_health(summary: Dict[str, Any], breaker_active: bool) -> str:
    stale_symbols = summary.get("stale_symbols", []) or []
    dead_feeds = summary.get("dead_feeds", []) or []
    total = int(summary.get("total", 0) or 0)
    stale_count = int(summary.get("stale_count", 0) or 0)
    stale_pct = float(summary.get("stale_pct", 0.0) or 0.0)
    return (
        f"stale_symbols={stale_count}/{total}"
        f" stale_pct={stale_pct * 100.0:.1f}%"
        f" max_cache_age={_age_label(summary.get('max_cache_age'))}"
        f" dead_feeds={','.join(dead_feeds) if dead_feeds else 'none'}"
        f" breaker_active={str(bool(breaker_active)).lower()}"
        f" symbols={','.join(stale_symbols[:8]) if stale_symbols else 'none'}"
    )


def signal_generation_pause_reason(health_block: bool) -> str:
    if PAUSE_NEW_SIGNALS:
        return "operator_pause"
    if health_block:
        return "data_health_circuit_breaker"
    return ""


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
        self.perp_feed = PerpDataFeed(coin=coin, interval="1h", max_candles=400)
        self.sent_feed = PerpSentimentFeed(coin=coin)
        self.engine = engine or _build_signal_engine()
        self.last_rejected_setup: Dict[str, float] = {}
        self.alert_mgr = AlertManager(
            min_flow_strength=0.7,
            min_imbalance_30m=0.7,
            min_funding_mag=0.01,
            cooldown_seconds=900,
        )
        tracked_tfs = ["1h"] + SWING_TIMEFRAMES
        self.last_signal: Dict[str, Dict[str, Optional[float]]] = {
            tf: {"side": None, "price": None} for tf in tracked_tfs
        }
        self._error_count: int = 0

    def start_feeds(self):
        candle_interval = max(PERP_FEED_INTERVAL_SECONDS, MIN_1H_CANDLE_POLL_SECONDS)
        jitter_cap = min(candle_interval, 30)
        candle_jitter = _stable_jitter_seconds(self.coin, jitter_cap)
        self.perp_feed.start(
            interval_sec=candle_interval,
            initial_delay_sec=candle_jitter,
        )
        self.sent_feed.start(interval_sec=SENTIMENT_FEED_INTERVAL_SECONDS)
        print(
            f"[{self.coin}] feeds started "
            f"candle_interval={candle_interval}s candle_jitter={candle_jitter:.1f}s"
        )

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
            str(meta.get("timeframe", "1h")),
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

    backoff_active, backoff_remaining, backoff_reason = api_backoff_status()
    if backoff_active:
        metadata = (
            f"remaining={backoff_remaining:.1f}s;"
            f"reason={backoff_reason}"
        )
        print(
            f"[{coin}] stale_data_skip reason=api_backoff_active "
            f"api_backoff_remaining={backoff_remaining:.1f}s"
        )
        _log_presignal_reject(
            coin,
            "api_backoff_active",
            metadata=metadata,
        )
        return False, True

    market_status = state.perp_feed.get_market_data_status(
        max_age_seconds=LIVE_MAX_CANDLE_CACHE_AGE_SECONDS
    )
    cache_age = market_status.get("cache_age")
    cache_age_label = "unknown" if cache_age is None else f"{float(cache_age):.1f}s"
    print(
        f"[{coin}] candle_status data_source={market_status.get('data_source')} "
        f"cache_age={cache_age_label} stale_skip={not market_status.get('fresh', False)}"
    )
    if not market_status.get("fresh", False):
        metadata = (
            f"data_source={market_status.get('data_source')};"
            f"cache_age={cache_age_label};"
            f"max_age={LIVE_MAX_CANDLE_CACHE_AGE_SECONDS:.1f}s;"
            f"fetch_source={market_status.get('fetch_source')};"
            f"last_error={market_status.get('last_error') or ''}"
        )
        print(
            f"[{coin}] stale_data_skip reason=stale_candle_cache "
            f"data_source={market_status.get('data_source')} "
            f"cache_age={cache_age_label} "
            f"max_age={LIVE_MAX_CANDLE_CACHE_AGE_SECONDS:.1f}s"
        )
        _log_presignal_reject(
            coin,
            "stale_candle_cache",
            metadata=metadata,
        )
        return False, True

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
    sent = state.sent_feed.get_snapshot(
        max_age_sec=LIVE_MAX_SENTIMENT_CACHE_AGE_SECONDS
    )
    if getattr(sent, "stale_neutralized", False):
        sent_age = getattr(sent, "cache_age_sec", None)
        sent_age_label = "unknown" if sent_age is None else f"{float(sent_age):.1f}s"
        metadata = (
            f"data_source={getattr(sent, 'data_source', 'neutralized')};"
            f"cache_age={sent_age_label};"
            f"max_age={LIVE_MAX_SENTIMENT_CACHE_AGE_SECONDS:.1f}s"
        )
        print(
            f"[{coin}] sentiment_stale_neutralized "
            f"data_source={getattr(sent, 'data_source', 'neutralized')} "
            f"cache_age={sent_age_label}"
        )
        _log_presignal_reject(
            coin,
            "sentiment_stale_neutralized",
            metadata=metadata,
        )
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
            tf_label="1h",
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
        notify_async(format_signal_message(coin, signal, flow_snapshot, sentiment, tf_label))

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

            notify_async(format_fill_message(
                coin=coin,
                signal=signal,
                fill_price=fill_price,
                fill_slippage_bps=fill_slippage_bps,
                fill_ratio=fill_ratio,
                position_id=position_id,
                size_usd=getattr(exec_result, "size_usd", 0.0),
                risk_usd=getattr(exec_result, "risk_usd", 0.0),
                entry_fee_usd=getattr(exec_result, "entry_fee_usd", 0.0),
                protection_status=getattr(exec_result, "protection_status", ""),
                stop_order_id=getattr(exec_result, "stop_order_id", ""),
                tp_order_id=getattr(exec_result, "tp_order_id", ""),
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


def build_states(session_factory) -> tuple[Dict[str, CoinState], AdaptiveSignalEngine]:
    """
    Build per-coin CoinState objects, each with its own FlowContext so that
    on-chain wallet tracking is scoped to the correct coin's FlowEvents.

    SOL  → FlowContext(coin="SOL")  — native SOL flow from tracked wallets
    JTO/WIF/PENGU → FlowContext(coin=X) — SPL token flow from same wallets
    HYPE/TAO/NEAR/SUI → no flow context (not Solana SPL tokens tracked here)
    """
    _FLOW_TRACKED_COINS = {"SOL", "JTO", "WIF", "PENGU"}

    states: Dict[str, CoinState] = {}
    shared_engine = _build_signal_engine()
    for coin in TRACKED_COINS:
        ctx = FlowContext(session_factory, coin=coin) if coin in _FLOW_TRACKED_COINS else None
        state = CoinState(coin, ctx, engine=shared_engine)
        state.start_feeds()
        states[coin] = state
        print(f"[BUILD] {coin}: flow_ctx={'enabled (' + coin + ')' if ctx else 'disabled'}")
    return states, shared_engine


# ── Entry point ────────────────────────────────────────────────────────────────

def run_agent():
    if not validate_env():
        print("[AGENT] Invalid environment. Exiting.")
        return

    validate_thresholds()

    init_db()
    init_signal_log()
    init_smc_live_log()

    states, shared_engine = build_states(SessionLocal)
    executor = Executor(notify_fn=notify_async, signal_engine=shared_engine)

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
    _last_data_health_heartbeat_ts: float = 0.0
    _last_data_health_block_log_ts: float = 0.0

    try:
        while True:
            cycle_started = time.time()
            any_signal = False
            skipped_symbols = 0

            health_summary = summarize_data_health(states)
            health_block = data_health_breaker_active(health_summary)
            _now = time.time()
            health_log_due = (
                DATA_HEALTH_HEARTBEAT_SECONDS > 0
                and _now - _last_data_health_heartbeat_ts >= DATA_HEALTH_HEARTBEAT_SECONDS
            )
            block_log_due = (
                health_block
                and _now - _last_data_health_block_log_ts >= DATA_HEALTH_BREAKER_LOG_SECONDS
            )
            if health_log_due or block_log_due:
                health_line = format_data_health(health_summary, health_block)
                print(f"[DATA_HEALTH] {health_line}")
                _last_data_health_heartbeat_ts = _now
                if block_log_due:
                    _last_data_health_block_log_ts = _now
                    _log_presignal_reject(
                        "ALL",
                        "data_health_circuit_breaker_active",
                        metadata=health_line,
                    )

            pause_reason = signal_generation_pause_reason(health_block)
            if pause_reason == "operator_pause":
                skipped_symbols = len(TRACKED_COINS)
                print("[AGENT] New signal generation paused by PAUSE_NEW_SIGNALS=true")
            elif pause_reason == "data_health_circuit_breaker":
                skipped_symbols = len(TRACKED_COINS)
                print("[DATA_HEALTH] New signal generation paused by stale-symbol circuit breaker")
            else:
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

            if skipped_symbols == len(TRACKED_COINS) and not PAUSE_NEW_SIGNALS:
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


def main():
    try:
        with agent_process_lock():
            run_agent()
    except AgentLockUnavailable as e:
        print(f"[AGENT_LOCK] {e}")


if __name__ == "__main__":
    main()

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
"""

import os
import time
from math import fabs
from typing import Dict, Any, Optional
from datetime import datetime

from config import POLL_INTERVAL_SECONDS
from db import init_db, SessionLocal
from perp_data import PerpDataFeed
from flow_context import FlowContext
from perp_sentiment import PerpSentimentFeed
from signal_engine import AdaptiveSignalEngine, Signal
from notifier import send_telegram_message
from signal_log import init_signal_log, append_signal
from alerts import AlertManager


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
# during exchange outages (at 15s poll = 240 msgs/hr per coin without this)
ERROR_NOTIFY_EVERY = int(os.getenv("ERROR_NOTIFY_EVERY", "10"))


# ── Helpers ────────────────────────────────────────────────────────────────────
from datetime import datetime, timezone

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


# ── Signal message builder ─────────────────────────────────────────────────────

def format_signal_message(
    coin: str,
    signal: Signal,
    flow_snapshot: Dict[str, Any],
    sentiment,
    tf_label: str = "15m",
) -> str:
    side  = signal.side
    emoji = "🟢" if side == "LONG" else "🔴"
    arrow = "📈" if side == "LONG" else "📉"

    price = float(signal.entry_price)
    sl    = float(signal.stop_price)
    tp    = float(signal.tp_price)
    rr    = fabs((tp - price) / (price - sl)) if price != sl else 0.0

    whale_pressure = float(flow_snapshot.get("whale_pressure", 0.0)) if flow_snapshot else 0.0

    if whale_pressure   >  0.4:  flow_bias = "Aggressive buyers"
    elif whale_pressure >  0.15: flow_bias = "Buyers stepping in"
    elif whale_pressure < -0.4:  flow_bias = "Heavy selling"
    elif whale_pressure < -0.15: flow_bias = "Sellers active"
    else:                        flow_bias = "Neutral flow"

    funding = float(getattr(sentiment, "funding_rate", 0.0))
    oi      = int(getattr(sentiment,   "open_interest", 0) or 0)

    if funding > 0:   funding_txt = "Longs crowded"
    elif funding < 0: funding_txt = "Shorts crowded"
    else:             funding_txt = "Balanced"

    reasons = (signal.reason or "").lower()
    if "sweep"  in reasons: setup = "Liquidity Sweep"
    elif "choch" in reasons: setup = "Structure Flip"
    elif "bos"   in reasons: setup = "Breakout"
    else:                    setup = "Flow Setup"

    strength     = int(min(100, max(50, signal.confidence * 100)))
    regime       = signal.regime.replace("_", " ")
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
        self.coin     = coin
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

        # Per-coin error counter — throttles Telegram error notifications
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

    def on_error(self, e: Exception) -> bool:
        """
        Increment error counter.
        Returns True if Telegram should be notified this time.
        Notifies on error #1 and every ERROR_NOTIFY_EVERY after.
        """
        self._error_count += 1
        return (
            self._error_count == 1 or
            self._error_count % ERROR_NOTIFY_EVERY == 0
        )

    def on_success(self):
        """Reset error counter after a clean cycle."""
        if self._error_count > 0:
            print(f"[{self.coin}] recovered after {self._error_count} consecutive errors")
        self._error_count = 0


# ── Per-coin processing ────────────────────────────────────────────────────────

def process_coin(state: CoinState) -> bool:
    coin   = state.coin
    warmup = WARMUP_BARS.get(coin, DEFAULT_WARMUP)

    df = state.perp_feed.get_ohlcv_df()
    if df is None or len(df) < warmup:
        print(f"[{coin}] warming up... {0 if df is None else len(df)}/{warmup}")
        return False

    flow_snapshot = state.get_flow_snapshot()
    sent          = state.sent_feed.get_snapshot()
    funding_rate  = float(getattr(sent, "funding_rate", 0.0))
    open_interest = int(getattr(sent,   "open_interest", 0) or 0)
    bias          = float(getattr(sent, "bias", 0.0))

    if coin == "SOL" and flow_snapshot:
        state.alert_mgr.maybe_alert_large_flow(flow_snapshot)
    state.alert_mgr.maybe_alert_funding_extreme(funding_rate)

    sig = state.engine.generate_signal(df, flow_snapshot, sent, coin=coin)
    if sig is None:
        return False

    price = float(sig.entry_price)
    if state.is_duplicate("15m", sig.side, price, 0.010):
        return False

    append_signal(
        coin=coin,
        sig_id=int(time.time() * 1000),
        signal=sig,
        flow_snapshot=flow_snapshot,
        funding_rate=funding_rate,
        open_interest=open_interest,
        long_short_bias=bias,
    )

    msg = format_signal_message(coin, sig, flow_snapshot, sent, "15m")
    print(f"[{coin}] 15m signal fired")
    notify(msg)
    state.mark_signal("15m", sig.side, price)
    return True


# ── State construction ─────────────────────────────────────────────────────────

def build_states(flow_ctx: FlowContext) -> Dict[str, CoinState]:
    states: Dict[str, CoinState] = {}
    for coin in TRACKED_COINS:
        ctx   = flow_ctx if coin == "SOL" else None
        state = CoinState(coin, ctx)
        state.start_feeds()
        states[coin] = state
    return states


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # Validate env before starting anything
    validate_env()

    init_db()
    init_signal_log()

    flow_ctx = FlowContext(SessionLocal)
    states   = build_states(flow_ctx)

    notify(
        f"🚀 *Cuan Sniffer LIVE*\n"
        f"Coins: {', '.join(TRACKED_COINS)}\n"
        f"Time: {utc_now()}"
    )

    try:
        while True:
            any_signal = False

            for coin in TRACKED_COINS:
                try:
                    if process_coin(states[coin]):
                        any_signal = True
                    states[coin].on_success()

                except Exception as e:
                    err_msg = str(e)[:180]
                    count   = states[coin]._error_count + 1
                    print(f"[{coin} ERROR #{count}] {err_msg}")

                    if states[coin].on_error(e):
                        notify(
                            f"❌ *{coin} error* (x{states[coin]._error_count})\n"
                            f"{utc_now()}\n"
                            f"`{err_msg}`"
                        )

            if not any_signal:
                print(f"[AGENT] No signals this cycle ({utc_now()})")

            time.sleep(POLL_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        notify(f"🛑 *Bot stopped cleanly*\n{utc_now()}")
        print("[AGENT] Stopped.")

    except Exception as e:
        err_msg = str(e)[:200]
        print(f"[AGENT ERROR] {err_msg}")
        notify(
            f"🚨 *Agent fatal error*\n"
            f"{utc_now()}\n"
            f"`{err_msg}`"
        )
        # Sleep so the Telegram HTTP request completes before process exits
        time.sleep(5)


if __name__ == "__main__":
    main()
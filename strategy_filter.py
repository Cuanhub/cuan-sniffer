# strategy_filter.py
"""
Strategy kill-switch + per-coin performance filter.

Tracks rolling win rates per (coin) and per (coin, setup_family, htf_regime).
Auto-pauses combinations that fall below thresholds with configurable cooldown.

Integrates with RiskManager.check_signal() — called before any trade is opened.
"""

import os
from collections import deque
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Tuple


# ── Config ────────────────────────────────────────────────────────────────────
KILL_MIN_TRADES      = int(os.getenv("KILL_MIN_TRADES",     "5"))    # min trades before kill-switch activates
KILL_WIN_RATE        = float(os.getenv("KILL_WIN_RATE",     "0.30")) # pause if WR drops below 30%
KILL_ROLLING_N       = int(os.getenv("KILL_ROLLING_N",      "10"))   # rolling window size
KILL_PAUSE_HOURS     = int(os.getenv("KILL_PAUSE_HOURS",    "24"))   # hours to pause after kill

COIN_MIN_TRADES      = int(os.getenv("COIN_MIN_TRADES",     "5"))
COIN_KILL_WIN_RATE   = float(os.getenv("COIN_KILL_WIN_RATE", "0.25")) # stricter per-coin threshold
COIN_ROLLING_N       = int(os.getenv("COIN_ROLLING_N",      "10"))
COIN_PAUSE_HOURS     = int(os.getenv("COIN_PAUSE_HOURS",    "12"))


def _utc() -> datetime:
    return datetime.now(timezone.utc)


class StrategyFilter:
    """
    Two-tier adaptive filter:

    Tier 1 — Per-coin filter:
        Tracks rolling win rate per coin.
        If WR < COIN_KILL_WIN_RATE after COIN_MIN_TRADES, pause coin for COIN_PAUSE_HOURS.

    Tier 2 — Setup kill-switch:
        Tracks rolling win rate per (coin, setup_family, htf_regime).
        If WR < KILL_WIN_RATE after KILL_MIN_TRADES, pause that specific combination.
    """

    def __init__(self):
        # coin → deque of bools (True=win, False=loss)
        self._coin_results: Dict[str, deque] = {}
        self._coin_paused_until: Dict[str, datetime] = {}

        # (coin, setup_family, htf_regime) → deque of bools
        self._setup_results: Dict[Tuple, deque] = {}
        self._setup_paused_until: Dict[Tuple, datetime] = {}

    # ── Check ─────────────────────────────────────────────────────────────────

    def is_allowed(self, coin: str, setup_family: str, htf_regime: str) -> Tuple[bool, str]:
        """
        Returns (allowed, reason).
        Call before accepting any signal.
        """
        now = _utc()

        # Tier 1: coin-level pause
        if coin in self._coin_paused_until:
            until = self._coin_paused_until[coin]
            if now < until:
                hrs = (until - now).total_seconds() / 3600
                return False, f"{coin} paused for {hrs:.1f}h (low win rate)"
            else:
                del self._coin_paused_until[coin]

        # Tier 2: setup-level kill-switch
        key = (coin, setup_family, htf_regime)
        if key in self._setup_paused_until:
            until = self._setup_paused_until[key]
            if now < until:
                hrs = (until - now).total_seconds() / 3600
                return False, f"{coin}/{setup_family}/{htf_regime} kill-switched for {hrs:.1f}h"
            else:
                del self._setup_paused_until[key]

        return True, "ok"

    # ── Record ────────────────────────────────────────────────────────────────

    def record_outcome(self, coin: str, setup_family: str, htf_regime: str, won: bool):
        """
        Call after every position closes (win = realized_r > 0).
        Updates rolling windows and triggers pauses if thresholds breached.
        """
        # Tier 1: coin
        if coin not in self._coin_results:
            self._coin_results[coin] = deque(maxlen=COIN_ROLLING_N)
        self._coin_results[coin].append(won)
        self._maybe_pause_coin(coin)

        # Tier 2: setup
        key = (coin, setup_family, htf_regime)
        if key not in self._setup_results:
            self._setup_results[key] = deque(maxlen=KILL_ROLLING_N)
        self._setup_results[key].append(won)
        self._maybe_pause_setup(key)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _maybe_pause_coin(self, coin: str):
        results = self._coin_results[coin]
        if len(results) < COIN_MIN_TRADES:
            return
        wr = sum(results) / len(results)
        if wr < COIN_KILL_WIN_RATE:
            until = _utc() + timedelta(hours=COIN_PAUSE_HOURS)
            self._coin_paused_until[coin] = until
            print(f"[FILTER] {coin} PAUSED {COIN_PAUSE_HOURS}h — "
                  f"WR={wr*100:.0f}% ({sum(results)}/{len(results)}) "
                  f"< {COIN_KILL_WIN_RATE*100:.0f}% threshold")

    def _maybe_pause_setup(self, key: Tuple):
        results = self._setup_results[key]
        if len(results) < KILL_MIN_TRADES:
            return
        wr = sum(results) / len(results)
        if wr < KILL_WIN_RATE:
            coin, setup, htf = key
            until = _utc() + timedelta(hours=KILL_PAUSE_HOURS)
            self._setup_paused_until[key] = until
            print(f"[FILTER] KILL-SWITCH {coin}/{setup}/{htf} for {KILL_PAUSE_HOURS}h — "
                  f"WR={wr*100:.0f}% ({sum(results)}/{len(results)}) "
                  f"< {KILL_WIN_RATE*100:.0f}% threshold")

    # ── Status ────────────────────────────────────────────────────────────────

    def summary(self) -> str:
        now = _utc()
        lines = []

        for coin, results in sorted(self._coin_results.items()):
            wr = sum(results) / len(results) * 100 if results else 0
            paused = coin in self._coin_paused_until and now < self._coin_paused_until[coin]
            tag = " [PAUSED]" if paused else ""
            lines.append(f"  {coin:6s} coin WR: {wr:5.1f}% n={len(results)}{tag}")

        for (coin, setup, htf), results in sorted(self._setup_results.items()):
            wr = sum(results) / len(results) * 100 if results else 0
            key = (coin, setup, htf)
            paused = key in self._setup_paused_until and now < self._setup_paused_until[key]
            tag = " [KILLED]" if paused else ""
            lines.append(f"  {coin:6s}/{setup:14s}/{htf:5s}: {wr:5.1f}% n={len(results)}{tag}")

        return "\n".join(lines) if lines else "  No data yet."
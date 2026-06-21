# strategy_filter.py
"""
Strategy kill-switch + per-coin performance filter.

Tracks rolling win rates per (coin) and per (coin, setup_family, htf_regime).
Auto-pauses combinations that fall below thresholds with configurable cooldown.

Integrates with RiskManager.check_signal() — called before any trade is opened.
"""

import os
import json
from collections import deque
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Tuple, Any


# ── Config ────────────────────────────────────────────────────────────────────
KILL_MIN_TRADES      = int(os.getenv("KILL_MIN_TRADES",     "5"))    # min trades before kill-switch activates (was 8)
KILL_WIN_RATE        = float(os.getenv("KILL_WIN_RATE",     "0.25")) # pause if WR drops below 25% (was 0.30)
KILL_ROLLING_N       = int(os.getenv("KILL_ROLLING_N",      "15"))   # rolling window size (was 10)
KILL_PAUSE_HOURS     = int(os.getenv("KILL_PAUSE_HOURS",    "8"))    # hours to pause after kill (was 24)

COIN_MIN_TRADES      = int(os.getenv("COIN_MIN_TRADES",     "5"))    # (was 8)
COIN_KILL_WIN_RATE   = float(os.getenv("COIN_KILL_WIN_RATE", "0.20")) # stricter per-coin threshold (was 0.25)
COIN_ROLLING_N       = int(os.getenv("COIN_ROLLING_N",      "15"))   # (was 10)
COIN_PAUSE_HOURS     = int(os.getenv("COIN_PAUSE_HOURS",    "6"))    # (was 12)
STRATEGY_FILTER_STATE_FILE = os.getenv("STRATEGY_FILTER_STATE_FILE", "strategy_filter_state.json")

# ── Per-coin adaptive confidence floor (Level 1 self-learning) ────────────────
# After COIN_FLOOR_MIN_TRADES recorded outcomes, the swing_conf gate uses a
# coin-specific floor rather than the global SWING_MIN_CONFIDENCE constant.
# Elite performers earn a lower bar; structural underperformers get a higher bar.
# Adjustment is capped at ±COIN_FLOOR_MAX_DELTA so no coin drifts far from base.
COIN_FLOOR_MIN_TRADES     = int(os.getenv("COIN_FLOOR_MIN_TRADES",    "5"))
COIN_FLOOR_ELITE_WR       = float(os.getenv("COIN_FLOOR_ELITE_WR",    "0.65"))  # ≥65% WR → −0.05
COIN_FLOOR_ELITE_BONUS    = float(os.getenv("COIN_FLOOR_ELITE_BONUS", "0.05"))
COIN_FLOOR_HIGH_WR        = float(os.getenv("COIN_FLOOR_HIGH_WR",     "0.55"))  # ≥55% WR → −0.03
COIN_FLOOR_HIGH_BONUS     = float(os.getenv("COIN_FLOOR_HIGH_BONUS",  "0.03"))
COIN_FLOOR_LOW_WR         = float(os.getenv("COIN_FLOOR_LOW_WR",      "0.35"))  # ≤35% WR → +0.03
COIN_FLOOR_LOW_PENALTY    = float(os.getenv("COIN_FLOOR_LOW_PENALTY", "0.03"))
COIN_FLOOR_MAX_DELTA      = float(os.getenv("COIN_FLOOR_MAX_DELTA",   "0.05"))


def _utc() -> datetime:  # TODO: identical helper in position.py and live_position_monitor.py — consolidate into a shared utils module
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
        self._state_file = STRATEGY_FILTER_STATE_FILE
        self._pause_state_loaded = self._load_pause_state()

    # ── Persistence ──────────────────────────────────────────────────────────

    @staticmethod
    def _dt_to_iso(dt: datetime) -> str:
        return dt.astimezone(timezone.utc).isoformat()

    @staticmethod
    def _iso_to_dt(raw: Any) -> Optional[datetime]:
        if not raw:
            return None
        try:
            dt = datetime.fromisoformat(str(raw))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            return None

    def _save_pause_state(self):
        payload = {
            "version": 1,
            "saved_at": self._dt_to_iso(_utc()),
            "coin_paused_until": {
                coin: self._dt_to_iso(until)
                for coin, until in self._coin_paused_until.items()
            },
            "setup_paused_until": [
                {
                    "coin": str(key[0]),
                    "setup_family": str(key[1]),
                    "htf_regime": str(key[2]),
                    "until": self._dt_to_iso(until),
                }
                for key, until in self._setup_paused_until.items()
                if len(key) == 3
            ],
        }
        try:
            with open(self._state_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
            print(
                f"[FILTER_STATE] saved coin_pauses={len(self._coin_paused_until)} "
                f"setup_pauses={len(self._setup_paused_until)}"
            )
        except Exception as e:
            print(f"[FILTER_STATE] save failed ({self._state_file}): {e}")

    def _load_pause_state(self) -> bool:
        if not os.path.exists(self._state_file):
            print(f"[FILTER_STATE] no persisted state file at {self._state_file}")
            return False

        try:
            with open(self._state_file, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception as e:
            print(f"[FILTER_STATE] load failed ({self._state_file}): {e}")
            return False

        now = _utc()
        coin_pauses: Dict[str, datetime] = {}
        setup_pauses: Dict[Tuple, datetime] = {}
        expired_pruned = 0

        raw_coin = raw.get("coin_paused_until", {})
        if isinstance(raw_coin, dict):
            for coin, until_raw in raw_coin.items():
                until = self._iso_to_dt(until_raw)
                if until is None:
                    continue
                if until > now:
                    coin_pauses[str(coin)] = until
                else:
                    expired_pruned += 1

        raw_setup = raw.get("setup_paused_until", [])
        if isinstance(raw_setup, list):
            for item in raw_setup:
                if not isinstance(item, dict):
                    continue
                coin = str(item.get("coin", "")).strip()
                setup = str(item.get("setup_family", "")).strip()
                htf = str(item.get("htf_regime", "")).strip()
                until = self._iso_to_dt(item.get("until"))
                if not coin or not setup or until is None:
                    continue
                key = (coin, setup, htf)
                if until > now:
                    setup_pauses[key] = until
                else:
                    expired_pruned += 1

        self._coin_paused_until = coin_pauses
        self._setup_paused_until = setup_pauses
        print(
            f"[FILTER_STATE] loaded coin_pauses={len(self._coin_paused_until)} "
            f"setup_pauses={len(self._setup_paused_until)} "
            f"expired_pruned={expired_pruned}"
        )
        if expired_pruned > 0:
            self._save_pause_state()
        return True

    def has_persisted_pause_state(self) -> bool:
        return bool(self._pause_state_loaded)

    # ── Check ─────────────────────────────────────────────────────────────────

    def is_allowed(self, coin: str, setup_family: str, htf_regime: str) -> Tuple[bool, str]:
        """
        Returns (allowed, reason).
        Call before accepting any signal.
        """
        now = _utc()
        state_changed = False

        # Tier 1: coin-level pause
        if coin in self._coin_paused_until:
            until = self._coin_paused_until[coin]
            if now < until:
                hrs = (until - now).total_seconds() / 3600
                return False, f"{coin} paused for {hrs:.1f}h (low win rate)"
            else:
                del self._coin_paused_until[coin]
                state_changed = True

        # Tier 2: setup-level kill-switch
        key = (coin, setup_family, htf_regime)
        if key in self._setup_paused_until:
            until = self._setup_paused_until[key]
            if now < until:
                hrs = (until - now).total_seconds() / 3600
                return False, f"{coin}/{setup_family}/{htf_regime} kill-switched for {hrs:.1f}h"
            else:
                del self._setup_paused_until[key]
                state_changed = True

        if state_changed:
            self._save_pause_state()

        return True, "ok"

    # ── Record ────────────────────────────────────────────────────────────────

    def record_outcome(
        self,
        coin: str,
        setup_family: str,
        htf_regime: str,
        won: bool,
        apply_pauses: bool = True,
    ):
        """
        Call after every position closes (win = realized_r > 0).
        Updates rolling windows and triggers pauses if thresholds breached.
        """
        # Tier 1: coin
        if coin not in self._coin_results:
            self._coin_results[coin] = deque(maxlen=COIN_ROLLING_N)
        self._coin_results[coin].append(won)
        if apply_pauses:
            self._maybe_pause_coin(coin)

        # Tier 2: setup
        key = (coin, setup_family, htf_regime)
        if key not in self._setup_results:
            self._setup_results[key] = deque(maxlen=KILL_ROLLING_N)
        self._setup_results[key].append(won)
        if apply_pauses:
            self._maybe_pause_setup(key)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _maybe_pause_coin(self, coin: str):
        results = self._coin_results[coin]
        if len(results) < COIN_MIN_TRADES:
            return
        wr = sum(results) / len(results)
        if wr < COIN_KILL_WIN_RATE:
            existing_until = self._coin_paused_until.get(coin)
            if existing_until is not None and _utc() < existing_until:
                return
            until = _utc() + timedelta(hours=COIN_PAUSE_HOURS)
            self._coin_paused_until[coin] = until
            self._save_pause_state()
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
            existing_until = self._setup_paused_until.get(key)
            if existing_until is not None and _utc() < existing_until:
                return
            until = _utc() + timedelta(hours=KILL_PAUSE_HOURS)
            self._setup_paused_until[key] = until
            self._save_pause_state()
            print(f"[FILTER] KILL-SWITCH {coin}/{setup}/{htf} for {KILL_PAUSE_HOURS}h — "
                  f"WR={wr*100:.0f}% ({sum(results)}/{len(results)}) "
                  f"< {KILL_WIN_RATE*100:.0f}% threshold")

    # ── Per-coin adaptive floor (Level 1 self-learning) ─────────────────────

    @staticmethod
    def _floor_delta(wr: float, n: int) -> float:
        """Single source of truth for the per-coin floor adjustment delta."""
        if n < COIN_FLOOR_MIN_TRADES:
            return 0.0
        if wr >= COIN_FLOOR_ELITE_WR:
            raw = -COIN_FLOOR_ELITE_BONUS
        elif wr >= COIN_FLOOR_HIGH_WR:
            raw = -COIN_FLOOR_HIGH_BONUS
        elif wr <= COIN_FLOOR_LOW_WR:
            raw = COIN_FLOOR_LOW_PENALTY
        else:
            raw = 0.0
        return max(-COIN_FLOOR_MAX_DELTA, min(COIN_FLOOR_MAX_DELTA, raw))

    def coin_confidence_floor(self, coin: str, base_floor: float) -> float:
        """
        Return a per-coin confidence floor adjusted by live rolling win rate.

        Returns base_floor unchanged until COIN_FLOOR_MIN_TRADES are recorded.
        Adjustments:
          WR ≥ COIN_FLOOR_ELITE_WR  → floor − COIN_FLOOR_ELITE_BONUS  (e.g. 0.90 → 0.85)
          WR ≥ COIN_FLOOR_HIGH_WR   → floor − COIN_FLOOR_HIGH_BONUS   (e.g. 0.90 → 0.87)
          WR ≤ COIN_FLOOR_LOW_WR    → floor + COIN_FLOOR_LOW_PENALTY   (e.g. 0.90 → 0.93)
          otherwise                 → base_floor unchanged
        Total adjustment capped at ±COIN_FLOOR_MAX_DELTA.
        """
        results = self._coin_results.get(str(coin).upper())
        if not results:
            return base_floor
        wr = sum(results) / len(results)
        return round(base_floor + self._floor_delta(wr, len(results)), 4)

    def coin_wr_stats(self) -> Dict[str, Dict]:
        """
        Return WR, trade count, and floor delta for all coins with recorded outcomes.
        Used by tools/research/param_suggester.py and boot status logging.
        """
        out: Dict[str, Dict] = {}
        for coin, results in self._coin_results.items():
            n = len(results)
            wr = sum(results) / n if n > 0 else 0.0
            out[coin] = {
                "n": n,
                "wr": round(wr, 3),
                "floor_delta": round(self._floor_delta(wr, n), 4),
                "active": n >= COIN_FLOOR_MIN_TRADES,
            }
        return out

    # ── Status ────────────────────────────────────────────────────────────────

    def summary(self) -> str:
        now = _utc()
        lines = []

        all_stats = self.coin_wr_stats()  # compute once — not inside the loop
        for coin, results in sorted(self._coin_results.items()):
            wr = sum(results) / len(results) * 100 if results else 0
            paused = coin in self._coin_paused_until and now < self._coin_paused_until[coin]
            tag = " [PAUSED]" if paused else ""
            s = all_stats.get(coin, {})
            delta = s.get("floor_delta", 0.0)
            floor_tag = f" floor_adj={delta:+.2f}" if delta != 0.0 and s.get("active") else ""
            lines.append(f"  {coin:8s} WR={wr:5.1f}% n={len(results)}{floor_tag}{tag}")

        for (coin, setup, htf), results in sorted(self._setup_results.items()):
            wr = sum(results) / len(results) * 100 if results else 0
            key = (coin, setup, htf)
            paused = key in self._setup_paused_until and now < self._setup_paused_until[key]
            tag = " [KILLED]" if paused else ""
            lines.append(f"  {coin:6s}/{setup:14s}/{htf:5s}: {wr:5.1f}% n={len(results)}{tag}")

        return "\n".join(lines) if lines else "  No data yet."

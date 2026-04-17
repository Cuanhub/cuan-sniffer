"""
Adaptive signal engine — 15m execution with 1h/4h context.

Changes vs prior version:
- Dedup TTL: 900s → 300s. Allows re-fire on same setup after 5 min.
  In trending/liquidation moves the setup repeats legitimately every candle.
  900s was locking out valid continuation entries for a full 15m cycle.
- Dedup ATR band: 0.75x → 0.50x. Expires sooner when price moves on momentum.
  Previously a 0.75×ATR move was still "the same setup" — too wide for alts.
- _build_feature_frame: drops forming candle (df_ohlcv.iloc[:-1]) before
  building features. Hyperliquid candleSnapshot always returns the open candle
  as the last row. Using it anchors entry_price to a stale mid-candle value,
  causing the executor's live-price check to see a large spurious gap.
- Required column names preserved as atr_14, rsi_14, vwap_dev, body_pct.
- _build_feature_frame correctly indented as a class method.
- Regime helpers receive full df_ohlcv — one forming 15m candle is negligible
  on resampled 1h/4h data.
"""

from dataclasses import dataclass
from collections import defaultdict, deque
from typing import Optional, Dict, Any, List, Tuple

import os
import pandas as pd

from features import add_features
from smc_structure import build_structure
from smc_zones import add_smc_zones
from smc_sweeps import add_sweep_features
from perp_sentiment import PerpSentimentSnapshot

# ── Market regime classification (signal-time) ───────────────────────────────
MARKET_REGIME_LOOKBACK = int(os.getenv("MARKET_REGIME_LOOKBACK", "30"))
MARKET_REGIME_ATR_EXPANSION_RATIO = float(os.getenv("MARKET_REGIME_ATR_EXPANSION_RATIO", "1.08"))
MARKET_REGIME_ATR_CONTRACTION_RATIO = float(os.getenv("MARKET_REGIME_ATR_CONTRACTION_RATIO", "0.94"))
MARKET_REGIME_EMA_SPREAD_STRONG_PCT = float(os.getenv("MARKET_REGIME_EMA_SPREAD_STRONG_PCT", "0.0025"))
MARKET_REGIME_BREAKOUT_ATR = float(os.getenv("MARKET_REGIME_BREAKOUT_ATR", "0.20"))
MARKET_REGIME_RANGE_MID_PCT = float(os.getenv("MARKET_REGIME_RANGE_MID_PCT", "0.20"))
MARKET_REGIME_STRONG_SCORE = int(os.getenv("MARKET_REGIME_STRONG_SCORE", "4"))
MARKET_REGIME_CHOP_SCORE = int(os.getenv("MARKET_REGIME_CHOP_SCORE", "1"))

# ── Continuation hardening ───────────────────────────────────────────────────
CONT_REGIME_STRONG_BONUS = float(os.getenv("CONT_REGIME_STRONG_BONUS", "0.08"))
CONT_REGIME_WEAK_PENALTY = -abs(float(os.getenv("CONT_REGIME_WEAK_PENALTY", "-0.12")))
CONT_REGIME_CHOP_PENALTY = -abs(float(os.getenv("CONT_REGIME_CHOP_PENALTY", "-0.22")))
CONT_REGIME_UNKNOWN_PENALTY = -abs(float(os.getenv("CONT_REGIME_UNKNOWN_PENALTY", "-0.03")))
CONT_WEAK_TREND_BLOCK_WITHOUT_DUAL_HTF = (
    os.getenv("CONT_WEAK_TREND_BLOCK_WITHOUT_DUAL_HTF", "true").lower() == "true"
)

CONT_LATE_ENTRY_BODY_PENALTY_ATR = float(os.getenv("CONT_LATE_ENTRY_BODY_PENALTY_ATR", "0.90"))
CONT_LATE_ENTRY_BODY_HARD_BLOCK_ATR = float(os.getenv("CONT_LATE_ENTRY_BODY_HARD_BLOCK_ATR", "1.50"))
CONT_LATE_ENTRY_HARD_BLOCK_RANGE_ATR = float(os.getenv("CONT_LATE_ENTRY_HARD_BLOCK_RANGE_ATR", "2.60"))
CONT_LATE_ENTRY_PENALTY_SCORE = -abs(float(os.getenv("CONT_LATE_ENTRY_PENALTY_SCORE", "-0.08")))
CONT_LATE_ENTRY_EXTENSION_HARD_BLOCK_ATR = float(
    os.getenv("CONT_LATE_ENTRY_EXTENSION_HARD_BLOCK_ATR", "1.90")
)

CONT_PULLBACK_SHALLOW_MIN_ATR = float(os.getenv("CONT_PULLBACK_SHALLOW_MIN_ATR", "0.12"))
CONT_PULLBACK_CLEAN_MAX_ATR = float(os.getenv("CONT_PULLBACK_CLEAN_MAX_ATR", "0.45"))
CONT_PULLBACK_TIGHT_MAX_ATR = float(os.getenv("CONT_PULLBACK_TIGHT_MAX_ATR", "0.80"))
CONT_PULLBACK_EXTENDED_ATR = float(os.getenv("CONT_PULLBACK_EXTENDED_ATR", "1.40"))
CONT_PULLBACK_CLEAN_REWARD = float(os.getenv("CONT_PULLBACK_CLEAN_REWARD", "0.08"))
CONT_PULLBACK_REWARD = float(os.getenv("CONT_PULLBACK_REWARD", "0.05"))
CONT_PULLBACK_PENALTY = -abs(float(os.getenv("CONT_PULLBACK_PENALTY", "-0.07")))
CONT_PULLBACK_SHALLOW_PENALTY = -abs(float(os.getenv("CONT_PULLBACK_SHALLOW_PENALTY", "-0.05")))
CONT_PULLBACK_SLOPPY_PENALTY = -abs(float(os.getenv("CONT_PULLBACK_SLOPPY_PENALTY", "-0.03")))
CONT_PULLBACK_EXTENDED_PENALTY_MULT = float(
    os.getenv("CONT_PULLBACK_EXTENDED_PENALTY_MULT", "1.5")
)
CONT_BELOW_EMA_PENALTY = -abs(float(os.getenv("CONT_BELOW_EMA_PENALTY", "-0.10")))

CONT_DIRECTIONAL_LOSS_PENALTY = -abs(float(os.getenv("CONT_DIRECTIONAL_LOSS_PENALTY", "-0.12")))
CONT_BREAKOUT_FAILURE_PENALTY = -abs(float(os.getenv("CONT_BREAKOUT_FAILURE_PENALTY", "-0.14")))
CONT_BREAKOUT_FAILURE_LOOKBACK = int(os.getenv("CONT_BREAKOUT_FAILURE_LOOKBACK", "12"))
CONT_BREAKOUT_FAILURE_MIN_COUNT = int(os.getenv("CONT_BREAKOUT_FAILURE_MIN_COUNT", "2"))

_DIRECTIONAL_OUTCOME_MAXLEN = int(os.getenv("DIRECTIONAL_OUTCOME_MAXLEN", "12"))

# ── Signal gating / dedup tuning ─────────────────────────────────────────────
LOW_VOL_SCORE_PENALTY = -abs(float(os.getenv("LOW_VOL_SCORE_PENALTY", "-0.05")))
REGIME_SCORE_THRESHOLD_STRONG = float(os.getenv("REGIME_SCORE_THRESHOLD_STRONG", "0.64"))
REGIME_SCORE_THRESHOLD_WEAK = float(os.getenv("REGIME_SCORE_THRESHOLD_WEAK", "0.66"))
REGIME_SCORE_THRESHOLD_CHOP = float(os.getenv("REGIME_SCORE_THRESHOLD_CHOP", "0.69"))

DEDUP_ANTI_SPAM_FLOOR_SEC = int(os.getenv("DEDUP_ANTI_SPAM_FLOOR_SEC", "60"))
DEDUP_PRICE_MOVE_ATR_MULT = float(os.getenv("DEDUP_PRICE_MOVE_ATR_MULT", "0.50"))
DEDUP_SCORE_IMPROVEMENT_MIN = float(os.getenv("DEDUP_SCORE_IMPROVEMENT_MIN", "0.08"))


@dataclass
class Signal:
    coin: str
    side: str
    entry_price: float
    stop_price: float
    tp_price: float
    confidence: float
    regime: str
    reason: str
    meta: Dict[str, Any]


class AdaptiveSignalEngine:
    """
    Full-feature signal engine.

    Philosophy:
    - 15m execution
    - 1h / 4h context
    - continuation + reversal families
    - execution-aware but not over-choked
    - orthogonal features preferred over stacked duplicates
    """

    def __init__(
        self,
        score_threshold: float = 0.72,
        atr_stop_mult: float = 1.3,
        atr_tp_mult: float = 4.0,
        min_stop_pct: float = 0.004,
        min_tp_pct: float = 0.010,
        swing_threshold_1h: float = 0.60,
        swing_threshold_4h: float = 0.65,
        debug: bool = True,
    ):
        self.score_threshold = score_threshold
        self.atr_stop_mult = atr_stop_mult
        self.atr_tp_mult = atr_tp_mult
        self.min_stop_pct = min_stop_pct
        self.min_tp_pct = min_tp_pct
        self.debug = debug

        self.swing_thresholds = {
            "1h": swing_threshold_1h,
            "4h": swing_threshold_4h,
        }

        self.last_swing_ts: Dict[Tuple[str, str], Optional[pd.Timestamp]] = {}

        self._last_emitted: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._directional_outcomes: Dict[Tuple[str, str, str], deque] = defaultdict(
            lambda: deque(maxlen=max(3, _DIRECTIONAL_OUTCOME_MAXLEN))
        )

    # ------------------------------------------------------------------
    # Feature frame
    # ------------------------------------------------------------------

    def _build_feature_frame(self, df_ohlcv: pd.DataFrame, min_len: int = 200) -> pd.DataFrame:
        """
        Build the feature frame from CLOSED candles only.

        Hyperliquid candleSnapshot always returns the currently forming candle
        as the final row (endTime=now). That row must be excluded before signal
        generation — using it anchors entry_price to a stale mid-candle value
        while the executor compares against the current live mid, causing large
        spurious gaps that trigger entry_outside_atr_buffer rejection.

        Requires min_len + 1 rows so that after dropping the forming candle we
        still have at least min_len closed candles for feature calculation.
        """
        if df_ohlcv is None or df_ohlcv.empty:
            return pd.DataFrame()

        if len(df_ohlcv) < (min_len + 1):
            return pd.DataFrame()

        # Drop the last (forming) candle. All signal logic uses only closed candles.
        df = df_ohlcv.iloc[:-1].copy()

        try:
            df = add_features(df)
            df = build_structure(df, lookback=5)
            df = add_smc_zones(df)
            df = add_sweep_features(df)
        except Exception as e:
            if self.debug:
                print("[FEATURE_FRAME_DEBUG] Pipeline build failed: " + str(e))
            return pd.DataFrame()

        flag_cols = [
            "trend_up", "trend_down",
            "vol_expansion", "vol_compression",
            "bos_bull", "bos_bear",
            "choch_bull", "choch_bear",
            "ob_bull", "ob_bear",
            "fvg_bull", "fvg_bear",
            "sweep_bull", "sweep_bear",
        ]

        for col in flag_cols:
            if col not in df.columns:
                df[col] = False
            df[col] = df[col].fillna(False).astype(bool)

        text_cols_defaults = {
            "trend": "neutral",
            "structure_label": "",
        }
        for col, default in text_cols_defaults.items():
            if col not in df.columns:
                df[col] = default
            df[col] = df[col].fillna(default)

        required_numeric = [
            "time", "open", "high", "low", "close", "volume",
            "atr_14", "vwap_dev", "body_pct", "rsi_14", "ema_50",
        ]

        missing_required = [c for c in required_numeric if c not in df.columns]
        if missing_required:
            if self.debug:
                print("[FEATURE_FRAME_DEBUG] Missing required columns: " + str(missing_required))
            return pd.DataFrame()

        df = df.dropna(subset=required_numeric).reset_index(drop=True)

        if len(df) < min_len:
            return pd.DataFrame()

        return df

    # ------------------------------------------------------------------
    # Engine-level dedup
    # ------------------------------------------------------------------

    def _is_stale_repeat(
        self,
        coin: str,
        side: str,
        setup_family: str,
        entry_price: float,
        regime: str,
        atr: float,
        market_regime: str,
        current_score: float,
    ) -> bool:
        """
        Suppress noisy repeats while allowing actionable re-fires.
        """
        last = self._last_emitted.get((coin, side))
        if last is None:
            return False

        if last["setup_family"] != setup_family:
            return False
        if last["regime"] != regime:
            return False

        elapsed = (pd.Timestamp.now(tz="UTC") - last["ts"]).total_seconds()
        if elapsed < DEDUP_ANTI_SPAM_FLOOR_SEC:
            if self.debug:
                print(
                    f"[DEDUP_BLOCK] {coin}"
                    f" | side={side}"
                    f" | family={setup_family}"
                    f" | reason=anti_spam_floor"
                    f" | elapsed={elapsed:.1f}s"
                    f" | floor={DEDUP_ANTI_SPAM_FLOOR_SEC}s"
                )
            return True

        ref_atr = float(last["atr"]) if float(last["atr"]) > 0 else float(atr)
        if ref_atr <= 0:
            return False

        ttl_sec = 120
        if setup_family == "continuation" and market_regime == "strong_trend":
            ttl_sec = 60
        elif setup_family == "continuation" and market_regime == "weak_trend":
            ttl_sec = 120
        elif setup_family == "reversal":
            ttl_sec = 120

        entry_last = float(last["entry_price"])
        price_move = abs(entry_price - entry_last)
        price_move_threshold = ref_atr * DEDUP_PRICE_MOVE_ATR_MULT
        last_score = float(last.get("score", 0.0) or 0.0)
        score_delta = float(current_score) - last_score

        bypass_reasons: List[str] = []
        if price_move > price_move_threshold:
            bypass_reasons.append(
                f"price_move={price_move:.6f}>{price_move_threshold:.6f}"
            )
        if setup_family == "continuation" and market_regime == "strong_trend":
            bypass_reasons.append("strong_trend_continuation")
        if score_delta >= DEDUP_SCORE_IMPROVEMENT_MIN:
            bypass_reasons.append(
                f"score_upgrade={score_delta:.3f}>={DEDUP_SCORE_IMPROVEMENT_MIN:.3f}"
            )

        if bypass_reasons:
            if self.debug:
                print(
                    f"[DEDUP_BYPASS] {coin}"
                    f" | side={side}"
                    f" | family={setup_family}"
                    f" | regime={market_regime}"
                    f" | elapsed={elapsed:.1f}s"
                    f" | reasons={bypass_reasons}"
                )
            return False

        if elapsed > ttl_sec:
            if self.debug:
                print(
                    f"[DEDUP_BYPASS] {coin}"
                    f" | side={side}"
                    f" | family={setup_family}"
                    f" | reason=ttl_expired"
                    f" | elapsed={elapsed:.1f}s"
                    f" | ttl={ttl_sec}s"
                )
            return False

        if self.debug:
            print(
                f"[DEDUP_BLOCK] {coin}"
                f" | side={side}"
                f" | family={setup_family}"
                f" | regime={regime}"
                f" | reason=ttl_active"
                f" | ttl={ttl_sec}s"
                f" | elapsed={elapsed:.1f}s"
                f" | entry_now={entry_price:.6f}"
                f" | entry_last={entry_last:.6f}"
                f" | atr_ref={ref_atr:.6f}"
                f" | score_now={float(current_score):.3f}"
                f" | score_last={last_score:.3f}"
            )

        return True

    def _record_emitted(
        self,
        coin: str,
        side: str,
        setup_family: str,
        entry_price: float,
        regime: str,
        atr: float,
        market_regime: str,
        score: float,
    ):
        self._last_emitted[(coin, side)] = {
            "setup_family": setup_family,
            "entry_price": float(entry_price),
            "regime": regime,
            "atr": float(atr),
            "market_regime": str(market_regime or "").strip().lower(),
            "score": float(score),
            "ts": pd.Timestamp.now(tz="UTC"),
        }

    def record_emitted_signal(
        self,
        signal: Any,
        entry_price: Optional[float] = None,
    ) -> None:
        """
        Record dedup state only after an entry is truly emitted (filled).
        """
        try:
            meta = getattr(signal, "meta", None) or {}
            timeframe = str(meta.get("timeframe", "15m")).strip().lower()
            if timeframe != "15m":
                return

            coin = str(getattr(signal, "coin", "") or "").upper().strip()
            side = str(getattr(signal, "side", "") or "").upper().strip()
            setup_family = str(meta.get("setup_family", meta.get("regime_local", ""))).strip().lower()
            regime = str(getattr(signal, "regime", "") or "").strip()
            market_regime = str(meta.get("market_regime", "unknown")).strip().lower()
            atr = float(meta.get("atr", 0.0) or 0.0)
            score = float(meta.get("total_score", getattr(signal, "confidence", 0.0)) or 0.0)
            emitted_entry = float(
                entry_price
                if entry_price is not None
                else getattr(signal, "entry_price", 0.0)
            )

            if not coin or side not in {"LONG", "SHORT"} or not setup_family or not regime:
                return
            if emitted_entry <= 0 or atr <= 0:
                return

            self._record_emitted(
                coin=coin,
                side=side,
                setup_family=setup_family,
                entry_price=emitted_entry,
                regime=regime,
                atr=atr,
                market_regime=market_regime,
                score=score,
            )
            if self.debug:
                print(
                    f"[DEDUP_RECORD] {coin}"
                    f" | side={side}"
                    f" | family={setup_family}"
                    f" | regime={market_regime}"
                    f" | score={score:.3f}"
                    f" | entry={emitted_entry:.6f}"
                )
        except Exception as e:
            if self.debug:
                print(f"[DEDUP_RECORD] skipped due to error: {e}")

    # ------------------------------------------------------------------
    # Regime helpers
    # ------------------------------------------------------------------

    def _compute_htf_regime(self, df_ohlcv: pd.DataFrame) -> Tuple[str, List[str]]:
        """
        Receives full df_ohlcv including the forming candle.
        Regime uses resampled 1h bars — one forming 15m candle is negligible.
        """
        notes: List[str] = []

        if df_ohlcv is None or df_ohlcv.empty:
            return "unknown", ["htf_no_data"]

        df_raw = df_ohlcv.copy()

        if "time" in df_raw.columns:
            df_raw["time"] = pd.to_datetime(df_raw["time"])
            df_raw = df_raw.set_index("time")
        elif not isinstance(df_raw.index, pd.DatetimeIndex):
            return "unknown", ["htf_no_time_index"]

        try:
            df_1h = (
                df_raw.resample("h")
                .agg({
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                })
                .dropna()
            )
        except Exception as e:
            if self.debug:
                print("[HTF_REGIME_DEBUG] Resample 1h failed: " + str(e))
            return "unknown", ["htf_resample_error"]

        if len(df_1h) < 30:
            return "unknown", ["htf_insufficient_bars"]

        df_1h["ema_fast"] = df_1h["close"].ewm(span=10, adjust=False).mean()
        df_1h["ema_slow"] = df_1h["close"].ewm(span=30, adjust=False).mean()

        last = df_1h.iloc[-1]
        prev = df_1h.iloc[-4] if len(df_1h) >= 4 else df_1h.iloc[0]

        ema_fast = float(last["ema_fast"])
        ema_slow = float(last["ema_slow"])
        ema_fast_prev = float(prev["ema_fast"])
        ema_slow_prev = float(prev["ema_slow"])

        fast_slope = ema_fast - ema_fast_prev
        slow_slope = ema_slow - ema_slow_prev

        if ema_fast > ema_slow and fast_slope > 0 and slow_slope >= 0:
            regime = "up"
            notes.append("htf_ema_trend_up")
        elif ema_fast < ema_slow and fast_slope < 0 and slow_slope <= 0:
            regime = "down"
            notes.append("htf_ema_trend_down")
        else:
            regime = "chop"
            notes.append("htf_ema_chop")

        if self.debug:
            print(
                "[HTF_REGIME_DEBUG] 1h regime=" + regime +
                " ema_fast=" + str(round(ema_fast, 2)) +
                " ema_slow=" + str(round(ema_slow, 2))
            )

        return regime, notes

    def _compute_macro_regime_4h(self, df_ohlcv: pd.DataFrame) -> Tuple[str, List[str]]:
        """
        Receives full df_ohlcv including the forming candle.
        Regime uses resampled 4h bars — one forming 15m candle is negligible.
        """
        notes: List[str] = []

        if df_ohlcv is None or df_ohlcv.empty:
            return "unknown", ["macro_no_data"]

        df_raw = df_ohlcv.copy()

        if "time" in df_raw.columns:
            df_raw["time"] = pd.to_datetime(df_raw["time"])
            df_raw = df_raw.set_index("time")
        elif not isinstance(df_raw.index, pd.DatetimeIndex):
            return "unknown", ["macro_no_time_index"]

        try:
            df_4h = (
                df_raw.resample("4h")
                .agg({
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                })
                .dropna()
            )
        except Exception as e:
            if self.debug:
                print("[MACRO_REGIME_DEBUG] Resample 4h failed: " + str(e))
            return "unknown", ["macro_resample_error"]

        if len(df_4h) < 18:
            return "unknown", ["macro_insufficient_bars"]

        df_4h["ema_fast"] = df_4h["close"].ewm(span=10, adjust=False).mean()
        df_4h["ema_slow"] = df_4h["close"].ewm(span=30, adjust=False).mean()

        last = df_4h.iloc[-1]
        prev = df_4h.iloc[-3] if len(df_4h) >= 3 else df_4h.iloc[0]

        ema_fast = float(last["ema_fast"])
        ema_slow = float(last["ema_slow"])
        ema_fast_prev = float(prev["ema_fast"])
        ema_slow_prev = float(prev["ema_slow"])

        fast_slope = ema_fast - ema_fast_prev
        slow_slope = ema_slow - ema_slow_prev

        if ema_fast > ema_slow and fast_slope > 0 and slow_slope >= 0:
            regime = "up"
            notes.append("macro_ema_trend_up")
        elif ema_fast < ema_slow and fast_slope < 0 and slow_slope <= 0:
            regime = "down"
            notes.append("macro_ema_trend_down")
        else:
            regime = "chop"
            notes.append("macro_ema_chop")

        if self.debug:
            print(
                "[MACRO_REGIME_DEBUG] 4h regime=" + regime +
                " ema_fast=" + str(round(ema_fast, 2)) +
                " ema_slow=" + str(round(ema_slow, 2))
            )

        return regime, notes

    def _compute_killzone(self, ts) -> Tuple[str, List[str]]:
        notes: List[str] = []

        if ts is None:
            return "unknown", ["kz_unknown_time"]

        hour = pd.to_datetime(ts, utc=True).hour

        if 0 <= hour < 4:
            label = "asia_open"
        elif 4 <= hour < 7:
            label = "asia_late"
        elif 7 <= hour < 10:
            label = "london_open"
        elif 10 <= hour < 13:
            label = "london_late"
        elif 13 <= hour < 16:
            label = "ny_open"
        elif 16 <= hour < 20:
            label = "ny_pm"
        else:
            label = "dead_zone"

        notes.append("kz_" + label)
        return label, notes

    def _compute_vol_state(self, row: pd.Series) -> Tuple[str, List[str], float]:
        notes: List[str] = []
        atr = float(row.get("atr_14", 0.0))
        close = float(row.get("close", 0.0))

        if close <= 0 or atr <= 0:
            return "unknown", ["vol_unknown"], 0.0

        vol_ratio = atr / close

        if vol_ratio < 0.003:
            state = "low"
        elif vol_ratio < 0.008:
            state = "normal"
        else:
            state = "high"

        notes.append("vol_" + state + "_" + str(round(vol_ratio, 4)))
        return state, notes, vol_ratio

    def _classify_market_regime(
        self,
        df: pd.DataFrame,
        row: pd.Series,
        triggers: Dict[str, bool],
        htf_regime: str,
        macro_regime: str,
    ) -> Tuple[str, Dict[str, Any], List[str]]:
        notes: List[str] = []

        close = float(row.get("close", 0.0))
        atr_now = float(row.get("atr_14", 0.0))
        if close <= 0 or atr_now <= 0 or df is None or df.empty:
            meta = {
                "market_regime": "unknown",
                "atr_ratio": 0.0,
                "trend_strength_proxy": 0,
                "range_mid_dev_pct": 0.0,
                "breakout_dist_atr": 0.0,
            }
            return "unknown", meta, ["market_regime_unknown"]

        lookback = max(8, min(MARKET_REGIME_LOOKBACK, len(df) - 1))
        hist = df.iloc[-(lookback + 1):-1]
        if hist.empty:
            meta = {
                "market_regime": "unknown",
                "atr_ratio": 0.0,
                "trend_strength_proxy": 0,
                "range_mid_dev_pct": 0.0,
                "breakout_dist_atr": 0.0,
            }
            return "unknown", meta, ["market_regime_unknown"]

        atr_ref = float(hist["atr_14"].tail(min(20, len(hist))).mean())
        if atr_ref <= 0:
            atr_ref = atr_now
        atr_ratio = atr_now / (atr_ref + 1e-12)

        ema_50 = float(row.get("ema_50", close))
        ema_200 = float(row.get("ema_200", close))
        ema_spread_pct = abs(ema_50 - ema_200) / close if close > 0 else 0.0

        trend_up = bool(row.get("trend_up", 0))
        trend_down = bool(row.get("trend_down", 0))
        local_trend_flag = trend_up or trend_down
        htf_macro_aligned = (
            (htf_regime in {"up", "down"})
            and (macro_regime == htf_regime)
        )

        prev_high = float(hist["high"].max())
        prev_low = float(hist["low"].min())
        range_width = max(prev_high - prev_low, 1e-12)
        range_mid = (prev_high + prev_low) * 0.5
        range_mid_dev_pct = abs(close - range_mid) / range_width

        breakout_up_dist = max(0.0, (close - prev_high) / (atr_now + 1e-12))
        breakout_down_dist = max(0.0, (prev_low - close) / (atr_now + 1e-12))
        breakout_dist_atr = max(breakout_up_dist, breakout_down_dist)
        structural_breakout = bool(
            triggers.get("bos_bull")
            or triggers.get("bos_bear")
            or triggers.get("choch_bull")
            or triggers.get("choch_bear")
        )
        breakout_flag = (
            breakout_dist_atr >= MARKET_REGIME_BREAKOUT_ATR
            or structural_breakout
        )

        volatility_expanding = (
            atr_ratio >= MARKET_REGIME_ATR_EXPANSION_RATIO
            or bool(row.get("vol_expansion", 0))
        )
        volatility_contracting = (
            atr_ratio <= MARKET_REGIME_ATR_CONTRACTION_RATIO
            or bool(row.get("vol_compression", 0))
        )

        trend_strength_proxy = 0
        if volatility_expanding:
            trend_strength_proxy += 1
        if ema_spread_pct >= MARKET_REGIME_EMA_SPREAD_STRONG_PCT:
            trend_strength_proxy += 1
        if local_trend_flag:
            trend_strength_proxy += 1
        if htf_macro_aligned:
            trend_strength_proxy += 1
        if breakout_flag:
            trend_strength_proxy += 1

        if volatility_contracting:
            trend_strength_proxy -= 1
        if (range_mid_dev_pct <= MARKET_REGIME_RANGE_MID_PCT) and (not breakout_flag):
            trend_strength_proxy -= 1

        chop_signature = (
            volatility_contracting
            and (ema_spread_pct < MARKET_REGIME_EMA_SPREAD_STRONG_PCT)
            and (not breakout_flag)
        )

        if chop_signature or trend_strength_proxy <= MARKET_REGIME_CHOP_SCORE:
            market_regime = "chop"
        elif trend_strength_proxy >= MARKET_REGIME_STRONG_SCORE:
            market_regime = "strong_trend"
        else:
            market_regime = "weak_trend"

        notes.append("market_regime_" + market_regime)
        meta = {
            "market_regime": market_regime,
            "atr_ratio": round(atr_ratio, 4),
            "trend_strength_proxy": int(trend_strength_proxy),
            "ema_spread_pct": round(ema_spread_pct, 6),
            "range_mid_dev_pct": round(range_mid_dev_pct, 4),
            "breakout_dist_atr": round(breakout_dist_atr, 4),
            "breakout_flag": bool(breakout_flag),
            "htf_macro_aligned": bool(htf_macro_aligned),
        }

        if self.debug:
            print(
                f"[MKT_REGIME_DEBUG] regime={market_regime}"
                f" atr_ratio={atr_ratio:.3f}"
                f" trend_proxy={trend_strength_proxy}"
                f" ema_spread_pct={ema_spread_pct:.4f}"
                f" breakout_atr={breakout_dist_atr:.3f}"
                f" range_mid_dev={range_mid_dev_pct:.3f}"
            )

        return market_regime, meta, notes

    # ------------------------------------------------------------------
    # Trigger extraction
    # ------------------------------------------------------------------

    def _extract_triggers(self, row: pd.Series) -> Dict[str, bool]:
        return {
            "bos_bull": bool(row.get("bos_bull", 0)),
            "bos_bear": bool(row.get("bos_bear", 0)),
            "choch_bull": bool(row.get("choch_bull", 0)),
            "choch_bear": bool(row.get("choch_bear", 0)),
            "ob_bull": bool(row.get("ob_bull", 0)),
            "ob_bear": bool(row.get("ob_bear", 0)),
            "fvg_bull": bool(row.get("fvg_bull", 0)),
            "fvg_bear": bool(row.get("fvg_bear", 0)),
            "sweep_bull": bool(row.get("sweep_bull", 0)),
            "sweep_bear": bool(row.get("sweep_bear", 0)),
        }

    # ------------------------------------------------------------------
    # RSI divergence helpers
    # ------------------------------------------------------------------

    def _is_pivot_low(self, df: pd.DataFrame, idx: int, left: int = 2, right: int = 2) -> bool:
        if idx - left < 0 or idx + right >= len(df):
            return False
        center = float(df.iloc[idx]["low"])
        left_min = float(df.iloc[idx - left:idx]["low"].min())
        right_min = float(df.iloc[idx + 1:idx + 1 + right]["low"].min())
        return center <= left_min and center <= right_min

    def _is_pivot_high(self, df: pd.DataFrame, idx: int, left: int = 2, right: int = 2) -> bool:
        if idx - left < 0 or idx + right >= len(df):
            return False
        center = float(df.iloc[idx]["high"])
        left_max = float(df.iloc[idx - left:idx]["high"].max())
        right_max = float(df.iloc[idx + 1:idx + 1 + right]["high"].max())
        return center >= left_max and center >= right_max

    def _get_recent_pivot_lows(self, df: pd.DataFrame, count: int = 2, lookback: int = 40) -> List[int]:
        start = max(2, len(df) - lookback)
        pivots: List[int] = []
        for i in range(start, len(df) - 2):
            if self._is_pivot_low(df, i):
                pivots.append(i)
        return pivots[-count:]

    def _get_recent_pivot_highs(self, df: pd.DataFrame, count: int = 2, lookback: int = 40) -> List[int]:
        start = max(2, len(df) - lookback)
        pivots: List[int] = []
        for i in range(start, len(df) - 2):
            if self._is_pivot_high(df, i):
                pivots.append(i)
        return pivots[-count:]

    def _score_rsi_divergence(
        self,
        df: pd.DataFrame,
        row: pd.Series,
        side: str,
        setup_family: str,
    ) -> Tuple[float, List[str], Dict[str, Any]]:
        notes: List[str] = []
        meta: Dict[str, Any] = {
            "rsi_divergence_detected": False,
            "rsi_divergence_type": None,
        }

        if setup_family != "reversal":
            return 0.0, notes, meta

        if "rsi_14" not in df.columns or len(df) < 30:
            return 0.0, notes, meta

        score = 0.0

        if side == "LONG":
            pivots = self._get_recent_pivot_lows(df, count=2, lookback=40)
            if len(pivots) < 2:
                return 0.0, notes, meta

            i1, i2 = pivots[-2], pivots[-1]
            p1 = float(df.iloc[i1]["low"])
            p2 = float(df.iloc[i2]["low"])
            r1 = float(df.iloc[i1]["rsi_14"])
            r2 = float(df.iloc[i2]["rsi_14"])
            atr_now = float(row.get("atr_14", 0.0))
            price_now = float(row.get("close", 0.0))

            made_lower_low = p2 < p1
            rsi_higher_low = r2 > r1
            pivot_distance_ok = (i2 - i1) >= 3
            proximity_ok = abs(price_now - p2) <= max(atr_now * 1.2, price_now * 0.003)

            if made_lower_low and rsi_higher_low and pivot_distance_ok and proximity_ok:
                rsi_delta = r2 - r1
                price_delta_pct = (p1 - p2) / p1 if p1 > 0 else 0.0

                score += 0.12
                if rsi_delta >= 4:
                    score += 0.05
                if price_delta_pct >= 0.003:
                    score += 0.03

                notes.append(f"bullish_rsi_divergence_rsi_{r1:.1f}_to_{r2:.1f}")
                notes.append(f"bullish_price_lower_low_{p1:.4f}_to_{p2:.4f}")
                meta.update({
                    "rsi_divergence_detected": True,
                    "rsi_divergence_type": "bullish",
                    "rsi_pivot_idx_1": i1, "rsi_pivot_idx_2": i2,
                    "rsi_pivot_price_1": p1, "rsi_pivot_price_2": p2,
                    "rsi_pivot_rsi_1": r1, "rsi_pivot_rsi_2": r2,
                })

        elif side == "SHORT":
            pivots = self._get_recent_pivot_highs(df, count=2, lookback=40)
            if len(pivots) < 2:
                return 0.0, notes, meta

            i1, i2 = pivots[-2], pivots[-1]
            p1 = float(df.iloc[i1]["high"])
            p2 = float(df.iloc[i2]["high"])
            r1 = float(df.iloc[i1]["rsi_14"])
            r2 = float(df.iloc[i2]["rsi_14"])
            atr_now = float(row.get("atr_14", 0.0))
            price_now = float(row.get("close", 0.0))

            made_higher_high = p2 > p1
            rsi_lower_high = r2 < r1
            pivot_distance_ok = (i2 - i1) >= 3
            proximity_ok = abs(price_now - p2) <= max(atr_now * 1.2, price_now * 0.003)

            if made_higher_high and rsi_lower_high and pivot_distance_ok and proximity_ok:
                rsi_delta = r1 - r2
                price_delta_pct = (p2 - p1) / p1 if p1 > 0 else 0.0

                score += 0.12
                if rsi_delta >= 4:
                    score += 0.05
                if price_delta_pct >= 0.003:
                    score += 0.03

                notes.append(f"bearish_rsi_divergence_rsi_{r1:.1f}_to_{r2:.1f}")
                notes.append(f"bearish_price_higher_high_{p1:.4f}_to_{p2:.4f}")
                meta.update({
                    "rsi_divergence_detected": True,
                    "rsi_divergence_type": "bearish",
                    "rsi_pivot_idx_1": i1, "rsi_pivot_idx_2": i2,
                    "rsi_pivot_price_1": p1, "rsi_pivot_price_2": p2,
                    "rsi_pivot_rsi_1": r1, "rsi_pivot_rsi_2": r2,
                })

        return score, notes, meta

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def _score_volume_context(self, row: pd.Series) -> Tuple[float, List[str]]:
        score = 0.0
        notes: List[str] = []
        vol_spike = bool(row.get("vol_spike", 0))
        vol_collapse = bool(row.get("vol_collapse", 0))
        if vol_spike:
            score += 0.08
            notes.append("vol_confirmed_trigger")
        elif vol_collapse:
            score -= 0.06
            notes.append("low_vol_trigger_penalty")
        return score, notes

    def _score_vwap_magnitude(self, row: pd.Series, side: str, setup_family: str) -> Tuple[float, List[str]]:
        score = 0.0
        notes: List[str] = []
        vwap_dev = float(row.get("vwap_dev", 0.0))
        close = float(row.get("close", 0.0))
        if close <= 0:
            return 0.0, []
        abs_dev_pct = abs(vwap_dev) / close

        if setup_family == "reversal":
            correct_side = (side == "LONG" and vwap_dev < 0) or (side == "SHORT" and vwap_dev > 0)
            if correct_side:
                if abs_dev_pct > 0.020:
                    score += 0.10
                    notes.append(f"deep_vwap_reversal_{abs_dev_pct:.3f}")
                elif abs_dev_pct > 0.010:
                    score += 0.06
                    notes.append(f"moderate_vwap_reversal_{abs_dev_pct:.3f}")
                elif abs_dev_pct > 0.005:
                    score += 0.03
                    notes.append(f"mild_vwap_reversal_{abs_dev_pct:.3f}")
        elif setup_family == "continuation":
            trending_correct = (side == "LONG" and vwap_dev > 0) or (side == "SHORT" and vwap_dev < 0)
            if trending_correct:
                if abs_dev_pct > 0.010:
                    score += 0.04
                    notes.append(f"vwap_trend_strength_{abs_dev_pct:.3f}")
                elif abs_dev_pct > 0.005:
                    score += 0.02
                    notes.append(f"vwap_trend_mild_{abs_dev_pct:.3f}")
        return score, notes

    def _score_trigger_quality(self, row: pd.Series) -> Tuple[float, List[str]]:
        score = 0.0
        notes: List[str] = []
        body_pct = float(row.get("body_pct", 0.5))
        if pd.isna(body_pct):
            body_pct = 0.5
        if body_pct > 0.60:
            score += 0.05
            notes.append(f"strong_body_{body_pct:.2f}")
        elif body_pct < 0.25:
            score -= 0.05
            notes.append(f"weak_body_{body_pct:.2f}")
        return score, notes

    def _score_flow_context(self, flow_snapshot: Dict[str, Any]) -> Tuple[float, List[str]]:
        score = 0.0
        notes: List[str] = []
        if not flow_snapshot:
            return score, notes
        whale_pressure = max(-2.0, min(2.0, float(flow_snapshot.get("whale_pressure", 0.0))))
        flow_momentum = max(-2.0, min(2.0, float(flow_snapshot.get("flow_momentum", 0.0))))
        if whale_pressure > 0.7:
            score += 0.20
            notes.append("whale_pressure_bull_" + str(round(whale_pressure, 2)))
        elif whale_pressure < -0.7:
            score -= 0.20
            notes.append("whale_pressure_bear_" + str(round(whale_pressure, 2)))
        elif whale_pressure > 0.2:
            score += 0.10
            notes.append("whale_pressure_mild_bull")
        elif whale_pressure < -0.2:
            score -= 0.10
            notes.append("whale_pressure_mild_bear")
        if flow_momentum > 0.15:
            score += 0.08
            notes.append("flow_momentum_up")
        elif flow_momentum < -0.15:
            score -= 0.08
            notes.append("flow_momentum_down")
        if "30m" in flow_snapshot:
            imbal_30m = float(flow_snapshot["30m"].get("imbalance", 0.0))
            if imbal_30m > 0.5:
                score += 0.10
                notes.append("net_inflow_30m_" + str(round(imbal_30m, 2)))
            elif imbal_30m < -0.5:
                score -= 0.10
                notes.append("net_outflow_30m_" + str(round(imbal_30m, 2)))
        return score, notes

    def _score_funding_context(self, sentiment: PerpSentimentSnapshot) -> Tuple[float, List[str]]:
        score = 0.0
        notes: List[str] = []
        fr = float(sentiment.funding_rate)
        abs_fr = abs(fr)
        if abs_fr > 0.01:
            bump, tier = 0.15, "extreme"
        elif abs_fr > 0.005:
            bump, tier = 0.08, "high"
        elif abs_fr > 0.001:
            bump, tier = 0.05, "mild"
        else:
            bump, tier = 0.0, ""
        if bump > 0:
            if fr < 0:
                score += bump
                notes.append(f"funding_neg_{tier}")
            else:
                score -= bump
                notes.append(f"funding_pos_{tier}")
        return score, notes

    def _score_oi_directional(self, sentiment: PerpSentimentSnapshot, side: str) -> Tuple[float, List[str]]:
        score = 0.0
        notes: List[str] = []
        oi = float(getattr(sentiment, "open_interest", 0.0) or 0.0)
        prev_oi = float(getattr(sentiment, "prev_open_interest", 0.0) or 0.0)
        if oi > 0 and prev_oi > 0:
            oi_pct = (oi - prev_oi) / prev_oi
            if oi_pct > 0.05:
                score += 0.08
                notes.append(f"oi_rising_{side.lower()}_conviction")
            elif oi_pct < -0.05:
                score -= 0.08
                notes.append(f"oi_falling_{side.lower()}_unwind")
        return score, notes

    def _score_rsi(self, row: pd.Series, side: str, setup_family: str) -> Tuple[float, List[str]]:
        score = 0.0
        notes: List[str] = []
        rsi_val = float(row.get("rsi_14", 50.0))
        if pd.isna(rsi_val):
            return 0.0, []
        if setup_family == "continuation":
            # Neutral RSI (45-65 LONG / 35-55 SHORT) no longer earns a bonus — those ranges
            # cover the entire non-extreme zone and don't discriminate quality setups.
            # Only extreme momentum readings confirm continuation edge.
            if side == "LONG" and rsi_val >= 60:
                score += 0.03
                notes.append(f"rsi_continuation_long_momentum_{rsi_val:.0f}")
            elif side == "SHORT" and rsi_val <= 40:
                score += 0.03
                notes.append(f"rsi_continuation_short_momentum_{rsi_val:.0f}")
        elif setup_family == "reversal":
            if side == "LONG" and rsi_val < 30:
                score += 0.10
                notes.append(f"rsi_oversold_{rsi_val:.0f}")
            elif side == "SHORT" and rsi_val > 70:
                score += 0.10
                notes.append(f"rsi_overbought_{rsi_val:.0f}")
            elif side == "LONG" and rsi_val < 40:
                score += 0.04
                notes.append(f"rsi_near_oversold_{rsi_val:.0f}")
            elif side == "SHORT" and rsi_val > 60:
                score += 0.04
                notes.append(f"rsi_near_overbought_{rsi_val:.0f}")
        return score, notes

    @staticmethod
    def _hard_chop_block_continuation(
        market_regime: str,
        htf_regime: str,
        macro_regime: str,
    ) -> bool:
        _ = (htf_regime, macro_regime)
        return market_regime == "chop"

    def _score_continuation_regime(self, market_regime: str) -> Tuple[float, List[str]]:
        notes: List[str] = []
        if market_regime == "strong_trend":
            return CONT_REGIME_STRONG_BONUS, ["continuation_regime_strong_trend"]
        if market_regime == "weak_trend":
            return CONT_REGIME_WEAK_PENALTY, ["continuation_regime_weak_trend_penalty"]
        if market_regime == "chop":
            return CONT_REGIME_CHOP_PENALTY, ["continuation_regime_chop_penalty"]
        return CONT_REGIME_UNKNOWN_PENALTY, ["continuation_regime_unknown_penalty"]

    def _score_late_entry(
        self,
        row: pd.Series,
        side: str,
    ) -> Tuple[float, List[str], bool]:
        notes: List[str] = []
        open_p = float(row.get("open", 0.0))
        close = float(row.get("close", 0.0))
        high = float(row.get("high", close))
        low = float(row.get("low", close))
        atr_now = float(row.get("atr_14", 0.0))

        if close <= 0 or atr_now <= 0 or open_p <= 0:
            return 0.0, notes, False

        ema_50_raw = row.get("ema_50", None)
        if ema_50_raw is not None and not pd.isna(ema_50_raw):
            ema_50 = float(ema_50_raw)
            if side == "LONG":
                extension_atr = (close - ema_50) / (atr_now + 1e-12)
            else:
                extension_atr = (ema_50 - close) / (atr_now + 1e-12)
            if extension_atr >= CONT_LATE_ENTRY_EXTENSION_HARD_BLOCK_ATR:
                notes.append(
                    f"continuation_hard_block_extension_atr={extension_atr:.2f}"
                )
                return 0.0, notes, True

        body_atr = abs(close - open_p) / (atr_now + 1e-12)
        range_atr = max(0.0, (high - low) / (atr_now + 1e-12))

        directional_body = (
            (side == "LONG" and close > open_p)
            or (side == "SHORT" and close < open_p)
        )

        if directional_body and (
            body_atr >= CONT_LATE_ENTRY_BODY_HARD_BLOCK_ATR
            or range_atr >= CONT_LATE_ENTRY_HARD_BLOCK_RANGE_ATR
        ):
            notes.append(
                f"continuation_hard_block_late_entry body_atr={body_atr:.2f} "
                f"range_atr={range_atr:.2f}"
            )
            return 0.0, notes, True

        if directional_body and body_atr >= CONT_LATE_ENTRY_BODY_PENALTY_ATR:
            notes.append(f"continuation_late_entry_penalty body_atr={body_atr:.2f}")
            return CONT_LATE_ENTRY_PENALTY_SCORE, notes, False

        return 0.0, notes, False

    def _score_pullback_quality(self, row: pd.Series, side: str) -> Tuple[float, List[str]]:
        notes: List[str] = []
        close = float(row.get("close", 0.0))
        atr_now = float(row.get("atr_14", 0.0))
        ema_50_raw = row.get("ema_50", None)

        if close <= 0 or atr_now <= 0 or ema_50_raw is None or pd.isna(ema_50_raw):
            notes.append("continuation_pullback_quality_unavailable")
            return 0.0, notes

        ema_50 = float(ema_50_raw)

        if side == "LONG":
            retrace_atr = (close - ema_50) / (atr_now + 1e-12)
            if retrace_atr < 0:
                notes.append(f"continuation_below_ema50_penalty_atr={retrace_atr:.2f}")
                return CONT_BELOW_EMA_PENALTY, notes
        else:
            retrace_atr = (ema_50 - close) / (atr_now + 1e-12)
            if retrace_atr < 0:
                notes.append(f"continuation_above_ema50_penalty_atr={retrace_atr:.2f}")
                return CONT_BELOW_EMA_PENALTY, notes

        if 0.0 <= retrace_atr < CONT_PULLBACK_SHALLOW_MIN_ATR:
            notes.append(f"continuation_pullback_too_shallow_atr={retrace_atr:.2f}")
            return CONT_PULLBACK_SHALLOW_PENALTY, notes
        if CONT_PULLBACK_SHALLOW_MIN_ATR <= retrace_atr <= CONT_PULLBACK_CLEAN_MAX_ATR:
            notes.append(f"continuation_pullback_clean_atr={retrace_atr:.2f}")
            return CONT_PULLBACK_CLEAN_REWARD, notes
        if CONT_PULLBACK_CLEAN_MAX_ATR < retrace_atr <= CONT_PULLBACK_TIGHT_MAX_ATR:
            notes.append(f"continuation_pullback_tight_atr={retrace_atr:.2f}")
            return CONT_PULLBACK_REWARD, notes
        if retrace_atr >= CONT_PULLBACK_EXTENDED_ATR:
            notes.append(f"continuation_pullback_extended_atr={retrace_atr:.2f}")
            return CONT_PULLBACK_PENALTY * CONT_PULLBACK_EXTENDED_PENALTY_MULT, notes
        notes.append(f"continuation_pullback_sloppy_atr={retrace_atr:.2f}")
        return CONT_PULLBACK_SLOPPY_PENALTY, notes

    def record_directional_outcome(
        self,
        coin: str,
        side: str,
        won: bool,
        setup_family: str = "",
    ) -> None:
        c = str(coin or "").upper().strip()
        s = str(side or "").upper().strip()
        f = str(setup_family or "").strip().lower()
        if not c or s not in {"LONG", "SHORT"}:
            return
        if not f:
            f = "unknown"

        self._directional_outcomes[(c, s, f)].append(
            {
                "won": bool(won),
                "setup_family": f,
                "ts": pd.Timestamp.now(tz="UTC"),
            }
        )

    def _score_directional_loss_memory(
        self,
        coin: str,
        side: str,
        setup_family: str,
    ) -> Tuple[float, List[str]]:
        notes: List[str] = []
        family = str(setup_family or "").strip().lower()
        if family != "continuation":
            return 0.0, notes

        key = (
            str(coin or "").upper().strip(),
            str(side or "").upper().strip(),
            family,
        )
        hist = list(self._directional_outcomes.get(key, []))
        if len(hist) < 3:
            return 0.0, notes

        last_three = hist[-3:]
        if all(not bool(item.get("won", False)) for item in last_three):
            notes.append("continuation_directional_3loss_penalty")
            return CONT_DIRECTIONAL_LOSS_PENALTY, notes
        return 0.0, notes

    def _score_breakout_failure_cluster(
        self,
        df: pd.DataFrame,
        side: str,
    ) -> Tuple[float, List[str]]:
        notes: List[str] = []
        if df is None or len(df) < 6:
            return 0.0, notes

        lookback = min(max(6, CONT_BREAKOUT_FAILURE_LOOKBACK), len(df) - 1)
        hist = df.iloc[-lookback:].reset_index(drop=True)
        failures = 0

        for i in range(len(hist) - 2):
            row_i = hist.iloc[i]
            next_row = hist.iloc[i + 1]
            next2_row = hist.iloc[i + 2]

            atr_i = float(row_i.get("atr_14", 0.0))
            close_i = float(row_i.get("close", 0.0))
            if atr_i <= 0 or close_i <= 0:
                continue

            if side == "LONG":
                trig = bool(row_i.get("bos_bull", 0) or row_i.get("choch_bull", 0))
                if not trig:
                    continue

                breakout_level = float(row_i.get("high", close_i))
                follow_through = max(
                    float(next_row.get("high", breakout_level)),
                    float(next2_row.get("high", breakout_level)),
                )
                pullback_low = min(
                    float(next_row.get("low", breakout_level)),
                    float(next2_row.get("low", breakout_level)),
                )

                weak_follow = (follow_through - breakout_level) < (0.25 * atr_i)
                failed_back_inside = pullback_low < (breakout_level - 0.50 * atr_i)

                if weak_follow or failed_back_inside:
                    failures += 1

            else:
                trig = bool(row_i.get("bos_bear", 0) or row_i.get("choch_bear", 0))
                if not trig:
                    continue

                breakout_level = float(row_i.get("low", close_i))
                follow_through = min(
                    float(next_row.get("low", breakout_level)),
                    float(next2_row.get("low", breakout_level)),
                )
                pullback_high = max(
                    float(next_row.get("high", breakout_level)),
                    float(next2_row.get("high", breakout_level)),
                )

                weak_follow = (breakout_level - follow_through) < (0.25 * atr_i)
                failed_back_inside = pullback_high > (breakout_level + 0.50 * atr_i)

                if weak_follow or failed_back_inside:
                    failures += 1

        if failures >= CONT_BREAKOUT_FAILURE_MIN_COUNT:
            notes.append(f"continuation_breakout_failure_cluster_{failures}")
            return CONT_BREAKOUT_FAILURE_PENALTY, notes

        return 0.0, notes

    def _effective_score_threshold(self, market_regime: str) -> float:
        base = float(self.score_threshold)
        regime = str(market_regime or "").strip().lower()
        if regime == "strong_trend":
            return float(REGIME_SCORE_THRESHOLD_STRONG)
        if regime == "weak_trend":
            return float(REGIME_SCORE_THRESHOLD_WEAK)
        if regime == "chop":
            return float(REGIME_SCORE_THRESHOLD_CHOP)
        return base

    def _fallback_setup_family(
        self,
        coin: str,
        row: pd.Series,
        triggers: Dict[str, bool],
        htf_regime: str,
        macro_regime: str,
        market_regime: str,
        df: Optional[pd.DataFrame] = None,
    ) -> Tuple[Optional[str], float, List[str], str]:
        notes: List[str] = []
        score = 0.0
        side: Optional[str] = None
        setup_family = "none"

        trend = str(row.get("trend", "neutral")).strip().lower()
        trend_up = bool(row.get("trend_up", 0))
        trend_dn = bool(row.get("trend_down", 0))
        local_bull = trend == "bull" or trend_up
        local_bear = trend == "bear" or trend_dn
        dual_up = htf_regime == "up" and macro_regime == "up"
        dual_down = htf_regime == "down" and macro_regime == "down"
        chop_like = (
            market_regime == "chop"
            or htf_regime == "chop"
            or macro_regime == "chop"
        )
        vwap_dev = float(row.get("vwap_dev", 0.0))
        mean_reversion_long = bool(triggers.get("sweep_bull", False)) or vwap_dev < 0
        mean_reversion_short = bool(triggers.get("sweep_bear", False)) or vwap_dev > 0

        ob_or_fvg_bull = bool(triggers.get("ob_bull", False) or triggers.get("fvg_bull", False))
        ob_or_fvg_bear = bool(triggers.get("ob_bear", False) or triggers.get("fvg_bear", False))

        # Continuation LONG: require at least one of htf/macro to be up or neutral (not both down).
        cont_long_regime_ok = not (htf_regime == "down" and macro_regime == "down")
        # Continuation SHORT: require at least one of htf/macro to be down or chop.
        # Prevents SHORT continuation signals when BOTH htf AND macro are bullish ("up").
        cont_short_regime_ok = (htf_regime in {"down", "chop"} or macro_regime in {"down", "chop"})

        if ob_or_fvg_bull and (local_bull or dual_up) and cont_long_regime_ok:
            side = "LONG"
            setup_family = "continuation"
            score = 0.56
            score += 0.03 if local_bull else 0.0
            score += 0.05 if dual_up else 0.0
            score += 0.04 if market_regime == "strong_trend" else 0.0
            notes.append(
                "fallback_family_continuation_ob_bull"
                if bool(triggers.get("ob_bull", False))
                else "fallback_family_continuation_fvg_bull"
            )
        elif ob_or_fvg_bear and (local_bear or dual_down) and cont_short_regime_ok:
            side = "SHORT"
            setup_family = "continuation"
            score = 0.56
            score += 0.03 if local_bear else 0.0
            score += 0.05 if dual_down else 0.0
            score += 0.04 if market_regime == "strong_trend" else 0.0
            notes.append(
                "fallback_family_continuation_ob_bear"
                if bool(triggers.get("ob_bear", False))
                else "fallback_family_continuation_fvg_bear"
            )
        elif ob_or_fvg_bull and chop_like and mean_reversion_long:
            side = "LONG"
            setup_family = "reversal"
            score = 0.54
            score += 0.05 if chop_like else 0.0
            score += 0.03 if mean_reversion_long else 0.0
            notes.append(
                "fallback_family_reversal_ob_bull"
                if bool(triggers.get("ob_bull", False))
                else "fallback_family_reversal_fvg_bull"
            )
        elif ob_or_fvg_bear and chop_like and mean_reversion_short:
            side = "SHORT"
            setup_family = "reversal"
            score = 0.54
            score += 0.05 if chop_like else 0.0
            score += 0.03 if mean_reversion_short else 0.0
            notes.append(
                "fallback_family_reversal_ob_bear"
                if bool(triggers.get("ob_bear", False))
                else "fallback_family_reversal_fvg_bear"
            )
        elif bool(triggers.get("choch_bull", False)):
            side = "LONG"
            setup_family = "reversal"
            score = 0.57
            score += 0.03 if mean_reversion_long else 0.0
            notes.append("fallback_family_reversal_choch_bull")
        elif bool(triggers.get("choch_bear", False)):
            side = "SHORT"
            setup_family = "reversal"
            score = 0.57
            score += 0.03 if mean_reversion_short else 0.0
            notes.append("fallback_family_reversal_choch_bear")

        if side is None:
            return None, 0.0, notes, setup_family

        if setup_family == "continuation":
            if self._hard_chop_block_continuation(market_regime, htf_regime, macro_regime):
                notes.append("fallback_continuation_blocked_hard_chop_stack")
                return None, 0.0, notes, "none"

            if market_regime == "weak_trend" and CONT_WEAK_TREND_BLOCK_WITHOUT_DUAL_HTF:
                dual_aligned = (
                    (side == "LONG" and htf_regime == "up" and macro_regime == "up")
                    or (side == "SHORT" and htf_regime == "down" and macro_regime == "down")
                )
                if not dual_aligned:
                    notes.append("fallback_continuation_blocked_weak_trend_without_dual_htf_alignment")
                    return None, 0.0, notes, "none"

            regime_s, regime_n = self._score_continuation_regime(market_regime)
            score += regime_s
            notes.extend(regime_n)

            rsi_s, rsi_n = self._score_rsi(row, side, "continuation")
            score += rsi_s
            notes.extend(rsi_n)

            vol_s, vol_n = self._score_volume_context(row)
            score += vol_s
            notes.extend(vol_n)

            body_s, body_n = self._score_trigger_quality(row)
            score += body_s
            notes.extend(body_n)

            vwap_s, vwap_n = self._score_vwap_magnitude(row, side, "continuation")
            score += vwap_s
            notes.extend(vwap_n)

            late_s, late_n, hard_block = self._score_late_entry(row, side)
            if hard_block:
                notes.append("fallback_late_entry_block")
                return None, 0.0, notes + late_n, "none"
            score += late_s
            notes.extend(late_n)

            pb_s, pb_n = self._score_pullback_quality(row, side)
            score += pb_s
            notes.extend(pb_n)

            mem_s, mem_n = self._score_directional_loss_memory(coin, side, "continuation")
            score += mem_s
            notes.extend(mem_n)
            if mem_s < 0:
                notes.append("fallback_directional_memory_penalty")

            if df is not None and not df.empty:
                bof_s, bof_n = self._score_breakout_failure_cluster(df, side)
                score += bof_s
                notes.extend(bof_n)
        else:
            rsi_s, rsi_n = self._score_rsi(row, side, "reversal")
            score += rsi_s
            notes.extend(rsi_n)

            vol_s, vol_n = self._score_volume_context(row)
            score += vol_s
            notes.extend(vol_n)

            body_s, body_n = self._score_trigger_quality(row)
            score += body_s
            notes.extend(body_n)

            vwap_s, vwap_n = self._score_vwap_magnitude(row, side, "reversal")
            score += vwap_s
            notes.extend(vwap_n)

        return side, score, notes, setup_family

    # ------------------------------------------------------------------
    # Setup families
    # ------------------------------------------------------------------

    def _build_continuation_signal(
        self,
        coin: str,
        row: pd.Series,
        triggers: Dict[str, bool],
        htf_regime: str,
        macro_regime: str,
        market_regime: str,
        flow_snapshot: Dict[str, Any],
        sentiment: PerpSentimentSnapshot,
        df: Optional[pd.DataFrame] = None,
    ) -> Tuple[Optional[str], float, List[str]]:
        if self._hard_chop_block_continuation(market_regime, htf_regime, macro_regime):
            return None, 0.0, ["continuation_blocked_hard_chop_stack"]

        htf_chop_penalty = -0.08
        macro_chop_penalty = -0.03
        disagreement_penalty = -0.08

        notes: List[str] = []
        score = 0.0
        side: Optional[str] = None

        trend = row.get("trend", "neutral")
        trend_up = bool(row.get("trend_up", 0))
        trend_dn = bool(row.get("trend_down", 0))

        flow_score, flow_notes = self._score_flow_context(flow_snapshot)
        funding_score, funding_notes = self._score_funding_context(sentiment)

        bullish_trigger = triggers["bos_bull"] or triggers["ob_bull"] or triggers["fvg_bull"]
        bearish_trigger = triggers["bos_bear"] or triggers["ob_bear"] or triggers["fvg_bear"]

        bullish_context = (trend == "bull" or trend_up) and (htf_regime == "up" or macro_regime == "up")
        bearish_context = (trend == "bear" or trend_dn) and (htf_regime == "down" or macro_regime == "down")

        if bullish_trigger and bullish_context:
            side = "LONG"
            score += 0.45
            notes.append("continuation_bull_trigger")
            if htf_regime == "up":
                score += 0.20
                notes.append("htf_up_aligned")
            elif htf_regime == "chop":
                score += htf_chop_penalty
                notes.append("htf_chop_penalty_continuation")

            if macro_regime == "up":
                score += 0.15
                notes.append("macro_up_aligned")
            elif macro_regime == "chop":
                score += macro_chop_penalty
                notes.append("macro_chop_penalty_continuation")

            if htf_regime == "up" and macro_regime == "down":
                score += disagreement_penalty
                notes.append("htf_macro_disagreement_penalty")

            if trend == "bull" or trend_up:
                score += 0.10
                notes.append("local_bull_trend")

            score += flow_score
            score += funding_score
            oi_score, oi_notes = self._score_oi_directional(sentiment, side)
            score += oi_score
            notes.extend(flow_notes)
            notes.extend(funding_notes)
            notes.extend(oi_notes)

        elif bearish_trigger and bearish_context:
            side = "SHORT"
            score += 0.45
            notes.append("continuation_bear_trigger")
            if htf_regime == "down":
                score += 0.20
                notes.append("htf_down_aligned")
            elif htf_regime == "chop":
                score += htf_chop_penalty
                notes.append("htf_chop_penalty_continuation")

            if macro_regime == "down":
                score += 0.15
                notes.append("macro_down_aligned")
            elif macro_regime == "chop":
                score += macro_chop_penalty
                notes.append("macro_chop_penalty_continuation")

            if htf_regime == "down" and macro_regime == "up":
                score += disagreement_penalty
                notes.append("htf_macro_disagreement_penalty")

            if trend == "bear" or trend_dn:
                score += 0.10
                notes.append("local_bear_trend")

            score += -flow_score
            score += -funding_score
            oi_score, oi_notes = self._score_oi_directional(sentiment, side)
            score += oi_score
            notes.extend(flow_notes)
            notes.extend(funding_notes)
            notes.extend(oi_notes)

        if side is not None:
            if market_regime == "weak_trend" and CONT_WEAK_TREND_BLOCK_WITHOUT_DUAL_HTF:
                dual_aligned = (
                    (side == "LONG" and htf_regime == "up" and macro_regime == "up")
                    or (side == "SHORT" and htf_regime == "down" and macro_regime == "down")
                )
                if not dual_aligned:
                    notes.append("continuation_blocked_weak_trend_without_dual_htf_alignment")
                    return None, 0.0, notes

            regime_s, regime_n = self._score_continuation_regime(market_regime)
            score += regime_s
            notes.extend(regime_n)

            rsi_s, rsi_n = self._score_rsi(row, side, "continuation")
            score += rsi_s
            notes.extend(rsi_n)

            vol_s, vol_n = self._score_volume_context(row)
            score += vol_s
            notes.extend(vol_n)

            body_s, body_n = self._score_trigger_quality(row)
            score += body_s
            notes.extend(body_n)

            vwap_s, vwap_n = self._score_vwap_magnitude(row, side, "continuation")
            score += vwap_s
            notes.extend(vwap_n)

            late_s, late_n, hard_block = self._score_late_entry(row, side)
            if hard_block:
                return None, 0.0, notes + late_n
            score += late_s
            notes.extend(late_n)

            pb_s, pb_n = self._score_pullback_quality(row, side)
            score += pb_s
            notes.extend(pb_n)

            mem_s, mem_n = self._score_directional_loss_memory(coin, side, "continuation")
            score += mem_s
            notes.extend(mem_n)

            if df is not None and not df.empty:
                bof_s, bof_n = self._score_breakout_failure_cluster(df, side)
                score += bof_s
                notes.extend(bof_n)

        return side, score, notes

    def _build_reversal_signal(
        self,
        row: pd.Series,
        triggers: Dict[str, bool],
        htf_regime: str,
        macro_regime: str,
        flow_snapshot: Dict[str, Any],
        sentiment: PerpSentimentSnapshot,
    ) -> Tuple[Optional[str], float, List[str]]:
        if htf_regime == "chop" and macro_regime == "down":
            return None, 0.0, ["reversal_blocked_chop_bear"]

        notes: List[str] = []
        score = 0.0
        side: Optional[str] = None

        vwap_dev = float(row.get("vwap_dev", 0.0))
        flow_score, flow_notes = self._score_flow_context(flow_snapshot)
        funding_score, funding_notes = self._score_funding_context(sentiment)

        bullish_trigger = (
            triggers["sweep_bull"] or triggers["choch_bull"] or
            triggers["ob_bull"] or triggers["fvg_bull"]
        )
        bearish_trigger = (
            triggers["sweep_bear"] or triggers["choch_bear"] or
            triggers["ob_bear"] or triggers["fvg_bear"]
        )

        bullish_context = vwap_dev < 0
        bearish_context = vwap_dev > 0
        allow_bull = (htf_regime != "down") or (flow_score > 0.25)
        allow_bear = (htf_regime != "up") or (-flow_score > 0.25)

        if bullish_trigger and bullish_context and allow_bull:
            side = "LONG"
            score += 0.45
            notes.append("reversal_bull_trigger")
            notes.append("below_vwap_reclaim_candidate")
            if htf_regime == "up":
                score += 0.15; notes.append("htf_up_supportive")
            elif htf_regime == "chop":
                score += 0.08; notes.append("htf_chop_allows_reversal")
            score += flow_score
            score += funding_score
            oi_score, oi_notes = self._score_oi_directional(sentiment, side)
            if oi_score > 0:
                score += min(0.10, oi_score + 0.02); notes.append("oi_supports_reversal")
            else:
                score += oi_score
            notes.extend(flow_notes); notes.extend(funding_notes); notes.extend(oi_notes)

        elif bearish_trigger and bearish_context and allow_bear:
            side = "SHORT"
            score += 0.45
            notes.append("reversal_bear_trigger")
            notes.append("above_vwap_rejection_candidate")
            if htf_regime == "down":
                score += 0.15; notes.append("htf_down_supportive")
            elif htf_regime == "chop":
                score += 0.08; notes.append("htf_chop_allows_reversal")
            score += -flow_score
            score += -funding_score
            oi_score, oi_notes = self._score_oi_directional(sentiment, side)
            if oi_score > 0:
                score += min(0.10, oi_score + 0.02); notes.append("oi_supports_reversal")
            else:
                score += oi_score
            notes.extend(flow_notes); notes.extend(funding_notes); notes.extend(oi_notes)

        if side is not None:
            rsi_s, rsi_n = self._score_rsi(row, side, "reversal")
            score += rsi_s; notes.extend(rsi_n)
            vol_s, vol_n = self._score_volume_context(row)
            score += vol_s; notes.extend(vol_n)
            body_s, body_n = self._score_trigger_quality(row)
            score += body_s; notes.extend(body_n)
            vwap_s, vwap_n = self._score_vwap_magnitude(row, side, "reversal")
            score += vwap_s; notes.extend(vwap_n)

        return side, score, notes

    # ------------------------------------------------------------------
    # Quality filter
    # ------------------------------------------------------------------

    def _quality_filter(self, row: pd.Series, session_label: str, vol_state: str) -> Tuple[bool, List[str]]:
        notes: List[str] = []

        if vol_state == "unknown":
            notes.append("unknown_vol_block")
            if self.debug:
                print(f"[QUALITY_DEBUG] blocked | session={session_label} | vol_state={vol_state}"
                      f" | close={float(row.get('close', 0.0)):.6f}"
                      f" | atr={float(row.get('atr_14', 0.0)):.6f}"
                      f" | rsi={float(row.get('rsi_14', 50.0)):.2f}")
            return False, notes

        if vol_state == "low":
            notes.append("low_vol_penalty")
            if self.debug:
                print(f"[QUALITY_DEBUG] low_vol_penalty (not blocked) | session={session_label}"
                      f" | close={float(row.get('close', 0.0)):.6f}"
                      f" | atr={float(row.get('atr_14', 0.0)):.6f}"
                      f" | rsi={float(row.get('rsi_14', 50.0)):.2f}")

        if session_label == "dead_zone":
            notes.append("dead_zone_penalty_only")
            if self.debug:
                print(f"[QUALITY_DEBUG] pass_with_penalty | session={session_label} | vol_state={vol_state}"
                      f" | close={float(row.get('close', 0.0)):.6f}"
                      f" | atr={float(row.get('atr_14', 0.0)):.6f}"
                      f" | rsi={float(row.get('rsi_14', 50.0)):.2f}")
            return True, notes

        notes.append("quality_pass")
        if self.debug:
            print(f"[QUALITY_DEBUG] pass | session={session_label} | vol_state={vol_state}"
                  f" | close={float(row.get('close', 0.0)):.6f}"
                  f" | atr={float(row.get('atr_14', 0.0)):.6f}"
                  f" | rsi={float(row.get('rsi_14', 50.0)):.2f}")
        return True, notes

    # ------------------------------------------------------------------
    # Trade level construction
    # ------------------------------------------------------------------

    def _build_trade_levels(
        self,
        side: str,
        price: float,
        row: pd.Series,
        df: pd.DataFrame,
    ) -> Tuple[Optional[float], Optional[float], float, float, float]:
        atr_val = float(row.get("atr_14", 0.0))
        if atr_val <= 0:
            return None, None, 0.0, 0.0, 0.0

        n = len(df)
        lookback = df.iloc[max(0, n - 3):]
        recent_low = float(lookback["low"].min())
        recent_high = float(lookback["high"].max())

        stop_dist_atr = self.atr_stop_mult * atr_val
        tp_dist_atr = self.atr_tp_mult * atr_val
        stop_dist_min = price * self.min_stop_pct
        tp_dist_min = price * self.min_tp_pct

        stop_dist = max(stop_dist_atr, stop_dist_min)
        tp_dist = max(tp_dist_atr, tp_dist_min)

        if side == "LONG":
            structure_stop = min(price - stop_dist, recent_low - 0.25 * atr_val)
            stop = structure_stop
            tp = price + tp_dist
            actual_stop_dist = price - stop
        else:
            structure_stop = max(price + stop_dist, recent_high + 0.25 * atr_val)
            stop = structure_stop
            tp = price - tp_dist
            actual_stop_dist = stop - price

        if actual_stop_dist <= 0:
            return None, None, atr_val, stop_dist, tp_dist

        if tp_dist / actual_stop_dist < 1.8:
            tp_dist = actual_stop_dist * 2.0
            tp = price + tp_dist if side == "LONG" else price - tp_dist

        return stop, tp, atr_val, actual_stop_dist, tp_dist

    # ------------------------------------------------------------------
    # Main 15m signal
    # ------------------------------------------------------------------

    def generate_signal(
        self,
        df_ohlcv: pd.DataFrame,
        flow_snapshot: Dict[str, Any],
        sentiment: PerpSentimentSnapshot,
        coin: str = "UNKNOWN",
    ) -> Optional[Signal]:
        df = self._build_feature_frame(df_ohlcv, min_len=200)
        if df.empty:
            if self.debug:
                print("[SIGNAL_DEBUG] Feature frame empty.")
            return None

        row = df.iloc[-1]
        price = float(row.get("close", 0.0))
        ts = row.get("time", None)

        htf_regime, notes_htf = self._compute_htf_regime(df_ohlcv)
        macro_regime, notes_macro = self._compute_macro_regime_4h(df_ohlcv)
        session_label, notes_kz = self._compute_killzone(ts)
        vol_state, notes_vol, vol_ratio = self._compute_vol_state(row)
        triggers = self._extract_triggers(row)
        market_regime, market_meta, market_notes = self._classify_market_regime(
            df=df,
            row=row,
            triggers=triggers,
            htf_regime=htf_regime,
            macro_regime=macro_regime,
        )

        if self.debug:
            print(
                f"[CANDIDATE_DEBUG] {coin}"
                f" | session={session_label}"
                f" | vol_state={vol_state}"
                f" | mkt={market_regime}"
                f" | htf={htf_regime}"
                f" | macro={macro_regime}"
                f" | close={price:.6f}"
                f" | atr={float(row.get('atr_14', 0.0)):.6f}"
                f" | rsi={float(row.get('rsi_14', 50.0)):.2f}"
                f" | vwap_dev={float(row.get('vwap_dev', 0.0)):.6f}"
                f" | body_pct={float(row.get('body_pct', 0.0)):.2f}"
                f" | triggers={triggers}"
            )

        # london_open bonus removed for continuation — historical data shows 0.95-conf
        # continuation signals in london_open have 8% WR / -8.88R (12 trades).
        # ny_pm penalty increased: 0% WR across all confidence tiers (14 trades, -15R).
        # ny_open bonus retained — 100% WR on limited sample.
        session_score_adj = {
            "london_open": 0.00,
            "ny_open": +0.05,
            "asia_open": 0.00,
            "london_late": 0.00,
            "asia_late": 0.00,
            "ny_pm": -0.05,
            "dead_zone": -0.05,
        }.get(session_label, 0.0)

        quality_ok, quality_notes = self._quality_filter(row, session_label, vol_state)
        if not quality_ok:
            if self.debug:
                print("[SIGNAL_DEBUG] Quality blocked: " + ", ".join(quality_notes))
            return None

        cont_side, cont_score, cont_notes = self._build_continuation_signal(
            coin=coin,
            row=row,
            triggers=triggers,
            htf_regime=htf_regime,
            macro_regime=macro_regime,
            market_regime=market_regime,
            flow_snapshot=flow_snapshot,
            sentiment=sentiment,
            df=df,
        )
        rev_side, rev_score, rev_notes = self._build_reversal_signal(
            row, triggers, htf_regime, macro_regime, flow_snapshot, sentiment,
        )

        chosen_side: Optional[str] = None
        chosen_score: float = 0.0
        chosen_notes: List[str] = []
        setup_family: str = "none"
        continuation_preference = 0.10

        if cont_side and cont_score >= (rev_score - continuation_preference):
            chosen_side, chosen_score, chosen_notes, setup_family = cont_side, cont_score, cont_notes, "continuation"
        elif rev_side:
            chosen_side, chosen_score, chosen_notes, setup_family = rev_side, rev_score, rev_notes, "reversal"

        if chosen_side is None:
            fb_side, fb_score, fb_notes, fb_family = self._fallback_setup_family(
                coin=coin,
                row=row,
                triggers=triggers,
                htf_regime=htf_regime,
                macro_regime=macro_regime,
                market_regime=market_regime,
                df=df,
            )
            if fb_side is None:
                if self.debug:
                    print("[SIGNAL_DEBUG] No valid setup family.")
                return None
            chosen_side, chosen_score, chosen_notes, setup_family = (
                fb_side,
                fb_score,
                fb_notes,
                fb_family,
            )
            if self.debug:
                print(
                    f"[SIGNAL_DEBUG] fallback_setup_family={setup_family}"
                    f" side={chosen_side} score={chosen_score:.3f}"
                    f" reasons={chosen_notes}"
                )

        divergence_meta: Dict[str, Any] = {"rsi_divergence_detected": False, "rsi_divergence_type": None}

        if setup_family == "reversal":
            div_score, div_notes, divergence_meta = self._score_rsi_divergence(
                df=df, row=row, side=chosen_side, setup_family=setup_family,
            )
            chosen_score += div_score
            chosen_notes.extend(div_notes)

        chosen_score += session_score_adj
        if session_score_adj != 0.0 and self.debug:
            print(f"[SIGNAL_DEBUG] session_adj={session_score_adj:+.2f} -> score={chosen_score:.3f}")

        if vol_state == "low":
            chosen_score += LOW_VOL_SCORE_PENALTY
            chosen_notes.append("low_vol_score_penalty")
            if self.debug:
                print(
                    f"[SIGNAL_DEBUG] low_vol_score_penalty {LOW_VOL_SCORE_PENALTY:+.2f}"
                    f" -> score={chosen_score:.3f}"
                )

        # Additional penalty: continuation signals in london_open consistently over-fire.
        # HTF-aligned setups accumulate max bonuses but face stop-hunts at the open.
        if setup_family == "continuation" and session_label == "london_open":
            chosen_score -= 0.05
            chosen_notes.append("london_open_continuation_penalty")
            if self.debug:
                print(f"[SIGNAL_DEBUG] london_open_continuation_penalty -0.05 -> score={chosen_score:.3f}")

        effective_threshold = self._effective_score_threshold(market_regime)

        if self.debug:
            print(
                f"[SCORE_DEBUG] {coin}"
                f" | chosen_side={chosen_side}"
                f" | family={setup_family}"
                f" | score={chosen_score:.3f}"
                f" | threshold={effective_threshold:.3f}"
                f" | base_threshold={float(self.score_threshold):.3f}"
                f" | market_regime={market_regime}"
                f" | session_adj={session_score_adj:+.2f}"
                f" | reasons={chosen_notes}"
            )

        if abs(chosen_score) < effective_threshold:
            if self.debug:
                print("[SIGNAL_DEBUG] score=" + str(round(chosen_score, 3)) +
                      " < threshold=" + str(round(effective_threshold, 3)))
            return None

        stop, tp, atr_val, stop_dist, tp_dist = self._build_trade_levels(
            side=chosen_side, price=price, row=row, df=df,
        )
        if stop is None or tp is None:
            if self.debug:
                print("[SIGNAL_DEBUG] Could not build valid trade levels.")
            return None

        rr = abs((tp - price) / (price - stop)) if price != stop else 0.0
        rr_floor = 1.8
        rr_floor_tolerance = 0.05
        rr_floor_effective = max(0.0, rr_floor - rr_floor_tolerance)
        if rr < rr_floor_effective:
            if self.debug:
                print("[SIGNAL_DEBUG] RR too low: " + str(round(rr, 2)))
            return None

        confidence = round(min(0.95, max(0.50, chosen_score)), 3)
        breakout_failure_note = next(
            (
                n for n in chosen_notes
                if str(n).startswith("continuation_breakout_failure_cluster_")
            ),
            "",
        )
        breakout_failure_count = 0
        if breakout_failure_note:
            parts = str(breakout_failure_note).rsplit("_", 1)
            if len(parts) == 2:
                try:
                    breakout_failure_count = int(parts[1])
                except ValueError:
                    breakout_failure_count = 0
        all_reasons = (
            chosen_notes
            + notes_htf
            + notes_macro
            + notes_kz
            + notes_vol
            + market_notes
            + quality_notes
        )
        reason_text = ", ".join(all_reasons)
        combined_regime = (
            setup_family
            + "|htf_" + htf_regime
            + "|macro_" + macro_regime
            + "|mkt_" + market_regime
        )

        if self.debug:
            print(
                f"[DEDUP_DEBUG] {coin}"
                f" | side={chosen_side}"
                f" | family={setup_family}"
                f" | regime={combined_regime}"
                f" | entry={price:.6f}"
                f" | atr={atr_val:.6f}"
            )

        if self._is_stale_repeat(
            coin=coin, side=chosen_side, setup_family=setup_family,
            entry_price=price, regime=combined_regime, atr=atr_val,
            market_regime=market_regime, current_score=chosen_score,
        ):
            if self.debug:
                print(f"[SIGNAL_DEBUG] {coin} {chosen_side} suppressed — "
                      f"dedup_active_for_repeat")
            return None

        meta = {
            "timeframe": "15m",
            "coin": coin,
            "total_score": round(chosen_score, 3),
            "regime_local": setup_family,
            "regime_htf_1h": htf_regime,
            "regime_macro_4h": macro_regime,
            "session": session_label,
            "close": price,
            "atr": atr_val,
            "stop_dist": stop_dist,
            "tp_dist": tp_dist,
            "effective_threshold": round(effective_threshold, 3),
            "vol_state": vol_state,
            "vol_ratio": vol_ratio,
            "setup_family": setup_family,
            "rr_planned": round(rr, 3),
            "session_score_adj": session_score_adj,
            "market_regime": market_regime,
            "continuation_breakout_failures": breakout_failure_count,
            "continuation_breakout_failure_penalized": bool(breakout_failure_note),
            **market_meta,
            **divergence_meta,
        }

        if self.debug:
            print("[SIGNAL_DEBUG] " + coin + " " + chosen_side +
                  " score=" + str(round(chosen_score, 3)) +
                  " rr=" + str(round(rr, 2)) +
                  " session=" + session_label +
                  " vol=" + vol_state)

        return Signal(
            coin=coin,
            side=chosen_side,
            entry_price=price,
            stop_price=stop,
            tp_price=tp,
            confidence=confidence,
            regime=combined_regime,
            reason=reason_text,
            meta=meta,
        )

    # ------------------------------------------------------------------
    # Swing signal (1h / 4h)
    # ------------------------------------------------------------------

    def generate_swing_signal(
        self,
        df_ohlcv: pd.DataFrame,
        flow_snapshot: Dict[str, Any],
        sentiment: PerpSentimentSnapshot,
        swing_tf: str = "1h",
        coin: str = "UNKNOWN",
    ) -> Optional[Signal]:
        if swing_tf not in ("1h", "4h"):
            return None
        if df_ohlcv is None or df_ohlcv.empty:
            return None

        df_raw = df_ohlcv.copy()
        if "time" in df_raw.columns:
            df_raw["time"] = pd.to_datetime(df_raw["time"])
            df_raw = df_raw.set_index("time")
        elif not isinstance(df_raw.index, pd.DatetimeIndex):
            return None

        rule = "h" if swing_tf == "1h" else "4h"
        try:
            df_tf = (
                df_raw.resample(rule)
                .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
                .dropna()
            )
        except Exception as e:
            if self.debug:
                print("[SWING_DEBUG] Resample " + swing_tf + " failed: " + str(e))
            return None

        min_raw_len = 50 if swing_tf == "1h" else 18
        if len(df_tf) < min_raw_len:
            if self.debug:
                print("[SWING_DEBUG] Not enough " + swing_tf + " bars (" + str(len(df_tf)) + ")")
            return None

        df_tf = df_tf.reset_index()
        df_feat = self._build_feature_frame(df_tf, min_len=min_raw_len)
        if df_feat.empty:
            return None

        row = df_feat.iloc[-1]
        ts = pd.to_datetime(row.get("time"))
        price = float(row.get("close", 0.0))

        swing_key = (str(coin).upper().strip(), swing_tf)
        last_ts = self.last_swing_ts.get(swing_key)
        if last_ts is not None and ts <= last_ts:
            return None

        triggers = self._extract_triggers(row)
        htf_regime, _ = self._compute_htf_regime(df_ohlcv)
        macro_regime, _ = self._compute_macro_regime_4h(df_ohlcv)
        vol_state, _, vol_ratio = self._compute_vol_state(row)
        market_regime, market_meta, market_notes = self._classify_market_regime(
            df=df_feat,
            row=row,
            triggers=triggers,
            htf_regime=htf_regime,
            macro_regime=macro_regime,
        )
        flow_score, flow_notes = self._score_flow_context(flow_snapshot)
        funding_score, funding_notes = self._score_funding_context(sentiment)

        side: Optional[str] = None
        score = 0.0
        reasons: List[str] = []

        if triggers["bos_bull"] or triggers["choch_bull"] or triggers["ob_bull"]:
            if htf_regime == "up" or macro_regime == "up":
                side = "LONG"
                score = 0.65
                score += flow_score
                score += funding_score
                oi_score, oi_notes = self._score_oi_directional(sentiment, side)
                score += oi_score
                reasons.append("swing_bull_setup")
                reasons.extend(flow_notes); reasons.extend(funding_notes); reasons.extend(oi_notes)
        elif triggers["bos_bear"] or triggers["choch_bear"] or triggers["ob_bear"]:
            if htf_regime == "down" or macro_regime == "down":
                side = "SHORT"
                score = 0.65
                score += -flow_score
                score += -funding_score
                oi_score, oi_notes = self._score_oi_directional(sentiment, side)
                score += oi_score
                reasons.append("swing_bear_setup")
                reasons.extend(flow_notes); reasons.extend(funding_notes); reasons.extend(oi_notes)

        if side is None:
            if self.debug:
                print("[SWING_DEBUG] No valid " + swing_tf + " swing setup.")
            return None

        vol_s, vol_n = self._score_volume_context(row)
        score += vol_s; reasons.extend(vol_n)
        body_s, body_n = self._score_trigger_quality(row)
        score += body_s; reasons.extend(body_n)
        reasons.extend(market_notes)

        threshold = self.swing_thresholds.get(swing_tf, 0.6)
        if score < threshold:
            if self.debug:
                print("[SWING_DEBUG] " + swing_tf + " score " + str(round(score, 3)) + " below threshold " + str(threshold))
            return None

        stop, tp, atr_val, stop_dist, tp_dist = self._build_trade_levels(
            side=side, price=price, row=row, df=df_feat,
        )
        if stop is None or tp is None:
            return None

        confidence = round(min(0.95, max(0.55, score)), 3)
        combined_regime = (
            "swing_" + swing_tf
            + "|htf_" + htf_regime
            + "|macro_" + macro_regime
            + "|mkt_" + market_regime
        )
        meta = {
            "timeframe": swing_tf, "coin": coin, "total_score": round(score, 3),
            "regime_local": "swing", "regime_htf_1h": htf_regime, "regime_macro_4h": macro_regime,
            "close": price, "atr": atr_val, "stop_dist": stop_dist, "tp_dist": tp_dist,
            "effective_threshold": round(threshold, 3), "vol_state": vol_state, "vol_ratio": vol_ratio,
            "market_regime": market_regime,
            **market_meta,
        }

        self.last_swing_ts[swing_key] = ts

        return Signal(
            coin=coin, side=side, entry_price=price, stop_price=stop, tp_price=tp,
            confidence=confidence, regime=combined_regime, reason=", ".join(reasons), meta=meta,
        )

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

import pandas as pd

from features import add_features
from smc_structure import build_structure
from smc_zones import add_smc_zones
from smc_sweeps import add_sweep_features
from perp_sentiment import PerpSentimentSnapshot


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
    Ship-version signal engine.

    Philosophy:
    - Keep product logic explainable
    - Use 15m for execution
    - Use 1h / 4h as context filters
    - Support only two setup families:
        1. continuation
        2. reversal
    """

    def __init__(
        self,
        score_threshold: float = 0.80,
        atr_stop_mult: float = 1.3,
        atr_tp_mult: float = 3.0,
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

        self.last_swing_ts: Dict[str, Optional[pd.Timestamp]] = {
            "1h": None,
            "4h": None,
        }

    # ------------------------------------------------------------------
    # Feature frame
    # ------------------------------------------------------------------

    def _build_feature_frame(self, df_ohlcv: pd.DataFrame, min_len: int = 200) -> pd.DataFrame:
        if df_ohlcv.empty or len(df_ohlcv) < min_len:
            return pd.DataFrame()

        df = df_ohlcv.copy()

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
            "atr_14", "vwap_dev",
        ]

        missing_required = [c for c in required_numeric if c not in df.columns]
        if missing_required:
            if self.debug:
                print("[FEATURE_FRAME_DEBUG] Missing required columns: " + str(missing_required))
            return pd.DataFrame()

        df = df.dropna(subset=required_numeric).reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # Regime helpers
    # ------------------------------------------------------------------

    def _compute_htf_regime(self, df_ohlcv: pd.DataFrame) -> Tuple[str, List[str]]:
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

        hour = pd.to_datetime(ts).hour

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
    # Flow + sentiment scoring
    # ------------------------------------------------------------------

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

    def _score_funding_context(
        self,
        sentiment: PerpSentimentSnapshot,
    ) -> Tuple[float, List[str]]:
        """
        Contrarian funding score:
          negative funding -> helps LONG setups
          positive funding -> helps SHORT setups

        Magnitude tiers:
          |fr| > 0.01  -> 0.15
          |fr| > 0.005 -> 0.08
          |fr| > 0.001 -> 0.05
        """
        score = 0.0
        notes: List[str] = []

        fr = float(sentiment.funding_rate)
        abs_fr = abs(fr)

        if abs_fr > 0.01:
            bump = 0.15
            tier = "extreme"
        elif abs_fr > 0.005:
            bump = 0.08
            tier = "high"
        elif abs_fr > 0.001:
            bump = 0.05
            tier = "mild"
        else:
            bump = 0.0
            tier = ""

        if bump > 0:
            if fr < 0:
                score += bump
                notes.append(f"funding_neg_{tier}")
            else:
                score -= bump
                notes.append(f"funding_pos_{tier}")

        return score, notes

    def _score_oi_directional(
        self,
        sentiment: PerpSentimentSnapshot,
        side: str,
    ) -> Tuple[float, List[str]]:
        """
        Direction-aware open interest scoring.

        Rising OI:
          supports active participation / conviction in the chosen setup.
        Falling OI:
          penalizes setups due to likely unwind / weak participation.

        This is intentionally symmetric by side.
        """
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

    def _score_rsi(
        self,
        row: pd.Series,
        side: str,
        setup_family: str,
    ) -> Tuple[float, List[str]]:
        score = 0.0
        notes: List[str] = []

        rsi_val = float(row.get("rsi_14", 50.0))
        if pd.isna(rsi_val):
            return 0.0, []

        if setup_family == "continuation":
            if side == "LONG" and 45 <= rsi_val <= 65:
                score += 0.05
                notes.append(f"rsi_continuation_long_{rsi_val:.0f}")
            elif side == "SHORT" and 35 <= rsi_val <= 55:
                score += 0.05
                notes.append(f"rsi_continuation_short_{rsi_val:.0f}")

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

    # ------------------------------------------------------------------
    # Setup families
    # ------------------------------------------------------------------

    def _build_continuation_signal(
        self,
        row: pd.Series,
        triggers: Dict[str, bool],
        htf_regime: str,
        macro_regime: str,
        flow_snapshot: Dict[str, Any],
        sentiment: PerpSentimentSnapshot,
    ) -> Tuple[Optional[str], float, List[str]]:
        if htf_regime == "chop":
            return None, 0.0, ["continuation_blocked_htf_chop"]

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
            if macro_regime == "up":
                score += 0.15
                notes.append("macro_up_aligned")
            if trend == "bull" or trend_up:
                score += 0.10
                notes.append("local_bull_trend")

            score += max(0.0, flow_score)
            score += max(0.0, funding_score)

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
            if macro_regime == "down":
                score += 0.15
                notes.append("macro_down_aligned")
            if trend == "bear" or trend_dn:
                score += 0.10
                notes.append("local_bear_trend")

            score += max(0.0, -flow_score)
            score += max(0.0, -funding_score)

            oi_score, oi_notes = self._score_oi_directional(sentiment, side)
            score += oi_score

            notes.extend(flow_notes)
            notes.extend(funding_notes)
            notes.extend(oi_notes)

        if side is not None:
            rsi_s, rsi_n = self._score_rsi(row, side, "continuation")
            score += rsi_s
            notes.extend(rsi_n)

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
                score += 0.15
                notes.append("htf_up_supportive")
            elif htf_regime == "chop":
                score += 0.08
                notes.append("htf_chop_allows_reversal")

            score += max(0.0, flow_score)
            score += max(0.0, funding_score)

            oi_score, oi_notes = self._score_oi_directional(sentiment, side)
            # Rising OI into reversal can be useful as crowding / fuel
            if oi_score > 0:
                score += min(0.10, oi_score + 0.02)
                notes.append("oi_supports_reversal")
            else:
                score += oi_score

            notes.extend(flow_notes)
            notes.extend(funding_notes)
            notes.extend(oi_notes)

        elif bearish_trigger and bearish_context and allow_bear:
            side = "SHORT"
            score += 0.45
            notes.append("reversal_bear_trigger")
            notes.append("above_vwap_rejection_candidate")
            if htf_regime == "down":
                score += 0.15
                notes.append("htf_down_supportive")
            elif htf_regime == "chop":
                score += 0.08
                notes.append("htf_chop_allows_reversal")

            score += max(0.0, -flow_score)
            score += max(0.0, -funding_score)

            oi_score, oi_notes = self._score_oi_directional(sentiment, side)
            if oi_score > 0:
                score += min(0.10, oi_score + 0.02)
                notes.append("oi_supports_reversal")
            else:
                score += oi_score

            notes.extend(flow_notes)
            notes.extend(funding_notes)
            notes.extend(oi_notes)

        if side is not None:
            rsi_s, rsi_n = self._score_rsi(row, side, "reversal")
            score += rsi_s
            notes.extend(rsi_n)

        return side, score, notes

    # ------------------------------------------------------------------
    # Quality filter
    # ------------------------------------------------------------------

    def _quality_filter(
        self,
        row: pd.Series,
        session_label: str,
        vol_state: str,
    ) -> Tuple[bool, List[str]]:
        notes: List[str] = []

        if session_label == "dead_zone":
            notes.append("dead_zone_block")
            return False, notes

        if vol_state in ("low", "unknown"):
            notes.append(vol_state + "_vol_block")
            return False, notes

        notes.append("quality_pass")
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

        session_score_adj = {
            "london_open": +0.05,
            "ny_open": +0.05,
            "asia_open": 0.00,
            "london_late": 0.00,
            "asia_late": 0.00,
            "ny_pm": -0.05,
            "dead_zone": -0.10,
        }.get(session_label, 0.0)

        quality_ok, quality_notes = self._quality_filter(row, session_label, vol_state)
        if not quality_ok:
            if self.debug:
                print("[SIGNAL_DEBUG] Quality blocked: " + ", ".join(quality_notes))
            return None

        cont_side, cont_score, cont_notes = self._build_continuation_signal(
            row, triggers, htf_regime, macro_regime, flow_snapshot, sentiment,
        )
        rev_side, rev_score, rev_notes = self._build_reversal_signal(
            row, triggers, htf_regime, macro_regime, flow_snapshot, sentiment,
        )

        chosen_side: Optional[str] = None
        chosen_score: float = 0.0
        chosen_notes: List[str] = []
        setup_family: str = "none"

        if cont_side and cont_score >= rev_score:
            chosen_side, chosen_score, chosen_notes, setup_family = cont_side, cont_score, cont_notes, "continuation"
        elif rev_side:
            chosen_side, chosen_score, chosen_notes, setup_family = rev_side, rev_score, rev_notes, "reversal"

        if chosen_side is None:
            if self.debug:
                print("[SIGNAL_DEBUG] No valid setup family.")
            return None

        chosen_score += session_score_adj
        if session_score_adj != 0.0 and self.debug:
            print(f"[SIGNAL_DEBUG] session_adj={session_score_adj:+.2f} -> score={chosen_score:.3f}")

        effective_threshold = self.score_threshold
        if htf_regime == "chop":
            effective_threshold += 0.05
        if macro_regime == "chop":
            effective_threshold += 0.03

        if abs(chosen_score) < effective_threshold:
            if self.debug:
                print(
                    "[SIGNAL_DEBUG] score=" + str(round(chosen_score, 3)) +
                    " < threshold=" + str(round(effective_threshold, 3))
                )
            return None

        stop, tp, atr_val, stop_dist, tp_dist = self._build_trade_levels(
            side=chosen_side, price=price, row=row, df=df,
        )
        if stop is None or tp is None:
            if self.debug:
                print("[SIGNAL_DEBUG] Could not build valid trade levels.")
            return None

        rr = abs((tp - price) / (price - stop)) if price != stop else 0.0
        if rr < 1.8:
            if self.debug:
                print("[SIGNAL_DEBUG] RR too low: " + str(round(rr, 2)))
            return None

        confidence = round(min(0.95, max(0.50, chosen_score)), 3)
        all_reasons = chosen_notes + notes_htf + notes_macro + notes_kz + notes_vol + quality_notes
        reason_text = ", ".join(all_reasons)
        combined_regime = setup_family + "|htf_" + htf_regime + "|macro_" + macro_regime

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
        }

        if self.debug:
            print(
                "[SIGNAL_DEBUG] " + coin + " " + chosen_side +
                " score=" + str(round(chosen_score, 3)) +
                " rr=" + str(round(rr, 2)) +
                " session=" + session_label +
                " vol=" + vol_state
            )

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

        last_ts = self.last_swing_ts.get(swing_tf)
        if last_ts is not None and ts <= last_ts:
            return None

        triggers = self._extract_triggers(row)
        htf_regime, _ = self._compute_htf_regime(df_ohlcv)
        macro_regime, _ = self._compute_macro_regime_4h(df_ohlcv)
        vol_state, _, vol_ratio = self._compute_vol_state(row)

        flow_score, flow_notes = self._score_flow_context(flow_snapshot)
        funding_score, funding_notes = self._score_funding_context(sentiment)

        side: Optional[str] = None
        score = 0.0
        reasons: List[str] = []

        if triggers["bos_bull"] or triggers["choch_bull"] or triggers["ob_bull"]:
            if htf_regime == "up" or macro_regime == "up":
                side = "LONG"
                score = 0.65
                score += max(0.0, flow_score)
                score += max(0.0, funding_score)

                oi_score, oi_notes = self._score_oi_directional(sentiment, side)
                score += oi_score

                reasons.append("swing_bull_setup")
                reasons.extend(flow_notes)
                reasons.extend(funding_notes)
                reasons.extend(oi_notes)

        elif triggers["bos_bear"] or triggers["choch_bear"] or triggers["ob_bear"]:
            if htf_regime == "down" or macro_regime == "down":
                side = "SHORT"
                score = 0.65
                score += max(0.0, -flow_score)
                score += max(0.0, -funding_score)

                oi_score, oi_notes = self._score_oi_directional(sentiment, side)
                score += oi_score

                reasons.append("swing_bear_setup")
                reasons.extend(flow_notes)
                reasons.extend(funding_notes)
                reasons.extend(oi_notes)

        if side is None:
            if self.debug:
                print("[SWING_DEBUG] No valid " + swing_tf + " swing setup.")
            return None

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
        combined_regime = "swing_" + swing_tf + "|htf_" + htf_regime + "|macro_" + macro_regime

        meta = {
            "timeframe": swing_tf,
            "coin": coin,
            "total_score": round(score, 3),
            "regime_local": "swing",
            "regime_htf_1h": htf_regime,
            "regime_macro_4h": macro_regime,
            "close": price,
            "atr": atr_val,
            "stop_dist": stop_dist,
            "tp_dist": tp_dist,
            "effective_threshold": round(threshold, 3),
            "vol_state": vol_state,
            "vol_ratio": vol_ratio,
        }

        self.last_swing_ts[swing_tf] = ts

        return Signal(
            coin=coin,
            side=side,
            entry_price=price,
            stop_price=stop,
            tp_price=tp,
            confidence=confidence,
            regime=combined_regime,
            reason=", ".join(reasons),
            meta=meta,
        )
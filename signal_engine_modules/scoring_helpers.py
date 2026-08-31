"""
Score component helpers — pure functions returning (score, notes).

Extracted from signal_engine.py. Each function takes primitive inputs
and returns a (float, List[str]) tuple representing the score
contribution and the reasoning tags.

Does NOT import signal_engine.py. Imports perp_sentiment for OI thresholds.
"""

import os
from typing import Any, Dict, List, Tuple

import pandas as pd

# ── Constants (same env reads as signal_engine.py) ───────────────────
WHALE_PRESSURE_DEAD_CALM_PENALTY = -abs(float(os.getenv("WHALE_PRESSURE_DEAD_CALM_PENALTY", "-0.05")))
WHALE_PRESSURE_STALE_AGE_SEC = float(os.getenv("WHALE_PRESSURE_STALE_AGE_SEC", "120.0"))

CONT_REGIME_STRONG_BONUS = float(os.getenv("CONT_REGIME_STRONG_BONUS", "0.08"))
CONT_REGIME_WEAK_PENALTY = -abs(float(os.getenv("CONT_REGIME_WEAK_PENALTY", "-0.12")))
CONT_REGIME_CHOP_PENALTY = -abs(float(os.getenv("CONT_REGIME_CHOP_PENALTY", "-0.22")))
CONT_REGIME_UNKNOWN_PENALTY = -abs(float(os.getenv("CONT_REGIME_UNKNOWN_PENALTY", "-0.03")))


def score_volume_context(row: pd.Series) -> Tuple[float, List[str]]:
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


def score_vwap_magnitude(row: pd.Series, side: str, setup_family: str) -> Tuple[float, List[str]]:
    score = 0.0
    notes: List[str] = []
    vwap_dev = float(row.get("vwap_dev", 0.0))
    vwap_dev_w1 = float(row.get("vwap_dev_w1", 0.0))
    close = float(row.get("close", 0.0))
    if close <= 0:
        return 0.0, []

    abs_dev_pct = abs(vwap_dev) / close
    abs_w1_pct = abs(vwap_dev_w1) / close

    if setup_family == "reversal":
        correct_side = (side == "LONG" and vwap_dev < 0) or (side == "SHORT" and vwap_dev > 0)
        if correct_side:
            if abs_dev_pct > 0.020:
                score += 0.10
                notes.append(f"deep_session_vwap_reversal_{abs_dev_pct:.3f}")
            elif abs_dev_pct > 0.010:
                score += 0.06
                notes.append(f"moderate_session_vwap_reversal_{abs_dev_pct:.3f}")
            elif abs_dev_pct > 0.005:
                score += 0.03
                notes.append(f"mild_session_vwap_reversal_{abs_dev_pct:.3f}")
    elif setup_family == "continuation":
        trending_correct = (side == "LONG" and vwap_dev > 0) or (side == "SHORT" and vwap_dev < 0)
        if trending_correct:
            if abs_dev_pct > 0.010:
                score += 0.04
                notes.append(f"session_vwap_trend_strength_{abs_dev_pct:.3f}")
            elif abs_dev_pct > 0.005:
                score += 0.02
                notes.append(f"session_vwap_trend_mild_{abs_dev_pct:.3f}")

    if vwap_dev_w1 != 0.0:
        w1_in_discount = (side == "LONG" and vwap_dev_w1 < 0)
        w1_in_premium = (side == "SHORT" and vwap_dev_w1 > 0)
        w1_aligned_cont = (
            (side == "LONG" and vwap_dev_w1 > 0) or
            (side == "SHORT" and vwap_dev_w1 < 0)
        )

        if setup_family == "reversal":
            if (w1_in_discount or w1_in_premium) and abs_w1_pct > 0.010:
                score += 0.04
                notes.append(f"w1_vwap_{'discount' if w1_in_discount else 'premium'}_{abs_w1_pct:.3f}")
        elif setup_family == "continuation" and w1_aligned_cont and abs_w1_pct > 0.005:
            score += 0.02
            notes.append(f"w1_vwap_momentum_aligned_{abs_w1_pct:.3f}")
        elif setup_family == "swing":
            if (w1_in_discount or w1_in_premium) and abs_w1_pct > 0.010:
                score += 0.04
                notes.append(f"w1_vwap_swing_{'discount' if w1_in_discount else 'premium'}_{abs_w1_pct:.3f}")

    return score, notes


def score_trigger_quality(row: pd.Series) -> Tuple[float, List[str]]:
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


def score_flow_context(flow_snapshot: Dict[str, Any]) -> Tuple[float, List[str]]:
    score = 0.0
    notes: List[str] = []
    if not flow_snapshot:
        # Coin has no on-chain flow tracking at all (not in TOKEN_MINTS) —
        # tagged distinctly from a tracked coin reading genuinely calm, so
        # telemetry/analysis doesn't conflate "no data" with "confirmed quiet".
        return score, ["flow_untracked"]
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


def score_flow_dead_calm(flow_snapshot: Dict[str, Any]) -> Tuple[float, List[str]]:
    if not flow_snapshot or WHALE_PRESSURE_DEAD_CALM_PENALTY >= 0:
        return 0.0, []
    snapshot_age = float(flow_snapshot.get("snapshot_age_sec", 9999.0))
    if snapshot_age >= WHALE_PRESSURE_STALE_AGE_SEC:
        return 0.0, []
    whale_pressure = float(flow_snapshot.get("whale_pressure", 0.0))
    if abs(whale_pressure) < 0.2:
        return WHALE_PRESSURE_DEAD_CALM_PENALTY, ["whale_pressure_dead_calm"]
    return 0.0, []


# Thresholds recalibrated 2026-08-25 against real Hyperliquid funding-rate
# history (fundingHistory endpoint, 30 days, all 8 currently-traded coins,
# 4,000 hourly readings): median=0.0013%, p95=0.0017%, p99=0.0040%,
# max observed=0.0085%. The previous thresholds (mild=0.1%, high=0.5%,
# extreme=1%) were 12-100x higher than anything actually observed in a
# month across the whole traded universe -- this function had never fired
# even its lowest tier in practice, not just "rarely". New thresholds are
# real percentile cutoffs, not round numbers picked from BTC/ETH funding
# lore that doesn't apply to these lower-cap alts on Hyperliquid.
FUNDING_MILD_THRESHOLD = float(os.getenv("FUNDING_MILD_THRESHOLD", "0.00002"))       # ~p93
FUNDING_HIGH_THRESHOLD = float(os.getenv("FUNDING_HIGH_THRESHOLD", "0.00004"))       # ~p99
FUNDING_EXTREME_THRESHOLD = float(os.getenv("FUNDING_EXTREME_THRESHOLD", "0.00008"))  # ~observed 30d max


def score_funding_context(funding_rate: float) -> Tuple[float, List[str]]:
    score = 0.0
    notes: List[str] = []
    fr = float(funding_rate)
    abs_fr = abs(fr)
    if abs_fr > FUNDING_EXTREME_THRESHOLD:
        bump, tier = 0.15, "extreme"
    elif abs_fr > FUNDING_HIGH_THRESHOLD:
        bump, tier = 0.08, "high"
    elif abs_fr > FUNDING_MILD_THRESHOLD:
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


def score_oi_directional(
    oi_delta_1h_pct: float,
    oi_delta_4h_pct: float,
    open_interest: float,
    prev_open_interest: float,
    side: str,
) -> Tuple[float, List[str]]:
    from perp_sentiment import OI_STRONG_1H, OI_MILD_1H, OI_STRONG_4H

    score = 0.0
    notes: List[str] = []

    oi_1h = float(oi_delta_1h_pct or 0.0)
    oi_4h = float(oi_delta_4h_pct or 0.0)

    if oi_1h == 0.0:
        oi = float(open_interest or 0.0)
        prev_oi = float(prev_open_interest or 0.0)
        if oi > 0 and prev_oi > 0:
            oi_1h = (oi - prev_oi) / prev_oi

    if oi_1h > OI_STRONG_1H:
        score += 0.08
        notes.append(f"oi_1h_surge_{side.lower()}_{oi_1h:.3f}")
    elif oi_1h > OI_MILD_1H:
        score += 0.04
        notes.append(f"oi_1h_build_{side.lower()}_{oi_1h:.3f}")
    elif oi_1h < -OI_STRONG_1H:
        score -= 0.08
        notes.append(f"oi_1h_unwind_{side.lower()}_{oi_1h:.3f}")
    elif oi_1h < -OI_MILD_1H:
        score -= 0.04
        notes.append(f"oi_1h_fade_{side.lower()}_{oi_1h:.3f}")

    if oi_4h > OI_STRONG_4H:
        score += 0.04
        notes.append(f"oi_4h_structural_build_{oi_4h:.3f}")
    elif oi_4h < -OI_STRONG_4H:
        score -= 0.04
        notes.append(f"oi_4h_structural_unwind_{oi_4h:.3f}")

    return score, notes


def score_rsi(row: pd.Series, side: str, setup_family: str) -> Tuple[float, List[str]]:
    score = 0.0
    notes: List[str] = []
    rsi_val = float(row.get("rsi_14", 50.0))
    if pd.isna(rsi_val):
        return 0.0, []
    if setup_family == "continuation":
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


def score_continuation_regime(market_regime: str) -> Tuple[float, List[str]]:
    if market_regime == "strong_trend":
        return CONT_REGIME_STRONG_BONUS, ["continuation_regime_strong_trend"]
    if market_regime == "weak_trend":
        return CONT_REGIME_WEAK_PENALTY, ["continuation_regime_weak_trend_penalty"]
    if market_regime == "chop":
        return CONT_REGIME_CHOP_PENALTY, ["continuation_regime_chop_penalty"]
    return CONT_REGIME_UNKNOWN_PENALTY, ["continuation_regime_unknown_penalty"]

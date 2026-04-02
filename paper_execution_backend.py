# paper_execution_backend.py
"""
Paper execution backend — realistic fill simulation.

Active-path accounting support:
- returns estimated entry/exit fee in FillResult.meta
- provides funding accrual helper for open positions
- preserves separate entry vs exit slippage profiles
- preserves fills.csv logging for reconciliation
"""

from __future__ import annotations

import csv
import os
import random
import time
from datetime import datetime, timezone
from typing import Optional, Dict, Any

import requests

from execution_backend import ExecutionBackend, FillResult

HYPERLIQUID_INFO_URL = "https://api.hyperliquid.xyz/info"

ENTRY_SLIP_MIN_BPS = float(os.getenv("ENTRY_SLIP_MIN_BPS", "5"))
ENTRY_SLIP_MAX_BPS = float(os.getenv("ENTRY_SLIP_MAX_BPS", "25"))
ENTRY_SLIP_MODE_BPS = float(os.getenv("ENTRY_SLIP_MODE_BPS", "12"))

EXIT_SLIP_MIN_BPS = float(os.getenv("EXIT_SLIP_MIN_BPS", "3"))
EXIT_SLIP_MAX_BPS = float(os.getenv("EXIT_SLIP_MAX_BPS", "18"))
EXIT_SLIP_MODE_BPS = float(os.getenv("EXIT_SLIP_MODE_BPS", "8"))

MIN_FILL_RATIO = float(os.getenv("MIN_FILL_RATIO", "0.60"))
FILL_BASE = float(os.getenv("FILL_BASE", "0.80"))
FILL_CONF_BONUS = float(os.getenv("FILL_CONF_BONUS", "0.15"))

MID_CACHE_TTL_SEC = float(os.getenv("MID_CACHE_TTL_SEC", "0.75"))
STOP_GAP_PROB = float(os.getenv("STOP_GAP_PROB", "0.08"))
STOP_GAP_EXTRA_BPS_MAX = float(os.getenv("STOP_GAP_EXTRA_BPS_MAX", "40"))

# Hyperliquid fee assumptions
TAKER_FEE_BPS = float(os.getenv("TAKER_FEE_BPS", "4.5"))  # 0.045%
MAKER_FEE_BPS = float(os.getenv("MAKER_FEE_BPS", "1.5"))  # optional / future

# Funding model for paper
# Positive funding rate means longs pay, shorts receive.
FUNDING_BPS_PER_8H = float(os.getenv("FUNDING_BPS_PER_8H", "0.0"))
FUNDING_INTERVAL_HOURS = float(os.getenv("FUNDING_INTERVAL_HOURS", "8.0"))

FILLS_LOG_FILE = os.getenv("FILLS_LOG_FILE", "fills.csv")
LOG_FILLS = os.getenv("LOG_FILLS", "true").lower() == "true"

FILLS_FIELDS = [
    "timestamp", "coin", "side", "reason",
    "requested_price", "fill_price", "slippage_bps",
    "requested_size_usd", "fill_size_usd", "fill_ratio",
    "gap_extra_bps", "confidence",
    "estimated_fee_usd", "fee_bps",
]


class PaperExecutionBackend(ExecutionBackend):
    def __init__(self, debug: bool = True):
        self.debug = debug
        self._mid_cache: Dict[str, float] = {}
        self._mid_cache_ts: float = 0.0

    # ── Entry ──────────────────────────────────────────────────────────────

    def execute_entry(
        self,
        coin: str,
        side: str,
        price: float,
        size_usd: float,
        confidence: float,
    ) -> FillResult:
        if price <= 0 or size_usd <= 0:
            return FillResult(
                filled=False,
                fill_price=0.0,
                fill_size_usd=0.0,
                slippage_bps=0.0,
                reject_reason="invalid_entry_request",
                requested_price=price,
                requested_size_usd=size_usd,
                fill_ratio=None,
                reason="entry",
                meta={},
            )

        slip_bps = self._triangular_bps(
            ENTRY_SLIP_MIN_BPS, ENTRY_SLIP_MAX_BPS, ENTRY_SLIP_MODE_BPS,
        )
        fill_price = self._apply_entry_slippage(price, side, slip_bps)
        fill_ratio = self._simulate_fill_ratio(confidence)
        fill_size = size_usd * fill_ratio

        if fill_size < 10.0:
            return FillResult(
                filled=False,
                fill_price=0.0,
                fill_size_usd=0.0,
                slippage_bps=slip_bps,
                reject_reason="fill_size_below_minimum",
                requested_price=price,
                requested_size_usd=size_usd,
                fill_ratio=fill_ratio,
                reason="entry",
                meta={},
            )

        estimated_fee_usd = self.estimate_entry_fee(fill_size)

        result = FillResult(
            filled=True,
            fill_price=fill_price,
            fill_size_usd=fill_size,
            slippage_bps=slip_bps,
            requested_price=price,
            requested_size_usd=size_usd,
            fill_ratio=fill_ratio,
            reason="entry",
            meta={
                "estimated_fee_usd": round(estimated_fee_usd, 8),
                "fee_bps": TAKER_FEE_BPS,
            },
        )

        self._log_fill(
            coin,
            side,
            "entry",
            result,
            confidence=confidence,
            estimated_fee_usd=estimated_fee_usd,
            fee_bps=TAKER_FEE_BPS,
        )
        return result

    # ── Exit ───────────────────────────────────────────────────────────────

    def execute_exit(
        self,
        coin: str,
        side: str,
        price: float,
        size_usd: float,
        reason: str = "",
    ) -> FillResult:
        if price <= 0 or size_usd <= 0:
            return FillResult(
                filled=False,
                fill_price=0.0,
                fill_size_usd=0.0,
                slippage_bps=0.0,
                reject_reason="invalid_exit_request",
                requested_price=price,
                requested_size_usd=size_usd,
                fill_ratio=None,
                reason=reason or "exit",
                meta={},
            )

        slip_bps = self._triangular_bps(
            EXIT_SLIP_MIN_BPS, EXIT_SLIP_MAX_BPS, EXIT_SLIP_MODE_BPS,
        )

        gap_extra = 0.0
        reason_lower = (reason or "").lower()
        if "stop" in reason_lower and random.random() < STOP_GAP_PROB:
            gap_extra = random.uniform(5, STOP_GAP_EXTRA_BPS_MAX)
            slip_bps += gap_extra

        fill_price = self._apply_exit_slippage(price, side, slip_bps)
        estimated_fee_usd = self.estimate_exit_fee(size_usd)

        result = FillResult(
            filled=True,
            fill_price=fill_price,
            fill_size_usd=size_usd,
            slippage_bps=slip_bps,
            requested_price=price,
            requested_size_usd=size_usd,
            fill_ratio=1.0,
            reason=reason or "exit",
            meta={
                "estimated_fee_usd": round(estimated_fee_usd, 8),
                "fee_bps": TAKER_FEE_BPS,
                "gap_extra_bps": round(gap_extra, 2),
            },
        )

        self._log_fill(
            coin,
            side,
            reason or "exit",
            result,
            gap_extra_bps=gap_extra,
            estimated_fee_usd=estimated_fee_usd,
            fee_bps=TAKER_FEE_BPS,
        )
        return result

    # ── Funding helpers ────────────────────────────────────────────────────

    def estimate_entry_fee(self, notional_usd: float) -> float:
        return max(0.0, notional_usd) * (TAKER_FEE_BPS / 10_000.0)

    def estimate_exit_fee(self, notional_usd: float) -> float:
        return max(0.0, notional_usd) * (TAKER_FEE_BPS / 10_000.0)

    def accrue_funding(
        self,
        coin: str,
        side: str,
        notional_usd: float,
        last_funding_ts: Optional[datetime],
        now: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Returns:
        {
            "funding_usd": signed float,
            "last_funding_ts": datetime
        }

        Positive funding_usd = cost to strategy
        Negative funding_usd = credit to strategy
        """
        now = now or datetime.now(timezone.utc)
        if last_funding_ts is None:
            return {"funding_usd": 0.0, "last_funding_ts": now}

        interval_sec = max(60.0, FUNDING_INTERVAL_HOURS * 3600.0)
        elapsed_sec = max(0.0, (now - last_funding_ts).total_seconds())
        if elapsed_sec <= 0 or FUNDING_BPS_PER_8H == 0.0 or notional_usd <= 0:
            return {"funding_usd": 0.0, "last_funding_ts": now}

        intervals = elapsed_sec / interval_sec
        raw_funding = notional_usd * (abs(FUNDING_BPS_PER_8H) / 10_000.0) * intervals

        side_upper = str(side).upper()
        if FUNDING_BPS_PER_8H > 0:
            # longs pay, shorts receive
            funding_usd = raw_funding if side_upper == "LONG" else -raw_funding
        else:
            # shorts pay, longs receive
            funding_usd = -raw_funding if side_upper == "LONG" else raw_funding

        return {
            "funding_usd": funding_usd,
            "last_funding_ts": now,
        }

    # ── Price feed ─────────────────────────────────────────────────────────

    def get_mid_price(self, coin: str) -> Optional[float]:
        now = time.time()
        if now - self._mid_cache_ts > MID_CACHE_TTL_SEC:
            try:
                resp = requests.post(
                    HYPERLIQUID_INFO_URL,
                    json={"type": "allMids"},
                    timeout=5,
                )
                resp.raise_for_status()
                mids = resp.json()
                if isinstance(mids, dict):
                    self._mid_cache = {
                        k.upper(): float(v) for k, v in mids.items()
                    }
                    self._mid_cache_ts = now
            except Exception as e:
                if self.debug:
                    print(f"[BACKEND] allMids fetch failed: {e}")

        return self._mid_cache.get(coin.upper())

    def shutdown(self):
        if self.debug:
            print("[BACKEND] PaperExecutionBackend shutdown")

    # ── Fill simulation internals ──────────────────────────────────────────

    def _simulate_fill_ratio(self, confidence: float) -> float:
        conf = max(0.50, min(0.95, confidence))
        conf_bonus = FILL_CONF_BONUS * (conf - 0.50) / 0.45
        jitter = random.uniform(-0.10, 0.10)
        ratio = FILL_BASE + conf_bonus + jitter
        return max(MIN_FILL_RATIO, min(1.0, ratio))

    @staticmethod
    def _triangular_bps(low: float, high: float, mode: float) -> float:
        return random.triangular(low, high, mode)

    @staticmethod
    def _apply_entry_slippage(price: float, side: str, bps: float) -> float:
        slip = bps / 10_000.0
        if str(side).upper() == "LONG":
            return price * (1.0 + slip)
        return price * (1.0 - slip)

    @staticmethod
    def _apply_exit_slippage(price: float, side: str, bps: float) -> float:
        slip = bps / 10_000.0
        if str(side).upper() == "LONG":
            return price * (1.0 - slip)
        return price * (1.0 + slip)

    # ── Fill event logging ─────────────────────────────────────────────────

    def _log_fill(
        self,
        coin: str,
        side: str,
        reason: str,
        fill: FillResult,
        confidence: float = 0.0,
        gap_extra_bps: float = 0.0,
        estimated_fee_usd: float = 0.0,
        fee_bps: float = 0.0,
    ):
        if not LOG_FILLS:
            return

        file_exists = os.path.exists(FILLS_LOG_FILE)
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "coin": coin,
            "side": str(side).upper(),
            "reason": reason,
            "requested_price": round(fill.requested_price, 8),
            "fill_price": round(fill.fill_price, 8),
            "slippage_bps": round(fill.slippage_bps, 2),
            "requested_size_usd": round(fill.requested_size_usd, 2),
            "fill_size_usd": round(fill.fill_size_usd, 2),
            "fill_ratio": round(fill.fill_ratio, 4) if fill.fill_ratio is not None else "",
            "gap_extra_bps": round(gap_extra_bps, 2),
            "confidence": round(confidence, 3),
            "estimated_fee_usd": round(estimated_fee_usd, 4),
            "fee_bps": round(fee_bps, 4),
        }
        try:
            with open(FILLS_LOG_FILE, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=FILLS_FIELDS)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)
        except Exception as e:
            if self.debug:
                print(f"[BACKEND] Fill log failed: {e}")
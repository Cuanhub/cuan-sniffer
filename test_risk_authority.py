import csv
import contextlib
import importlib
import io
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock


def _clean_env(**overrides) -> dict:
    env = dict(os.environ)
    env.update({
        "STARTING_BALANCE": "10000",
        "RISK_PCT_PER_TRADE": "1.00",
        "RISK_PCT_MULT_INTRADAY": "1.0",
        "RISK_PCT_MULT_SWING": "1.0",
        "UNIVERSAL_MIN_CONFIDENCE": "0.80",
        "MIN_SIGNAL_CONFIDENCE": "0.80",
        "SCORE_SIZE_BASE_MULT": "1.00",
        "SCORE_SIZE_MID_THRESHOLD": "0.75",
        "SCORE_SIZE_MID_MULT": "1.25",
        "SCORE_SIZE_HIGH_THRESHOLD": "0.90",
        "SCORE_SIZE_HIGH_MULT": "1.50",
        "CONT_STRONG_TREND_BONUS_MULT": "1.15",
        "SCORE_SIZE_OVERLAY_MAX_MULT": "1.50",
        "CONTINUATION_MAX_SIZE_MULT": "1.50",
        "WEAK_TREND_SIZE_MULT": "0.70",
        "FINAL_RISK_AUTHORITY_TOLERANCE": "0.001",
        "RESEARCH_ONLY_MODE": "false",
    })
    env.update(overrides)
    return env


def _reload_executor(env: dict):
    with unittest.mock.patch.dict(os.environ, env, clear=True):
        for name in list(sys.modules):
            if name == "executor" or name == "risk_manager" or name.startswith("executor_modules"):
                sys.modules.pop(name, None)
        return importlib.import_module("executor")


def _decision(mod, *, size=1000.0, risk=10.0, mult=1.0):
    risk_manager = importlib.import_module("risk_manager")
    return risk_manager.RiskDecision(
        approved=True,
        reason="approved",
        size_usd=size,
        risk_usd=risk,
        size_multiplier=mult,
        approved_size_usd=size,
        approved_max_risk_usd=risk,
        approved_size_multiplier=mult,
        track="intraday",
    )


def _signal(**overrides):
    meta = {
        "timeframe": "15m",
        "setup_family": "continuation",
        "market_regime": "strong_trend",
        "session": "ny_open",
        "active_quality_model": "v3",
        "active_quality_score": 0.95,
        "active_quality_threshold": 0.80,
        "total_score": 0.95,
        "rr_planned": 2.0,
    }
    meta.update(overrides.pop("meta", {}))
    data = {
        "coin": "ZZZ",
        "side": "LONG",
        "entry_price": 100.0,
        "stop_price": 99.0,
        "tp_price": 103.0,
        "confidence": 0.95,
        "regime": "continuation|htf_up|macro_up|mkt_strong_trend",
        "meta": meta,
    }
    data.update(overrides)
    return types.SimpleNamespace(**data)


class TestRiskAuthorityMutations(unittest.TestCase):
    def test_final_risk_cannot_exceed_approved_after_score_overlay(self):
        mod = _reload_executor(_clean_env())
        ex = object.__new__(mod.Executor)
        sig = _signal()
        decision = _decision(mod, mult=1.25)

        info = ex._apply_score_size_overlay_reduce_only(
            decision,
            sig,
            setup_family="continuation",
            market_regime="strong_trend",
        )

        self.assertGreater(info["raw_overlay_mult"], 1.0)
        self.assertLessEqual(decision.risk_usd, decision.approved_max_risk_usd)
        self.assertLessEqual(decision.size_multiplier, decision.approved_size_multiplier)

    def test_final_risk_cannot_exceed_approved_after_margin_sizing(self):
        mod = _reload_executor(_clean_env())
        ex = object.__new__(mod.Executor)
        ex._live_mode = False
        ex.backend = MagicMock()
        ex._get_venue_available_margin = lambda: (10000.0, 0.0, 500.0)
        ex._apply_runtime_balance_from_venue = MagicMock()
        decision = _decision(mod, size=1000.0, risk=10.0, mult=1.0)

        reject = ex._apply_available_margin_sizing(decision, coin="ZZZ")

        self.assertIsNone(reject)
        self.assertLessEqual(decision.risk_usd, decision.approved_max_risk_usd)
        self.assertLess(decision.size_usd, decision.approved_size_usd)

    def test_final_risk_cannot_exceed_approved_after_continuation_cap(self):
        mod = _reload_executor(_clean_env(CONTINUATION_MAX_SIZE_MULT="1.50"))
        decision = _decision(mod, size=1000.0, risk=10.0, mult=2.0)

        applied = mod.Executor._apply_continuation_size_cap_reduce_only(
            decision,
            setup_family="continuation",
        )

        self.assertTrue(applied)
        self.assertLessEqual(decision.risk_usd, decision.approved_max_risk_usd)
        self.assertAlmostEqual(decision.size_multiplier, 1.50)

    def test_final_risk_cannot_exceed_approved_after_weak_trend_mult(self):
        mod = _reload_executor(_clean_env(WEAK_TREND_SIZE_MULT="0.70"))
        decision = _decision(mod, size=1000.0, risk=10.0, mult=1.0)

        reason = mod.Executor._apply_weak_trend_size_mult_reduce_only(
            decision,
            market_regime="weak_trend",
        )

        self.assertIsNone(reason)
        self.assertLessEqual(decision.risk_usd, decision.approved_max_risk_usd)
        self.assertAlmostEqual(decision.risk_usd, 7.0)

    def test_v3_high_score_cannot_double_count_quality_into_risk(self):
        mod = _reload_executor(_clean_env())
        risk_manager = importlib.import_module("risk_manager")
        risk = risk_manager.RiskManager(strategy_filter=None)
        sig = _signal()

        decision = risk.check_signal(sig)
        self.assertTrue(decision.approved, decision.reason)
        approved_risk = decision.approved_max_risk_usd
        ex = object.__new__(mod.Executor)
        info = ex._apply_score_size_overlay_reduce_only(
            decision,
            sig,
            setup_family="continuation",
            market_regime="strong_trend",
        )

        self.assertGreater(info["raw_overlay_mult"], 1.0)
        self.assertAlmostEqual(decision.risk_usd, approved_risk)
        self.assertAlmostEqual(approved_risk, 125.0)


class TestRiskAuthoritySlippageAndReject(unittest.TestCase):
    def test_final_risk_cannot_exceed_approved_after_fill_slippage_simulation(self):
        with tempfile.TemporaryDirectory() as tmp:
            reject_path = str(Path(tmp) / "executor_rejects.csv")
            mod = _reload_executor(_clean_env(EXECUTOR_REJECTS_PATH=reject_path))
            ex = object.__new__(mod.Executor)
            ex.notify = MagicMock()
            sig = _signal()
            decision = _decision(mod, size=1000.0, risk=10.0, mult=1.0)

            pos = ex._build_position(
                signal=sig,
                sig_id=7,
                decision=decision,
                fill_price=101.0,
                fill_size_usd=1000.0,
                entry_order_ids=["abc"],
            )

            self.assertEqual(pos.risk_authority_status, "actual_risk_exceeded_approved")
            self.assertGreater(pos.actual_risk_usd, pos.approved_max_risk_usd)
            self.assertAlmostEqual(pos.risk_usd, pos.approved_max_risk_usd)
            self.assertAlmostEqual(pos.r_value, pos.approved_max_risk_usd)
            self.assertEqual(pos.protection_status, "protection_critical")
            ex.notify.assert_called_once()
            with open(reject_path, newline="") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(rows[-1]["reject_reason"], "actual_risk_exceeded_approved")

    def test_final_risk_authority_violation_rejects_before_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            reject_path = str(Path(tmp) / "executor_rejects.csv")
            mod = _reload_executor(_clean_env(EXECUTOR_REJECTS_PATH=reject_path))
            mod.evaluate_regime_block = lambda **kwargs: (False, "")
            mod.HARD_BLOCKED_COINS = set()
            mod.HARD_BLOCKED_TIMEFRAMES = set()
            mod.SOFT_BLOCKED_SESSIONS = set()
            mod.HARD_BLOCK_UNKNOWN_SESSION = False
            mod.HARD_BLOCK_CONTINUATION = False
            mod.HARD_BLOCK_CHOP = False
            mod.BLOCK_CONTINUATION_IN_CHOP = False
            mod.BLOCK_REVERSAL_AGAINST_DUAL_TREND = False
            mod.SWING_MIN_CONFIDENCE = 0.0

            ex = object.__new__(mod.Executor)
            ex._RESEARCH_ONLY_MODE = False
            ex._live_mode = False
            ex._balance_ready = True
            ex.venue_sync_unhealthy = False
            ex._last_margin_block_reason = ""
            ex._cooldowns = {}
            ex._coin_last_fill_ts = {}
            ex._bucket_last_fill_ts = {}
            ex._pending_positions = {}
            ex.notify = MagicMock()
            ex.backend = MagicMock()
            ex.backend.execute_entry = MagicMock()
            ex.order_tracker = MagicMock()
            ex.signal_engine = None

            class FakeRisk:
                open_positions = {}

                def check_signal(self, signal):
                    return _decision(mod, size=1000.0, risk=10.0, mult=1.0)

            ex.risk = FakeRisk()
            ex._try_mark_coin_pending_open = lambda coin: True
            ex._clear_coin_pending_open = MagicMock()
            ex._has_open_position_for_coin = lambda coin: False
            ex._log_missed = MagicMock()
            ex._apply_reject_throttle = lambda signal, side, reason: reason
            ex._refresh_runtime_balance_from_venue = MagicMock()
            ex._apply_entry_stop_redesign = lambda signal, track: None
            ex._apply_available_margin_sizing = lambda decision, coin: None
            ex._validate_entry = lambda signal, session: None
            ex._get_hard_blocked_sessions = lambda bucket: set()
            ex._can_override_soft_block = lambda signal, session: True
            ex._is_capacity_reject_reason = lambda reason: False

            def unsafe_overlay(decision, signal, setup_family, market_regime):
                decision.risk_usd = 10.05
                return {
                    "score_for_sizing": 0.95,
                    "score_mult": 1.5,
                    "trend_bonus_mult": 1.0,
                    "raw_overlay_mult": 1.5,
                    "overlay_mult": 1.0,
                    "trend_bonus_applied": False,
                    "old_size": 1000.0,
                    "old_risk": 10.0,
                }

            ex._apply_score_size_overlay_reduce_only = unsafe_overlay
            sig = _signal(meta={"setup_family": "reversal", "market_regime": "strong_trend"})

            result = ex._on_signal_inner(sig, sig_id=42)

            self.assertFalse(result.traded)
            self.assertIn("final_risk_authority_violation", result.reason)
            ex.backend.execute_entry.assert_not_called()
            with open(reject_path, newline="") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(rows[-1]["reject_reason"], "final_risk_authority_violation")


class TestRiskAuthorityStressBanner(unittest.TestCase):
    def test_stress_banner_reflects_true_max_risk(self):
        mod = _reload_executor(_clean_env())
        ex = object.__new__(mod.Executor)
        ex._RESEARCH_ONLY_MODE = False
        ex._boot_balance = 10000.0
        ex._boot_hwm = 10000.0
        ex._boot_open = []

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ex._print_boot_banner()
        output = buf.getvalue()

        self.assertIn("Base risk:", output)
        self.assertIn("Confidence multiplier max:", output)
        self.assertIn("Score overlay max:", output)
        self.assertIn("applied=1.00x reduce_only", output)
        self.assertIn("Max true per-trade risk:", output)
        self.assertIn("Max simultaneous open risk:", output)
        self.assertIn("Worst-case account drawdown", output)


if __name__ == "__main__":
    unittest.main()

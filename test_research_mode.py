"""
Tests for RESEARCH_ONLY_MODE and broad chop lane.
"""

import importlib
import os
import sys
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock

from executor_modules.chop_exception_shadow import log_broad_chop_lane


@dataclass
class FakeSignal:
    coin: str = "SOL"
    side: str = "LONG"
    entry_price: float = 100.0
    stop_price: float = 98.0
    tp_price: float = 104.0
    confidence: float = 0.85
    regime: str = "reversal|htf_chop|macro_up|mkt_chop"
    meta: Dict[str, Any] = field(default_factory=dict)


def _reload_executor(env: dict):
    for key in list(os.environ.keys()):
        if key.startswith(("STOP_", "MIN_STOP", "MIN_EXECUTION", "HIGH_CONF_STOP",
                           "REGIME_TP", "STARTING_BALANCE", "UNIVERSAL_MIN",
                           "RESEARCH_ONLY")):
            os.environ.pop(key, None)
    os.environ.update(env)
    for mod in list(sys.modules):
        if mod in ("executor",) or mod.startswith("executor_modules"):
            del sys.modules[mod]
    for dep in ("risk_manager", "strategy_filter", "bootstrap",
                "execution_backend", "execution_backend_factory",
                "live_execution_backend", "live_position_monitor",
                "order_tracker", "protection_manager", "position",
                "trade_log", "live_data_guard"):
        sys.modules.pop(dep, None)
    return importlib.import_module("executor")


class TestResearchOnlyMode(unittest.TestCase):

    def test_research_mode_blocks_live_orders(self):
        mod = _reload_executor({"RESEARCH_ONLY_MODE": "true", "STARTING_BALANCE": "10000"})
        self.assertTrue(mod.Executor._RESEARCH_ONLY_MODE)

    def test_research_mode_off_by_default(self):
        env = {"STARTING_BALANCE": "10000"}
        mod = _reload_executor(env)
        self.assertFalse(mod.Executor._RESEARCH_ONLY_MODE)

    def test_research_mode_reason_in_result(self):
        mod = _reload_executor({"RESEARCH_ONLY_MODE": "true", "STARTING_BALANCE": "10000"})
        ex = object.__new__(mod.Executor)
        ex._live_mode = False
        ex.notify = MagicMock()
        ex.backend = MagicMock()
        ex.signal_engine = None

        sig = FakeSignal(meta={"session": "ny_open", "setup_family": "continuation",
                               "market_regime": "chop", "timeframe": "1h"})
        result = ex._on_signal_inner(sig, sig_id=0)
        self.assertFalse(result.traded)
        self.assertEqual(result.reason, "research_only_mode")

    def test_telemetry_still_runs_in_research_mode(self):
        """The executor reject telemetry should still log in research mode."""
        mod = _reload_executor({"RESEARCH_ONLY_MODE": "true", "STARTING_BALANCE": "10000"})
        # Verify the gate logs to executor_rejects before returning
        import inspect
        src = inspect.getsource(mod.Executor._on_signal_inner)
        # The research_only block must call log_executor_reject
        self.assertIn("log_executor_reject", src)
        # And it must appear before the return
        rom_idx = src.index("research_only_mode")
        ler_idx = src.index("log_executor_reject", rom_idx)
        ret_idx = src.index("return ExecutorResult", ler_idx)
        self.assertLess(ler_idx, ret_idx)

    def test_research_mode_logs_broad_chop_lane_before_blocking_order(self):
        path = tempfile.mktemp(suffix=".csv")
        mod = _reload_executor({
            "RESEARCH_ONLY_MODE": "true",
            "STARTING_BALANCE": "10000",
            "HARD_BLOCK_CHOP": "true",
            "SHADOW_CHOP_LANE_PATH": path,
        })
        ex = object.__new__(mod.Executor)
        ex._live_mode = False
        ex.notify = MagicMock()
        ex.backend = MagicMock()
        ex.signal_engine = None

        try:
            sig = FakeSignal(
                coin="WIF",
                side="LONG",
                confidence=0.82,
                regime="reversal|htf_down|macro_chop|mkt_chop",
                meta={
                    "session": "ny_open",
                    "setup_family": "reversal",
                    "market_regime": "chop",
                    "regime_htf_1h": "down",
                    "regime_macro_4h": "chop",
                    "timeframe": "1h",
                    "total_score": 0.62,
                    "score_v2": 0.71,
                    "score_v3": 0.84,
                    "active_quality_model": "v3",
                    "active_quality_score": 0.84,
                },
            )
            result = ex._on_signal_inner(sig, sig_id=7)
            self.assertFalse(result.traded)
            self.assertEqual(result.reason, "research_only_mode")
            with open(path) as fh:
                lines = fh.readlines()
            self.assertEqual(len(lines), 2)
            self.assertIn("WIF", lines[1])
            self.assertIn("preferred_symbol", lines[1])
        finally:
            if os.path.exists(path):
                os.unlink(path)


class TestBroadChopLane(unittest.TestCase):

    def test_good_symbol_logged(self):
        path = tempfile.mktemp(suffix=".csv")
        import importlib
        for m in list(sys.modules):
            if "chop_exception_shadow" in m:
                del sys.modules[m]
        import executor_modules.chop_exception_shadow as mod
        mod.SHADOW_CHOP_LANE_PATH = path
        try:
            sig = FakeSignal(
                coin="WIF", side="LONG", confidence=0.72,
                meta={"session": "ny_open", "setup_family": "reversal",
                      "market_regime": "chop"},
            )
            mod.log_broad_chop_lane(sig, "market_regime_block:chop")
            with open(path) as fh:
                lines = fh.readlines()
            self.assertEqual(len(lines), 2)
            self.assertIn("WIF", lines[1])
            self.assertIn("preferred_symbol", lines[1])
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def _fresh_mod(self):
        for m in list(sys.modules):
            if "chop_exception_shadow" in m:
                del sys.modules[m]
        import executor_modules.chop_exception_shadow as mod
        return mod

    def test_eth_blocked(self):
        path = tempfile.mktemp(suffix=".csv")
        mod = self._fresh_mod()
        mod.SHADOW_CHOP_LANE_PATH = path
        try:
            sig = FakeSignal(coin="ETH", meta={"session": "ny_open", "setup_family": "reversal"})
            mod.log_broad_chop_lane(sig, "market_regime_block:chop")
            self.assertFalse(os.path.exists(path))
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_reversal_accepted(self):
        path = tempfile.mktemp(suffix=".csv")
        mod = self._fresh_mod()
        mod.SHADOW_CHOP_LANE_PATH = path
        try:
            sig = FakeSignal(
                coin="SOL", side="SHORT", confidence=0.65,
                meta={"session": "dead_zone", "setup_family": "reversal",
                      "market_regime": "chop"},
            )
            mod.log_broad_chop_lane(sig, "market_regime_block:chop")
            with open(path) as fh:
                lines = fh.readlines()
            self.assertEqual(len(lines), 2)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_no_confidence_filter(self):
        path = tempfile.mktemp(suffix=".csv")
        mod = self._fresh_mod()
        mod.SHADOW_CHOP_LANE_PATH = path
        try:
            sig = FakeSignal(
                coin="SOL", confidence=0.55,
                meta={"session": "ny_pm", "setup_family": "continuation",
                      "market_regime": "chop"},
            )
            mod.log_broad_chop_lane(sig, "market_regime_block:chop")
            with open(path) as fh:
                lines = fh.readlines()
            self.assertEqual(len(lines), 2)
        finally:
            if os.path.exists(path):
                os.unlink(path)


class TestAnalyzerHandlesEmpty(unittest.TestCase):

    def test_analyzer_no_crash_on_empty(self):
        import subprocess
        result = subprocess.run(
            [sys.executable, "tools/research/analyze_chop_lane.py"],
            capture_output=True, text=True, timeout=10,
            cwd=str(Path(__file__).resolve().parent),
            env={**os.environ,
                 "SHADOW_CHOP_LANE_PATH": "/tmp/nonexistent_chop_lane.csv",
                 "SHADOW_CHOP_EXCEPTION_PATH": "/tmp/nonexistent_chop_ex.csv"},
        )
        self.assertEqual(result.returncode, 0, f"Crashed: {result.stderr}")
        self.assertIn("No chop lane data", result.stdout)


if __name__ == "__main__":
    unittest.main()

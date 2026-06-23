"""Tests for chop exception shadow lane."""

import os
import sys
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock

from executor_modules.chop_exception_shadow import evaluate_and_log_chop_exception

sys.path.insert(0, str(Path(__file__).resolve().parent / "tools" / "research"))
from replay_shadow_trades import is_chop_exception_candidate  # type: ignore[import-not-found]


@dataclass
class FakeSignal:
    coin: str = "SOL"
    side: str = "LONG"
    entry_price: float = 100.0
    stop_price: float = 98.0
    tp_price: float = 104.0
    confidence: float = 0.92
    regime: str = "continuation|htf_chop|macro_up|mkt_chop"
    meta: Dict[str, Any] = field(default_factory=dict)


class TestChopExceptionEvaluator(unittest.TestCase):

    def test_qualifying_signal_logged(self):
        path = tempfile.mktemp(suffix=".csv")
        for m in list(sys.modules):
            if "chop_exception_shadow" in m:
                del sys.modules[m]
        import executor_modules.chop_exception_shadow as mod
        mod.SHADOW_CHOP_PATH = path
        try:
            sig = FakeSignal(
                coin="SOL", side="LONG", confidence=0.92,
                meta={
                    "session": "ny_open",
                    "setup_family": "continuation",
                    "fvg_bull": True,
                    "market_regime": "chop",
                    "regime_htf_1h": "chop",
                    "regime_macro_4h": "up",
                    "total_score": 0.92,
                },
            )
            mod.evaluate_and_log_chop_exception(sig, "market_regime_block:chop")
            self.assertTrue(os.path.exists(path))
            with open(path) as fh:
                lines = fh.readlines()
            self.assertEqual(len(lines), 2)
            self.assertIn("SOL", lines[1])
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_low_confidence_not_logged(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        os.unlink(path)

        import executor_modules.chop_exception_shadow as mod
        orig = mod.SHADOW_CHOP_PATH
        mod.SHADOW_CHOP_PATH = path
        try:
            sig = FakeSignal(
                coin="SOL", confidence=0.70,
                meta={"session": "ny_open", "setup_family": "continuation", "fvg_bull": True},
            )
            evaluate_and_log_chop_exception(sig, "market_regime_block:chop")
            self.assertFalse(os.path.exists(path))
        finally:
            mod.SHADOW_CHOP_PATH = orig

    def test_bad_symbol_not_logged(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        os.unlink(path)

        import executor_modules.chop_exception_shadow as mod
        orig = mod.SHADOW_CHOP_PATH
        mod.SHADOW_CHOP_PATH = path
        try:
            sig = FakeSignal(
                coin="ETH", confidence=0.92,
                meta={"session": "ny_open", "setup_family": "continuation", "fvg_bull": True},
            )
            evaluate_and_log_chop_exception(sig, "market_regime_block:chop")
            self.assertFalse(os.path.exists(path))
        finally:
            mod.SHADOW_CHOP_PATH = orig

    def test_bad_session_not_logged(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        os.unlink(path)

        import executor_modules.chop_exception_shadow as mod
        orig = mod.SHADOW_CHOP_PATH
        mod.SHADOW_CHOP_PATH = path
        try:
            sig = FakeSignal(
                coin="SOL", confidence=0.92,
                meta={"session": "ny_pm", "setup_family": "continuation", "fvg_bull": True},
            )
            evaluate_and_log_chop_exception(sig, "market_regime_block:chop")
            self.assertFalse(os.path.exists(path))
        finally:
            mod.SHADOW_CHOP_PATH = orig

    def test_no_fvg_not_logged(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        os.unlink(path)

        import executor_modules.chop_exception_shadow as mod
        orig = mod.SHADOW_CHOP_PATH
        mod.SHADOW_CHOP_PATH = path
        try:
            sig = FakeSignal(
                coin="SOL", confidence=0.92,
                meta={"session": "ny_open", "setup_family": "continuation"},
            )
            evaluate_and_log_chop_exception(sig, "market_regime_block:chop")
            self.assertFalse(os.path.exists(path))
        finally:
            mod.SHADOW_CHOP_PATH = orig

    def test_reversal_not_logged(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        os.unlink(path)

        import executor_modules.chop_exception_shadow as mod
        orig = mod.SHADOW_CHOP_PATH
        mod.SHADOW_CHOP_PATH = path
        try:
            sig = FakeSignal(
                coin="SOL", confidence=0.92,
                meta={"session": "ny_open", "setup_family": "reversal", "fvg_bull": True},
            )
            evaluate_and_log_chop_exception(sig, "market_regime_block:chop")
            self.assertFalse(os.path.exists(path))
        finally:
            mod.SHADOW_CHOP_PATH = orig

    def test_wrong_reason_not_logged(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        os.unlink(path)

        import executor_modules.chop_exception_shadow as mod
        orig = mod.SHADOW_CHOP_PATH
        mod.SHADOW_CHOP_PATH = path
        try:
            sig = FakeSignal(
                coin="SOL", confidence=0.92,
                meta={"session": "ny_open", "setup_family": "continuation", "fvg_bull": True},
            )
            evaluate_and_log_chop_exception(sig, "swing_conf_gate:too_low")
            self.assertFalse(os.path.exists(path))
        finally:
            mod.SHADOW_CHOP_PATH = orig


class TestReplayChopDetection(unittest.TestCase):

    def test_qualifying_row(self):
        row = {
            "live_reject_reason": "market_regime_block:chop",
            "symbol": "SOL", "session": "ny_open",
            "setup_family": "continuation", "confidence_v1": "0.92",
            "score_v3_tags": "+fvg,+ny_open",
        }
        self.assertTrue(is_chop_exception_candidate(row))

    def test_bad_symbol_excluded(self):
        row = {
            "live_reject_reason": "market_regime_block:chop",
            "symbol": "ETH", "session": "ny_open",
            "setup_family": "continuation", "confidence_v1": "0.92",
            "score_v3_tags": "+fvg",
        }
        self.assertFalse(is_chop_exception_candidate(row))

    def test_bad_session_excluded(self):
        row = {
            "live_reject_reason": "market_regime_block:chop",
            "symbol": "SOL", "session": "london_late",
            "setup_family": "continuation", "confidence_v1": "0.92",
            "score_v3_tags": "+fvg",
        }
        self.assertFalse(is_chop_exception_candidate(row))

    def test_low_conf_excluded(self):
        row = {
            "live_reject_reason": "market_regime_block:chop",
            "symbol": "SOL", "session": "ny_open",
            "setup_family": "continuation", "confidence_v1": "0.70",
            "score_v3_tags": "+fvg",
        }
        self.assertFalse(is_chop_exception_candidate(row))

    def test_no_fvg_excluded(self):
        row = {
            "live_reject_reason": "market_regime_block:chop",
            "symbol": "SOL", "session": "ny_open",
            "setup_family": "continuation", "confidence_v1": "0.92",
            "score_v3_tags": "+ob_stop",
        }
        self.assertFalse(is_chop_exception_candidate(row))

    def test_non_chop_reason_excluded(self):
        row = {
            "live_reject_reason": "swing_conf_gate",
            "symbol": "SOL", "session": "ny_open",
            "setup_family": "continuation", "confidence_v1": "0.92",
            "score_v3_tags": "+fvg",
        }
        self.assertFalse(is_chop_exception_candidate(row))


if __name__ == "__main__":
    unittest.main()

"""
Acceptance test for the TRADING_DISABLED kill switch (2026-09-01 data-only
pivot) -- independent, redundant hard-stop separate from RESEARCH_ONLY_MODE.
Executor must refuse to construct entirely when it's set, before ever
reaching build_execution_backend, so no write-capable venue client exists.
"""
import importlib
import os
import sys
import unittest
from unittest.mock import patch


class TestTradingDisabledKillSwitch(unittest.TestCase):

    def _reload_executor_with_env(self, env):
        for mod in list(sys.modules):
            if mod == "executor":
                del sys.modules[mod]
        with patch.dict(os.environ, env):
            import executor
            importlib.reload(executor)
            return executor

    def test_trading_disabled_true_refuses_construction_before_backend_build(self):
        executor = self._reload_executor_with_env({"TRADING_DISABLED": "true"})
        with patch("executor.build_execution_backend") as mock_build:
            with self.assertRaises(RuntimeError):
                executor.Executor()
            mock_build.assert_not_called()

    def test_trading_disabled_defaults_true_when_env_var_absent(self):
        with patch.dict(os.environ):
            os.environ.pop("TRADING_DISABLED", None)
            for mod in list(sys.modules):
                if mod == "executor":
                    del sys.modules[mod]
            import executor
            importlib.reload(executor)
            self.assertTrue(executor.Executor._TRADING_DISABLED)
            with patch("executor.build_execution_backend") as mock_build:
                with self.assertRaises(RuntimeError):
                    executor.Executor()
                mock_build.assert_not_called()

    def test_trading_disabled_false_does_not_fire_the_gate(self):
        executor = self._reload_executor_with_env({"TRADING_DISABLED": "false"})
        self.assertFalse(executor.Executor._TRADING_DISABLED)


if __name__ == "__main__":
    unittest.main()

"""
Hard-constraint enforcement for research/wallet_tracking/: read-only
Etherscan data endpoints only, never anything capable of placing,
modifying, or cancelling an order, and no private key / signing
capability anywhere in the client. Same two-layer verification as the
funding and cascade collectors' isolation tests.
"""

import ast
import sys
import unittest
from pathlib import Path

FORBIDDEN_MODULES = {
    "executor", "risk_manager", "protection_manager", "order_tracker",
    "live_position_monitor", "execution_backend", "execution_backend_factory",
    "live_execution_backend", "bootstrap", "agent",
}

PACKAGE_DIR = Path(__file__).resolve().parent / "research" / "wallet_tracking"


class TestResearchWalletTrackingIsolation(unittest.TestCase):

    def test_no_forbidden_module_in_runtime_import_graph(self):
        for mod in list(sys.modules):
            if mod.startswith("research."):
                del sys.modules[mod]

        before = set(sys.modules.keys())
        import research.wallet_tracking.analyze  # noqa: F401
        after = set(sys.modules.keys())

        newly_loaded = after - before
        offending = FORBIDDEN_MODULES & newly_loaded
        self.assertEqual(
            offending, set(),
            f"importing research.wallet_tracking.analyze newly loaded forbidden "
            f"write-capable module(s): {offending}",
        )

    def test_no_forbidden_import_statement_in_source(self):
        py_files = list(PACKAGE_DIR.glob("*.py"))
        self.assertGreater(len(py_files), 0, "no source files found to check")

        for path in py_files:
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = {alias.name.split(".")[0] for alias in node.names}
                elif isinstance(node, ast.ImportFrom):
                    names = {node.module.split(".")[0]} if node.module else set()
                else:
                    continue
                offending = FORBIDDEN_MODULES & names
                self.assertEqual(
                    offending, set(),
                    f"{path.name} imports forbidden module(s): {offending}",
                )

    def test_etherscan_client_holds_no_write_capability(self):
        import os
        os.environ.setdefault("ETHERSCAN_API_KEY", "test_key_for_isolation_check")
        from research.wallet_tracking.etherscan_client import EtherscanReadOnlyClient
        client = EtherscanReadOnlyClient()
        forbidden_attrs = {"private_key", "wallet", "signer", "sign", "account"}
        present = forbidden_attrs & set(dir(client))
        self.assertEqual(present, set())

    def test_etherscan_client_has_no_order_or_write_methods(self):
        import os
        os.environ.setdefault("ETHERSCAN_API_KEY", "test_key_for_isolation_check")
        from research.wallet_tracking.etherscan_client import EtherscanReadOnlyClient
        client = EtherscanReadOnlyClient()
        forbidden_methods = {"execute_entry", "execute_exit", "place_order",
                              "cancel_order", "send_transaction", "sign_transaction"}
        present = forbidden_methods & set(dir(client))
        self.assertEqual(present, set())


if __name__ == "__main__":
    unittest.main()

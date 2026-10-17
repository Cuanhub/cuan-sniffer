"""
Hard-constraint enforcement for research/levels/ (support/resistance level
scanner, 2026-09-17): read-only venue info endpoints only, never anything
capable of placing/modifying/cancelling an order. Same two-layer
verification as the funding and cascade isolation tests.

This package suggests entry, stop and take-profit prices, which makes the
constraint matter more here than in the other two rather than less: those
are numbers a reader could mistake for instructions to act on. Nothing in
this package can act on them.
"""

import ast
import sys
import unittest
from pathlib import Path

FORBIDDEN_MODULES = {
    "executor", "risk_manager", "protection_manager", "order_tracker",
    "live_position_monitor", "execution_backend_factory", "bootstrap", "agent",
    # core/ places orders. research/ is read-only by design and must never
    # reach it, the same reason the retired trading modules are listed here.
    "core.venue", "core.execute", "core.safety", "core.protect",
}

PACKAGE_DIR = Path(__file__).resolve().parent / "research" / "levels"


class TestResearchLevelsImportIsolation(unittest.TestCase):

    def test_no_forbidden_module_in_runtime_import_graph(self):
        for mod in list(sys.modules):
            if mod.startswith("research."):
                del sys.modules[mod]

        before = set(sys.modules.keys())
        import research.levels.scanner  # noqa: F401
        after = set(sys.modules.keys())

        newly_loaded = after - before
        offending = FORBIDDEN_MODULES & newly_loaded
        self.assertEqual(
            offending, set(),
            f"importing research.levels.scanner newly loaded forbidden "
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

    def test_venue_client_holds_no_write_capability(self):
        from research.levels.venue_client import HyperliquidReadOnlyClient
        client = HyperliquidReadOnlyClient()
        forbidden_attrs = {"api_key", "private_key", "wallet", "signer", "sign", "account"}
        present = forbidden_attrs & set(dir(client))
        self.assertEqual(present, set())

    def test_venue_client_has_no_order_methods(self):
        from research.levels.venue_client import HyperliquidReadOnlyClient
        client = HyperliquidReadOnlyClient()
        forbidden_methods = {"execute_entry", "execute_exit", "place_order",
                              "cancel_order", "place_native_protection", "place_stop_only"}
        present = forbidden_methods & set(dir(client))
        self.assertEqual(present, set())


if __name__ == "__main__":
    unittest.main()

"""
Hard-constraint enforcement for research/funding/ (Phase 3, 2026-09-01
data-only pivot): it may call read-only venue info endpoints only, and
must never import anything capable of placing, modifying, or cancelling
an order. This is not a style preference -- it's the actual safety
boundary the brief specified, verified against the real import graph at
runtime, not just by inspecting import statements textually.
"""

import ast
import sys
import unittest
from pathlib import Path

FORBIDDEN_MODULES = {
    "executor",
    "risk_manager",
    "protection_manager",
    "order_tracker",
    "live_position_monitor",
    "execution_backend",
    "execution_backend_factory",
    "live_execution_backend",
    "bootstrap",
    "agent",
}

PACKAGE_DIR = Path(__file__).resolve().parent / "research" / "funding"


class TestResearchFundingImportIsolation(unittest.TestCase):

    def test_no_forbidden_module_in_runtime_import_graph(self):
        """The real test: actually import the collector and check whether
        THAT import caused any forbidden module to load. Compares a
        before/after delta rather than the final sys.modules state --
        running inside the full suite, an unrelated earlier test may have
        already loaded executor/risk_manager/etc. for its own purposes,
        and that pre-existing pollution isn't a violation this package
        caused. Catches indirect/transitive imports a textual grep could
        miss, without producing false positives from test order."""
        for mod in list(sys.modules):
            if mod.startswith("research."):
                del sys.modules[mod]

        before = set(sys.modules.keys())
        import research.funding.collector  # noqa: F401
        after = set(sys.modules.keys())

        newly_loaded = after - before
        offending = FORBIDDEN_MODULES & newly_loaded
        self.assertEqual(
            offending, set(),
            f"importing research.funding.collector newly loaded forbidden "
            f"write-capable module(s): {offending}",
        )

    def test_no_forbidden_import_statement_in_source(self):
        """Static check as a second, independent layer -- catches a
        forbidden import even in a code path not exercised by the runtime
        check above (e.g. behind a conditional the test doesn't trigger)."""
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
        """No API key, wallet, or signing-related attribute exists on the
        client at all -- it's not just unused, it's structurally absent."""
        from research.funding.venue_client import HyperliquidReadOnlyClient
        client = HyperliquidReadOnlyClient()
        forbidden_attrs = {"api_key", "private_key", "wallet", "signer", "sign", "account"}
        present = forbidden_attrs & set(dir(client))
        self.assertEqual(present, set())

    def test_venue_client_has_no_order_methods(self):
        from research.funding.venue_client import HyperliquidReadOnlyClient
        client = HyperliquidReadOnlyClient()
        forbidden_methods = {"execute_entry", "execute_exit", "place_order",
                              "cancel_order", "place_native_protection", "place_stop_only"}
        present = forbidden_methods & set(dir(client))
        self.assertEqual(present, set())


if __name__ == "__main__":
    unittest.main()

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
import subprocess
import sys
import unittest
from pathlib import Path

FORBIDDEN_MODULES = {
    "executor", "risk_manager", "protection_manager", "order_tracker",
    "live_position_monitor", "execution_backend_factory", "bootstrap", "agent",
    # core/ places orders. research/ is read-only by design and must never
    # reach it, the same reason the retired trading modules are listed here.
    "core", "notify", "tools", "hyperliquid.exchange", "eth_account",
}

PACKAGE_DIR = Path(__file__).resolve().parent / "research" / "levels"


class TestResearchLevelsImportIsolation(unittest.TestCase):
    """Both checks were rewritten after the 2026-10-27 audit, when a smuggled
    `from core.venue import Venue` in research/levels/zones.py PASSED them:

      - the static check compared only the FIRST dotted component ("core")
        against names like "core.venue", so a dotted import never matched;
      - the runtime check asked whether importing the scanner NEWLY loaded a
        forbidden module -- inside a pytest run where earlier tests had
        already imported core.venue, so it was never new.

    A guard that cannot fail is not a guard. Names now match by prefix, and
    the runtime check runs in a fresh interpreter with no history.
    """

    @staticmethod
    def _forbidden(name: str) -> bool:
        return any(name == f or name.startswith(f + ".") for f in FORBIDDEN_MODULES)

    def test_no_forbidden_module_in_runtime_import_graph(self):
        code = ("import sys, pkgutil, importlib, research.levels as P\n"
                "for m in pkgutil.iter_modules(P.__path__):\n"
                "    importlib.import_module('research.levels.' + m.name)\n"
                "print('\\n'.join(sorted(sys.modules)))\n")
        out = subprocess.run([sys.executable, "-c", code], cwd=PACKAGE_DIR.parents[1],
                             capture_output=True, text=True, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr[-2000:])
        loaded = out.stdout.split()
        self.assertIn("research.levels.scanner", loaded, "the check imported nothing")
        offending = sorted(m for m in loaded if self._forbidden(m))
        self.assertEqual(offending, [], f"research/levels loads write-capable module(s): "
                                        f"{offending}")

    def test_no_forbidden_import_statement_in_source(self):
        py_files = list(PACKAGE_DIR.glob("*.py"))
        self.assertGreater(len(py_files), 0, "no source files found to check")

        for path in py_files:
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module] if node.module else []
                else:
                    continue
                offending = [n for n in names if self._forbidden(n)]
                self.assertEqual(offending, [],
                                 f"{path.name} imports forbidden module(s): {offending}")

    def test_the_static_check_matches_a_dotted_import(self):
        self.assertTrue(self._forbidden("core.venue"))
        self.assertTrue(self._forbidden("core"))
        self.assertFalse(self._forbidden("research.levels.zones"))

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

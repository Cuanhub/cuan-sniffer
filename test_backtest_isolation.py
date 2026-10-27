"""research/backtest/ cannot reach anything that places orders.

Checked in a FRESH interpreter. The sibling test for research/levels runs in
the pytest process, where `core.venue` is usually already imported by an
earlier test -- so importing it again is not "newly loaded" and the check
passes whatever the code does. A subprocess has no such history.
"""

import ast
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PKG = ROOT / "research" / "backtest"

ORDER_CAPABLE = ("core", "hyperliquid.exchange", "eth_account", "tools")


class ImportGraph(unittest.TestCase):
    def test_nothing_order_capable_is_loaded_by_the_experiment(self):
        code = (
            "import sys\n"
            "import research.backtest.experiments.trendline_breakout\n"
            "import research.backtest.band, research.backtest.costs\n"
            "bad = sorted(m for m in sys.modules if m == 'core' or m.startswith('core.')\n"
            "             or m.startswith('hyperliquid.exchange') or m.startswith('eth_account')\n"
            "             or m.startswith('tools.'))\n"
            "print(','.join(bad))\n")
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                             capture_output=True, text=True, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.strip(), "",
                         f"research/backtest loaded: {out.stdout.strip()}")

    def test_no_import_statement_names_an_order_capable_module(self):
        files = list(PKG.rglob("*.py"))
        self.assertGreater(len(files), 3)
        for path in files:
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                else:
                    continue
                for name in names:
                    for bad in ORDER_CAPABLE:
                        self.assertFalse(
                            name == bad or name.startswith(bad + "."),
                            f"{path.relative_to(ROOT)} imports {name}")

    def test_the_static_check_would_catch_a_dotted_import(self):
        """The research/levels static check compares only the FIRST dotted
        component against names like "core.venue", so `from core.venue
        import Venue` sails through it. This one must not."""
        tree = ast.parse("from core.venue import Venue")
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom))
        self.assertTrue(any(node.module == b or node.module.startswith(b + ".")
                            for b in ORDER_CAPABLE))


if __name__ == "__main__":
    unittest.main()

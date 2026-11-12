"""research/trend/ cannot reach anything that places orders. Checked in a
fresh interpreter, as test_backtest_isolation.py is, and for its reason."""

import ast
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PKG = ROOT / "research" / "trend"
ORDER_CAPABLE = ("core", "hyperliquid", "eth_account", "tools")


class ImportGraph(unittest.TestCase):
    def test_nothing_order_capable_is_loaded_by_the_paper_pass(self):
        code = ("import sys\nimport research.trend.paper\n"
                "bad = sorted(m for m in sys.modules if m.split('.')[0] in "
                f"{ORDER_CAPABLE!r})\nprint(','.join(bad))\n")
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                             capture_output=True, text=True, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.strip(), "")

    def test_no_import_statement_names_an_order_capable_module(self):
        files = list(PKG.rglob("*.py"))
        self.assertGreaterEqual(len(files), 2)
        for path in files:
            for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                else:
                    continue
                for name in names:
                    self.assertNotIn(name.split(".")[0], ORDER_CAPABLE,
                                     f"{path.name} imports {name}")

    def test_the_only_venue_client_it_names_is_the_read_only_one(self):
        src = (PKG / "paper.py").read_text()
        self.assertIn("HyperliquidReadOnlyClient", src)
        for word in ("send_ioc", "send_limit", "send_trigger", "Exchange(", "private_key"):
            self.assertNotIn(word, src)


if __name__ == "__main__":
    unittest.main()

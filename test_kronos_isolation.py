"""Kronos stays a research instrument.

  - the live code (core/, tools/) never imports it, nor PyTorch;
  - nothing in research/kronos imports anything that can place an order;
  - the Kronos environment's lock file has no Hyperliquid SDK or signing
    library, so nothing run from it could sign an order;
  - inputs.py and score.py need no PyTorch, so the main suite tests them;
  - the vendored model code is upstream's byte for byte.
"""

import ast
import hashlib
import re
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KRONOS = ROOT / "research" / "kronos"
ORDER_CAPABLE = ("core", "tools", "hyperliquid", "eth_account", "eth_keys", "web3")


def imported(path: Path):
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module


def names(path: Path, prefixes):
    return [n for n in imported(path)
            if any(n == p or n.startswith(p + ".") for p in prefixes)]


def sources(top: Path):
    return [p for p in top.rglob("*.py") if ".venv" not in p.parts and "hf" not in p.parts]


class KronosIsolation(unittest.TestCase):
    def test_live_code_never_imports_kronos_or_torch(self):
        files = sources(ROOT / "core") + sources(ROOT / "tools")
        self.assertGreater(len(files), 10)
        for f in files:
            self.assertEqual(names(f, ("research.kronos", "torch")), [],
                             f"{f.relative_to(ROOT)}")

    def test_kronos_code_imports_nothing_order_capable(self):
        files = sources(KRONOS) + [ROOT / "research/backtest/experiments/kronos_filter.py"]
        self.assertGreater(len(files), 6)
        for f in files:
            self.assertEqual(names(f, ORDER_CAPABLE), [], f"{f.relative_to(ROOT)}")

    def test_the_scoring_half_needs_no_torch(self):
        for f in ("inputs.py", "score.py"):
            self.assertEqual(names(KRONOS / f, ("torch", "research.kronos.forecast",
                                                "research.kronos.fast")), [], f)

    def test_the_experiment_module_loads_without_torch_or_order_capable_code(self):
        code = ("import sys\n"
                "import research.backtest.experiments.kronos_filter\n"
                "bad = sorted(m for m in sys.modules if m.split('.')[0] in "
                "('torch', 'core', 'tools', 'eth_account') or m.startswith('hyperliquid'))\n"
                "print(','.join(bad))\n")
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                             capture_output=True, text=True, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.strip(), "")

    def test_the_kronos_environment_cannot_sign(self):
        lock = (KRONOS / "requirements.txt").read_text().lower()
        self.assertIn("torch==", lock)
        for pkg in ("hyperliquid", "eth-account", "eth_account", "eth-keys", "web3",
                    "python-dotenv"):
            self.assertNotIn(pkg, lock)

    def test_the_vendored_code_is_upstreams(self):
        prov = (KRONOS / "vendor" / "PROVENANCE.md").read_text()
        want = dict(re.findall(r"\| (model/\w+\.py|LICENSE) \| ([0-9a-f]{64}) \|", prov))
        self.assertEqual(len(want), 4)
        for rel, sha in want.items():
            got = hashlib.sha256((KRONOS / "vendor" / rel).read_bytes()).hexdigest()
            self.assertEqual(got, sha, rel)


if __name__ == "__main__":
    unittest.main()

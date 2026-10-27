"""The OpenClaw lab package: built clean, refuses unsafe contents, self-contained.

The package is meant to run unattended under an LLM agent, so what it must
NOT contain matters more than what it does: anything that can trade, any
secret, the live data. These tests build it for real and check.
"""

import json
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tools import build_openclaw_package as B

REPO = Path(__file__).resolve().parent


class BuiltPackage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.out = B.build(cls.tmp / "cuan-lab", tarball=False, log=lambda *_: None)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_the_build_passes_its_own_checks(self):
        self.assertEqual(B.check(self.out), [])

    def test_no_order_capable_code_is_present(self):
        lab = self.out / "lab"
        self.assertFalse((lab / "core").exists())
        for name in ("live_levels_trader.py", "telegram_control.py", "protect_loop.py"):
            self.assertFalse((lab / "tools" / name).exists(), name)
        self.assertFalse((lab / "notify.py").exists())

    def test_no_live_data_is_shipped(self):
        data = self.out / "lab" / "research" / "levels" / "data"
        self.assertEqual(sorted(p.name for p in data.iterdir()), [".gitkeep"])

    def test_the_workspace_bootstrap_files_exist(self):
        for name in ("AGENTS.md", "SOUL.md", "USER.md", "MEMORY.md", "HEARTBEAT.md",
                     "AUTOMATIONS.md", "README.md"):
            self.assertTrue((self.out / name).is_file(), name)

    def test_every_skill_has_valid_openclaw_frontmatter(self):
        skills = sorted((self.out / "skills").glob("*/SKILL.md"))
        self.assertGreaterEqual(len(skills), 5)
        for p in skills:
            text = p.read_text()
            m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
            self.assertIsNotNone(m, f"{p}: no frontmatter")
            fm = dict(line.split(": ", 1) for line in m.group(1).splitlines())
            self.assertEqual(fm.get("name"), p.parent.name, p)
            self.assertTrue(fm.get("description"), p)
            self.assertLessEqual(len(fm["description"]), 512, p)
            meta = json.loads(fm["metadata"])       # single-line JSON, as OpenClaw requires
            self.assertIn("openclaw", meta)
            self.assertIn("{baseDir}", text, f"{p}: should locate the lab via {{baseDir}}")

    def test_the_seeded_ledger_is_intact_and_honest(self):
        sys.path.insert(0, str(self.out / "lab"))
        try:
            from labcore import ledger
            path = self.out / "lab" / "ledger" / "ledger.jsonl"
            self.assertIsNone(ledger.verify(path))
            self.assertGreaterEqual(ledger.tested_count(path), 41)
            ids = {r["id"] for r in ledger.entries(path)}
            self.assertIn("H-TRENDLINE", ids)
        finally:
            sys.path.remove(str(self.out / "lab"))
            for m in [m for m in sys.modules if m == "labcore" or m.startswith("labcore.")]:
                del sys.modules[m]

    def test_the_manifest_covers_every_file(self):
        man = json.loads((self.out / "MANIFEST.json").read_text())
        # Bytecode caches appear as soon as anything imports from the package
        # (other tests here do); they are not shipped content.
        files = {str(p.relative_to(self.out)) for p in self.out.rglob("*")
                 if p.is_file() and p.name != "MANIFEST.json"
                 and "__pycache__" not in p.parts and ".pytest_cache" not in p.parts}
        self.assertEqual(set(man["files"]), files)

    def test_the_lab_suite_passes_inside_the_package(self):
        """Self-contained: run from the built lab with nothing from this repo
        on the path."""
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
                           cwd=self.out / "lab", capture_output=True, text=True, timeout=900,
                           env={"PATH": "/usr/bin:/bin", "HOME": str(self.tmp)})
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-2000:])


class RefusesUnsafeContents(unittest.TestCase):
    def setUp(self):
        self.out = Path(tempfile.mkdtemp()) / "pkg"
        (self.out / "lab").mkdir(parents=True)
        (self.out / "lab" / "ok.py").write_text("import json\n")

    def test_order_capable_code_is_refused(self):
        (self.out / "lab" / "core").mkdir()
        (self.out / "lab" / "core" / "venue.py").write_text("x = 1\n")
        self.assertTrue(any("core/" in p for p in B.check(self.out)))

    def test_an_import_of_core_is_refused(self):
        (self.out / "lab" / "sneaky.py").write_text("from core.execute import open\n")
        self.assertTrue(any("imports core.execute" in p for p in B.check(self.out)))

    def test_a_secrets_file_is_refused(self):
        (self.out / ".env").write_text("TELEGRAM_BOT_TOKEN=x\n")
        self.assertTrue(any("secrets" in p for p in B.check(self.out)))

    def test_something_shaped_like_a_private_key_is_refused(self):
        (self.out / "lab" / "notes.md").write_text("key 0x" + "ab" * 32 + "\n")
        self.assertTrue(any("private key" in p for p in B.check(self.out)))

    def test_the_account_address_is_refused(self):
        addr = B._account_address()
        if not addr:
            self.skipTest("no account address configured on this machine")
        (self.out / "lab" / "notes.md").write_text(f"wallet {addr.upper()}\n")
        self.assertTrue(any("account address" in p for p in B.check(self.out)))

    def test_a_cache_directory_is_refused(self):
        d = self.out / "lab" / ".mypy_cache" / "3.13"
        d.mkdir(parents=True)
        (d / "cache.db").write_bytes(b"x")
        self.assertTrue(any("build artifact" in p for p in B.check(self.out)))

    def test_live_data_is_refused(self):
        d = self.out / "lab" / "research" / "levels" / "data"
        d.mkdir(parents=True)
        (d / "live_trades.csv").write_text("x\n")
        self.assertTrue(any("live path" in p for p in B.check(self.out)))


if __name__ == "__main__":
    unittest.main()

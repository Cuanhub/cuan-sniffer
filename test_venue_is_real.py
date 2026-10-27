"""The live system talks to a stated venue and invents nothing about it.

From the 2026-10-27 audit: three places where the venue layer substituted an
assumption for a fact. Each test here fails on the code as it was.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import core.venue as V
from core import config_audit, protect
from tests.fake_venue import FakeHL, build_venue


class TheNetworkIsStated(unittest.TestCase):
    """It defaulted to testnet: a missing variable would have run the live bot
    on testnet while every log and alert looked normal."""

    def construct(self, raw):
        with mock.patch.object(V, "HL_ACCOUNT_ADDRESS", "0x" + "11" * 20), \
             mock.patch.object(V, "HL_SECRET_KEY", "0x" + "22" * 32), \
             mock.patch.object(V, "HL_TESTNET_RAW", raw), \
             mock.patch.object(V, "Info", side_effect=AssertionError("reached the network")), \
             mock.patch.object(V, "Exchange", side_effect=AssertionError("reached the network")):
            return V.Venue()

    def test_unset_refuses_before_touching_the_network(self):
        with self.assertRaises(ValueError) as cm:
            self.construct("")
        self.assertIn("HL_TESTNET", str(cm.exception))

    def test_anything_but_true_or_false_refuses(self):
        for raw in ("yes", "0", "mainnet", "flase"):
            with self.assertRaises(ValueError, msg=raw):
                self.construct(raw)

    def test_an_explicit_value_proceeds(self):
        for raw in ("true", "false"):
            with self.assertRaises(AssertionError, msg=raw) as cm:
                self.construct(raw)                 # got as far as building Info
            self.assertIn("network", str(cm.exception))

    def test_with_the_variable_truly_unset_there_is_no_network(self):
        """Behavioural, in a fresh interpreter: patching the parsed value (as
        the tests above do) cannot see a default reintroduced in getenv."""
        import os
        import subprocess
        import sys
        env = {k: v for k, v in os.environ.items() if k != "HL_TESTNET"}
        code = ("import core.venue as V\n"
                "print(repr(V.HL_TESTNET_RAW), V.HL_TESTNET)\n")
        out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                             text=True, cwd=Path(__file__).resolve().parent, timeout=60)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.strip(), "'' False")

    def test_the_live_env_states_mainnet(self):
        """This repo's deployment trades real money; its .env must say so."""
        env = Path(__file__).resolve().parent / ".env"
        if not env.exists():
            self.skipTest("no .env on this machine")
        lines = [l.strip() for l in env.read_text().splitlines()
                 if l.strip().startswith("HL_TESTNET=")]
        self.assertEqual(lines, ["HL_TESTNET=false"])


class PrecisionIsReadNotInvented(unittest.TestCase):
    def test_a_coin_without_szdecimals_is_unknown_not_three(self):
        fake = FakeHL()
        real_meta = fake.meta

        def meta(dex=""):
            m = real_meta(dex)
            for a in m["universe"]:
                if a.get("name") == "BTC":
                    a.pop("szDecimals", None)
            return m
        fake.meta = meta
        v = build_venue(fake)
        self.assertFalse(v.precision_known("BTC"))
        self.assertEqual(v.round_size("BTC", 0.0123), 0.0)        # refused, not guessed
        self.assertTrue(v.precision_known("ETH"))


class AgeIsReadNotInvented(unittest.TestCase):
    def order(self, ts):
        return V.Order(coin="HYPE", oid=1, size=1.0, is_buy=True, reduce_only=False,
                       is_trigger=False, trigger_px=None, is_stop=False,
                       limit_px=70.0, timestamp_ms=ts)

    def test_unknown_age_is_none_not_zero(self):
        self.assertIsNone(self.order(0).age_hours())
        self.assertAlmostEqual(self.order(1_000_000).age_hours(now_ms=1_000_000 + 7_200_000), 2.0)

    def test_a_bid_of_unknown_age_is_pulled(self):
        fake = FakeHL()
        fake.rest_plain_order("HYPE", 1.0, 70.0)
        fake.resting[-1]["timestamp"] = 0              # the venue did not say
        fake.rest_plain_order("ETH", 0.1, 2000.0)      # a fresh bid: kept
        v = build_venue(fake)
        with mock.patch.object(v, "cancel", return_value={"status": "ok"}) as cancel:
            pulled = protect.expire_bids(v)
        self.assertEqual(pulled, ["HYPE"])
        self.assertEqual(cancel.call_count, 1)


class DeadConfiguration(unittest.TestCase):
    def env(self, text):
        p = Path(tempfile.mkdtemp()) / ".env"
        p.write_text(text)
        return p

    def test_the_keys_that_matter_are_seen_as_read(self):
        read = config_audit.keys_read()
        for k in ("HL_TESTNET", "HL_ACCOUNT_ADDRESS", "HL_SECRET_KEY", "HL_VAULT_ADDRESS",
                  "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "LEVELS_LIVE_ARMED",
                  "LEVELS_MIN_RR", "LEVELS_ZONE_STOP_ATR_MULT", "SAFETY_READ_DEAD_SEC"):
            self.assertIn(k, read)

    def test_every_documented_key_is_seen_as_read(self):
        """.env.example offers only real knobs (test_static_checks), so every
        one of them must pass the guard -- or setting a documented knob would
        stop the trader arming."""
        example = Path(__file__).resolve().parent / ".env.example"
        self.assertEqual(config_audit.dead_keys(example), [])

    def test_legacy_controls_are_caught_and_comments_ignored(self):
        p = self.env("HL_TESTNET=false\n# PAUSE_NEW_SIGNALS=true\nRESEARCH_ONLY_MODE=true\n"
                     "export TRADING_DISABLED=true\nLEVELS_MIN_RR=2.0\n")
        self.assertEqual(config_audit.dead_keys(p), ["RESEARCH_ONLY_MODE", "TRADING_DISABLED"])

    def test_the_refusal_names_them(self):
        msg = config_audit.refusal(["RESEARCH_ONLY_MODE"])
        self.assertIn("RESEARCH_ONLY_MODE", msg)
        self.assertEqual(config_audit.refusal([]), "")

    def test_the_trader_checks_before_announcing_its_mode(self):
        import inspect
        import tools.live_levels_trader as T
        src = inspect.getsource(T.main)
        self.assertLess(src.index("config_audit.refusal"), src.index('mode: {'))


if __name__ == "__main__":
    unittest.main()

"""Tests for notify.py.

Alerting is the one part of this system whose failure is silent by
construction: if a message never arrives, nothing anywhere reports that. So
the delivery rules are pinned here rather than trusted.

Two of them carry real weight. A send must never raise, because every call
site is inside the trading loop and a Telegram outage is not a trading
condition. And the cooldown must be stamped on success only -- a message that
failed was seen by nobody, so suppressing its retry would turn one dropped
packet into permanent silence.
"""

import os
import unittest

import notify


class FakeResp:
    def __init__(self, ok):
        self._ok = ok

    def json(self):
        return {"ok": self._ok}


class NotifyTest(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self._orig_transport = notify._transport
        self._orig_env = {k: os.environ.get(k)
                          for k in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID")}
        os.environ["TELEGRAM_BOT_TOKEN"] = "test-token"
        os.environ["TELEGRAM_CHAT_ID"] = "test-chat"
        notify.reset_cooldowns()

    def tearDown(self):
        notify._transport = self._orig_transport
        for k, v in self._orig_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        notify.reset_cooldowns()

    def transport(self, *results):
        """Each result is True/False for Telegram's ok flag, or an Exception."""
        it = iter(results)

        def _t(url, payload):
            self.calls.append(payload)
            r = next(it)
            if isinstance(r, Exception):
                raise r
            return FakeResp(r)

        notify._transport = _t

    # -- delivery ----------------------------------------------------------

    def test_markdown_is_tried_first(self):
        self.transport(True)
        self.assertTrue(notify.send("hello"))
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0]["parse_mode"], "Markdown")

    def test_falls_back_to_plain_when_markdown_is_rejected(self):
        # An unescaped _ or * in a coin name or a quoted venue error.
        self.transport(False, True)
        self.assertTrue(notify.send("PUMP_FUN broke *everything*"))
        self.assertEqual(len(self.calls), 2)
        self.assertNotIn("parse_mode", self.calls[1])

    def test_falls_back_when_markdown_raises(self):
        self.transport(RuntimeError("connection reset"), True)
        self.assertTrue(notify.send("hi"))
        self.assertEqual(len(self.calls), 2)

    def test_returns_false_when_both_attempts_fail(self):
        self.transport(False, False)
        self.assertFalse(notify.send("hi"))

    def test_never_raises(self):
        self.transport(RuntimeError("down"), RuntimeError("still down"))
        self.assertFalse(notify.send("hi"))   # must return, not raise

    def test_unconfigured_sends_nothing(self):
        os.environ.pop("TELEGRAM_BOT_TOKEN", None)
        self.transport(True)
        self.assertFalse(notify.send("hi"))
        self.assertEqual(self.calls, [])

    # -- rate limiting -----------------------------------------------------

    def test_cooldown_suppresses_a_repeat(self):
        self.transport(True, True)
        self.assertTrue(notify.send("halted", key="halt", cooldown_sec=60))
        self.assertFalse(notify.send("halted", key="halt", cooldown_sec=60))
        self.assertEqual(len(self.calls), 1)

    def test_cooldown_is_not_stamped_on_failure(self):
        # A message nobody received must not suppress its own retry.
        self.transport(False, False, True)
        self.assertFalse(notify.send("halted", key="halt", cooldown_sec=60))
        self.assertTrue(notify.send("halted", key="halt", cooldown_sec=60))

    def test_unkeyed_sends_are_never_suppressed(self):
        self.transport(True, True)
        self.assertTrue(notify.send("a"))
        self.assertTrue(notify.send("a"))
        self.assertEqual(len(self.calls), 2)


if __name__ == "__main__":
    unittest.main()

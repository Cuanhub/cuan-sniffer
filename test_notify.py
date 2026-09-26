"""Tests for notify.py.

Alerting is the one part of this system whose failure is silent by
construction: if a message never arrives, nothing anywhere reports that. So
the delivery rules are pinned here rather than trusted.

Two of them carry real weight. A send must never raise, because every call
site is inside the trading loop and a Telegram outage is not a trading
condition. And the cooldown must be stamped on success only -- a message that
failed was seen by nobody, so suppressing its retry would turn one dropped
packet into permanent silence.

The third is newer. A transport failure is retried and a Telegram rejection is
not, because they mean opposite things: a raised exception says the request
never landed, while an ok:false says it landed and the text was refused.
Retrying a refusal just burns the same error three times, and NOT retrying a
blip loses the message -- which is how both OPENED alerts for 2026-09-23 were
lost while the trades themselves went through fine.
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
        self.sleeps = []
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
        """Each result is True/False for Telegram's ok flag, or an Exception.
        Running off the end is an error, not a silent extra failure: the point
        of several of these tests is the exact number of attempts made, and an
        exhausted iterator would otherwise read as one more transport blip."""
        it = iter(results)

        def _t(url, payload):
            self.calls.append(payload)
            try:
                r = next(it)
            except StopIteration:
                raise AssertionError(
                    f"transport called {len(self.calls)} times, "
                    f"only {len(results)} results were scripted") from None
            if isinstance(r, Exception):
                raise r
            return FakeResp(r)

        notify._transport = _t

    def send(self, text, **kw):
        """Every send in here records its backoff instead of serving it."""
        kw.setdefault("sleep", self.sleeps.append)
        return notify.send(text, **kw)

    # -- delivery ----------------------------------------------------------

    def test_markdown_is_tried_first(self):
        self.transport(True)
        self.assertTrue(self.send("hello"))
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0]["parse_mode"], "Markdown")

    def test_falls_back_to_plain_when_markdown_is_rejected(self):
        # An unescaped _ or * in a coin name or a quoted venue error.
        self.transport(False, True)
        self.assertTrue(self.send("PUMP_FUN broke *everything*"))
        self.assertEqual(len(self.calls), 2)
        self.assertNotIn("parse_mode", self.calls[1])

    def test_returns_false_when_both_attempts_fail(self):
        self.transport(False, False)
        self.assertFalse(self.send("hi"))

    def test_never_raises(self):
        self.transport(RuntimeError("down"), RuntimeError("down"),
                       RuntimeError("down"))
        self.assertFalse(self.send("hi"))   # must return, not raise

    # -- retrying a blip ---------------------------------------------------

    def test_a_transient_failure_is_retried(self):
        # The 2026-09-23 case: one bad POST, then the network is fine again.
        self.transport(OSError("connection reset by peer"), True)
        self.assertTrue(self.send("OPENED DOGE"))
        self.assertEqual(len(self.calls), 2)

    def test_a_transport_failure_retries_markdown_rather_than_plain(self):
        # An exception means the request never landed, so the formatting was
        # never the problem and dropping to plain text would lose the markup
        # for no reason.
        self.transport(OSError("timed out"), True)
        self.assertTrue(self.send("*bold*"))
        self.assertEqual(self.calls[1].get("parse_mode"), "Markdown")

    def test_a_rejection_is_not_retried(self):
        # Telegram answered twice and refused both forms. A third round trip
        # would fail identically, and send() is inside the trading pass.
        self.transport(False, False)
        self.assertFalse(self.send("hi", attempts=3))
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(self.sleeps, [])

    def test_it_gives_up_after_the_attempt_limit(self):
        self.transport(*[OSError("down")] * 3)
        self.assertFalse(self.send("hi", attempts=3))
        self.assertEqual(len(self.calls), 3)

    def test_backoff_grows_and_none_follows_the_last_attempt(self):
        # A pass must not be delayed by a sleep that buys nothing.
        self.transport(*[OSError("down")] * 3)
        self.assertFalse(self.send("hi", attempts=3))
        self.assertEqual(self.sleeps, [notify.SEND_BACKOFF_SEC,
                                       notify.SEND_BACKOFF_SEC * 2])

    def test_an_unreadable_reply_is_retried(self):
        # A proxy or captive portal answering with HTML, not JSON.
        class Garbage:
            def json(self):
                raise ValueError("Expecting value: line 1 column 1")

        seen = []

        def _t(url, payload):
            seen.append(payload)
            return FakeResp(True) if len(seen) > 1 else Garbage()

        notify._transport = _t
        self.assertTrue(self.send("hi"))
        self.assertEqual(len(seen), 2)

    def test_the_whole_retry_budget_cannot_stall_a_pass(self):
        self.transport(*[OSError("down")] * notify.SEND_ATTEMPTS)
        self.assertFalse(self.send("hi"))
        self.assertLess(sum(self.sleeps), 10.0)

    def test_unconfigured_sends_nothing(self):
        os.environ.pop("TELEGRAM_BOT_TOKEN", None)
        self.transport(True)
        self.assertFalse(self.send("hi"))
        self.assertEqual(self.calls, [])

    # -- rate limiting -----------------------------------------------------

    def test_cooldown_suppresses_a_repeat(self):
        self.transport(True, True)
        self.assertTrue(self.send("halted", key="halt", cooldown_sec=60))
        self.assertFalse(self.send("halted", key="halt", cooldown_sec=60))
        self.assertEqual(len(self.calls), 1)

    def test_cooldown_is_not_stamped_on_failure(self):
        # A message nobody received must not suppress its own retry.
        self.transport(False, False, True)
        self.assertFalse(self.send("halted", key="halt", cooldown_sec=60))
        self.assertTrue(self.send("halted", key="halt", cooldown_sec=60))

    def test_unkeyed_sends_are_never_suppressed(self):
        self.transport(True, True)
        self.assertTrue(self.send("a"))
        self.assertTrue(self.send("a"))
        self.assertEqual(len(self.calls), 2)


if __name__ == "__main__":
    unittest.main()

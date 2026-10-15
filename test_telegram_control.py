"""The Telegram control channel, and the boundary around it.

This is the first thing in the project that lets an outside message move
money, so most of what is pinned here is refusal rather than function.

Shape of the design, because the tests only make sense against it:

  read-only          answered in-process, writes nothing
  risk-reducing      executed against the venue directly, safe because the
                     trader already treats a vanished position as a normal
                     case it reconciles
  state-changing     appended to a queue the trader drains, so
                     _live_state.json keeps exactly one writer
"""

import json
import os
import pathlib
import time
import unittest
from unittest import mock

import notify
import tools.telegram_control as C
import tools.live_levels_trader as T


class FakeResp:
    def __init__(self, body):
        self._body = body

    def json(self):
        return self._body


class Authorisation(unittest.TestCase):
    def setUp(self):
        self._env = dict(os.environ)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._env)

    def test_the_configured_chat_is_authorised(self):
        os.environ["TELEGRAM_CHAT_ID"] = "12345"
        self.assertTrue(notify.authorised("12345"))
        self.assertTrue(notify.authorised(" 12345 "))

    def test_any_other_chat_is_not(self):
        os.environ["TELEGRAM_CHAT_ID"] = "12345"
        for other in ("54321", "", "123456", "1234", "-12345"):
            self.assertFalse(notify.authorised(other), other)

    def test_an_unset_chat_id_authorises_nobody(self):
        """A missing credential must not read as 'allow everyone', which is
        what comparing two empty strings would do."""
        os.environ.pop("TELEGRAM_CHAT_ID", None)
        self.assertFalse(notify.authorised(""))
        self.assertFalse(notify.authorised("12345"))

    def test_a_blank_chat_id_authorises_nobody(self):
        os.environ["TELEGRAM_CHAT_ID"] = "   "
        self.assertFalse(notify.authorised(""))
        self.assertFalse(notify.authorised("   "))


class Polling(unittest.TestCase):
    def setUp(self):
        self._env = dict(os.environ)
        os.environ["TELEGRAM_BOT_TOKEN"] = "t"

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._env)

    def poll(self, body, offset=0):
        with mock.patch.object(notify, "_transport",
                               lambda u, p: FakeResp(body)):
            return notify.get_updates(offset)

    def test_messages_are_parsed(self):
        ups, nxt, ok = self.poll({"ok": True, "result": [
            {"update_id": 7, "message": {"text": "/status",
                                         "chat": {"id": 99},
                                         "from": {"id": 42}}}]})
        self.assertEqual(len(ups), 1)
        self.assertEqual(ups[0]["text"], "/status")
        self.assertEqual(ups[0]["chat_id"], "99")
        self.assertEqual(nxt, 8, "offset must advance past what was handled")
        self.assertTrue(ok)

    def test_the_offset_advances_even_past_messages_with_no_text(self):
        """Telegram re-delivers anything unacknowledged, so a photo in the
        chat must not wedge the loop forever."""
        ups, nxt, ok = self.poll({"ok": True, "result": [
            {"update_id": 11, "message": {"chat": {"id": 1}}}]})
        self.assertEqual(ups, [])
        self.assertEqual(nxt, 12)
        self.assertTrue(ok, "an empty result is a successful poll")

    def test_a_refusal_reports_failure_so_the_caller_backs_off(self):
        """Telegram 409s a second concurrent getUpdates while the first still
        hangs, so a caller that retries fast collides with its own open poll
        and 409s forever. It did exactly that on first install, which is why
        `ok` exists -- an empty list alone cannot tell the two apart."""
        self.assertEqual(self.poll({"ok": False, "description": "nope"},
                                   offset=5), ([], 5, False))

    def test_a_409_conflict_reports_failure(self):
        self.assertEqual(
            self.poll({"ok": False, "error_code": 409,
                       "description": "Conflict: terminated by other "
                                      "getUpdates request"}, offset=2),
            ([], 2, False))

    def test_a_transport_failure_is_not_fatal(self):
        def boom(u, p):
            raise OSError("down")
        with mock.patch.object(notify, "_transport", boom):
            self.assertEqual(notify.get_updates(5), ([], 5, False))

    def test_no_token_means_no_poll(self):
        os.environ.pop("TELEGRAM_BOT_TOKEN", None)
        with mock.patch.object(notify, "_transport",
                               mock.Mock(side_effect=AssertionError("called"))):
            self.assertEqual(notify.get_updates(3), ([], 3, False))


class Parsing(unittest.TestCase):
    def test_commands_and_arguments(self):
        self.assertEqual(C.parse("/close HYPE"), ("close", "HYPE"))
        self.assertEqual(C.parse("/status"), ("status", ""))
        self.assertEqual(C.parse("  /PAUSE 6 "), ("pause", "6"))
        self.assertEqual(C.parse(""), ("", ""))

    def test_a_group_mention_is_stripped(self):
        self.assertEqual(C.parse("/status@CuanBot"), ("status", ""))


class Queueing(unittest.TestCase):
    """The controller appends; only the trader applies."""

    def setUp(self):
        self.tmp = pathlib.Path(
            os.environ.get("TMPDIR", "/tmp")) / f"q{os.getpid()}.json"
        self._q, self._t = C.QUEUE, T.CONTROL_QUEUE
        C.QUEUE = T.CONTROL_QUEUE = self.tmp

    def tearDown(self):
        C.QUEUE, T.CONTROL_QUEUE = self._q, self._t
        self.tmp.unlink(missing_ok=True)

    def ctl(self):
        return C.Controller(None)

    def test_halt_is_queued_without_confirmation(self):
        """Halting only ever reduces exposure, so it does not need a token."""
        reply = self.ctl().handle("/halt", {"owned": {}})
        self.assertIn("queued", reply.lower())
        self.assertEqual(json.loads(self.tmp.read_text())[0]["cmd"], "halt")

    def test_the_trader_applies_a_queued_halt(self):
        self.tmp.write_text(json.dumps([{"cmd": "halt", "arg": "by hand"}]))
        st = {}
        with mock.patch.object(T, "save_state"), \
                mock.patch.object(T.notify, "send"):
            done = T.drain_control_queue(st)
        self.assertTrue(st["halted"])
        self.assertEqual(st["halt_kind"], "manual")
        self.assertTrue(done)

    def test_a_telegram_halt_is_never_auto_released(self):
        """It is kind 'manual' on purpose: the rails clear themselves, a
        human's halt does not."""
        st = {"halted": True, "halt_kind": "manual",
              "halt_reason": "halted from Telegram", "owned": {}, "pending": {},
              "realised_today": 0.0}
        with mock.patch.object(T, "save_state"):
            self.assertFalse(T.release_halt_if_clear(st, 704.79))
        self.assertTrue(st["halted"])

    def test_the_queue_is_cleared_so_commands_cannot_repeat(self):
        self.tmp.write_text(json.dumps([{"cmd": "pause", "arg": "2"}]))
        st = {}
        with mock.patch.object(T, "save_state"):
            T.drain_control_queue(st)
            first = st.get("streak_cooldown_until")
            st.pop("streak_cooldown_until", None)
            T.drain_control_queue(st)
        self.assertTrue(first)
        self.assertNotIn("streak_cooldown_until", st,
                         "a drained command must not be applied twice")

    def test_an_unclearable_queue_is_not_acted_on(self):
        """Acting without clearing would replay every command every pass."""
        self.tmp.write_text(json.dumps([{"cmd": "halt", "arg": ""}]))
        st = {}
        with mock.patch.object(pathlib.Path, "write_text",
                               side_effect=OSError("read-only")):
            done = T.drain_control_queue(st)
        self.assertEqual(done, [])
        self.assertFalse(st.get("halted"))

    def test_an_unknown_command_is_dropped_not_retried(self):
        self.tmp.write_text(json.dumps([{"cmd": "selfdestruct", "arg": ""}]))
        st = {}
        with mock.patch.object(T, "save_state"):
            done = T.drain_control_queue(st)
        self.assertIn("ignored unknown command 'selfdestruct'", done)
        self.assertEqual(json.loads(self.tmp.read_text()), [])

    def test_junk_in_the_queue_does_not_crash_the_pass(self):
        for junk in ("not json", "{}", "[1, 2, null]", '[{"cmd": null}]'):
            self.tmp.write_text(junk)
            with mock.patch.object(T, "save_state"):
                T.drain_control_queue({})      # must not raise

    def test_a_missing_queue_is_fine(self):
        self.tmp.unlink(missing_ok=True)
        self.assertEqual(T.drain_control_queue({}), [])

    def test_pause_rejects_nonsense_and_out_of_range(self):
        ctl = self.ctl()
        for bad in ("abc", "-1", "0", "9999"):
            ctl.pending = {"cmd": "pause", "arg": bad, "token": "x",
                           "expires": time.time() + 60}
            reply = ctl.handle("/confirm x", {"owned": {}})
            self.assertFalse(self.tmp.exists() and
                             json.loads(self.tmp.read_text()),
                             f"queued a bad pause: {bad} -> {reply}")


class Confirmation(unittest.TestCase):
    def setUp(self):
        self.tmp = pathlib.Path(
            os.environ.get("TMPDIR", "/tmp")) / f"c{os.getpid()}.json"
        self._q = C.QUEUE
        C.QUEUE = self.tmp
        self.st = {"owned": {"HYPE": {"entry": 90.0, "size_usd": 180.0,
                                      "stop": 88.0, "tp": 95.0}},
                   "pending": {}, "halted": True,
                   "halt_reason": "daily loss stop"}

    def tearDown(self):
        C.QUEUE = self._q
        self.tmp.unlink(missing_ok=True)

    def test_a_risky_command_does_nothing_until_confirmed(self):
        ctl = C.Controller(None)
        reply = ctl.handle("/resume", self.st)
        self.assertIn("/confirm", reply)
        self.assertFalse(self.tmp.exists(),
                         "nothing may be queued before confirmation")

    def test_confirming_with_the_right_token_executes(self):
        ctl = C.Controller(None)
        reply = ctl.handle("/resume", self.st)
        token = reply.split("/confirm ")[1].split("`")[0].strip()
        out = ctl.handle(f"/confirm {token}", self.st)
        self.assertIn("queued", out.lower())
        self.assertEqual(json.loads(self.tmp.read_text())[0]["cmd"], "resume")

    def test_a_wrong_token_does_nothing(self):
        ctl = C.Controller(None)
        ctl.handle("/resume", self.st)
        out = ctl.handle("/confirm deadbeef", self.st)
        self.assertIn("Wrong token", out)
        self.assertFalse(self.tmp.exists())

    def test_an_expired_token_does_nothing(self):
        ctl = C.Controller(None)
        ctl.handle("/resume", self.st)
        ctl.pending["expires"] = time.time() - 1
        out = ctl.handle(f"/confirm {ctl.pending['token']}", self.st)
        self.assertIn("expired", out)
        self.assertFalse(self.tmp.exists())

    def test_confirming_nothing_says_so(self):
        self.assertIn("Nothing", C.Controller(None).handle("/confirm abc",
                                                           self.st))

    def test_a_token_is_single_use(self):
        ctl = C.Controller(None)
        reply = ctl.handle("/resume", self.st)
        token = reply.split("/confirm ")[1].split("`")[0].strip()
        ctl.handle(f"/confirm {token}", self.st)
        self.tmp.unlink(missing_ok=True)
        again = ctl.handle(f"/confirm {token}", self.st)
        self.assertIn("Nothing", again)
        self.assertFalse(self.tmp.exists(), "a replayed token acted twice")

    def test_a_new_risky_command_replaces_the_pending_one(self):
        """Otherwise an old token could execute a command the operator has
        since changed their mind about."""
        ctl = C.Controller(None)
        first = ctl.handle("/resume", self.st)
        t1 = first.split("/confirm ")[1].split("`")[0].strip()
        ctl.handle("/close HYPE", self.st)
        out = ctl.handle(f"/confirm {t1}", self.st)
        self.assertIn("Wrong token", out)

    def test_close_names_what_it_will_do(self):
        ctl = C.Controller(None)
        reply = ctl.handle("/close HYPE", self.st)
        self.assertIn("HYPE", reply)
        self.assertIn("180", reply)

    def test_close_on_a_coin_not_held_says_so(self):
        ctl = C.Controller(None)
        self.assertIn("not open", ctl.handle("/close DOGE", self.st))


class NoBlindOrders(unittest.TestCase):
    class Backend:
        def __init__(self, mid=None):
            self.mid, self.exits = mid, []

        def get_mid_price(self, coin):
            return self.mid

        def execute_exit(self, **kw):
            self.exits.append(kw)
            return type("R", (), {"status": "filled"})()

    def st(self):
        return {"owned": {"HYPE": {"entry": 90.0, "size_usd": 180.0}}}

    def test_an_unpriceable_coin_is_not_exited(self):
        b = self.Backend(mid=None)
        out = C.Controller(b).close_one("HYPE", self.st())
        self.assertIn("Could not price", out)
        self.assertEqual(b.exits, [])

    def test_a_priced_coin_is_exited_once(self):
        b = self.Backend(mid=89.0)
        out = C.Controller(b).close_one("HYPE", self.st())
        self.assertEqual(len(b.exits), 1)
        self.assertEqual(b.exits[0]["coin"], "HYPE")
        self.assertEqual(b.exits[0]["size_usd"], 180.0)
        self.assertIn("✅", out)

    def test_no_backend_means_no_close(self):
        self.assertIn("No venue connection",
                      C.Controller(None).close_one("HYPE", self.st()))

    def test_a_raising_backend_is_reported_not_swallowed(self):
        class Angry(self.Backend):
            def execute_exit(self, **kw):
                raise RuntimeError("venue said no")
        out = C.Controller(Angry(mid=89.0)).close_one("HYPE", self.st())
        self.assertIn("venue said no", out)


class ThereIsNoRawOpen(unittest.TestCase):
    def test_open_is_not_a_command(self):
        """A long with no level has no structural stop and no target.
        /take forces a coin the scanner already has; /open would invent one."""
        ctl = C.Controller(None)
        self.assertIn("Unknown command", ctl.handle("/open HYPE", {"owned": {}}))

    def test_take_only_grants_a_window_and_decides_nothing(self):
        """The trader is what knows whether a candidate exists, so the drain
        records permission and refuses nothing here."""
        st = {}
        tmp = pathlib.Path(os.environ.get("TMPDIR", "/tmp")) / f"t{os.getpid()}.json"
        saved = T.CONTROL_QUEUE
        T.CONTROL_QUEUE = tmp
        try:
            tmp.write_text(json.dumps([{"cmd": "take", "arg": "hype"}]))
            with mock.patch.object(T, "save_state"):
                T.drain_control_queue(st)
            self.assertIn("HYPE", st["forced"])
        finally:
            T.CONTROL_QUEUE = saved
            tmp.unlink(missing_ok=True)


class Rendering(unittest.TestCase):
    def test_status_reports_a_halt_and_its_kind(self):
        out = C.render_status({"halted": True, "halt_kind": "drawdown",
                               "halt_reason": "account drawdown 11.1%",
                               "owned": {}, "pending": {}}, 704.79)
        self.assertIn("HALTED", out)
        self.assertIn("drawdown", out)

    def test_status_reports_the_rolling_drawdown(self):
        out = C.render_status(
            {"owned": {}, "pending": {}, "realised_today": -5.0,
             "equity_samples": [["2026-10-15T00:00:00Z", 800.0]]}, 760.0)
        self.assertIn("5.00%", out)
        self.assertIn("800", out)

    def test_status_reports_a_stand_down(self):
        from datetime import datetime, timedelta, timezone
        until = (datetime.now(timezone.utc) + timedelta(hours=5)
                 ).isoformat().replace("+00:00", "Z")
        out = C.render_status({"owned": {}, "pending": {},
                               "streak_cooldown_until": until}, 700.0)
        self.assertIn("standing down", out)

    def test_status_survives_a_junk_state_file(self):
        for st in ({}, {"owned": None, "equity_samples": "nope"},
                   {"realised_today": "x", "loss_streak": None}):
            C.render_status(st, None)        # must not raise

    def test_positions_when_flat(self):
        self.assertIn("Flat", C.render_positions({"owned": {}}, None))

    def test_positions_shows_live_pnl(self):
        class B:
            def get_mid_price(self, c):
                return 99.0
        out = C.render_positions(
            {"owned": {"HYPE": {"entry": 90.0, "size_usd": 180.0,
                                "stop": 88.0, "tp": 100.0}}}, B())
        self.assertIn("HYPE", out)
        self.assertIn("+10.00%", out)

    def test_positions_survives_a_backend_that_raises(self):
        class B:
            def get_mid_price(self, c):
                raise RuntimeError("down")
        out = C.render_positions(
            {"owned": {"HYPE": {"entry": 90.0}}}, B())
        self.assertIn("HYPE", out)


if __name__ == "__main__":
    unittest.main()


class TheLoopBacksOffRatherThanFightingItself(unittest.TestCase):
    def test_the_failure_backoff_exceeds_the_poll_window(self):
        """Pinned as a number, because getting this wrong is not a crash --
        it is a log full of 409s and a control channel that never answers."""
        src = open("tools/telegram_control.py").read()
        self.assertIn("fail_backoff = notify.POLL_TIMEOUT_SEC + 5", src)
        self.assertIn("time.sleep(1 if ok else fail_backoff)", src)

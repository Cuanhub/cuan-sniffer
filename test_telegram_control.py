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

import contextlib
import io
import json
import os
import pathlib
import sys
import time
import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

import notify
import tools.telegram_control as C
import tempfile
from pathlib import Path
from unittest import mock

from core import execute, safety
from tests.fake_venue import FakeHL, build_venue


def _venue(**kw):
    execute.SETTLE_SEC = 0.0
    safety.FLATTEN_PAUSE_SEC = 0.0
    return build_venue(FakeHL(**kw))


def _halt():
    return safety.HaltState(Path(tempfile.mkdtemp()) / "halt.json")

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
        seen = {}

        def fake(u, p, timeout=None):
            seen["timeout"] = timeout
            return FakeResp(body)

        with mock.patch.object(notify, "_transport", fake):
            out = notify.get_updates(offset)
        self.seen = seen
        return out

    def test_the_read_timeout_outlasts_the_long_poll(self):
        """A 10s read timeout against a 25s long-poll aborts every quiet poll
        before Telegram answers. Observed as a log of nothing but "Read timed
        out" the moment the 409 conflict cleared."""
        self.poll({"ok": True, "result": []})
        self.assertGreater(self.seen["timeout"], notify.POLL_TIMEOUT_SEC,
                           "the HTTP read timeout must outlast the poll")

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
        def boom(u, p, timeout=None):
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
        st, h = {}, _halt()
        with mock.patch.object(T, "save_state"), \
                mock.patch.object(T.notify, "send"):
            done = T.drain_control_queue(st, _venue(), h)
        active = h.active()
        self.assertIsNotNone(active, "the queued halt was not applied")
        self.assertEqual(active["kind"], "manual")
        self.assertIn("by hand", active["reason"])
        self.assertTrue(done)

    def test_a_telegram_halt_is_never_auto_released(self):
        """It is kind 'manual' on purpose: the rails clear themselves, a
        human's halt does not."""
        h = _halt()
        h.set(safety.MANUAL, "halted from Telegram")
        # Not even a new UTC day clears a manual halt. Only --resume does.
        with mock.patch.object(safety.HaltState, "_today",
                               staticmethod(lambda: "2099-01-01")):
            self.assertIsNotNone(h.active())
        h.clear()
        self.assertIsNone(h.active())

    def test_the_queue_is_cleared_so_commands_cannot_repeat(self):
        """A timed pause is an AUTO halt now -- a halt that clears itself is
        exactly what AUTO means, so it is one rather than a third state with
        its own timestamp and its own way of getting stuck."""
        self.tmp.write_text(json.dumps([{"cmd": "pause", "arg": "2"}]))
        st, h = {}, _halt()
        with mock.patch.object(T, "save_state"), \
                mock.patch.object(T.notify, "send"):
            first = T.drain_control_queue(st, _venue(), h)
            self.assertTrue(first, "the pause was not applied")
            self.assertEqual(h.active()["kind"], "auto")
            h.clear()
            again = T.drain_control_queue(st, _venue(), h)
        self.assertEqual(again, [], "a drained command must not run twice")
        self.assertIsNone(h.active())

    def test_an_unclearable_queue_is_not_acted_on(self):
        """Acting without clearing would replay every command every pass."""
        self.tmp.write_text(json.dumps([{"cmd": "halt", "arg": ""}]))
        st = {}
        with mock.patch.object(pathlib.Path, "write_text",
                               side_effect=OSError("read-only")):
            done = T.drain_control_queue(st, _venue(), _halt())
        self.assertEqual(done, [])
        self.assertFalse(st.get("halted"))

    def test_an_unknown_command_is_dropped_not_retried(self):
        self.tmp.write_text(json.dumps([{"cmd": "selfdestruct", "arg": ""}]))
        st = {}
        with mock.patch.object(T, "save_state"):
            done = T.drain_control_queue(st, _venue(), _halt())
        self.assertIn("ignored unknown command 'selfdestruct'", done)
        self.assertEqual(json.loads(self.tmp.read_text()), [])

    def test_junk_in_the_queue_does_not_crash_the_pass(self):
        for junk in ("not json", "{}", "[1, 2, null]", '[{"cmd": null}]'):
            self.tmp.write_text(junk)
            with mock.patch.object(T, "save_state"):
                T.drain_control_queue({}, _venue(), _halt())      # must not raise

    def test_a_missing_queue_is_fine(self):
        self.tmp.unlink(missing_ok=True)
        self.assertEqual(T.drain_control_queue({}, _venue(), _halt()), [])

    def confirm_pause(self, arg):
        ctl = self.ctl()
        ctl.pending = {"cmd": "pause", "arg": arg, "token": "x",
                       "expires": time.time() + 60}
        return ctl.handle("/confirm x", {"owned": {}})

    def test_pause_rejects_nonsense_and_absurd_lengths(self):
        for bad in ("abc", "", "9999"):
            self.tmp.unlink(missing_ok=True)
            reply = self.confirm_pause(bad)
            self.assertFalse(self.tmp.exists() and
                             json.loads(self.tmp.read_text()),
                             f"queued a bad pause: {bad!r} -> {reply}")

    def test_pause_zero_lifts_a_stand_down(self):
        """0 means "stop pausing" -- one verb for the dial, and zero is the
        obvious way to turn it off. Until 2026-10-16 nothing could lift one."""
        for arg in ("0", "-1"):
            self.tmp.unlink(missing_ok=True)
            reply = self.confirm_pause(arg)
            self.assertIn("lifting", reply.lower())
            q = json.loads(self.tmp.read_text())
            self.assertEqual((q[0]["cmd"], q[0]["arg"]), ("pause", "0"))

    def test_pause_with_hours_still_sets_one(self):
        self.tmp.unlink(missing_ok=True)
        reply = self.confirm_pause("6")
        self.assertIn("6h", reply)
        self.assertEqual(json.loads(self.tmp.read_text())[0]["arg"], "6.0")


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
        """From the VENUE, so the confirmation describes what is actually
        held rather than what the state file remembers."""
        v = build_venue(FakeHL(positions={"HYPE": 2.0}, mid=90.0))
        reply = C.Controller(v).handle("/close HYPE", self.st)
        self.assertIn("HYPE", reply)
        self.assertIn("90", reply)

    def test_close_on_a_coin_not_held_says_so(self):
        v = build_venue(FakeHL(positions={"HYPE": 2.0}, mid=90.0))
        self.assertIn("not open",
                      C.Controller(v).handle("/close DOGE", self.st))


class ClosingReportsTheVenue(unittest.TestCase):
    """On 2026-10-17 a /closeall of four positions reported four identical
    warnings while three closed only to dust and ZRO did not close at all --
    $178.70 of live exposure behind a message that read like progress.

    The fake is now the recorded-payload venue, so "did it close" is answered
    by what the venue HOLDS rather than by a reply shape.
    """

    def setUp(self):
        self._o = (execute.SETTLE_SEC, safety.FLATTEN_PAUSE_SEC,
                   safety.MAX_FLATTEN_SEC)
        execute.SETTLE_SEC = 0.0
        safety.FLATTEN_PAUSE_SEC = 0.0

    def tearDown(self):
        (execute.SETTLE_SEC, safety.FLATTEN_PAUSE_SEC,
         safety.MAX_FLATTEN_SEC) = self._o

    def venue(self, **kw):
        self.fake = FakeHL(positions={"HYPE": 2.0}, mid=90.0, **kw)
        return build_venue(self.fake)

    def test_a_clean_close_is_confirmed_against_the_venue(self):
        """It used to assert the word "flat", and that was the bug's
        fingerprint: `/close COIN` reported ACCOUNT-wide state because it
        called flatten_all, which closes everything. A single-coin command
        now names the coin it closed and what is still held, which is only
        the same sentence when nothing else was open."""
        v = self.venue()
        out = C.Controller(v).close_one("HYPE", {})
        self.assertIn("✅", out)
        self.assertIn("HYPE", out)
        self.assertIn("closed", out.lower())
        self.assertEqual(self.fake.pos, {})

    def test_a_position_still_open_is_reported_LOUDLY(self):
        """The case that cost $178.70 of unnoticed exposure."""
        safety.MAX_FLATTEN_SEC = 0.05
        v = self.venue()
        self.fake.order_fills = False        # accepted, nothing moves
        out = C.Controller(v).close_one("HYPE", {})
        self.assertIn("🚨", out)
        self.assertIn("STILL OPEN", out)
        self.assertNotIn("✅", out)

    def test_a_partial_close_is_not_reported_as_success(self):
        """`filled` is True for a partial too, so the old code called a
        half-closed position closed."""
        safety.MAX_FLATTEN_SEC = 0.05
        v = self.venue()
        self.fake.positions = {"HYPE": 2.0}
        self.fake.fill_fraction = 0.0001     # moves, but never finishes
        out = C.Controller(v).close_one("HYPE", {})
        self.assertNotIn("✅", out)

    def test_a_coin_not_held_is_not_sent_to_the_venue(self):
        v = self.venue()
        out = C.Controller(v).close_one("ZRO", {})
        self.assertIn("not open", out)
        self.assertEqual(self.fake.sent, [])

    def test_a_coin_the_state_file_never_knew_is_still_closeable(self):
        """REVIEW.md finding 5. The old close_one read state["owned"] first
        and answered "No open position" while the venue held it -- which is
        exactly what a timeout-after-fill leaves behind."""
        v = self.venue()
        out = C.Controller(v).close_one("HYPE", {})     # empty state
        self.assertIn("✅", out)
        self.assertEqual(self.fake.pos, {})

    def test_an_unreadable_venue_sends_nothing_and_says_so(self):
        v = self.venue()
        self.fake.fail_info_always = True
        out = C.Controller(v).close_one("HYPE", {})
        self.assertIn("Could not read", out)
        self.assertEqual(self.fake.sent, [])

    def test_closeall_closes_everything_the_venue_holds(self):
        v = self.venue()
        self.fake.pos.update({"ZRO": 50.0, "NIL": 900.0})
        out = C.Controller(v).close_all()
        self.assertIn("✅", out)
        self.assertEqual(self.fake.pos, {})

    def test_closeall_cancels_resting_orders_too(self):
        v = self.venue()
        self.fake.rest_plain_order("HYPE", 1.0, 80.0)
        C.Controller(v).close_all()
        self.assertEqual(self.fake.resting, [],
                         "a resting bid could re-open what was just closed")

    def test_close_one_does_NOT_read_a_field_a_reply_lacks(self):
        """The root cause, pinned. Nothing parses an order response now."""
        src = open("tools/telegram_control.py").read()
        fn = src[src.index("    def close_one("):src.index("    def close_all(")]
        code = fn.split('"""', 2)[2]
        for bad in ('getattr(res, "status"', "res.status", 'res.get("status")'):
            self.assertNotIn(bad, code, f"close_one reads {bad}")



# ThereIsNoRawOpen retired: /take was dropped on instruction, along with
# FORCEABLE and the state["forced"] plumbing it tested. There is no longer any
# way to force an entry past a rail, which is a stronger guarantee than the
# one this class checked.


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


class SharedTokenIsDetected(unittest.TestCase):
    """Two pollers on one bot token steal each other's updates, so a command
    sent to this trader can be consumed by the other process and never
    arrive. Found on first install: the same token was in use by another
    project on this machine that had been polling for four days. A sequential
    loop cannot 409 itself, so repeated failures mean exactly this.
    """

    def test_the_loop_reports_a_shared_token_rather_than_retrying_forever(self):
        src = open("tools/telegram_control.py").read()
        self.assertIn("conflicts += 1", src)
        self.assertIn("if conflicts == 3:", src)
        self.assertIn("BotFather", src,
                      "the alert must say what actually fixes it")
        self.assertIn("conflicts = 0", src,
                      "a successful poll must reset the counter")
class Rendering(unittest.TestCase):
    def venue(self, positions=None, usdc=700.0, **kw):
        self.fake = FakeHL(positions=positions or {}, mid=99.0, usdc=usdc, **kw)
        return build_venue(self.fake)

    def test_status_reports_a_halt_and_its_kind(self):
        d = Path(tempfile.mkdtemp())
        h = safety.HaltState(d / "halt.json")
        h.set(safety.MANUAL, "halted from Telegram: test")
        eq = safety.EquityLog(d / "eq.json")
        with mock.patch.object(safety, "HaltState", lambda *a, **k: h), \
                mock.patch.object(safety, "EquityLog", lambda *a, **k: eq):
            out = C.render_status({}, 704.79, self.venue())
        self.assertIn("HALTED", out)
        self.assertIn("manual", out)

    def test_status_reports_the_rolling_drawdown(self):
        d = Path(tempfile.mkdtemp())
        log = safety.EquityLog(d / "eq.json")
        log.observe(1000.0)
        halt = safety.HaltState(d / "h.json")
        with mock.patch.object(safety, "HaltState", lambda *a, **k: halt), \
                mock.patch.object(safety, "EquityLog", lambda *a, **k: log):
            out = C.render_status({}, 900.0, self.venue())
        self.assertIn("drawdown", out)
        self.assertIn("10.00%", out)

    def test_status_survives_no_equity(self):
        C.render_status({}, None, None)      # must not raise

    def test_positions_when_flat(self):
        self.assertIn("Flat", C.render_positions({}, self.venue()))

    def test_positions_shows_live_pnl(self):
        v = self.venue({"HYPE": 2.0})
        self.fake.unrealized = {"HYPE": 18.0}
        out = C.render_positions({}, v)
        self.assertIn("HYPE", out)
        self.assertIn("+18.00", out)

    def test_positions_flags_a_position_with_no_stop(self):
        """The screen the operator checks first must say this."""
        out = C.render_positions({}, self.venue({"HYPE": 2.0}))
        self.assertIn("NO STOP", out)

    def test_positions_shows_the_stop_when_there_is_one(self):
        v = self.venue({"HYPE": 2.0})
        self.fake.rest_protection("HYPE", 2.0, stop_px=88.0)
        out = C.render_positions({}, v)
        self.assertIn("88", out)
        self.assertNotIn("NO STOP", out)

    def test_positions_warns_when_the_stop_undercovers(self):
        v = self.venue({"HYPE": 2.0})
        self.fake.rest_protection("HYPE", 1.0, stop_px=88.0)
        self.assertIn("covers only", C.render_positions({}, v))

    def test_positions_survives_a_backend_that_raises(self):
        v = self.venue({"HYPE": 2.0})
        self.fake.fail_info_always = True
        self.assertIn("Could not read", C.render_positions({}, v))

    def test_positions_with_no_backend(self):
        self.assertIn("No venue", C.render_positions({}, None))



class HelpIsReachable(unittest.TestCase):
    """Regression for a real breakage during the execution rewrite.

    Editing render_positions removed the HELP constant along with it, so
    /help and /start raised NameError. The module still imported, because
    HELP is only referenced inside handle() -- mypy caught it and no test
    did, because nothing exercised /help. Now something does.
    """

    def test_help_and_start_return_the_help_text(self):
        ctl = C.Controller(None)
        for cmd in ("/help", "/start"):
            out = ctl.handle(cmd, {})
            self.assertIn("Cuan Sniffer controls", out)
            self.assertIn("/closeall", out)

    @staticmethod
    def advertised():
        """Commands listed as available, not ones the prose mentions.

        The help text deliberately explains that /open and /take do NOT
        exist, so a naive scan of every `/word` in it picks those up too.
        Only a line that STARTS with a command is an advertisement.
        """
        import re
        out = set()
        for line in C.HELP.splitlines():
            m = re.match(r"\s*`/([a-z]+)", line)
            if m:
                out.add(m.group(1))
        return out

    def test_help_does_not_advertise_a_command_that_was_removed(self):
        self.assertNotIn("take", self.advertised())
        self.assertNotIn("open", self.advertised())
        # and the prose still explains why neither exists
        out = C.Controller(None).handle("/help", {})
        self.assertIn("no `/open COIN`", out)
        self.assertIn("no `/take`", out)

    def test_every_advertised_command_is_handled(self):
        """A help text that lists a command nothing implements is a lie."""
        ctl = C.Controller(None)
        found = self.advertised()
        self.assertTrue({"status", "positions", "halt", "close", "closeall",
                         "resume", "pause", "help"} <= found, found)
        for cmd in sorted(found):
            out = ctl.handle(f"/{cmd}", {})
            self.assertNotIn("Unknown command", out,
                             f"/{cmd} is advertised but not handled")


class RepeatedCommandsCollapse(unittest.TestCase):
    """Found by a real drain: four identical `halt` commands were waiting and
    each ran its own flatten_all -- four venue sweeps for one intent."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp()) / "q.json"
        self._saved = T.CONTROL_QUEUE
        T.CONTROL_QUEUE = self.tmp

    def tearDown(self):
        T.CONTROL_QUEUE = self._saved

    def test_four_halts_flatten_once(self):
        self.tmp.write_text(json.dumps(
            [{"cmd": "halt", "arg": "a"}] * 4))
        fake = FakeHL(positions={"HYPE": 2.0}, mid=100.0)
        execute.SETTLE_SEC = 0.0
        safety.FLATTEN_PAUSE_SEC = 0.0
        v = build_venue(fake)
        h = _halt()
        with mock.patch.object(T, "save_state"), \
                mock.patch.object(T.notify, "send"):
            done = T.drain_control_queue({}, v, h)
        self.assertEqual(len(done), 1, f"acted {len(done)} times: {done}")
        self.assertIsNotNone(h.active())

    def test_distinct_commands_all_run(self):
        self.tmp.write_text(json.dumps(
            [{"cmd": "halt", "arg": "x"}, {"cmd": "resume", "arg": ""}]))
        v = _venue()
        h = _halt()
        with mock.patch.object(T, "save_state"), \
                mock.patch.object(T.notify, "send"):
            done = T.drain_control_queue({}, v, h)
        self.assertEqual(len(done), 2, done)

    def test_the_last_of_a_repeated_verb_wins(self):
        self.tmp.write_text(json.dumps(
            [{"cmd": "pause", "arg": "2"}, {"cmd": "pause", "arg": "9"}]))
        v = _venue()
        h = _halt()
        with mock.patch.object(T, "save_state"), \
                mock.patch.object(T.notify, "send"):
            done = T.drain_control_queue({}, v, h)
        self.assertEqual(len(done), 1)
        self.assertIn("9", h.active()["reason"])


class AQueuedHaltDoesNotFlatten(unittest.TestCase):
    """The regression that cost a live position on 2026-10-17.

    `/halt`'s own reply says "blocks new entries only -- open positions keep
    their stops. Use /closeall to flatten." It flattened anyway: every queued
    halt used to run with backend=None (REVIEW.md finding 4), so the flatten
    inside safety.halt was a silent no-op and the promise held by accident.
    Passing the venue in to fix the protection check armed a flatten that had
    never once fired, and it market-closed ARB.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp()) / "q.json"
        self._t = T.CONTROL_QUEUE
        T.CONTROL_QUEUE = self.tmp

    def tearDown(self):
        T.CONTROL_QUEUE = self._t

    def _drain(self, cmd, arg=""):
        self.tmp.write_text(json.dumps([{"cmd": cmd, "arg": arg}]))
        execute.SETTLE_SEC = 0.0
        safety.FLATTEN_PAUSE_SEC = 0.0
        fake = FakeHL(positions={"ARB": 974.2}, mid=0.18539)
        h = _halt()
        with mock.patch.object(T, "save_state"), \
                mock.patch.object(T.notify, "send"):
            done = T.drain_control_queue({}, build_venue(fake), h)
        return fake, h, done

    def test_the_position_survives_the_halt(self):
        fake, h, done = self._drain("halt", "requested from Telegram")
        self.assertIsNotNone(h.active(), "the halt must still be applied")
        self.assertEqual(h.active()["kind"], "manual")
        self.assertEqual(fake.pos.get("ARB"), 974.2,
                         "a /halt market-closed the position it promised to "
                         "leave alone")
        self.assertTrue(done)

    def test_the_halt_places_and_cancels_nothing(self):
        """Not just "the position is still open" -- nothing was sent at all.
        A close that failed would also leave the position open."""
        fake, _, _ = self._drain("halt", "requested from Telegram")
        self.assertEqual(fake.sent, [], f"orders were sent: {fake.sent}")
        self.assertEqual(fake.cancelled, [],
                         f"orders were cancelled: {fake.cancelled}")

    def test_a_pause_leaves_the_position_alone_too(self):
        """It already did. Pinned so the two branches cannot drift apart."""
        fake, h, _ = self._drain("pause", "2")
        self.assertEqual(fake.pos.get("ARB"), 974.2)
        self.assertEqual(h.active()["kind"], "auto")

    def test_the_reply_the_operator_reads_still_matches_the_behaviour(self):
        """The bug was a promise the code stopped keeping, so the promise is
        part of the test. If someone restores the flatten, this fails next to
        the assertion above and says which of the two is wrong."""
        with mock.patch.object(C, "QUEUE", self.tmp):
            reply = C.Controller(None).handle("/halt", {"owned": {}})
        self.assertIn("new entries only", reply)
        self.assertIn("keep their stops", reply)
        self.assertIn("closeall", reply)

    def test_the_equity_rails_still_flatten(self):
        """Flattening is not gone, it is only where it is asked for. A rail
        is breached ON exposure, so its halt must remove it."""
        execute.SETTLE_SEC = 0.0
        safety.FLATTEN_PAUSE_SEC = 0.0
        fake = FakeHL(positions={"ARB": 974.2}, mid=0.18539)
        with mock.patch.object(safety.notify, "send"):
            safety.halt(build_venue(fake), _halt(), "drawdown rail",
                        kind=safety.AUTO)
        self.assertEqual(fake.pos.get("ARB", 0.0), 0.0,
                         "an AUTO rail halt must flatten")


class AReplayedCommandIsRefused(unittest.TestCase):
    """Telegram re-delivers anything unacknowledged, so an instance that
    starts at offset 0 replays the whole backlog. That is how a `/halt` sent
    hours earlier during a disarm was re-enqueued at 14:17:11Z on 2026-10-17
    -- while two processes fought over the same bot token (409 Conflict,
    warned about three times in that same log) and left it unacknowledged.

    A replay is identical to a fresh command by content, so age is the only
    thing that can tell them apart.
    """

    def setUp(self):
        self._env = dict(os.environ)
        os.environ["TELEGRAM_BOT_TOKEN"] = "t"
        os.environ["TELEGRAM_CHAT_ID"] = "42"
        self.dir = Path(tempfile.mkdtemp())
        self._q, self._o = C.QUEUE, C.OFFSET
        C.QUEUE = self.dir / "q.json"
        C.OFFSET = self.dir / "off.json"

    def tearDown(self):
        C.QUEUE, C.OFFSET = self._q, self._o
        os.environ.clear()
        os.environ.update(self._env)

    def _poll(self, age_sec, text="/halt"):
        """Run one pass of the real poll loop over one update."""
        upd = [{"update_id": 1, "chat_id": "42", "user_id": "7",
                "text": text, "at": int(time.time() - age_sec)}]
        sent = []
        with mock.patch.object(C.notify, "get_updates",
                               return_value=(upd, 2, True)), \
                mock.patch.object(C.notify, "send",
                                  side_effect=lambda m, **k: sent.append(m)), \
                mock.patch.object(sys, "argv",
                                  ["telegram_control", "--once",
                                   "--no-venue"]):
            C.main()
        queued = (json.loads(C.QUEUE.read_text())
                  if C.QUEUE.exists() else [])
        return queued, sent

    def test_a_fresh_command_is_obeyed(self):
        queued, _ = self._poll(age_sec=5)
        self.assertEqual([q["cmd"] for q in queued], ["halt"])

    def test_a_replayed_command_is_not_queued(self):
        queued, _ = self._poll(age_sec=C.COMMAND_MAX_AGE_SEC + 60)
        self.assertEqual(queued, [],
                         "a stale /halt reached the queue and would halt and "
                         "-- before the fix -- flatten")

    def test_the_operator_is_told_it_was_dropped(self):
        """Silently swallowing a command is its own trap: the operator would
        believe the bot was halted when it was not."""
        _, sent = self._poll(age_sec=C.COMMAND_MAX_AGE_SEC + 60)
        self.assertTrue(any("stale" in m.lower() for m in sent), sent)

    def test_the_stale_command_is_acknowledged_so_it_cannot_return(self):
        """Dropping it without advancing the offset would replay it for
        ever, which is worse than obeying it once."""
        self._poll(age_sec=C.COMMAND_MAX_AGE_SEC + 60)
        self.assertEqual(json.loads(C.OFFSET.read_text())["offset"], 2)

    def test_an_update_with_no_date_is_still_obeyed(self):
        """Old Telegram payloads and the recorded fixtures carry no `date`.
        Refusing those would break the channel to fix a replay."""
        upd = [{"update_id": 1, "chat_id": "42", "user_id": "7",
                "text": "/halt"}]
        with mock.patch.object(C.notify, "get_updates",
                               return_value=(upd, 2, True)), \
                mock.patch.object(C.notify, "send"), \
                mock.patch.object(sys, "argv",
                                  ["telegram_control", "--once",
                                   "--no-venue"]):
            C.main()
        self.assertEqual(
            [q["cmd"] for q in json.loads(C.QUEUE.read_text())], ["halt"])

    def test_the_window_outlasts_a_confirmation(self):
        """A token the operator is given 2 minutes to return must not be
        killable by the staleness check instead."""
        self.assertGreater(C.COMMAND_MAX_AGE_SEC, C.CONFIRM_TTL_SEC * 2)

    def test_get_updates_carries_the_send_time(self):
        """The check is only as good as the field it reads."""
        body = {"ok": True, "result": [
            {"update_id": 5, "message": {"text": "/halt", "date": 1760000000,
                                         "chat": {"id": 42},
                                         "from": {"id": 7}}}]}
        with mock.patch.object(notify, "_transport",
                               return_value=FakeResp(body)):
            out, nxt, ok = notify.get_updates(0)
        self.assertTrue(ok)
        self.assertEqual(out[0]["at"], 1760000000)


class AStaleQueuedCommandIsDropped(unittest.TestCase):
    """The controller refuses to ENQUEUE a replayed update, which cannot help
    a command already sitting in the queue -- written before that check
    existed, or while the trader was down. On 2026-10-18 a `halt` enqueued at
    00:14:19Z was still waiting to fire, which is the same hazard arriving by
    a different door.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp()) / "q.json"
        self._t = T.CONTROL_QUEUE
        T.CONTROL_QUEUE = self.tmp

    def tearDown(self):
        T.CONTROL_QUEUE = self._t

    @staticmethod
    def _stamp(age_sec):
        when = (datetime.now(timezone.utc)
                - timedelta(seconds=age_sec))
        return when.isoformat().replace("+00:00", "Z")

    def _drain(self, *items):
        self.tmp.write_text(json.dumps(list(items)))
        h = _halt()
        with mock.patch.object(T, "save_state"), \
                mock.patch.object(T.notify, "send"):
            done = T.drain_control_queue({}, _venue(), h)
        return done, h

    def test_a_fresh_queued_command_is_applied(self):
        done, h = self._drain({"cmd": "halt", "arg": "x",
                               "at": self._stamp(30)})
        self.assertTrue(done)
        self.assertIsNotNone(h.active())

    def test_a_stale_queued_command_is_not_applied(self):
        done, h = self._drain(
            {"cmd": "halt", "arg": "x",
             "at": self._stamp(T.CONTROL_COMMAND_MAX_AGE_SEC + 60)})
        self.assertEqual(done, [])
        self.assertIsNone(h.active(), "a stale queued halt was applied")

    def test_the_queue_is_still_cleared_so_it_cannot_retry(self):
        self._drain({"cmd": "halt", "arg": "x",
                     "at": self._stamp(T.CONTROL_COMMAND_MAX_AGE_SEC + 60)})
        self.assertEqual(json.loads(self.tmp.read_text()), [])

    def test_a_command_with_no_timestamp_is_still_applied(self):
        """The `at` field postdates the queue format, and refusing a command
        for predating a timestamp is worse than the problem it guards."""
        done, h = self._drain({"cmd": "halt", "arg": "x"})
        self.assertTrue(done)
        self.assertIsNotNone(h.active())

    def test_an_unparseable_timestamp_is_treated_as_fresh(self):
        done, _ = self._drain({"cmd": "halt", "arg": "x", "at": "whenever"})
        self.assertTrue(done, "a bad timestamp must not silently drop a halt")

    def test_a_stale_command_does_not_block_a_fresh_one_beside_it(self):
        done, h = self._drain(
            {"cmd": "pause", "arg": "2",
             "at": self._stamp(T.CONTROL_COMMAND_MAX_AGE_SEC + 60)},
            {"cmd": "halt", "arg": "fresh", "at": self._stamp(10)})
        self.assertEqual(len(done), 1, done)
        self.assertIn("fresh", h.active()["reason"])

    def test_the_drain_window_is_looser_than_the_controllers(self):
        """A queued command is one the operator was told applies "on the
        trader's next pass", and a pass is 450s -- so this side must tolerate
        more delay than the side that reads Telegram's own send time."""
        self.assertGreater(T.CONTROL_COMMAND_MAX_AGE_SEC,
                           C.COMMAND_MAX_AGE_SEC)

    def test_age_of_the_real_recorded_entry(self):
        """The exact entry that was waiting in the live queue."""
        age = T._command_age_sec({"at": "2026-10-18T00:14:19.752034Z"})
        self.assertGreater(age, 0.0)
        self.assertEqual(T._command_age_sec({}), 0.0)


class CloseOneClosesOne(unittest.TestCase):
    """`/close COIN` must close that coin and nothing else.

    It called safety.flatten_all, which takes no coin and closes EVERY
    position while cancelling EVERY resting order -- so `/close WLD` would
    have flattened the whole account, while HELP promised "market-close one
    position". Found on 2026-10-19 about to be used on exactly that command,
    with SOL held and two entry bids resting.
    """

    def setUp(self):
        execute.SETTLE_SEC = 0.0
        safety.FLATTEN_PAUSE_SEC = 0.0
        self.fake = FakeHL(positions={"WLD": 385.3, "SOL": 1.76}, mid=0.48)
        self.fake.resting = [
            {"coin": "XMR", "oid": 1, "sz": "0.324", "limitPx": "548.5",
             "side": "B", "isTrigger": False, "reduceOnly": False,
             "timestamp": self.fake.now_ms},
        ]
        self.v = build_venue(self.fake)

    def ctl(self):
        return C.Controller(self.v)

    def test_the_named_coin_is_closed(self):
        out = self.ctl().close_one("WLD", {})
        self.assertNotIn("WLD", self.fake.pos, out)
        self.assertIn("closed", out.lower())

    def test_the_other_position_is_left_alone(self):
        """The whole bug, as one assertion."""
        self.ctl().close_one("WLD", {})
        self.assertEqual(self.fake.pos.get("SOL"), 1.76,
                         "closing WLD also closed SOL")

    def test_resting_bids_elsewhere_are_not_cancelled(self):
        """flatten_all cancels every resting order, which would have pulled
        entry bids on coins the operator never named."""
        self.ctl().close_one("WLD", {})
        self.assertEqual(self.fake.cancelled, [],
                         f"orders were cancelled: {self.fake.cancelled}")

    def test_the_reply_names_what_is_still_held(self):
        out = self.ctl().close_one("WLD", {})
        self.assertIn("SOL", out)

    def test_a_coin_that_is_not_open_is_reported_not_sent(self):
        out = self.ctl().close_one("DOGE", {})
        self.assertIn("not open", out.lower())
        self.assertEqual(self.fake.sent, [], "an order was sent anyway")

    def test_an_unreadable_venue_sends_nothing(self):
        self.fake.fail_info_always = True
        out = self.ctl().close_one("WLD", {})
        self.assertIn("could not read", out.lower())
        self.assertEqual(self.fake.sent, [])

    def test_a_close_that_does_not_flatten_says_so(self):
        """It must not claim success off a sent order. A partial exit leaves
        the position open and the operator has to know."""
        self.fake.order_fills = False
        out = self.ctl().close_one("WLD", {})
        self.assertIn("STILL OPEN", out)
        self.assertIn("WLD", self.fake.pos)

    def test_closeall_is_still_flatten_all(self):
        """The scoped fix must not weaken the kill switch."""
        out = self.ctl().close_all()
        self.assertEqual(self.fake.pos, {}, out)


class StatusSaysWhatIsHeld(unittest.TestCase):
    """/status reported a COUNT -- "positions 1/4" -- which answers "how
    many" and not the question anyone actually has.
    """

    def venue(self):
        fake = FakeHL(positions={"XRP": 135.0}, mid=1.3206,
                      unrealized={"XRP": 3.23})
        v = build_venue(fake)
        fake.rest_trigger("XRP", 135.0, 1.2678, is_stop=True, is_buy=False) \
            if hasattr(fake, "rest_trigger") else None
        return fake, v

    def test_each_position_is_named_with_its_pnl(self):
        fake, v = self.venue()
        out = C.render_status({}, 700.0, v)
        self.assertIn("XRP", out)
        self.assertIn("135", out)
        self.assertIn("+3.23", out)

    def test_an_unprotected_position_is_called_out_in_status(self):
        """The one thing worth seeing without having to ask a second
        command."""
        fake, v = self.venue()
        fake.resting = []                      # no stop at the venue
        out = C.render_status({}, 700.0, v)
        self.assertIn("NO STOP", out)

    def test_resting_bids_are_named_with_their_age(self):
        fake, v = self.venue()
        fake.rest_plain_order("HYPE", 2.36, 74.722, is_buy=True)
        out = C.render_status({}, 700.0, v)
        self.assertIn("HYPE", out)
        self.assertIn("74.722", out)

    def test_it_still_reports_an_unreadable_venue_rather_than_flat(self):
        fake = FakeHL(positions={"XRP": 1.0})
        fake.fail_info_always = True
        out = C.render_status({}, 700.0, build_venue(fake))
        self.assertIn("unreadable", out)


class PositionsReadsTheVenueNotThePlan(unittest.TestCase):
    """/positions took the target from the PLAN. The plan says what was
    intended; only a resting order will actually sell."""

    def test_a_planned_target_that_is_not_resting_is_flagged(self):
        from core.meta import MetaStore, TradeMeta
        import tempfile
        from pathlib import Path
        fake = FakeHL(positions={"XRP": 135.0}, mid=1.3206)
        v = build_venue(fake)
        store = MetaStore(Path(tempfile.mkdtemp()) / "m.json")
        store.put("XRP", TradeMeta(reason="x", planned_stop=1.2678,
                                   planned_tp=1.3573, opened_at=""))
        with mock.patch("core.meta.MetaStore", return_value=store):
            out = C.render_positions({}, v)
        self.assertIn("not resting", out,
                      "a target that exists only in the plan was shown as a tp")


class TokenAndConflict(unittest.TestCase):
    TOKEN = "123456789:AAabcDEF_ghi-jkl"

    def test_a_network_error_never_prints_the_token(self):
        def boom(u, p, timeout=None):
            raise OSError(f"Max retries exceeded with url: {u}")
        out = io.StringIO()
        with mock.patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": self.TOKEN}), \
                mock.patch.object(notify, "_transport", boom), \
                contextlib.redirect_stdout(out):
            notify.get_updates(1)
        self.assertIn("bot<redacted>", out.getvalue())
        self.assertNotIn(self.TOKEN, out.getvalue())

    def _poll(self, transport):
        with mock.patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": self.TOKEN}), \
                mock.patch.object(notify, "_transport", transport):
            notify.get_updates(1)
        return notify.poll_conflicted()

    def test_only_a_409_counts_as_a_shared_token(self):
        def down(u, p, timeout=None):
            raise OSError("down")

        class Reply:
            @staticmethod
            def json():
                return {"ok": False, "error_code": 409, "description": "Conflict"}
        self.assertFalse(self._poll(down))
        self.assertTrue(self._poll(lambda u, p, timeout=None: Reply))

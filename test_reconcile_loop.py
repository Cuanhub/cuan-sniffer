"""The fast reconciliation loop. Audit finding B-1 (interim).

There is no websocket anywhere in this project -- both SDK clients are built
with skip_ws=True -- so every piece of market and account data arrives by
REST poll on the trader's 450-second cadence. A gap through a stop, a partial
fill, or a venue-side liquidation is therefore invisible for up to seven and
a half minutes.

Websockets are the real answer and are a later batch. This is the interim:
a loop that does nothing but compare positions and protective orders against
the venue every RECONCILE_POLL_SEC, independent of the signal loop, and halt
plus alert on any divergence it cannot explain.

It is deliberately NOT allowed to place entries, and deliberately does not
share the trader's state file for writes -- one writer stays one writer. It
raises the alarm; the trader acts.
"""

import json
import pathlib
import tempfile
import unittest
from unittest import mock

import tools.reconcile_loop as R


class FakeBackend:
    def __init__(self, positions=None, orders=None, equity=1000.0):
        self.positions = positions or {}
        self.orders = orders or {}
        self.equity = equity

    def get_margin_summary(self):
        return (self.equity, 0.0, self.equity)

    def _user_state_query_address(self):
        return "0xfake"

    def get_open_orders(self, coin=""):
        return self.orders.get(coin.upper(), [])

    @property
    def info(self):
        outer = self

        class I:
            @staticmethod
            def user_state(addr, dex=""):
                return {"assetPositions": [
                    {"position": {"coin": c, "szi": str(s),
                                  "positionValue": str(abs(s) * 100)}}
                    for c, s in outer.positions.items()]}
        return I()


def state(owned=None, halted=False):
    return {"owned": owned or {}, "pending": {}, "halted": halted,
            "realised_today": 0.0}


class Agreement(unittest.TestCase):
    def test_agreement_reports_nothing(self):
        b = FakeBackend(positions={"HYPE": 2.0},
                        orders={"HYPE": [{"coin": "HYPE", "reduceOnly": True,
                                          "isTrigger": True}]})
        problems = R.check(b, state({"HYPE": {"size_coin": 2.0}}))
        self.assertEqual(problems, [])

    def test_flat_everywhere_reports_nothing(self):
        self.assertEqual(R.check(FakeBackend(), state()), [])


class Divergence(unittest.TestCase):
    def test_a_position_the_venue_has_and_state_does_not(self):
        """The C-1 timeout shape: untracked and probably unprotected."""
        b = FakeBackend(positions={"ZRO": 50.0})
        problems = R.check(b, state())
        self.assertTrue(any("ZRO" in p and "untracked" in p for p in problems),
                        problems)

    def test_a_position_state_has_and_the_venue_does_not(self):
        b = FakeBackend()
        problems = R.check(b, state({"HYPE": {"size_coin": 2.0}}))
        self.assertTrue(any("HYPE" in p and "gone" in p for p in problems),
                        problems)

    def test_an_open_position_with_no_protective_order(self):
        b = FakeBackend(positions={"HYPE": 2.0}, orders={"HYPE": []})
        problems = R.check(b, state({"HYPE": {"size_coin": 2.0}}))
        self.assertTrue(any("HYPE" in p and "unprotected" in p
                            for p in problems), problems)

    def test_a_size_that_has_drifted(self):
        b = FakeBackend(positions={"HYPE": 1.0},
                        orders={"HYPE": [{"coin": "HYPE", "isTrigger": True}]})
        problems = R.check(b, state({"HYPE": {"size_coin": 2.0}}))
        self.assertTrue(any("HYPE" in p and "size" in p for p in problems),
                        problems)

    def test_several_problems_are_all_reported(self):
        b = FakeBackend(positions={"ZRO": 5.0})
        problems = R.check(b, state({"HYPE": {"size_coin": 2.0}}))
        self.assertGreaterEqual(len(problems), 2)


class CannotRead(unittest.TestCase):
    def test_an_unreadable_venue_is_not_a_divergence(self):
        """A failed query must not be reported as a missing position -- that
        would halt the account on a rate limit."""
        class Broken(FakeBackend):
            @property
            def info(self):
                class I:
                    @staticmethod
                    def user_state(addr, dex=""):
                        raise RuntimeError("429")
                return I()
        self.assertIsNone(R.check(Broken(), state({"HYPE": {}})))

    def test_unreadable_orders_do_not_report_unprotected(self):
        class NoOrders(FakeBackend):
            def get_open_orders(self, coin=""):
                return None
        b = NoOrders(positions={"HYPE": 2.0})
        problems = R.check(b, state({"HYPE": {"size_coin": 2.0}}))
        self.assertFalse(any("unprotected" in p for p in problems), problems)


class ItRaisesTheAlarmAndDoesNotTrade(unittest.TestCase):
    def test_it_queues_a_halt_rather_than_writing_the_state_file(self):
        """One writer stays one writer: the trader owns _live_state.json."""
        with tempfile.TemporaryDirectory() as d:
            q = pathlib.Path(d) / "q.json"
            with mock.patch.object(R, "QUEUE", q):
                R.raise_alarm(["HYPE untracked at the venue"])
            cmds = json.loads(q.read_text())
        self.assertEqual(cmds[0]["cmd"], "halt")
        self.assertIn("reconcile", cmds[0]["arg"])

    def test_it_alerts(self):
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.object(R, "QUEUE", pathlib.Path(d) / "q.json"), \
                    mock.patch.object(R.notify, "send") as sent:
                R.raise_alarm(["ZRO untracked at the venue"])
        self.assertTrue(sent.called)
        self.assertIn("ZRO", sent.call_args[0][0])

    def test_it_cannot_place_an_entry(self):
        src = open("tools/reconcile_loop.py").read()
        for banned in ("execute_entry", "place_limit_entry", "size_from_stop"):
            self.assertNotIn(banned, src,
                             f"the reconciler must not be able to {banned}")

    def test_it_does_not_write_the_trader_state_file(self):
        src = open("tools/reconcile_loop.py").read()
        self.assertNotIn("save_state", src)
        self.assertNotIn("_live_state.json", src.split("STATE =")[-1][:200]
                         .replace('"_live_state.json"', ""))


class Cadence(unittest.TestCase):
    def test_the_poll_is_far_faster_than_the_signal_loop(self):
        from research.levels.config import POLL_SEC, RECONCILE_POLL_SEC
        self.assertLessEqual(RECONCILE_POLL_SEC, 60)
        self.assertLess(RECONCILE_POLL_SEC * 5, POLL_SEC,
                        "the point is to be much faster than 450s")

    def test_repeated_identical_problems_do_not_spam(self):
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.object(R, "QUEUE", pathlib.Path(d) / "q.json"), \
                    mock.patch.object(R.notify, "send") as sent:
                R.raise_alarm(["HYPE untracked"])
                R.raise_alarm(["HYPE untracked"])
        self.assertEqual(sent.call_count, 1, "the same alarm fired twice")


if __name__ == "__main__":
    unittest.main()

"""backend.flatten: "make me flat" is a loop that verifies, not one order.

On 2026-10-17 a hand-flatten of four positions sent one execute_exit each and
reported all four as sent. The venue told a different story: SUI, WLD and NIL
had closed only down to dust ($0.20, $5.94, $1.90) and ZRO was entirely
untouched at $178.70. A second pass closed all four.

execute_exit sends one marketable IOC with bounded slippage. Partial fills and
no-fills are both normal, so a single call is not a close. Worse, FillResult
reports `filled=True` for a PARTIAL fill, so every caller that branched on
`filled` and then released the position's slot would orphan live size at the
venue -- which is what two of the three trader paths did.

Size is read from the VENUE on each attempt rather than from what the caller
believes it holds, because the gap between those two is the condition this
exists to resolve.
"""

import unittest
from unittest import mock

import live_execution_backend as LB
from live_execution_backend import LiveExecutionBackend


class Fake(LiveExecutionBackend):
    """Only the four venue touchpoints flatten uses, no network, no __init__."""

    def __init__(self, sizes, mid=2.0, fill_fraction=1.0):
        # Deliberately NOT calling super().__init__ -- it opens a venue.
        self.sizes = list(sizes)        # size seen on each successive read
        self.mid = mid
        self.fill_fraction = fill_fraction
        self.exits = []
        self.reads = 0

    def _get_venue_signed_size(self, coin):
        i = min(self.reads, len(self.sizes) - 1)
        self.reads += 1
        return self.sizes[i]

    def get_mid_price(self, coin, **kw):
        return self.mid

    def execute_exit(self, coin, side, price, size_usd, reason=""):
        self.exits.append({"coin": coin, "size_usd": size_usd,
                           "reason": reason})

        class R:
            filled = True
            fill_size_usd = size_usd * self.fill_fraction
            fill_ratio = self.fill_fraction
            reject_reason = ""
        return R()


class _NoSleep(unittest.TestCase):
    """flatten waits before re-reading a failed venue query. Tests must not."""

    def setUp(self):
        self._p = mock.patch.object(LB.time, "sleep")
        self.slept = self._p.start()
        self.addCleanup(self._p.stop)


class ReadFailureBacksOff(_NoSleep):
    def test_an_unreadable_size_waits_before_re_asking(self):
        """Three reads inside a millisecond is one read with extra steps."""
        b = Fake(sizes=[None])
        b.flatten("ZRO", attempts=3)
        self.assertEqual(self.slept.call_count, 3)
        self.assertTrue(all(c.args[0] > 0 for c in self.slept.call_args_list))

    def test_a_partial_fill_does_NOT_wait(self):
        """After a real exit attempt the venue has already answered."""
        b = Fake(sizes=[176.5, 0.2, 0.0])
        b.flatten("SUI")
        self.assertEqual(self.slept.call_count, 0)

    def test_the_wait_is_a_bounded_fraction_of_a_pass(self):
        from research.levels.config import POLL_SEC
        self.assertLess(LB.RETRY_READ_SLEEP_SEC * 3, POLL_SEC * 0.05,
                        "flatten could delay a trading pass")


class AlreadyFlat(_NoSleep):
    def test_no_order_is_sent(self):
        b = Fake(sizes=[0.0])
        res = b.flatten("ZRO")
        self.assertTrue(res["flat"])
        self.assertEqual(b.exits, [])
        self.assertEqual(res["closed_usd"], 0.0)

    def test_dust_below_the_epsilon_counts_as_flat(self):
        b = Fake(sizes=[1e-12])
        self.assertTrue(b.flatten("ZRO")["flat"])
        self.assertEqual(b.exits, [])


class OneAttemptIsEnough(_NoSleep):
    def test_a_clean_close_sends_one_order(self):
        b = Fake(sizes=[89.5, 0.0])
        res = b.flatten("ZRO")
        self.assertTrue(res["flat"])
        self.assertEqual(len(b.exits), 1)
        self.assertAlmostEqual(b.exits[0]["size_usd"], 89.5 * 2.0)
        self.assertEqual(res["attempts_used"], 1)

    def test_the_reason_is_passed_through(self):
        b = Fake(sizes=[1.0, 0.0])
        b.flatten("ZRO", reason="telegram")
        self.assertEqual(b.exits[0]["reason"], "telegram")


class RetryUntilFlat(_NoSleep):
    def test_a_partial_fill_is_retried(self):
        """The SUI/WLD/NIL shape: closes to dust, then finishes."""
        b = Fake(sizes=[176.5, 0.2, 0.0])
        res = b.flatten("SUI")
        self.assertTrue(res["flat"])
        self.assertEqual(len(b.exits), 2, "it stopped after one partial")

    def test_size_is_re_read_from_the_venue_each_attempt(self):
        """Not from what the caller believes. The second order must be sized
        on the dust that is left, not on the original position."""
        b = Fake(sizes=[176.5, 0.2, 0.0])
        b.flatten("SUI")
        self.assertAlmostEqual(b.exits[0]["size_usd"], 176.5 * 2.0)
        self.assertAlmostEqual(b.exits[1]["size_usd"], 0.2 * 2.0)

    def test_a_no_fill_is_retried(self):
        """The ZRO shape: the first order did nothing at all."""
        b = Fake(sizes=[89.5, 89.5, 0.0])
        res = b.flatten("ZRO")
        self.assertTrue(res["flat"])
        self.assertEqual(len(b.exits), 2)


class RefusesToClaimFlat(_NoSleep):
    def test_a_position_that_will_not_close_reports_not_flat(self):
        b = Fake(sizes=[89.5])          # never moves
        res = b.flatten("ZRO")
        self.assertFalse(res["flat"])
        self.assertIn("STILL OPEN", res["detail"])
        self.assertEqual(len(b.exits), 3, "it should have used its attempts")

    def test_an_unreadable_venue_is_never_reported_as_flat(self):
        """Saying 'flat' on a failed read is the one answer that orphans a
        position."""
        b = Fake(sizes=[None])
        res = b.flatten("ZRO")
        self.assertFalse(res["flat"])
        self.assertEqual(b.exits, [], "it sent an order it could not size")
        self.assertIn("unreadable", res["detail"])

    def test_a_missing_price_does_not_send_a_blind_order(self):
        b = Fake(sizes=[89.5], mid=0.0)
        res = b.flatten("ZRO")
        self.assertFalse(res["flat"])
        self.assertEqual(b.exits, [])
        self.assertIn("no price", res["detail"])

    def test_attempts_is_respected(self):
        b = Fake(sizes=[89.5])
        b.flatten("ZRO", attempts=1)
        self.assertEqual(len(b.exits), 1)

    def test_closed_usd_accumulates_across_attempts(self):
        b = Fake(sizes=[100.0, 50.0, 0.0], fill_fraction=0.5)
        res = b.flatten("ZRO")
        self.assertGreater(res["closed_usd"], 0)


class CallersBranchOnFlat(unittest.TestCase):
    def test_no_close_path_branches_on_filled_any_more(self):
        """FillResult.filled is True for a PARTIAL fill, so a caller that
        reads it and then releases the slot orphans live size. All three
        trader close paths went through this; two did not even check."""
        src = open("tools/live_levels_trader.py").read()
        self.assertNotIn('getattr(exit_fill, "filled"', src)
        # Count is a FLOOR, not an equality. The original assertion pinned it
        # at exactly 3 and then failed when the audit's orphan sweep and
        # halt-flatten legitimately added two more -- a test that resists
        # correct change is a test that gets deleted.
        self.assertGreaterEqual(src.count("backend.flatten("), 3,
                                "a close path stopped going through flatten")
        # The property that actually matters: nothing outside the backend
        # calls execute_exit directly, so no caller can mistake one IOC for
        # a close.
        self.assertNotIn("backend.execute_exit(", src)

    def test_the_telegram_path_uses_it_too(self):
        src = open("tools/telegram_control.py").read()
        self.assertIn("self.backend.flatten(", src)
        self.assertNotIn("self.backend.execute_exit(", src)


if __name__ == "__main__":
    unittest.main()


class AShortIsNeverCalledFlat(_NoSleep):
    """Found in the pre-resume review on 2026-10-17.

    _signed_size_for_side("LONG", -5) returns 0, so judging flatness by SIDE
    reported a live short as flat. Nothing in this system takes shorts, so it
    would have sat there until the account held one from somewhere else -- a
    hand-placed trade on the same wallet -- and then /close would have said
    the position was gone while it was still on.

    An exit in the wrong direction would also ADD to the position, so the
    answer is to refuse and say so, not to send it.
    """

    def test_a_short_is_reported_open_not_flat(self):
        b = Fake(sizes=[-5.0])
        res = b.flatten("ZRO", side="LONG")
        self.assertFalse(res["flat"], "a live short was reported flat")
        self.assertIn("not a LONG", res["detail"])

    def test_no_wrong_way_order_is_sent(self):
        b = Fake(sizes=[-5.0])
        b.flatten("ZRO", side="LONG")
        self.assertEqual(b.exits, [],
                         "selling while short would increase the position")

    def test_a_short_can_be_flattened_as_a_short(self):
        b = Fake(sizes=[-5.0, 0.0])
        res = b.flatten("ZRO", side="SHORT")
        self.assertTrue(res["flat"])
        self.assertEqual(len(b.exits), 1)

    def test_the_final_verdict_also_uses_the_raw_size(self):
        """A position that flips sign mid-flatten must not read as flat."""
        b = Fake(sizes=[5.0, -5.0, -5.0, -5.0])
        res = b.flatten("ZRO", side="LONG")
        self.assertFalse(res["flat"])

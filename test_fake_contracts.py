"""Fakes must not invent interfaces the real classes do not have.

This is the generalised form of the 2026-10-17 bug. close_one branched on
`res.status`; FillResult has no such field, so the branch was dead and a
half-filled close read as a success. The test that should have caught it used
a fake that RETURNED a `status` attribute -- it modelled an interface that
never existed, so it certified the broken code.

A fake that is more permissive than the real thing does not test the code, it
tests the fake. These assertions are cheap and they close that whole class.
"""

import unittest

from execution_backend import FillResult
from live_execution_backend import LiveExecutionBackend


def fields(dc):
    return set(dc.__dataclass_fields__)


class FillResultFakes(unittest.TestCase):
    def test_FakeFill_claims_nothing_FillResult_lacks(self):
        from test_live_trader import FakeFill
        extra = fields(FakeFill) - fields(FillResult)
        self.assertEqual(extra, set(),
                         f"FakeFill invents {sorted(extra)}; a fake that is "
                         f"more permissive than the real class certifies "
                         f"code the real class would break")

    def test_FillResult_still_has_no_status(self):
        """The field the dead branch read. If it ever gains one, the comments
        in close_one and test_flatten need revisiting."""
        self.assertNotIn("status", fields(FillResult))

    def test_FillResult_still_carries_what_callers_branch_on(self):
        for name in ("filled", "fill_ratio", "fill_size_usd", "reject_reason"):
            self.assertIn(name, fields(FillResult))


class BackendFakes(unittest.TestCase):
    """Stand-ins for LiveExecutionBackend, checked against the real surface.

    `user_state` is allowed: FakeBackend doubles as the SDK `info` object,
    which the backend reaches through `self.info`.
    """

    ALLOWED_EXTRA = {"user_state"}

    def public_methods(self, cls):
        return {n for n in vars(cls)
                if not n.startswith("_") and callable(vars(cls)[n])}

    def check(self, cls, label):
        real = set(dir(LiveExecutionBackend))
        extra = self.public_methods(cls) - real - self.ALLOWED_EXTRA
        self.assertEqual(extra, set(),
                         f"{label} defines {sorted(extra)}, which "
                         f"LiveExecutionBackend does not have")

    def test_the_trader_test_backend(self):
        from test_live_trader import FakeBackend
        self.check(FakeBackend, "test_live_trader.FakeBackend")

    def test_the_telegram_test_backend(self):
        import test_telegram_control as t
        self.check(t.ClosingReportsTheVenue.Backend,
                   "test_telegram_control Backend")

    def test_the_flatten_test_backend_is_a_real_subclass(self):
        """The strongest form: it inherits, so it cannot drift at all."""
        from test_flatten import Fake
        self.assertTrue(issubclass(Fake, LiveExecutionBackend))

    def test_every_backend_fake_implements_flatten_or_inherits_it(self):
        """A fake missing `flatten` silently exercises a path production no
        longer takes."""
        from test_live_trader import FakeBackend
        import test_telegram_control as t
        for cls, label in ((FakeBackend, "FakeBackend"),
                           (t.ClosingReportsTheVenue.Backend, "tg Backend")):
            self.assertTrue(hasattr(cls, "flatten"),
                            f"{label} has no flatten")


if __name__ == "__main__":
    unittest.main()

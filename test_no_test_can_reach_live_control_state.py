"""No test may hold a path into the live data directory.

Three times now a test has written to research/levels/data/ because a module
global was not in conftest's redirect list:

  - 70 fabricated rows in live_trades.csv, the real-money audit trail, and
    realised_today left at -10.00 where the daily loss stop would count it;
  - four rows in episodes.csv for AAA, BBB, CCC and DDD -- fixture names, not
    perps -- which the scanner's outcome resolver then retried forever, about
    14,500 wasted venue requests;
  - a genuine `halt` appended to _control_queue.json by
    test_every_advertised_command_is_handled, which calls every command the
    help text advertises. The trader drained it ~50s later and market-closed
    an open ARB position on 2026-10-17.

Each was fixed by adding one more name to the list, which is how there came
to be a third. This checks the property instead: after conftest's autouse
fixture has run, no module under test may still name a path inside the real
data directory. It fails on the next module without anyone knowing it exists.
"""

from __future__ import annotations

import importlib
import pathlib

import pytest

# Computed from this file, NOT from the modules under test -- the whole point
# is to disagree with them when one of them is pointing at the live tree.
LIVE_DATA = (pathlib.Path(__file__).resolve().parent
             / "research" / "levels" / "data")

# Every module that reads or writes the data directory. A module missing from
# here is not checked, so it is also listed in test_the_watched_list_is_whole.
WATCHED = (
    "tools.live_levels_trader",
    "tools.telegram_control",
    "tools.protect_loop",
    "core.meta",
    "core.safety",
    "research.levels.storage",
    # Both found by test_the_watched_list_is_whole on its first run rather
    # than by anything going wrong -- retention rotates live_trades.csv.
    "tools.retention",
    "tools.screen_context",
)


def _live_paths(mod):
    """Module-level Paths that resolve inside the live data directory."""
    out = []
    for name, val in vars(mod).items():
        if not isinstance(val, pathlib.Path):
            continue
        try:
            val.resolve().relative_to(LIVE_DATA)
        except ValueError:
            continue
        out.append(f"{mod.__name__}.{name} = {val}")
    return out


@pytest.mark.parametrize("name", WATCHED)
def test_no_module_still_points_at_the_live_data_directory(name):
    leaks = _live_paths(importlib.import_module(name))
    assert not leaks, (
        "these names still reach the live data directory during tests; add "
        "them to conftest.isolated_runtime_files:\n  " + "\n  ".join(leaks))


def test_the_live_control_queue_is_not_what_a_test_writes():
    """The specific one that cost a position, named so it cannot come back.

    The control queue is an INSTRUCTION to a running trader, not a record of
    one. A stray write to live_trades.csv corrupts history; a stray write
    here acts on the account.
    """
    import tools.telegram_control as control
    import tools.live_levels_trader as trader

    assert control.QUEUE.resolve() != (LIVE_DATA / "_control_queue.json")
    assert trader.CONTROL_QUEUE.resolve() != (LIVE_DATA / "_control_queue.json")


def test_calling_every_advertised_command_writes_nothing_live():
    """The exact loop that did it, run again with the real file watched.

    Reads the live queue directly rather than through the module, because
    reading it through the module is what hid the bug.
    """
    import tools.telegram_control as control

    live = LIVE_DATA / "_control_queue.json"
    before = live.read_bytes() if live.exists() else None

    ctl = control.Controller(None)
    for line in control.HELP.splitlines():
        text = line.strip()
        if text.startswith("`/"):
            ctl.handle(text[1:].split("`")[0], {"owned": {}})

    after = live.read_bytes() if live.exists() else None
    assert after == before, (
        "driving the advertised commands changed the LIVE control queue")


def test_the_watched_list_is_whole():
    """A module that writes into the data directory and is not WATCHED is
    unguarded, so finding one is itself the failure."""
    import pkgutil

    missing = []
    for pkg in ("tools", "core"):
        for info in pkgutil.iter_modules([str(pathlib.Path(__file__).parent
                                              / pkg)]):
            dotted = f"{pkg}.{info.name}"
            if dotted in WATCHED:
                continue
            try:
                mod = importlib.import_module(dotted)
            except Exception:
                continue        # not importable in isolation; not our problem
            if _live_paths(mod):
                missing.append(dotted)
    assert not missing, (
        f"these modules reach the live data directory and are not WATCHED: "
        f"{missing}")

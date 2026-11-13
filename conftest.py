"""Suite-wide safety: no alert may leave this machine during a test.

tools/live_levels_trader calls load_dotenv() at import, so a test process
holds the operator's real TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID. Several
tests deliberately drive the trader into a halt, and halt() alerts on the
transition -- so the first run of the suite after alerting was added sent
real messages to a real phone. Twice, before this file existed.

Autouse, so a test cannot forget to opt in, and applied to every test rather
than only the ones that look like they alert: the whole point is that the
next person to add a halt path does not have to know this file is here.
"""

import os

# Read at import by the modules below. These waits are for the real venue's
# lag; a fake answers at once, and the suite would take many minutes otherwise.
os.environ.setdefault("PROTECT_CONFIRM_WAIT_SEC", "0")
os.environ.setdefault("LEVELS_MIN_REQUEST_GAP_SEC", "0")
# Forced, not defaulted: the trader loads the operator's .env at import, and an
# armed trend book changes what the band trader and protect do with ten coins.
# A test that needs it armed says so with monkeypatch.setenv.
os.environ["TREND_LIVE_ARMED"] = "false"
os.environ["TREND_CAPITAL_USD"] = ""

import pytest

import notify


class _OkResponse:
    @staticmethod
    def json():
        return {"ok": True}


@pytest.fixture(autouse=True)
def isolated_runtime_files(tmp_path, monkeypatch):
    """No test may write to the live state file or the real trade log.

    Both are module-level Paths pointing at research/levels/data/, and the
    functions under test save state and append trade rows as a matter of
    course -- reconcile() does both. Tests that built their own state dict
    and called reconcile() directly wrote 70 fabricated rows into
    live_trades.csv, the real-money audit trail, and left realised_today at
    -10.00 on the live account, where the daily loss stop would have counted
    it. Redirecting per-test was the arrangement that allowed that; this is
    autouse so the next test cannot opt out by not knowing.

    This covered the trader's two paths and MISSED storage.DATA_DIR, which is
    how episodes.csv acquired four rows for the coins AAA, BBB, CCC and DDD --
    test fixture names, not perps; the venue has 234 and none of them is in
    it. They carried no detected_at_utc and never resolved, so the scanner's
    outcome resolver retried them on every pass forever: 3,630 failed
    candleSnapshot fetches per coin, about 14,500 wasted venue requests,
    found in the 2026-10-16 audit. Any module that writes into the data
    directory has to be redirected here, not just the ones that were doing it
    at the time.

    It then MISSED tools.telegram_control's QUEUE, and that one cost money.
    `test_every_advertised_command_is_handled` walks the help text and calls
    every command it finds, `/halt` among them, straight into the real
    handler -- which appended a genuine halt to the live control queue. On
    2026-10-17 the trader drained it about 50 seconds later and market-closed
    an open ARB position. The operator was told the position had been closed
    "from Telegram" and knew he had not touched Telegram.

    Three instances of one mistake is a pattern, so there is now a structural
    guard as well as this list: test_no_test_can_reach_live_control_state.py
    walks these modules for any Path still inside the real data directory and
    fails, which catches the next module without anyone remembering to.
    """
    import tools.live_levels_trader as trader
    import tools.telegram_control as control
    from research.levels import storage
    from core import meta as core_meta
    from core import safety as core_safety
    from tools import liveness

    monkeypatch.setattr(liveness, "DATA", tmp_path / "data")
    monkeypatch.setattr(liveness, "STATE", tmp_path / "_liveness.json")

    monkeypatch.setattr(trader, "STATE_PATH", tmp_path / "_live_state.json")
    monkeypatch.setattr(trader, "TRADE_LOG", tmp_path / "live_trades.csv")
    monkeypatch.setattr(trader, "CONTROL_QUEUE", tmp_path / "_control_queue.json")
    monkeypatch.setattr(storage, "DATA_DIR", tmp_path / "data")
    # core.meta and core.safety both default their paths off this directory.
    # A MetaStore(), HaltState() or EquityLog() built with no argument in a
    # test would otherwise read -- and on any write, clobber -- the live files.
    # core.safety binds DATA at import, so patching core.meta alone misses it.
    monkeypatch.setattr(core_meta, "DATA", tmp_path / "data")
    monkeypatch.setattr(core_safety, "DATA", tmp_path / "data")
    # The control queue is an INSTRUCTION to the live trader, not a record of
    # it, which is what made this the expensive one: a stray write here does
    # not corrupt history, it acts.
    monkeypatch.setattr(control, "DATA", tmp_path / "data")
    monkeypatch.setattr(control, "STATE", tmp_path / "_live_state.json")
    monkeypatch.setattr(control, "QUEUE", tmp_path / "_control_queue.json")
    monkeypatch.setattr(control, "OFFSET", tmp_path / "_control_offset.json")
    # Found by that structural guard on its first run, not by a failure:
    # retention ROTATES and archives live_trades.csv, so a test reaching it
    # would move the real audit trail aside rather than merely append to it.
    import tools.retention as retention
    import tools.screen_context as screen_context
    monkeypatch.setattr(retention, "DATA", tmp_path / "data")
    monkeypatch.setattr(retention, "ARCHIVE", tmp_path / "data" / "archive")
    monkeypatch.setattr(retention, "TRADE_LOG",
                        tmp_path / "data" / "live_trades.csv")
    monkeypatch.setattr(screen_context, "DATA", tmp_path / "data")


@pytest.fixture(autouse=True)
def no_live_venue(monkeypatch):
    """No test may reach Hyperliquid.

    The trader builds its read-only client lazily into a module global, so a
    test that exercises the armed path without supplying candles silently
    fetches them from the real API -- three new tests did exactly that and
    took 64 seconds doing it. Fail loudly instead; tests that need candles
    assign their own fake, as ArmedPathTest does.
    """
    class _Refuse:
        def get_candles(self, *a, **kw):
            raise AssertionError(
                "test reached the live venue: set T._RO_CLIENT to a fake")

    import tools.live_levels_trader as trader
    monkeypatch.setattr(trader, "_RO_CLIENT", _Refuse())


@pytest.fixture(autouse=True)
def captured_alerts(monkeypatch):
    """Swallow outbound alerts and hand the payloads back for assertions."""
    sent = []

    def _capture(url, payload):
        sent.append(payload)
        return _OkResponse()

    # notify.send returns before the transport when these are unset, so without
    # them a machine with no .env captures nothing and these tests fail.
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "test-chat")
    monkeypatch.setattr(notify, "_transport", _capture)
    notify.reset_cooldowns()
    yield sent
    notify.reset_cooldowns()

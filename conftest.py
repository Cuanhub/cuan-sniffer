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
    """
    import tools.live_levels_trader as trader
    from research.levels import storage
    from core import meta as core_meta
    from core import safety as core_safety

    monkeypatch.setattr(trader, "STATE_PATH", tmp_path / "_live_state.json")
    monkeypatch.setattr(trader, "TRADE_LOG", tmp_path / "live_trades.csv")
    monkeypatch.setattr(storage, "DATA_DIR", tmp_path / "data")
    # core.meta and core.safety both default their paths off this directory.
    # A MetaStore(), HaltState() or EquityLog() built with no argument in a
    # test would otherwise read -- and on any write, clobber -- the live files.
    # core.safety binds DATA at import, so patching core.meta alone misses it.
    monkeypatch.setattr(core_meta, "DATA", tmp_path / "data")
    monkeypatch.setattr(core_safety, "DATA", tmp_path / "data")


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

    monkeypatch.setattr(notify, "_transport", _capture)
    notify.reset_cooldowns()
    yield sent
    notify.reset_cooldowns()

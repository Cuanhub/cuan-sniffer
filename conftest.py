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
    """
    import tools.live_levels_trader as trader

    monkeypatch.setattr(trader, "STATE_PATH", tmp_path / "_live_state.json")
    monkeypatch.setattr(trader, "TRADE_LOG", tmp_path / "live_trades.csv")


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

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

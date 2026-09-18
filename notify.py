"""Telegram alerts for the things a human has to see.

The level trader runs unattended under launchd and prints to a log file in
~/Library/Logs. Nothing reads that file. On 2026-09-18 its output moved to a
new path and the old one simply stopped growing -- and the gap went unnoticed
for hours, because nobody was watching either location. Every safety rail in
this system ends in a print statement, which means every safety rail ends
somewhere nobody is looking.

What belongs here is state a human must act on: the bot halted and will not
resume by itself, or money moved in a way it did not intend. Nothing else.
Ninety-six passes a day printing "wait BTC" is what the log is for.

A failure here must never reach the trading loop. A Telegram outage is not a
trading condition, so send() catches everything and returns False.

Credentials come from TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID, read at call
time rather than import time so that a .env loaded by the caller still counts.
"""

import os
import time
from typing import Callable, Dict, Optional

import requests

TIMEOUT_SEC = 10

# key -> unix seconds of last CONFIRMED delivery
_last_sent: Dict[str, float] = {}


def _post(url: str, payload: dict):
    return requests.post(url, json=payload, timeout=TIMEOUT_SEC)


# Replaced in tests so nothing leaves the machine.
_transport: Callable = _post


def configured() -> bool:
    return bool(os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
                and os.getenv("TELEGRAM_CHAT_ID", "").strip())


def reset_cooldowns() -> None:
    _last_sent.clear()


def send(text: str, key: Optional[str] = None, cooldown_sec: float = 0.0) -> bool:
    """Deliver `text`. Returns True only when Telegram confirms delivery.

    `key` names a recurring alert so it can be rate limited. The trader polls
    every 15 minutes and a halt persists until someone clears it, so an
    unkeyed halt alert would send ninety-six identical messages a day and
    train the reader to ignore all of them.

    The cooldown is stamped on success only. A send that failed has not been
    seen by anyone, so the next pass should try again.
    """
    if key and cooldown_sec > 0:
        if time.time() - _last_sent.get(key, 0.0) < cooldown_sec:
            return False

    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat:
        print(f"[ALERT unsent -- telegram not configured] {text}")
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    base = {"chat_id": chat, "disable_web_page_preview": True}

    # Markdown first, then plain. An unescaped _ or * -- in a coin name, or in
    # a venue error string quoted verbatim -- makes Telegram reject the whole
    # message, and the fallback is the difference between a mangled alert and
    # no alert.
    for payload in ({**base, "text": text, "parse_mode": "Markdown"},
                    {**base, "text": text}):
        try:
            resp = _transport(url, payload)
            if resp.json().get("ok"):
                if key:
                    _last_sent[key] = time.time()
                return True
        except Exception as e:
            print(f"[ALERT] send failed: {str(e)[:160]}")

    print(f"[ALERT undelivered] {text}")
    return False

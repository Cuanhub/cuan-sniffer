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

# Transport failures are retried; a Telegram rejection is not. On 2026-09-23 a
# brief DNS/connection failure lost three messages outright -- including both
# OPENED alerts for that day's trades, which is why the positions came as a
# surprise. There was no retry of any kind: one failed POST and the message
# was gone.
#
# Three attempts with 1s then 2s of backoff. Deliberately small: send() is
# called from inside the trading pass, so the ceiling has to be a bounded
# fraction of POLL_SEC. A transport failure breaks out after one request, so
# the worst case is 3 timeouts plus the backoff -- 3*TIMEOUT_SEC + 3s = 33s,
# against a 450s pass. Raising either number is what would make an outage
# start delaying trades.
SEND_ATTEMPTS = 3
SEND_BACKOFF_SEC = 1.0

# key -> unix seconds of last CONFIRMED delivery
_last_sent: Dict[str, float] = {}


def _post(url: str, payload: dict, timeout: float = TIMEOUT_SEC):
    """`timeout` is a parameter because the two callers need different ones.

    TIMEOUT_SEC (10s) is right for sendMessage, which answers immediately and
    is called from inside the trading pass. A getUpdates long-poll deliberately
    HANGS for up to its own timeout, so a 10s read timeout aborts every quiet
    poll before Telegram ever answers -- which is exactly what it did once the
    409s cleared: a log of nothing but "Read timed out".
    """
    return requests.post(url, json=payload, timeout=timeout)


# Replaced in tests so nothing leaves the machine.
_transport: Callable = _post


def configured() -> bool:
    return bool(os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
                and os.getenv("TELEGRAM_CHAT_ID", "").strip())


def reset_cooldowns() -> None:
    _last_sent.clear()


def send(text: str, key: Optional[str] = None, cooldown_sec: float = 0.0,
         attempts: int = SEND_ATTEMPTS, sleep=time.sleep) -> bool:
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
    # Markdown first, then plain. A rejection means the text upset Telegram, so
    # the fallback is worth trying immediately; a raised exception means the
    # request never landed, which is worth trying AGAIN.
    variants = ({**base, "text": text, "parse_mode": "Markdown"},
                {**base, "text": text})
    for attempt in range(1, max(1, attempts) + 1):
        transport_failed = False
        for payload in variants:
            try:
                resp = _transport(url, payload)
            except Exception as e:
                transport_failed = True
                print(f"[ALERT] send failed (attempt {attempt}): {str(e)[:160]}")
                break          # the network is the problem, not the formatting
            try:
                if resp.json().get("ok"):
                    if key:
                        _last_sent[key] = time.time()
                    return True
            except Exception as e:
                transport_failed = True
                print(f"[ALERT] unreadable reply (attempt {attempt}): {str(e)[:120]}")
                break
        if not transport_failed:
            # Telegram answered and refused both forms. Retrying will not help.
            break
        if attempt < max(1, attempts):
            sleep(SEND_BACKOFF_SEC * attempt)

    print(f"[ALERT undelivered] {text}")
    return False


# ---- receiving ------------------------------------------------------------
# Long-polling getUpdates rather than a webhook, deliberately. A webhook needs
# an inbound port and a public address; this machine has neither and should
# not grow them for a trading bot. getUpdates is an ordinary outbound HTTPS
# request to a host this module already talks to, so the control channel adds
# no new attack surface and no new listening socket.
#
# The bot token is now a credential that can close positions and halt trading,
# not just a way to send messages. It cannot move money -- nothing here has
# transfer or withdrawal capability -- which bounds what a leaked token costs
# to "someone can flatten the book and stop the bot".
POLL_TIMEOUT_SEC = 25


def get_updates(offset: int = 0, timeout: int = POLL_TIMEOUT_SEC):
    """Fetch pending messages. Returns (updates, next_offset, ok).

    `updates` is a list of dicts with update_id, chat_id, user_id, text.
    Returns ([], offset, False) on any failure: an unreachable Telegram must
    not raise into a control loop, for the same reason send() never raises
    into the trading loop.

    `ok` exists because a caller cannot otherwise tell "no messages" from
    "the poll failed" -- both are an empty list -- and the two need opposite
    responses. Telegram answers a second concurrent getUpdates with 409 while
    the first is still hanging, so retrying a failure quickly makes the loop
    collide with its OWN open long-poll and 409 forever. Observed doing
    exactly that on first install. The caller must back off longer than
    `timeout` on a failure.

    The caller must advance `offset` past what it has handled, because
    Telegram re-delivers anything unacknowledged. Not advancing it on a
    command that failed would retry that command forever.
    """
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        return [], offset, False
    url = f"https://api.telegram.org/bot{token}/getUpdates"
    payload = {"timeout": timeout, "allowed_updates": ["message"]}
    if offset:
        payload["offset"] = offset
    try:
        # Read timeout must outlast the long-poll it is waiting on, with
        # enough margin for Telegram to actually reply at the end of it.
        resp = _transport(url, payload, timeout=timeout + 10)
        body = resp.json()
    except Exception as e:
        print(f"[CONTROL] poll failed: {str(e)[:160]}")
        return [], offset, False
    if not body.get("ok"):
        print(f"[CONTROL] poll refused: {str(body)[:160]}")
        return [], offset, False

    out = []
    nxt = offset
    for item in body.get("result") or []:
        try:
            uid = int(item["update_id"])
        except (KeyError, TypeError, ValueError):
            continue
        nxt = max(nxt, uid + 1)
        msg = item.get("message") or {}
        text = (msg.get("text") or "").strip()
        chat = str(((msg.get("chat") or {}).get("id")) or "")
        user = str(((msg.get("from") or {}).get("id")) or "")
        if text:
            out.append({"update_id": uid, "chat_id": chat,
                        "user_id": user, "text": text})
    return out, nxt, True


def authorised(chat_id: str) -> bool:
    """Whether a message came from the one chat allowed to command this bot.

    Compared against TELEGRAM_CHAT_ID, the same value alerts are sent to. An
    unconfigured chat id authorises NOTHING -- a missing credential must not
    read as "allow everyone", which is what `chat_id == os.getenv(...)` does
    when both sides are empty.
    """
    allowed = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    return bool(allowed) and str(chat_id).strip() == allowed

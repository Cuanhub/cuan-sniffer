import requests
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

BASE_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"


def send_telegram_message(text: str):
    """
    Bulletproof Telegram sender.

    - Tries Markdown first
    - Falls back to plain text if formatting breaks
    - Never silently fails
    """

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[WARN] Telegram not configured. Message would be:")
        print(text)
        return

    url = f"{BASE_URL}/sendMessage"

    def send(payload):
        return requests.post(url, json=payload, timeout=10)

    # ── Attempt 1: Markdown ─────────────────────────────
    payload_md = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True,
    }

    try:
        resp = send(payload_md)
        data = resp.json()

        if data.get("ok"):
            print("[TELEGRAM] Message sent (Markdown).")
            return

        else:
            print(f"[WARN] Markdown failed: {data}")

    except Exception as e:
        print(f"[ERROR] Markdown send failed: {e}")

    # ── Attempt 2: Plain text fallback ───────────────────
    payload_plain = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "disable_web_page_preview": True,
    }

    try:
        resp = send(payload_plain)
        data = resp.json()

        if data.get("ok"):
            print("[TELEGRAM] Message sent (plain fallback).")
        else:
            print(f"[ERROR] Telegram API error (plain): {data}")

    except Exception as e:
        print(f"[CRITICAL] Telegram completely failed: {e}")
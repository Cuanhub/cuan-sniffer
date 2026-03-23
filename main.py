import time
import random
import requests

from config import TRACKED_WALLETS, POLL_INTERVAL_SECONDS
from db import init_db
from engine import SolFlowEngine


def main():
    if not TRACKED_WALLETS:
        print("❌ No TRACKED_WALLETS configured")
        return

    print("🚀 Initializing database...")
    init_db()

    engine = SolFlowEngine(tracked_wallets=TRACKED_WALLETS)

    cycle = 0
    backoff = 5

    BATCH_SIZE = 5

    print("✅ Engine live\n")

    try:
        while True:
            cycle += 1
            print(f"[ENGINE] 🔁 Cycle {cycle}")

            start = (cycle * BATCH_SIZE) % len(TRACKED_WALLETS)
            batch = TRACKED_WALLETS[start:start + BATCH_SIZE]

            for addr in batch:
                try:
                    engine.process_wallet(addr)
                    backoff = 5  # reset on success

                except requests.HTTPError as e:
                    if getattr(e.response, "status_code", None) == 429:
                        print(f"[RATE LIMIT] Cooling down {backoff}s...")
                        time.sleep(backoff)
                        backoff = min(backoff * 2, 60)
                        continue

                    print(f"[ENGINE ERROR] {addr}: {e}")

                except Exception as e:
                    print(f"[ENGINE ERROR] {addr}: {e}")

                time.sleep(0.6 + random.uniform(0, 0.4))

            print(f"[ENGINE] 💤 Sleeping {POLL_INTERVAL_SECONDS}s\n")
            time.sleep(POLL_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\n🛑 Engine stopped.")


if __name__ == "__main__":
    main()
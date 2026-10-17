"""Record real Hyperliquid payloads for the test fixtures. Read-only.

Run: python3 -m tests.record_fixtures

Every fixture under tests/fixtures/hyperliquid/ comes from here. The fake venue
in the tests is built ONLY from those files, so a test can never assert a payload
shape the venue does not actually produce -- which is how the previous suite
passed while the code misread `sz` (see REVIEW.md finding 1).

This module makes no exchange calls. It POSTs to /info only.

Account identifiers are scrubbed: the address becomes a fixed placeholder and
cloids are dropped. Sizes, prices, statuses, field names and nesting are
untouched, because those are the whole point.

PROVENANCE is written alongside the fixtures and records where each file came
from. One file (clearinghouse_state_with_position.json) is the official
documented example rather than a recording: this account was flat at record time,
so no live payload with an open position was available. It is marked as such.
"""

from __future__ import annotations

import json
import os
import pathlib
import time
from typing import Any, Dict, List

from dotenv import load_dotenv

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "tests/fixtures/hyperliquid"
PLACEHOLDER_ADDR = "0x" + "11" * 20


def _scrub(obj: Any, real_addr: str) -> Any:
    """Replace the account address and drop cloids, recursively."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k == "cloid":
                out[k] = None
                continue
            out[k] = _scrub(v, real_addr)
        return out
    if isinstance(obj, list):
        return [_scrub(v, real_addr) for v in obj]
    if isinstance(obj, str) and real_addr and real_addr.lower() in obj.lower():
        return PLACEHOLDER_ADDR
    return obj


def _write(name: str, payload: Any, real_addr: str) -> int:
    path = OUT / name
    path.write_text(json.dumps(_scrub(payload, real_addr), indent=2) + "\n")
    n = len(payload) if isinstance(payload, list) else 1
    print(f"  wrote {name} ({n} record(s), {path.stat().st_size} bytes)")
    return n


def main() -> int:
    load_dotenv(ROOT / ".env", override=False)
    addr = os.getenv("HL_ACCOUNT_ADDRESS", "").strip()
    if not addr:
        print("HL_ACCOUNT_ADDRESS not set -- cannot record")
        return 1

    from hyperliquid.info import Info
    from hyperliquid.utils import constants

    testnet = os.getenv("HL_TESTNET", "true").lower() == "true"
    base = constants.TESTNET_API_URL if testnet else constants.MAINNET_API_URL
    info = Info(base, skip_ws=True, spot_meta={"tokens": [], "universe": []})
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"recording from {'TESTNET' if testnet else 'MAINNET'} (read-only)")

    hist = info.historical_orders(addr)
    if not isinstance(hist, list):
        print(f"historical_orders returned {type(hist)} -- aborting")
        return 1

    # One record per distinct inner status, which is the axis the old code got
    # wrong. Keep them all: 'filled' (sz=0) and 'perpMarginRejected' (sz=origSz)
    # are the two that must never be confused.
    by_status: Dict[str, List[Dict[str, Any]]] = {}
    for h in hist:
        by_status.setdefault(str(h.get("status")), []).append(h)
    print(f"  inner statuses seen: "
          f"{ {k: len(v) for k, v in sorted(by_status.items())} }")

    samples = [v[0] for _, v in sorted(by_status.items())]
    _write("order_status_by_inner_status.json", samples, addr)

    # The full orderStatus envelope, as query_order_by_oid returns it. The
    # envelope status is 'order' for every one of the statuses above, which is
    # exactly why reading only the envelope was wrong.
    envelopes = []
    for st, rows in sorted(by_status.items()):
        oid = (rows[0].get("order") or {}).get("oid")
        if oid is None:
            continue
        env = info.query_order_by_oid(addr, int(oid))
        envelopes.append({"_inner_status": st, "payload": env})
        time.sleep(0.12)                      # stay well under the rate limit
    _write("order_status_envelopes.json", envelopes, addr)

    _write("clearinghouse_state_flat.json", info.user_state(addr), addr)
    _write("spot_user_state.json", info.spot_user_state(addr), addr)
    _write("frontend_open_orders.json", info.frontend_open_orders(addr), addr)
    _write("user_fills_30d.json",
           info.user_fills_by_time(addr, int(time.time() * 1000) - 30 * 86_400_000),
           addr)
    meta = info.meta()
    _write("meta.json", meta, addr)
    _write("abstraction_state.json",
           {"state": info.query_user_abstraction_state(addr)}, addr)

    (OUT / "PROVENANCE.md").write_text(
        "# Fixture provenance\n\n"
        "Recorded by `tests/record_fixtures.py` (read-only `/info` POSTs) from "
        f"{'TESTNET' if testnet else 'MAINNET'} on "
        f"{time.strftime('%Y-%m-%d', time.gmtime())}.\n"
        "Account address replaced with a placeholder; `cloid` dropped. Sizes, "
        "prices, statuses, field names and nesting are verbatim.\n\n"
        "| file | source |\n|---|---|\n"
        "| `order_status_by_inner_status.json` | `historicalOrders`, one record "
        "per distinct inner status |\n"
        "| `order_status_envelopes.json` | `orderStatus` by oid, one per inner "
        "status |\n"
        "| `clearinghouse_state_flat.json` | `clearinghouseState` (account was "
        "flat) |\n"
        "| `clearinghouse_state_with_position.json` | **Official documented "
        "example**, not a recording -- this account held no position at record "
        "time. Source: Hyperliquid docs, info endpoint / perpetuals. |\n"
        "| `spot_user_state.json` | `spotClearinghouseState` |\n"
        "| `frontend_open_orders.json` | `frontendOpenOrders` |\n"
        "| `user_fills_30d.json` | `userFillsByTime`, last 30 days |\n"
        "| `meta.json` | `meta` (perp universe) |\n"
        "| `abstraction_state.json` | `userAbstraction` |\n")
    print(f"  wrote PROVENANCE.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Cuan Lab -- the one command the agent (and the operator) drives.

    lab status                     where everything stands
    lab refresh [--quiet-ok]       fetch new real bars into the archive, sync live records
    lab new <slug>                 scaffold an experiment from the template
    lab register <ID>              validate + test + commit the design (before any real data)
    lab null <ID>                  run it on random walks: is the test itself trustworthy?
    lab run <ID>                   development data only (everything before the seal)
    lab holdout <ID>               the sealed recent data -- once, and only if dev passed
    lab propose <ID>               write the proposal for the operator (holdout passed)
    lab decide <ID> accepted|declined [--note ...]   record the OPERATOR's decision
    lab ledger                     every experiment and its verdict
    lab why                        why the live strategy is losing (read-only, guarded)
    lab selftest [--quick]         the lab's own tests; run before trusting it
    lab verify                     ledger chain + shipped files untouched

Run from the lab directory (the folder this file is in). Nothing here can
place an order, hold a key, or write to the live system.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))
os.chdir(LAB)

from labcore import evaluate as EV            # noqa: E402
from labcore import ledger, paths, runner, seal  # noqa: E402


def _utc(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M")


def cmd_status(_a) -> int:
    man = json.loads(paths.ARCHIVE_MANIFEST.read_text()) if paths.ARCHIVE_MANIFEST.exists() else {}
    coins = man.get("coins", {})
    print("== data")
    if coins:
        newest = max(c["last_t"] for c in coins.values())
        oldest = min(c["first_t"] for c in coins.values())
        # From the newest bar's CLOSE (open + 1h): that is when it became data.
        age_h = (time.time() * 1000 - (newest + 3_600_000)) / 3_600_000
        print(f"  {len(coins)} coins, {sum(c['bars'] for c in coins.values()):,} closed 1h bars, "
              f"{_utc(oldest)} -> {_utc(newest)} UTC ({age_h:.1f}h old)")
        if age_h > 30:
            print("  !! archive is more than a day stale -- run `lab refresh`")
        s = seal.load()
        if s:
            print(f"  seal {s['seal_id']} at {s['seal_utc']}: holdout opened "
                  f"{ledger.holdout_spends(s['seal_id'])} time(s)")
    else:
        print("  archive empty -- run `lab refresh`")
    sm = paths.LAB / "research" / "levels" / "data" / "_sync_manifest.json"
    print(f"  live records: " + (f"synced {json.loads(sm.read_text())['at_utc']}"
                                 if sm.exists() else "not synced (CUAN_LIVE_DATA unset?)"))
    print("== ledger")
    bad = ledger.verify()
    print(f"  chain: {'INTACT' if bad is None else 'BROKEN -- ' + bad}")
    exps = ledger.experiments()
    states: dict = {}
    for e in exps.values():
        states[_state(e)] = states.get(_state(e), 0) + 1
    print(f"  {len(exps)} experiments: " + ", ".join(f"{k} {v}" for k, v in sorted(states.items())))
    print(f"  hypotheses tested on real data to date: {ledger.tested_count()}")
    waiting = sorted(p.name for p in paths.PROPOSALS.glob("E*.md"))
    print(f"  proposals awaiting the operator: {', '.join(waiting) or 'none'}")
    q = (paths.LAB / "ledger" / "queue.md")
    if q.exists():
        items = [l for l in q.read_text().splitlines() if l.startswith(("1.", "2.", "3."))]
        print("== queue (top 3)")
        for l in items[:3]:
            print("  " + l[:150])
    return 0


def _state(e) -> str:
    for k in ("operator_decision", "proposed", "holdout_result", "dev_result",
              "null_calibrated", "registered", "historical", "retired"):
        if k in e:
            return k
    return "note"


def cmd_refresh(a) -> int:
    from labcore import archive, sync
    from research.levels.universe import select_universe
    from research.levels.venue_client import HyperliquidReadOnlyClient
    client = HyperliquidReadOnlyClient()
    try:
        coins = [str(r["symbol"]).upper() for r in select_universe(client)]
    except Exception as e:                                      # noqa: BLE001
        print(f"refresh FAILED: universe unavailable ({str(e)[:120]})")
        return 1
    known = [p.stem for p in paths.ARCHIVE.glob("*.json")]
    coins = sorted(set(coins) | set(known))      # keep archiving coins that left the top 45
    rep = archive.update(coins, client, log=lambda *_: None)
    # 15m too: the 15m test (H-LTF15M) was limited to 52 days, the venue's
    # ceiling, and could not rule out a material effect. Archived daily, the
    # same pre-registered test has three times the trades in six months.
    rep15 = archive.update(coins, client, log=lambda *_: None, interval="15m")
    rep["failed"].update({f"{c} (15m)": w for c, w in rep15["failed"].items()})
    syn = sync.sync_live()
    man = json.loads(paths.ARCHIVE_MANIFEST.read_text())
    newest = max(c["last_t"] for c in man["coins"].values()) + 3_600_000
    rolled = seal.roll_if_due(newest)
    added = sum(c["added"] for c in rep["coins"].values())
    revised = sum(c["revised"] for c in rep["coins"].values())
    notable = rep["failed"] or revised or rolled
    if a.quiet_ok and not notable:
        print("NO_REPLY")
        return 0
    print(f"refresh: {len(rep['coins'])} coins, +{added} bars, {revised} revised, "
          f"{len(rep['failed'])} failed; live sync {'ok' if syn.get('synced') else syn.get('reason')}"
          + (f"; seal rolled to {rolled}" if rolled else ""))
    for c, why in list(rep["failed"].items())[:8]:
        print(f"  failed {c}: {why}")
    if revised:
        print(f"  !! {revised} archived bar(s) changed at the venue -- see data/archive/revisions.jsonl")
    return 1 if rep["failed"] and not rep["coins"] else 0


def cmd_new(a) -> int:
    exp_id, d = runner.new(a.slug)
    print(f"created {exp_id} at {d.relative_to(LAB)}")
    print("next: write PREREG.md and experiment.py, then `lab register " + exp_id + "`")
    return 0


def cmd_register(a) -> int:
    runner.register(a.id)
    return 0


def cmd_null(a) -> int:
    cal = runner.null(a.id, datasets=a.datasets)
    for k, v in cal["framings"].items():
        print(f"  {k}: fired {v['fired']}/{cal['datasets']}, mean {v['mean']:+.4f} +/- {v['se']:.4f} "
              f"-> {'MISCALIBRATED' if v['miscalibrated'] else 'ok'}")
    return 0


def cmd_run(a) -> int:
    res = runner.run(a.id)
    print(EV.render(res, f"{a.id.upper()} -- development"))
    return 0


def cmd_holdout(a) -> int:
    res = runner.holdout(a.id)
    print(EV.render(res, f"{a.id.upper()} -- holdout"))
    return 0


def cmd_propose(a) -> int:
    p = runner.propose(a.id)
    print(f"wrote {p.relative_to(LAB)} -- fill in the change and risks, then tell the operator")
    return 0


def cmd_decide(a) -> int:
    """Record what the operator decided about a proposal. Only on their word:
    the agent must never record a decision the operator did not state."""
    exp_id = a.id.upper()
    if "proposed" not in ledger.experiments().get(exp_id, {}):
        raise SystemExit(f"{exp_id} has no proposal to decide on")
    ledger.append("operator_decision", exp_id, {"decision": a.decision, "note": a.note})
    print(f"recorded: {exp_id} {a.decision}")
    return 0


def cmd_ledger(_a) -> int:
    for k, e in sorted(ledger.experiments().items()):
        title = (e.get("registered") or e.get("historical") or {}).get("title", "")
        verdict = ""
        for stage in ("holdout_result", "dev_result"):
            if stage in e:
                ds = e[stage]["decisions"]
                verdict = f"{stage.split('_')[0]}: " + ", ".join(
                    f"{n} {'EDGE' if d['adds_edge'] else 'no'}" for n, d in ds.items())
                break
        if "historical" in e:
            verdict = e["historical"].get("verdict", "")
        print(f"{k:6} {_state(e):16} {title[:60]:60} {verdict}")
    return 0


def cmd_why(_a) -> int:
    return subprocess.call([sys.executable, "-m", "tools.why"], cwd=LAB)


def cmd_selftest(a) -> int:
    args = [sys.executable, "-m", "pytest", "-q", "tests"]
    if a.quick:
        args += ["-x"]
    rc = subprocess.call(args, cwd=LAB)
    bad = ledger.verify()
    print(f"ledger chain: {'intact' if bad is None else 'BROKEN -- ' + bad}")
    return rc or (1 if bad else 0)


def cmd_verify(_a) -> int:
    rc = 0
    bad = ledger.verify()
    print(f"ledger chain: {'intact' if bad is None else 'BROKEN -- ' + bad}")
    rc |= bad is not None
    man = LAB.parent / "MANIFEST.json"
    if man.exists():
        import hashlib
        m = json.loads(man.read_text())
        changed = [f for f, h in m["files"].items()
                   if (LAB.parent / f).exists()
                   and hashlib.sha256((LAB.parent / f).read_bytes()).hexdigest() != h]
        missing = [f for f in m["files"] if not (LAB.parent / f).exists()]
        print(f"shipped files: {len(m['files'])}, changed since build: {len(changed)}, "
              f"missing: {len(missing)}")
        for f in (changed + missing)[:20]:
            print(f"  {f}")
    return int(rc)


def main() -> int:
    ap = argparse.ArgumentParser(prog="lab", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    p = sub.add_parser("refresh")
    p.add_argument("--quiet-ok", action="store_true",
                   help="print NO_REPLY when nothing needs attention (for automations)")
    p.set_defaults(fn=cmd_refresh)
    p = sub.add_parser("new")
    p.add_argument("slug")
    p.set_defaults(fn=cmd_new)
    for name, fn in (("register", cmd_register), ("run", cmd_run),
                     ("holdout", cmd_holdout), ("propose", cmd_propose)):
        p = sub.add_parser(name)
        p.add_argument("id")
        p.set_defaults(fn=fn)
    p = sub.add_parser("null")
    p.add_argument("id")
    p.add_argument("--datasets", type=int, default=None,
                   help="null datasets (default: labcore.nullcal.DATASETS, 20)")
    p.set_defaults(fn=cmd_null)
    p = sub.add_parser("decide")
    p.add_argument("id")
    p.add_argument("decision", choices=("accepted", "declined"))
    p.add_argument("--note", default="")
    p.set_defaults(fn=cmd_decide)
    sub.add_parser("ledger").set_defaults(fn=cmd_ledger)
    sub.add_parser("why").set_defaults(fn=cmd_why)
    p = sub.add_parser("selftest")
    p.add_argument("--quick", action="store_true")
    p.set_defaults(fn=cmd_selftest)
    sub.add_parser("verify").set_defaults(fn=cmd_verify)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())

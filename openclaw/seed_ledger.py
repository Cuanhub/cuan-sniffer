#!/usr/bin/env python3
"""Seed the lab's ledger with what this project tested before the lab existed.

    python openclaw/seed_ledger.py        # writes openclaw/workspace/lab/ledger/ledger.jsonl

The lab reports "hypotheses tested to date" as the denominator for anything
it calls a discovery. Starting that count at zero would pretend the dozens of
ideas tested by hand in October 2026 never happened -- the exact forgetting
the ledger exists to prevent. Every entry below is a finding from this repo's
history, with its numbers, and `hypotheses` counts how many separate tests it
represents.

Refuses to overwrite an existing ledger with experiments in it: once the lab
has run, its ledger belongs to the lab.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAB = HERE / "workspace" / "lab"
sys.path.insert(0, str(LAB))

from labcore import ledger, paths      # noqa: E402

HISTORY = [
    ("H-STOPATR", "Stop distance in ATR multiples", 1, "ESTABLISHED",
     "t = +9.75; win 18.6% below 1.32x ATR vs 60.8% above; 10/10 coins agree; survives "
     "holding the target fixed. Basis of the live 1.5x ATR stop."),
    ("H-BASIS", "Perp above oracle (basis) as a veto, plus 19 price features", 20,
     "ESTABLISHED (veto only)",
     "-0.524R vs +0.044R at or below, p = 0.0035, 248 resolved episodes; the 19 price "
     "features tested alongside it all failed."),
    ("H-KILLZONE", "Killzone (session) timing", 1, "DEAD", "reversed under control"),
    ("H-HTF", "HTF bias proxy", 1, "DEAD", "vanished under control"),
    ("H-SWEEP", "Liquidity sweep confirmation", 1, "DEAD", "+0.080R at n = 15: noise"),
    ("H-PULLRUN", "pull_over_run", 1, "DEAD", "inverts under control"),
    ("H-PERCOIN", "Per-coin edge", 1, "DEAD", "did not replicate across halves"),
    ("H-GATE", "The old level gate", 1, "DEAD",
     "-0.0413R vs random entry, z = -0.93, n = 688 backfill trades"),
    ("H-TIGHTSTOP", "Tight fixed 0.25% stops", 1, "DEAD",
     "7 consecutive live stop-outs (-$20.31), all sub-1.5% stops"),
    ("H-LABELS", "Operator take/skip verdicts vs outcome", 1, "NOT ESTABLISHED",
     "takes -0.281R (n 23), skips +0.880R (n 10); split by month the gap reverses; "
     "t = 1.33. Skips had tighter stop/ATR (1.28x vs 1.59x)."),
    ("H-BANDCFG", "Band stop/rr configurations (scratch harness, 9 configs)", 9,
     "SUPERSEDED",
     "old -43.7R/786 vs 1.5x ATR +25.2R/173 -- none significant, and that harness "
     "resolved market entries on their own decision bar and selected plans[0]; "
     "re-measure on research/backtest (queue item 1)."),
]


def main() -> int:
    if paths.LEDGER.exists() and any(e["event"] != "historical"
                                     for e in ledger.entries()):
        raise SystemExit(f"{paths.LEDGER} already holds lab experiments; refusing to reseed")
    if paths.LEDGER.exists():
        paths.LEDGER.unlink()
    for hid, title, n, verdict, evidence in HISTORY:
        ledger.append("historical", hid, {"title": title, "hypotheses": n, "verdict": verdict,
                                          "evidence": evidence, "source": "CuanSniffer repo, "
                                          "Claude Code sessions, 2026-09/10"})
    tl = HERE.parent / "research" / "backtest" / "results" / "trendline_breakout" / "summary.json"
    if tl.exists():
        s = json.loads(tl.read_text())
        dec = s["decision"]
        verdict = ("ADDS EDGE: " + ", ".join(k for k, v in dec.items() if v["adds_edge"])
                   if any(v["adds_edge"] for v in dec.values()) else
                   "NO EDGE: no framing passed the pre-registered rule (H1 wrong sign, "
                   "H3 dead; H2 +0.035R not established, queued for a forward re-test)")
        ledger.append("historical", "H-TRENDLINE", {
            "title": "Trendline breakout: filter / confirmation entry / standalone",
            "hypotheses": 3, "verdict": verdict,
            "evidence": {k: {"estimate": s[k]["cluster"]["estimate"],
                             "ci": [s[k]["cluster"]["ci_lo"], s[k]["cluster"]["ci_hi"]],
                             "p": s[k]["cluster"]["p"]} for k in ("H1", "H2", "H3")},
            "prereg": "research/backtest/PREREG_trendline_breakout.md",
            "report": "research/backtest/results/trendline_breakout/report.md",
            "source": "CuanSniffer repo, pre-registered 2026-10-27"})
    res = HERE.parent / "research" / "backtest" / "results"
    later = [
        ("H-RRFLOOR", "Reward:risk floor -- do refused bands carry edge?", "rr_floor",
         "has_edge", "NO EDGE in any bucket (live rr>=2 +0.055 p 0.60; rr 1-2 -0.060; rr 0.3-1 "
         "-0.024). Every floor loses money; lowering it loses more.",
         "research/backtest/PREREG_rr_floor.md"),
        ("H-LTF15M", "15m break-of-structure confirmation (band stop / 15m stop)", "ltf_15m",
         "adds_edge", "NO EDGE (B1 -0.067 p 0.53; B2 -0.006 p 0.95) on 52 days -- cannot rule "
         "out a material effect; re-test when the 15m archive has ~6 months. As measurement: "
         "0 of 281 outcomes change at 15m.", "research/backtest/PREREG_ltf_15m.md"),
    ]
    for hid, title, folder, flag, verdict, prereg in later:
        f = res / folder / "summary.json"
        if not f.exists():
            continue
        s = json.loads(f.read_text())
        fr = s["framings"]
        ledger.append("historical", hid, {
            "title": title, "hypotheses": len(fr), "verdict": verdict,
            "evidence": {k: {"estimate": v.get("estimate"), "ci": [v.get("ci_lo"), v.get("ci_hi")],
                             "p": v.get("p"), "n": v.get("n")} for k, v in fr.items()},
            "edge": {k: s["decision"][k][flag] for k in fr},
            "prereg": prereg, "report": f"research/backtest/results/{folder}/report.md",
            "source": "CuanSniffer repo, pre-registered 2026-10-27"})
    print(f"seeded {len(ledger.entries())} entries; hypotheses to date: {ledger.tested_count()}; "
          f"chain {'intact' if ledger.verify() is None else 'BROKEN'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

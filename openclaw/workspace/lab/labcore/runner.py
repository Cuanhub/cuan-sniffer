"""Running an experiment, in the only order the lab allows.

    new -> register -> null -> run (development) -> holdout (once) -> propose

Each step refuses unless the previous one is on the ledger with the SAME
design hashes. The agent can write any experiment it likes; it cannot run
one on real data before its design is committed, cannot change the design
afterwards without starting a new experiment, and cannot open the holdout
twice.
"""

from __future__ import annotations

import importlib
import json
import random
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from labcore import evaluate as EV
from labcore import ledger, nullcal, paths, pool, prereg, seal

TEMPLATE = paths.EXPERIMENTS / "TEMPLATE"


@dataclass
class Ctx:
    """What an experiment's records() receives."""
    bars_by: Dict[str, List[Dict[str, Any]]]
    executor: Any
    _baseline: Optional[Dict[str, list]] = field(default=None, repr=False)

    def coins(self) -> List[str]:
        return sorted(self.bars_by)

    def map(self, fn: Callable, args: List[Any], chunksize: int = 1) -> List[Any]:
        return list(self.executor.map(fn, args, chunksize=chunksize))

    def baseline(self) -> Dict[str, list]:
        """The live band strategy's trades per coin, computed once per run."""
        if self._baseline is None:
            coins = self.coins()
            self._baseline = dict(zip(coins, self.map(pool.replay_band, coins)))
        return self._baseline


# ------------------------------------------------------------- locating --
def exp_dir(exp_id: str) -> Path:
    exp_id = exp_id.upper()
    hits = sorted(paths.EXPERIMENTS.glob(f"{exp_id.lower()}_*"))
    if len(hits) != 1:
        raise SystemExit(f"no unique experiment directory for {exp_id} under {paths.EXPERIMENTS}")
    return hits[0]


def load(exp_id: str):
    d = exp_dir(exp_id)
    if str(paths.LAB) not in sys.path:
        sys.path.insert(0, str(paths.LAB))
    mod = importlib.import_module(f"experiments.{d.name}.experiment")
    for attr in ("TITLE", "FRAMINGS", "FEATURES", "records"):
        if not hasattr(mod, attr):
            raise SystemExit(f"{d.name}/experiment.py must define {attr}")
    EV.check_framings(mod.FRAMINGS)
    return mod


def hashes(exp_id: str) -> Dict[str, str]:
    d = exp_dir(exp_id)
    return {"prereg_sha": prereg.sha(d / "PREREG.md"),
            "code_sha": prereg.sha(d / "experiment.py"),
            "harness_sha": prereg.harness_digest()}


# ------------------------------------------------------------- creating --
def new(slug: str) -> Tuple[str, Path]:
    slug = re.sub(r"[^a-z0-9_]+", "_", slug.lower()).strip("_")[:40]
    if not slug:
        raise SystemExit("slug must contain letters or digits")
    exp_id = ledger.next_id()
    taken = {p.name[:4].upper() for p in paths.EXPERIMENTS.glob("e[0-9][0-9][0-9]_*")}
    while exp_id in taken:
        exp_id = f"E{int(exp_id[1:]) + 1:03d}"
    d = paths.EXPERIMENTS / f"{exp_id.lower()}_{slug}"
    d.mkdir(parents=True)
    (d / "__init__.py").write_text("")
    for name in ("PREREG.md", "experiment.py", "test_experiment.py"):
        text = (TEMPLATE / name).read_text().replace("{{ID}}", exp_id).replace("{{SLUG}}", slug)
        (d / name).write_text(text)
    return exp_id, d


# ------------------------------------------------------------ causality --
def causality_problems(mod, bars_by: Dict[str, List[Dict[str, Any]]],
                       samples: int = 40) -> List[str]:
    """Every declared feature must give the same value at t whether or not
    the bars after t exist. Checked on REAL recorded bars."""
    out = []
    rng = random.Random(7)
    for name, fn in mod.FEATURES.items():
        for coin in sorted(bars_by)[:4]:
            bars = bars_by[coin]
            full = fn(bars)
            if len(full) != len(bars):
                out.append(f"{name}: returned {len(full)} values for {len(bars)} bars")
                break
            for t in sorted(rng.sample(range(50, len(bars)), min(samples, len(bars) - 50))):
                cut = fn(bars[:t + 1])
                if cut[t] != full[t]:
                    out.append(f"{name}: value at bar {t} of {coin} changes when later "
                               f"bars are removed ({cut[t]!r} vs {full[t]!r}) -- lookahead")
                    break
    return out


def fixture_bars() -> Dict[str, List[Dict[str, Any]]]:
    raw = json.loads((paths.LAB / "tests" / "fixtures" / "zones" / "bars_1h.json").read_text())
    return {c: [dict(b, t=1_700_006_400_000 + k * 3_600_000) for k, b in enumerate(bs)]
            for c, bs in raw.items()}


# -------------------------------------------------------------- running --
def records_on(mod, bars_by: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    ex = pool.make(bars_by)
    try:
        return mod.records(Ctx(bars_by, ex))
    finally:
        ex.shutdown()


def register(exp_id: str, log=print) -> Dict[str, Any]:
    exp_id = exp_id.upper()
    d = exp_dir(exp_id)
    mod = load(exp_id)
    problems = prereg.validate((d / "PREREG.md").read_text(), mod.FRAMINGS)
    if not mod.FEATURES and "no per-bar feature" not in (d / "PREREG.md").read_text().lower():
        problems.append("FEATURES is empty: declare the per-bar features, or state "
                        "'no per-bar feature' in the PREREG and say why")
    problems += causality_problems(mod, fixture_bars())
    t = subprocess.run([sys.executable, "-m", "pytest", "-q", str(d)], cwd=paths.LAB,
                       capture_output=True, text=True)
    if t.returncode != 0:
        problems.append("the experiment's own tests fail:\n" + t.stdout[-1500:])
    if problems:
        raise SystemExit("NOT REGISTERED:\n  - " + "\n  - ".join(problems))
    h = hashes(exp_id)
    commit = prereg.git_commit([d], f"{exp_id}: register {mod.TITLE}")
    row = ledger.append("registered", exp_id, {
        "title": mod.TITLE, "dir": d.name, "framings": mod.FRAMINGS,
        "features": sorted(mod.FEATURES), "commit": commit, **h})
    log(f"[REGISTERED] {exp_id} {mod.TITLE}  commit {commit or '(no git)'}")
    return row


def _require(exp_id: str, step: str) -> Dict[str, Any]:
    e = ledger.experiments().get(exp_id)
    if not e or "registered" not in e:
        raise SystemExit(f"{exp_id} is not registered -- run `lab register {exp_id}` first")
    now = hashes(exp_id)
    reg = e["registered"]
    for k in ("prereg_sha", "code_sha"):
        if reg[k] != now[k]:
            raise SystemExit(f"{exp_id}: {k[:-4]} changed since registration. An edited design "
                             f"is a new experiment -- `lab new` it and register that.")
    if reg["harness_sha"] != now["harness_sha"]:
        raise SystemExit(f"{exp_id}: the lab's own code changed since registration, so the "
                         f"registered design no longer means what it meant. Re-register it "
                         f"as a new experiment after `lab selftest` passes.")
    return e


def null(exp_id: str, datasets: Optional[int] = None, log=print) -> Dict[str, Any]:
    exp_id = exp_id.upper()
    _require(exp_id, "null")
    mod = load(exp_id)

    def once(bars_by):
        return EV.evaluate(records_on(mod, bars_by), mod.FRAMINGS, B=2000)
    cal = nullcal.calibrate(once, datasets=datasets, log=log)
    ledger.append("null_calibrated", exp_id, {**cal, **hashes(exp_id)})
    return cal


def _snapshot(until_ms: Optional[int] = None):
    from labcore import archive
    return archive.snapshot(until_ms=until_ms)


def _write(exp_id: str, kind: str, res: Dict[str, Any], meta: Dict[str, Any]) -> Path:
    out = paths.RESULTS / exp_id
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{kind}.json").write_text(json.dumps({**res, **meta}, indent=1, default=str))
    (out / f"{kind}.md").write_text(EV.render(res, f"{exp_id} -- {kind}"))
    return out


def run(exp_id: str, log=print) -> Dict[str, Any]:
    exp_id = exp_id.upper()
    e = _require(exp_id, "run")
    if "dev_result" in e:
        raise SystemExit(f"{exp_id} already has a development result. Results are final; "
                         f"a different design is a new experiment.")
    if "null_calibrated" not in e or e["null_calibrated"].get("code_sha") != e["registered"]["code_sha"]:
        raise SystemExit(f"{exp_id}: run `lab null {exp_id}` first")
    mod = load(exp_id)
    full = _snapshot()
    newest = max(b[-1]["t"] for b in full.bars.values()) + 3_600_000
    s = seal.current(newest)
    dev = _snapshot(until_ms=s["seal_ms"])
    log(f"[DEV] {len(dev.bars)} coins, bars before the seal {s['seal_utc']} ({s['seal_id']})")
    rows = records_on(mod, dev.bars)
    res = EV.evaluate(rows, mod.FRAMINGS, miscalibrated=e["null_calibrated"]["miscalibrated"])
    meta = {"seal": s, "coins": {c: m["sha256"] for c, m in dev.manifest["coins"].items()},
            "rows": len(rows), **hashes(exp_id)}
    out = _write(exp_id, "dev", res, meta)
    commit = prereg.git_commit([out], f"{exp_id}: development result")
    ledger.append("dev_result", exp_id, {"framings": res["framings"],
                                         "decisions": res["decisions"], "seal_id": s["seal_id"],
                                         "rows": len(rows), "commit": commit})
    return res


def holdout(exp_id: str, log=print) -> Dict[str, Any]:
    exp_id = exp_id.upper()
    e = _require(exp_id, "holdout")
    if "holdout_result" in e:
        raise SystemExit(f"{exp_id}: the holdout was already opened once. It cannot be reopened.")
    dev = e.get("dev_result")
    if not dev:
        raise SystemExit(f"{exp_id}: no development result yet")
    passed = [k for k, d in dev["decisions"].items() if d["adds_edge"]]
    if not passed:
        raise SystemExit(f"{exp_id}: nothing passed in development, so there is nothing to "
                         f"confirm -- and opening the holdout would only spend it")
    mod = load(exp_id)
    full = _snapshot()
    newest = max(b[-1]["t"] for b in full.bars.values()) + 3_600_000
    s = seal.current(newest)
    if dev["seal_id"] != s["seal_id"]:
        raise SystemExit(f"{exp_id}: developed under seal {dev['seal_id']}, but the seal has "
                         f"rolled to {s['seal_id']} -- its old holdout is now development data. "
                         f"Re-run it as a new experiment under the current seal.")
    k = ledger.holdout_spends(s["seal_id"]) + 1
    alpha = EV.ALPHA / k
    framings = {n: v for n, v in mod.FRAMINGS.items()
                if n in passed or (v["role"] == "must_agree" and v.get("with") in passed)}
    log(f"[HOLDOUT] {exp_id}: opening {s['seal_id']} (opening #{k}, alpha {alpha:.4f}) "
        f"for {', '.join(passed)}")
    rows = records_on(mod, full.bars)
    res = EV.evaluate(rows, framings, window=(s["seal_ms"], newest + 1), alpha=alpha,
                      miscalibrated=e["null_calibrated"]["miscalibrated"])
    out = _write(exp_id, "holdout", res, {"seal": s, "opening": k, **hashes(exp_id)})
    commit = prereg.git_commit([out], f"{exp_id}: holdout result")
    ledger.append("holdout_result", exp_id, {"framings": res["framings"],
                                             "decisions": res["decisions"], "seal_id": s["seal_id"],
                                             "opening": k, "alpha": alpha, "commit": commit})
    return res


def propose(exp_id: str) -> Path:
    exp_id = exp_id.upper()
    e = ledger.experiments().get(exp_id, {})
    ho = e.get("holdout_result")
    if not ho:
        raise SystemExit(f"{exp_id}: no holdout result -- nothing is proposed on development alone")
    winners = [k for k, d in ho["decisions"].items() if d["adds_edge"] and d["material"]]
    if not winners:
        raise SystemExit(f"{exp_id}: nothing passed the holdout with a material effect "
                         f"(>= {EV.MIN_EFFECT}R)")
    paths.PROPOSALS.mkdir(parents=True, exist_ok=True)
    p = paths.PROPOSALS / f"{exp_id}.md"
    dev = e["dev_result"]
    lines = [f"# Proposal {exp_id}: {e['registered']['title']}", "",
             "Status: **awaiting the operator**. Nothing has been changed. The lab cannot "
             "change the live system; this is a request for a human to.", "",
             "## Evidence (filled by the lab, not editable)", ""]
    for k in winners:
        dv, hv = dev["framings"][k], ho["framings"][k]
        lines.append(f"- `{k}`: development {dv['estimate']:+.3f}R "
                     f"[{dv['ci_lo']:+.3f}, {dv['ci_hi']:+.3f}] p={dv['p']:.4f}; "
                     f"holdout {hv['estimate']:+.3f}R [{hv['ci_lo']:+.3f}, {hv['ci_hi']:+.3f}] "
                     f"p={hv['p']:.4f} at alpha {ho['alpha']:.4f}")
    lines += ["", f"Hypotheses tested by this lab to date (the denominator): "
              f"{ledger.tested_count()}", "",
              f"Registered design: commit {e['registered'].get('commit')}, PREREG sha "
              f"{e['registered']['prereg_sha'][:12]}", "",
              "## The change (written by the agent)", "", "<the exact change: file, function, "
              "constant, or a unified diff against the live repo>", "",
              "## Risks and what to watch after shipping", "", "<...>", ""]
    p.write_text("\n".join(lines))
    ledger.append("proposed", exp_id, {"file": str(p.relative_to(paths.LAB)),
                                       "framings": winners})
    return p

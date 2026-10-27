"""The lab's machinery: everything the agent relies on to keep it honest.

Synthetic bars here test MACHINERY only; no claim about the market comes
from them.
"""

import ast
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))

from labcore import (archive, controls, evaluate as EV, ledger, nullcal,  # noqa: E402
                     paths, prereg, runner, seal, sync)


# ------------------------------------------------------------- ledger --
class TestLedger:
    def test_chain_is_intact_after_appends(self):
        for i in range(3):
            ledger.append("note", f"E00{i}", {"i": i})
        assert ledger.verify() is None

    def test_editing_a_line_breaks_the_chain(self):
        ledger.append("registered", "E001", {"title": "a"})
        ledger.append("dev_result", "E001", {"decisions": {}})
        ledger.append("note", "E002", {})
        lines = paths.LEDGER.read_text().splitlines()
        lines[1] = lines[1].replace('"dev_result"', '"note"')
        paths.LEDGER.write_text("\n".join(lines) + "\n")
        assert "line 3" in ledger.verify()

    def test_deleting_a_failure_breaks_the_chain(self):
        ledger.append("dev_result", "E001", {"decisions": {"x": {"adds_edge": False}}})
        ledger.append("dev_result", "E002", {"decisions": {"x": {"adds_edge": True}}})
        lines = paths.LEDGER.read_text().splitlines()
        paths.LEDGER.write_text(lines[1] + "\n")
        assert ledger.verify() is not None

    def test_unknown_events_are_refused(self):
        with pytest.raises(ValueError):
            ledger.append("deleted", "E001", {})

    def test_ids_and_denominator(self):
        ledger.append("historical", "H001", {"title": "x", "hypotheses": 4})
        ledger.append("registered", "E001", {"title": "y"})
        ledger.append("dev_result", "E001", {"framings": {"a": {}, "b": {}}, "decisions": {}})
        assert ledger.next_id() == "E002"
        assert ledger.tested_count() == 6


# ---------------------------------------------------------- evaluate --
def rows_diff(effect, n=400, seed=0, weeks=40, halves_flip=False):
    rng = np.random.default_rng(seed)
    out = []
    t0 = 1_760_000_000_000
    for i in range(n):
        g = i % 2
        t = t0 + int(i / n * weeks * 7 * 86_400_000)
        eff = effect * (-1 if halves_flip and i < n // 2 else 1)
        out.append({"framing": "f", "t": t, "coin": "X",
                    "value": float(rng.normal(0, 1) + (eff if g else 0)), "group": g})
    return out


F = {"f": {"role": "primary", "stat": "diff"}}


class TestEvaluate:
    def test_a_clear_effect_passes_every_criterion(self):
        r = EV.evaluate(rows_diff(0.6), F, B=2000)
        d = r["decisions"]["f"]
        assert d["adds_edge"], d["criteria"]
        assert d["material"]

    def test_no_effect_does_not_pass(self):
        r = EV.evaluate(rows_diff(0.0, seed=3), F, B=2000)
        assert not r["decisions"]["f"]["adds_edge"]

    def test_an_effect_that_flips_between_halves_fails(self):
        r = EV.evaluate(rows_diff(0.9, halves_flip=True), F, B=2000)
        assert not r["decisions"]["f"]["criteria"]["both_halves_positive"]
        assert not r["decisions"]["f"]["adds_edge"]

    def test_a_negative_must_agree_framing_vetoes(self):
        fr = {"f": {"role": "primary", "stat": "diff"},
              "g": {"role": "must_agree", "with": "f", "stat": "diff"}}
        rows = rows_diff(0.6) + [dict(r, framing="g", value=-r["value"]) for r in rows_diff(0.6)]
        d = EV.evaluate(rows, fr, B=1000)["decisions"]["f"]
        assert not d["criteria"]["must_agree_positive"]
        assert not d["adds_edge"]

    def test_too_few_units_fail(self):
        d = EV.evaluate(rows_diff(2.0, n=40), F, B=1000)["decisions"]["f"]
        assert not d["criteria"]["enough_units"]

    def test_miscalibrated_vetoes(self):
        d = EV.evaluate(rows_diff(0.6), F, B=1000, miscalibrated={"f": True})["decisions"]["f"]
        assert not d["adds_edge"]

    def test_window_selects_by_time(self):
        rows = rows_diff(0.6)
        mid = sorted(r["t"] for r in rows)[len(rows) // 2]
        r = EV.evaluate(rows, F, window=(mid, 10 ** 15), B=500)
        assert r["framings"]["f"]["n"] == len([x for x in rows if x["t"] >= mid])

    def test_malformed_rows_are_refused(self):
        bad = [{"framing": "f", "t": 1_760_000_000_000, "coin": "X", "value": float("nan"), "group": 1}]
        with pytest.raises(ValueError):
            EV.evaluate(bad, F)
        with pytest.raises(ValueError):
            EV.evaluate([{"framing": "f", "t": 5, "coin": "X", "value": 1.0, "group": 1}], F)
        with pytest.raises(ValueError):
            EV.evaluate([{"framing": "zz", "t": 1_760_000_000_000, "coin": "X", "value": 1.0}], F)

    def test_framings_need_a_primary_and_valid_links(self):
        with pytest.raises(ValueError):
            EV.check_framings({"g": {"role": "must_agree", "with": "f", "stat": "diff"}})
        with pytest.raises(ValueError):
            EV.check_framings({"f": {"role": "primary", "stat": "median"}})

    def test_thresholds_are_read_at_call_time(self):
        """Not captured as default arguments at import."""
        old = EV.MIN_N
        try:
            EV.MIN_N = 10_000
            d = EV.evaluate(rows_diff(0.6), F, B=500)["decisions"]["f"]
            assert not d["criteria"]["enough_units"]
        finally:
            EV.MIN_N = old


# ----------------------------------------------------------- prereg --
GOOD = "\n".join(f"## {s}\n" + "Concrete content that is long enough to count. " * 2
                 for s in prereg.SECTIONS) + "\nframing f is described here.\n"


class TestPrereg:
    def test_a_complete_prereg_validates(self):
        text = GOOD.replace("## Framings\n", "## Framings\nThe framing f: primary, diff.\n")
        assert prereg.validate(text, {"f": {}}) == []

    def test_missing_sections_and_placeholders_are_caught(self):
        text = GOOD.replace("## Controls", "## Something else") + "\nTODO later\n"
        probs = prereg.validate(text, {})
        assert any("Controls" in p for p in probs)
        assert any("placeholder" in p for p in probs)

    def test_the_template_cannot_be_registered_as_is(self):
        text = (LAB / "experiments" / "TEMPLATE" / "PREREG.md").read_text()
        assert prereg.validate(text, {"x": {}})

    def test_undescribed_framings_are_caught(self):
        probs = prereg.validate(GOOD, {"mystery": {}})
        assert any("mystery" in p for p in probs)


# ------------------------------------------------------------- seal --
class TestSeal:
    def test_first_seal_sits_holdout_days_back(self):
        newest = 1_800_000_000_000
        s = seal.current(newest)
        assert s["seal_id"] == "S1"
        assert newest - s["seal_ms"] >= seal.HOLDOUT_DAYS * seal.DAY_MS
        assert newest - s["seal_ms"] < seal.HOLDOUT_DAYS * seal.DAY_MS + 3_600_000

    def test_rolls_only_after_a_second_holdout_window_accumulates(self):
        newest = 1_800_000_000_000
        s1 = seal.current(newest)
        assert seal.roll_if_due(newest + 40 * seal.DAY_MS) is None
        assert seal.roll_if_due(s1["seal_ms"] + 2 * seal.HOLDOUT_DAYS * seal.DAY_MS) == "S2"
        assert seal.load()["history"][0]["seal_id"] == "S1"


# ---------------------------------------------------------- archive --
def bar(t, c=10.0):
    return {"t": t, "o": c, "h": c + 1, "l": c - 1, "c": c, "v": 1.0, "n": 1}


class TestArchive:
    def test_merge_appends_and_never_silently_overwrites(self):
        H = 3_600_000
        old = [bar(1_700_006_400_000 + k * H) for k in range(5)]
        new = [bar(1_700_006_400_000 + k * H) for k in range(3, 8)]
        new[1] = dict(new[1], c=10.5, h=11.5)            # bar 4 changed at the venue
        merged, counts, revs = archive.merge(old, new, "X")
        assert counts == {"added": 3, "revised": 1, "bars": 8, "gaps": 0, "missing_bars": 0}
        assert revs[0]["old"]["c"] == 10.0 and revs[0]["new"]["c"] == 10.5

    def test_a_gap_is_counted_not_filled(self):
        H = 3_600_000
        old = [bar(1_700_006_400_000 + k * H) for k in range(3)]
        new = [bar(1_700_006_400_000 + k * H) for k in range(10, 12)]
        merged, counts, _ = archive.merge(old, new)
        assert counts["missing_bars"] == 7 and len(merged) == 5

    def test_snapshot_cut_never_includes_a_bar_after_the_seal(self):
        H = 3_600_000
        paths.ARCHIVE.mkdir(parents=True)
        bars = [bar(1_700_006_400_000 + k * H) for k in range(1200)]
        (paths.ARCHIVE / "X.json").write_text(json.dumps(bars))
        cut = bars[1000]["t"]
        snap = archive.snapshot(until_ms=cut)
        assert max(b["t"] + H for b in snap.bars["X"]) <= cut


# -------------------------------------------------------- controls --
class TestControls:
    def test_draws_are_strictly_after_the_signal(self):
        for i in (900, 3000, 4800):
            d = controls.draws_after(5000, i, coin="ETH")
            assert (d > i).all() and (d <= min(4998, i + 240)).all()

    def test_no_room_means_no_draws(self):
        assert len(controls.draws_after(1000, 998, coin="ETH")) == 0

    def test_unresolved_controls_are_dropped_not_zeroed(self):
        vals = iter([1.0, None, None] * 40)
        m = controls.control_mean(lambda u: next(vals), np.arange(120), min_resolved=10)
        assert m == 1.0

    def test_circular_shift_keeps_count(self):
        s = [i % 17 == 0 for i in range(200)]
        assert sum(controls.circular_shift(s, 33)) == sum(s)


# ------------------------------------------------------------- sync --
class TestSync:
    def test_copies_records_and_never_control_files(self, tmp_path):
        src = tmp_path / "live"
        src.mkdir()
        for name in ("zone_episodes.csv", "labels.csv", "_control_queue.json", "_halt.json",
                     "_live_state.json"):
            (src / name).write_text("x")
        dst = tmp_path / "replica"
        m = sync.sync_live(str(src), dst)
        assert set(m["files"]) == {"zone_episodes.csv", "labels.csv"}
        assert not (dst / "_control_queue.json").exists()
        assert (src / "_control_queue.json").read_text() == "x"

    def test_refuses_to_sync_a_directory_onto_itself(self, tmp_path):
        assert not sync.sync_live(str(tmp_path), tmp_path)["synced"]

    def test_nothing_in_the_lab_writes_to_the_live_source(self):
        """Static: LIVE_SOURCE is only ever passed to Path() and read."""
        for p in list((LAB / "labcore").glob("*.py")) + [LAB / "lab.py"]:
            for node in ast.walk(ast.parse(p.read_text())):
                if isinstance(node, ast.Call) and getattr(node.func, "attr", "") in (
                        "write_text", "write_bytes", "unlink", "rmtree", "mkdir"):
                    src = ast.unparse(node.func.value)
                    assert "LIVE_SOURCE" not in src and "src" != src, f"{p.name}: {ast.unparse(node)}"


# ----------------------------------------------------------- nullcal --
class TestNullData:
    def test_random_walks_read_module_values_at_call_time(self):
        old = nullcal.COINS
        try:
            nullcal.COINS = 3
            assert len(nullcal.random_walks(0, n=50)) == 3
        finally:
            nullcal.COINS = old

    def test_random_walks_are_a_martingale(self):
        bars = nullcal.random_walks(1, coins=4, n=3000)
        r = [b2["c"] / b1["c"] for bs in bars.values() for b1, b2 in zip(bs, bs[1:])]
        assert abs(np.mean(r) - 1) < 4 * np.std(r) / math.sqrt(len(r))


# --------------------------------------------------------- isolation --
ORDER_CAPABLE = ("core", "tools.live_levels_trader", "tools.telegram_control",
                 "tools.protect_loop", "hyperliquid.exchange", "eth_account", "notify")


class TestIsolation:
    def test_no_lab_code_imports_anything_order_capable(self):
        files = (list((LAB / "labcore").rglob("*.py")) + list((LAB / "experiments").rglob("*.py"))
                 + list((LAB / "research").rglob("*.py")) + list((LAB / "tools").rglob("*.py"))
                 + [LAB / "lab.py"])
        for p in files:
            for node in ast.walk(ast.parse(p.read_text())):
                names = ([a.name for a in node.names] if isinstance(node, ast.Import) else
                         [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
                for n in names:
                    assert not any(n == b or n.startswith(b + ".") for b in ORDER_CAPABLE), \
                        f"{p.relative_to(LAB)} imports {n}"

    def test_no_order_capable_module_is_even_present(self):
        assert not (LAB / "core").exists()
        for name in ("live_levels_trader.py", "telegram_control.py", "protect_loop.py"):
            assert not (LAB / "tools" / name).exists()

    def test_no_secrets_file_is_shipped(self):
        assert not list(LAB.rglob(".env")) and not list(LAB.parent.rglob(".env"))

    def test_a_fresh_interpreter_loads_nothing_order_capable(self):
        code = ("import sys, lab\n"
                "import labcore.runner, labcore.archive, labcore.sync\n"
                "bad = [m for m in sys.modules if m == 'core' or m.startswith(('core.', "
                "'hyperliquid.exchange', 'eth_account', 'tools.live', 'tools.telegram'))]\n"
                "print(','.join(sorted(bad)))\n")
        out = subprocess.run([sys.executable, "-c", code], cwd=LAB, capture_output=True,
                             text=True, timeout=120)
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip() == ""


# --------------------------------------------------------- lifecycle --
def _archive_from(bars_by):
    paths.ARCHIVE.mkdir(parents=True, exist_ok=True)
    for c, b in bars_by.items():
        (paths.ARCHIVE / f"{c}.json").write_text(json.dumps(b))


@pytest.fixture
def lab_with_template_experiment(monkeypatch, tmp_path):
    """A throwaway copy of the experiments dir holding E001 = the template,
    with a valid PREREG, over a synthetic archive (machinery test only)."""
    exps = tmp_path / "experiments"
    shutil.copytree(LAB / "experiments", exps)
    monkeypatch.setattr(paths, "EXPERIMENTS", exps)
    monkeypatch.setattr(runner, "TEMPLATE", exps / "TEMPLATE")
    monkeypatch.setattr(sys, "path", [str(tmp_path)] + [p for p in sys.path])
    for m in [m for m in sys.modules if m == "experiments" or m.startswith("experiments.")]:
        monkeypatch.delitem(sys.modules, m)
    exp_id, d = runner.new("vol_regime")
    text = (d / "PREREG.md").read_text()
    import re
    text = re.sub(r"<fill[^>]*>", "Concrete pre-registered content for the machinery test. " * 2,
                  text, flags=re.S)
    text = text.replace("## Framings\n", "## Framings\nhigh_vol_minus_low_vol and "
                        "same_vs_random_entries are described here.\n")
    (d / "PREREG.md").write_text(text)
    _archive_from(nullcal.random_walks(77, coins=3, n=2400))
    return exp_id, d


class TestLifecycle:
    def test_full_order_is_enforced(self, lab_with_template_experiment, monkeypatch):
        exp_id, d = lab_with_template_experiment
        monkeypatch.setattr(nullcal, "COINS", 3)
        monkeypatch.setattr(nullcal, "BARS", 1700)
        with pytest.raises(SystemExit, match="not registered"):
            runner.run(exp_id)
        runner.register(exp_id, log=lambda *_: None)
        with pytest.raises(SystemExit, match="null"):
            runner.run(exp_id)
        runner.null(exp_id, datasets=2, log=lambda *_: None)
        res = runner.run(exp_id, log=lambda *_: None)
        assert "high_vol_minus_low_vol" in res["decisions"]
        with pytest.raises(SystemExit, match="already has a development result"):
            runner.run(exp_id, log=lambda *_: None)
        e = ledger.experiments()[exp_id]
        assert {"registered", "null_calibrated", "dev_result"} <= set(e["events"])
        assert ledger.verify() is None

    def test_an_edited_design_cannot_run_under_the_old_registration(
            self, lab_with_template_experiment):
        exp_id, d = lab_with_template_experiment
        runner.register(exp_id, log=lambda *_: None)
        (d / "experiment.py").write_text((d / "experiment.py").read_text() + "\n# tweak\n")
        with pytest.raises(SystemExit, match="changed since registration"):
            runner.null(exp_id, datasets=1, log=lambda *_: None)

    def test_the_holdout_needs_a_dev_pass_and_opens_once(self, lab_with_template_experiment):
        exp_id, d = lab_with_template_experiment
        runner.register(exp_id, log=lambda *_: None)
        h = runner.hashes(exp_id)
        ledger.append("null_calibrated", exp_id, {"miscalibrated": {}, **h})
        ledger.append("dev_result", exp_id, {"framings": {}, "seal_id": "S1",
                                             "decisions": {"high_vol_minus_low_vol":
                                                           {"adds_edge": False}}})
        with pytest.raises(SystemExit, match="nothing passed"):
            runner.holdout(exp_id, log=lambda *_: None)
        ledger.append("holdout_result", exp_id, {"decisions": {}, "seal_id": "S1"})
        with pytest.raises(SystemExit, match="already opened once"):
            runner.holdout(exp_id, log=lambda *_: None)

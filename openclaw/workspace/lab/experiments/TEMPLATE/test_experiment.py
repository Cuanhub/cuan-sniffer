"""Tests every experiment must keep. Add your own below them."""

import importlib
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAB = HERE.parents[1]
sys.path.insert(0, str(LAB))

from labcore import evaluate as EV          # noqa: E402
from labcore import nullcal, runner         # noqa: E402

mod = importlib.import_module(f"experiments.{HERE.name}.experiment")


def test_framings_are_well_formed():
    EV.check_framings(mod.FRAMINGS)


def test_every_feature_is_causal_on_real_bars():
    assert runner.causality_problems(mod, runner.fixture_bars()) == []


def test_records_have_the_contract_shape():
    bars = nullcal.random_walks(0, coins=2, n=1600)
    rows = runner.records_on(mod, bars)
    assert rows, "no rows on 2 x 1600 bars -- is records() wired up?"
    EV.check_rows(rows, mod.FRAMINGS)

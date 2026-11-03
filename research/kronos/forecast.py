"""Sampled future candles from Kronos, for decision bars, in batches.

Runs only in research/kronos/.venv. What it does, and what it deliberately
does not:

  - the model is Kronos-small with Kronos-Tokenizer-base, loaded from
    safetensors at the pinned Hugging Face revisions in REVISIONS;
  - the model code is upstream commit 67b630e byte for byte (vendor/);
  - the input is built by inputs.py: the `lookback` closed bars ending AT the
    decision bar, normalised as upstream normalises;
  - upstream's predict() AVERAGES its sampled paths into one forecast. The
    experiment needs the paths themselves, so each window is repeated S times
    in the batch and sampled once per row (sample_count=1): one row, one path;
  - sampling goes through fast.sample, a cached copy of upstream's loop that
    test_kronos_fast holds to upstream. cached=False runs upstream's own.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(HERE / "hf"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
sys.path.insert(0, str(HERE / "vendor"))

import torch                                                     # noqa: E402
from model.kronos import (Kronos, KronosTokenizer,               # noqa: E402
                          auto_regressive_inference)

from research.kronos import inputs as I                          # noqa: E402

MODEL = "NeoQuasar/Kronos-small"
TOKENIZER = "NeoQuasar/Kronos-Tokenizer-base"
REVISIONS = {
    MODEL: "901c26c1332695a2a8f243eb2f37243a37bea320",
    TOKENIZER: "0e0117387f39004a9016484a186a908917e22426",
}
MAX_CONTEXT = 512


def load(device: str = "") -> Tuple[Any, Any, str]:
    """(tokenizer, model, device), in eval mode: dropout off.

    Kronos-small's config sets dropout from 0.1 to 0.25, and upstream's
    predictor never calls eval(). Sampling with dropout live would add noise
    nobody chose, so it is set here rather than trusted to the loader.
    """
    if not device:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = KronosTokenizer.from_pretrained(TOKENIZER, revision=REVISIONS[TOKENIZER])
    mdl = Kronos.from_pretrained(MODEL, revision=REVISIONS[MODEL])
    tok.eval()
    mdl.eval()
    return tok.to(device), mdl.to(device), device


def sample_paths(tok, mdl, device: str, windows: Sequence[Sequence[Dict[str, Any]]],
                 horizon: int, samples: int, seed: int, T: float = 1.0,
                 top_p: float = 0.9, top_k: int = 0, cached: bool = True) -> np.ndarray:
    """(B, samples, horizon, 6) sampled future candles in price units, one
    block of `samples` paths per window. Deterministic for a given batch,
    seed, device and library versions."""
    xs, means, stds, xst, yst = [], [], [], [], []
    for win in windows:
        xn, m, s = I.normalise(I.features(win))
        xs.append(xn)
        means.append(m)
        stds.append(s)
        ts = [int(b["t"]) for b in win]
        xst.append(I.stamps(ts))
        yst.append(I.stamps(I.future_times(ts[-1], horizon)))
    rep = lambda a: np.repeat(np.stack(a).astype(np.float32), samples, axis=0)  # noqa: E731
    x_t = torch.from_numpy(rep(xs)).to(device)
    xs_t = torch.from_numpy(rep(xst)).to(device)
    ys_t = torch.from_numpy(rep(yst)).to(device)
    torch.manual_seed(seed)
    if cached:
        from research.kronos import fast
        preds = fast.sample(tok, mdl, x_t, xs_t, ys_t, horizon, MAX_CONTEXT, I.CLIP,
                            T, top_k, top_p)
    else:
        preds = auto_regressive_inference(tok, mdl, x_t, xs_t, ys_t, MAX_CONTEXT, horizon,
                                          I.CLIP, T, top_k, top_p, sample_count=1)
    preds = preds[:, -horizon:, :].reshape(len(windows), samples, horizon, 6)
    m = np.stack(means)[:, None, None, :]
    s = np.stack(stds)[:, None, None, :]
    return preds.astype(np.float64) * (s + 1e-5) + m

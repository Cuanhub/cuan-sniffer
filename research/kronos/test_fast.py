"""fast.sample against upstream's auto_regressive_inference.

Runs in research/kronos/.venv (`research/kronos/.venv/bin/python -m pytest
research/kronos`); the main suite has no PyTorch and skips it. Inputs are
random walks: this checks arithmetic, not forecasting.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from research.backtest import common as C                       # noqa: E402
from research.kronos import fast, forecast as Fc, inputs as I   # noqa: E402
from model.kronos import auto_regressive_inference              # noqa: E402


@pytest.fixture(scope="module")
def cpu_model():
    tok, mdl, dev = Fc.load("cpu")
    return tok, mdl


def batch(L, H, coins=2, device="cpu"):
    bars = C.random_walks(3, coins=coins, n=L + 50)
    xs, xst, yst = [], [], []
    for c in sorted(bars):
        win = I.window(bars[c], L + 10, L)
        xs.append(I.normalise(I.features(win))[0])
        ts = [b["t"] for b in win]
        xst.append(I.stamps(ts))
        yst.append(I.stamps(I.future_times(ts[-1], H)))
    f = lambda a: torch.from_numpy(np.stack(a).astype(np.float32)).to(device)  # noqa: E731
    return f(xs), f(xst), f(yst)


def upstream_logits(mdl, pre, post, stamp, s1_next):
    """Upstream's step: decode_s1 over the whole prefix, decode_s2 on it."""
    l1, ctx = mdl.decode_s1(pre, post, stamp)
    l2 = mdl.decode_s2(ctx, s1_next)
    return l1[:, -1, :], l2[:, -1, :]


def teacher_forced(tok, mdl, L, H, device):
    """Max |difference| in s1 and s2 logits over H steps, feeding both the
    same continuation (upstream's own greedy tokens)."""
    x, xs, ys = batch(L, H, device=device)
    full_stamp = torch.cat([xs, ys], dim=1)
    with torch.no_grad():
        tk = tok.encode(torch.clip(x, -5, 5), half=True)
        pre, post = tk[0], tk[1]
        cache = fast.Cache(mdl, x.size(0), L + H, x.device)
        c1 = cache.prefill(pre, post, full_stamp[:, :L])
        worst1 = worst2 = 0.0
        for i in range(H):
            n = L + i
            u1, _ = upstream_logits(mdl, pre, post, full_stamp[:, :n], pre[:, :1])
            s1 = u1.argmax(-1, keepdim=True)
            _, u2 = upstream_logits(mdl, pre, post, full_stamp[:, :n], s1)
            c2 = cache.s2_logits(s1)
            worst1 = max(worst1, float((u1 - c1).abs().max()))
            worst2 = max(worst2, float((u2 - c2).abs().max()))
            s2 = u2.argmax(-1, keepdim=True)
            pre = torch.cat([pre, s1], dim=1)
            post = torch.cat([post, s2], dim=1)
            if i + 1 < H:
                c1 = cache.step(s1, s2, full_stamp[:, n:n + 1])
    return worst1, worst2


def test_the_model_is_in_eval_mode(cpu_model):
    tok, mdl = cpu_model
    assert not mdl.training and not tok.training


def test_teacher_forced_logits_match_upstream_on_cpu(cpu_model):
    tok, mdl = cpu_model
    w1, w2 = teacher_forced(tok, mdl, L=120, H=10, device="cpu")
    assert w1 < 1e-3 and w2 < 1e-3, (w1, w2)


def test_teacher_forced_logits_match_at_the_experiments_size(cpu_model):
    tok, mdl = cpu_model
    w1, w2 = teacher_forced(tok, mdl, L=464, H=4, device="cpu")
    assert w1 < 1e-3 and w2 < 1e-3, (w1, w2)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS")
def test_teacher_forced_logits_match_upstream_on_mps():
    tok, mdl, _ = Fc.load("mps")
    w1, w2 = teacher_forced(tok, mdl, L=200, H=8, device="mps")
    assert w1 < 5e-3 and w2 < 5e-3, (w1, w2)


def test_sampled_paths_equal_upstreams_with_the_same_seed(cpu_model):
    tok, mdl = cpu_model
    x, xs, ys = batch(L=100, H=12, coins=3)
    torch.manual_seed(7)
    up = auto_regressive_inference(tok, mdl, x, xs, ys, 512, 12, 5.0, 1.0, 0, 0.9,
                                   sample_count=1)
    torch.manual_seed(7)
    ca = fast.sample(tok, mdl, x, xs, ys, 12, 512, 5.0, 1.0, 0, 0.9)
    assert up.shape == ca.shape
    np.testing.assert_allclose(ca, up, atol=1e-4)


def test_different_seeds_give_different_paths(cpu_model):
    """Non-vacuity for the test above: if the draws ignored the seed, equal
    paths would prove nothing."""
    tok, mdl = cpu_model
    x, xs, ys = batch(L=100, H=12, coins=3)
    torch.manual_seed(7)
    a = fast.sample(tok, mdl, x, xs, ys, 12, 512, 5.0, 1.0, 0, 0.9)
    torch.manual_seed(8)
    b = fast.sample(tok, mdl, x, xs, ys, 12, 512, 5.0, 1.0, 0, 0.9)
    assert np.abs(a[:, -12:] - b[:, -12:]).max() > 1e-3


def test_a_window_that_would_slide_is_refused(cpu_model):
    tok, mdl = cpu_model
    x, xs, ys = batch(L=500, H=13)
    with pytest.raises(ValueError, match="slide"):
        fast.sample(tok, mdl, x, xs, ys, 13)

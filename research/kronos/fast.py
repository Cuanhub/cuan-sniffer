"""Kronos's sampler with a key/value cache: the same model and the same draws,
for a fraction of the work.

Upstream's auto_regressive_inference re-runs the whole transformer over the
whole window for every candle it generates -- a 48-candle forecast is 48 full
passes over ~500 positions, ~15s per decision on this machine. In a causal
transformer the hidden state of an earlier position cannot change when a later
one is appended, so all but the newest position is repetition. That holds only
while the window does not slide: past max_context upstream rolls the buffer,
every position's rotary phase moves, and nothing can be reused. So sample()
refuses lookback + horizon > max_context rather than approximate it.

What is upstream's, called rather than copied: the tokenizer's encode and
decode, the embeddings, the norms, the feed-forward layers, the heads,
sample_from_logits, and decode_s1 for the whole context (prefill() runs it and
captures each layer's keys and values on the way through). What is new: one
position at a time through the attention layers, and the dependency layer's
cross-attention evaluated at the last position only -- the only one upstream
reads. The draws happen in upstream's order, s1 then s2 each step, so a batch
given the same seed consumes the random stream identically.

test_kronos_fast (Kronos venv) holds this to upstream: teacher-forced logits
at every step, and whole sampled paths with the same seed.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F

from research.kronos import forecast as _setup      # noqa: F401  vendor path, HF_HOME
from model.kronos import sample_from_logits          # noqa: E402


def _heads(x: torch.Tensor, n_heads: int) -> torch.Tensor:
    n, s, d = x.shape
    return x.view(n, s, n_heads, d // n_heads).transpose(1, 2)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class Cache:
    """One batch of windows. prefill() fills positions 0..L-1; each step()
    writes one more and returns the s1 logits for the position after it.

    Buffers are allocated once at full length and written in place, with a
    mask hiding the positions not filled yet. Growing them with torch.cat
    allocated a new, larger copy of every buffer at every step, which the MPS
    allocator did not reuse: 30 GB and swapping for a batch of 128 rows.
    """

    def __init__(self, mdl, n_rows: int, max_len: int, device):
        self.mdl = mdl
        attn = mdl.transformer[0].self_attn
        self.n_heads, self.head_dim = attn.n_heads, attn.head_dim
        # Upstream's RotaryPositionalEmbedding formula, for every position at
        # once; position p's row is what upstream computes for index p.
        rot = attn.rotary
        t = torch.arange(max_len, device=rot.inv_freq.device).type_as(rot.inv_freq)
        freqs = torch.einsum("i,j->ij", t, rot.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos, self.sin = emb.cos(), emb.sin()
        layers = len(mdl.transformer)
        self.k = torch.zeros(layers, n_rows, self.n_heads, max_len, self.head_dim,
                             device=device)
        self.v = torch.zeros_like(self.k)
        ca = mdl.dep_layer.cross_attn
        self.ck = torch.zeros(n_rows, ca.n_heads, max_len, ca.head_dim, device=device)
        self.cv = torch.zeros_like(self.ck)
        self.last: Optional[torch.Tensor] = None    # final-norm state, newest position
        self.index = torch.arange(max_len, device=device)
        self.pos = 0

    def _rope(self, x: torch.Tensor, start: int) -> torch.Tensor:
        n = x.size(-2)
        cos = self.cos[start:start + n][None, None]
        sin = self.sin[start:start + n][None, None]
        return x * cos + _rotate_half(x) * sin

    def _visible(self, upto: int) -> torch.Tensor:
        """Positions 0..upto-1 take part in attention (SDPA: True = attend)."""
        return (self.index < upto)[None, None, None, :]

    def _write_dep_kv(self, h: torch.Tensor, at: int) -> None:
        ca = self.mdl.dep_layer.cross_attn
        n = h.size(1)
        self.ck[:, :, at:at + n] = _heads(ca.k_proj(h), ca.n_heads)
        self.cv[:, :, at:at + n] = _heads(ca.v_proj(h), ca.n_heads)

    def prefill(self, s1: torch.Tensor, s2: torch.Tensor, stamp: torch.Tensor) -> torch.Tensor:
        """Upstream's own decode_s1 over the context, with each attention
        layer's k_proj and v_proj outputs captured for reuse."""
        caps: List[dict] = [{} for _ in self.mdl.transformer]
        hooks = []
        for layer, store in zip(self.mdl.transformer, caps):
            a = layer.self_attn
            hooks.append(a.k_proj.register_forward_hook(
                lambda m, i, o, s=store: s.__setitem__("k", o)))
            hooks.append(a.v_proj.register_forward_hook(
                lambda m, i, o, s=store: s.__setitem__("v", o)))
        try:
            logits, ctx = self.mdl.decode_s1(s1, s2, stamp)
        finally:
            for h in hooks:
                h.remove()
        L = s1.size(1)
        for li, c in enumerate(caps):
            self.k[li, :, :, :L] = self._rope(_heads(c["k"], self.n_heads), 0)
            self.v[li, :, :, :L] = _heads(c["v"], self.n_heads)
        self._write_dep_kv(ctx, 0)
        self.last = ctx[:, -1:, :]
        self.pos = L
        return logits[:, -1, :]

    def s2_logits(self, s1_next: torch.Tensor) -> torch.Tensor:
        """decode_s2 at the last position. Upstream rotates the dependency
        layer's q and k with the rotary cache for q's length, which is 1:
        position 0, cos 1 and sin 0 exactly, so the rotation is the identity.
        It is non-causal outside training: every filled position is visible."""
        mdl = self.mdl
        ca = mdl.dep_layer.cross_attn
        sib = mdl.embedding.emb_s1(s1_next)
        q = _heads(ca.q_proj(sib), ca.n_heads)
        out = F.scaled_dot_product_attention(q, self.ck, self.cv,
                                             attn_mask=self._visible(self.pos))
        out = out.transpose(1, 2).contiguous().view(sib.size(0), 1, -1)
        x2 = mdl.dep_layer.norm(self.last + ca.out_proj(out))
        return mdl.head.cond_forward(x2)[:, -1, :]

    def step(self, s1: torch.Tensor, s2: torch.Tensor, stamp: torch.Tensor) -> torch.Tensor:
        """Write the tokens (N, 1) at the next position, stamped (N, 1, 5)."""
        mdl = self.mdl
        p = self.pos
        visible = self._visible(p + 1)
        x = mdl.token_drop(mdl.embedding([s1, s2]) + mdl.time_emb(stamp))
        for li, layer in enumerate(mdl.transformer):
            a = layer.self_attn
            h = layer.norm1(x)
            q = self._rope(_heads(a.q_proj(h), self.n_heads), p)
            self.k[li, :, :, p:p + 1] = self._rope(_heads(a.k_proj(h), self.n_heads), p)
            self.v[li, :, :, p:p + 1] = _heads(a.v_proj(h), self.n_heads)
            # One query at the newest position: upstream's causal mask allows
            # it every earlier key, and `visible` allows exactly those.
            o = F.scaled_dot_product_attention(q, self.k[li], self.v[li], attn_mask=visible)
            o = o.transpose(1, 2).contiguous().view(x.size(0), 1, -1)
            x = x + a.resid_dropout(a.out_proj(o))
            x = x + layer.ffn(layer.norm2(x))
        h = mdl.norm(x)
        self._write_dep_kv(h, p)
        self.last = h
        self.pos = p + 1
        return mdl.head(h)[:, -1, :]


def sample(tok, mdl, x: torch.Tensor, x_stamp: torch.Tensor, y_stamp: torch.Tensor,
           pred_len: int, max_context: int = 512, clip: float = 5.0, T: float = 1.0,
           top_k: int = 0, top_p: float = 0.9):
    """auto_regressive_inference(..., sample_count=1), cached. Returns what
    upstream returns: the decoded context plus forecast, (N, L + pred_len, 6),
    as a numpy array."""
    if x.size(1) + pred_len > max_context:
        raise ValueError(f"lookback {x.size(1)} + horizon {pred_len} exceeds "
                         f"max_context {max_context}: the window would slide")
    with torch.no_grad():
        x = torch.clip(x, -clip, clip)
        L = x.size(1)
        x_token = tok.encode(x, half=True)
        full_stamp = torch.cat([x_stamp, y_stamp], dim=1)
        cache = Cache(mdl, x.size(0), L + pred_len, x.device)
        logits = cache.prefill(x_token[0], x_token[1], full_stamp[:, :L])
        pre, post = [], []
        for i in range(pred_len):
            s1 = sample_from_logits(logits, temperature=T, top_k=top_k, top_p=top_p,
                                    sample_logits=True)
            s2 = sample_from_logits(cache.s2_logits(s1), temperature=T, top_k=top_k,
                                    top_p=top_p, sample_logits=True)
            pre.append(s1)
            post.append(s2)
            if i + 1 < pred_len:
                logits = cache.step(s1, s2, full_stamp[:, L + i:L + i + 1])
        full_pre = torch.cat([x_token[0]] + pre, dim=1)
        full_post = torch.cat([x_token[1]] + post, dim=1)
        z = tok.decode([full_pre, full_post], half=True)
        return z.cpu().numpy()

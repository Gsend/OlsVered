"""
diagnostic/models/tiny_transformer.py
======================================

TinyTransformer — 2-block pre-LN encoder for synthetic sequence
classification (E8). Probes attention (the bilinear softmax mixing) and
in-chain LayerNorm.

Design choices (Q2 resolution):
  - The Q/K/V/O and MLP projections are plain nn.Linear -> they are exactly
    the objects the OLS framework fits.
  - Softmax self-attention (softmax(QK^T/sqrt(d)) @ V) is treated as an
    OPAQUE, non-invertible op (like max-pool in E3): we never back-propagate
    a label target through the softmax. It is replayed in the forward sweep.
  - LayerNorm is standalone (no following ReLU), distinct from LNReLU in E2.

Architecture (pre-LN):
  h = embed(tokens) + pos                                  (B, T, d_model)
  per block:
     a = Wo( MHSA( LN1(h) ) );   h = h + a
     m = fc_out( ReLU( fc_in( LN2(h) ) ) );   h = h + m
  pooled = mean_t( LN_f(h) )                               (B, d_model)
  logits = head(pooled)                                    (B, n_classes)

The Linear layers OLS-fits target, deepest-first:
  head, b1.fc_out, b1.fc_in, b1.Wo, b1.Wv, b1.Wk, b1.Wq,
        b0.fc_out, b0.fc_in, b0.Wo, b0.Wv, b0.Wk, b0.Wq
(b0 = first block, b1 = second block).

Synthetic task (self-contained, reproducible): "majority" — each token is an
integer in [0, V); label = 1 if more than half the tokens are >= V/2, else 0.
Solvable by uniform averaging (attention can implement it) and reliably
learnable by backprop, so distillation / retrain numbers are meaningful.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# synthetic data
# ---------------------------------------------------------------------------

def make_majority_dataset(
    n: int, seq_len: int, vocab: int, *, seed: int = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (tokens (n, seq_len) long, labels (n,) long in {0,1}).

    label = 1 iff count(token >= vocab/2) > seq_len/2.
    """
    g = torch.Generator().manual_seed(seed)
    tokens = torch.randint(0, vocab, (n, seq_len), generator=g)
    high = (tokens >= (vocab // 2)).sum(dim=1)
    labels = (high > (seq_len / 2)).long()
    return tokens, labels


def make_pointer_dataset(
    n: int, seq_len: int, vocab: int, *, seed: int = 0,
    ptr_range: Tuple[int, int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (tokens (n, seq_len) long, labels (n,) long in {0,1}).

    token[:, 0] = pointer p drawn uniformly from [1, seq_len-1] (or ptr_range).
    token[:, 1:] = payload tokens drawn uniformly from [0, vocab).
    label = 1 if token[p] >= vocab//2 else 0.

    Requires content-based attention: the model must read the pointer at
    position 0, then attend to position p to fetch and threshold the payload.
    Uniform averaging cannot solve this task.
    """
    if ptr_range is None:
        ptr_range = (1, seq_len - 1)
    lo, hi = ptr_range
    g = torch.Generator().manual_seed(seed)
    ptr = torch.randint(lo, hi + 1, (n,), generator=g)          # pointer values
    payload = torch.randint(0, vocab, (n, seq_len - 1), generator=g)
    tokens = torch.cat([ptr.unsqueeze(1), payload], dim=1)       # (n, seq_len)
    # Fetch token at position ptr[i] for each sample i
    fetched = tokens.gather(1, ptr.unsqueeze(1)).squeeze(1)
    labels = (fetched >= (vocab // 2)).long()
    return tokens, labels


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

class _Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.ln1 = nn.LayerNorm(d_model)
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)

        self.ln2 = nn.LayerNorm(d_model)
        self.fc_in = nn.Linear(d_model, d_ff)
        self.act = nn.ReLU()
        self.fc_out = nn.Linear(d_ff, d_model)

    def attention(self, z):
        """z: (B,T,d) -> context (B,T,d). Returns (context, attn_weights)."""
        B, T, d = z.shape
        q = self.Wq(z).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.Wk(z).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.Wv(z).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn = scores.softmax(dim=-1)
        ctx = attn @ v                                  # (B,nh,T,dh)
        ctx = ctx.transpose(1, 2).reshape(B, T, d)      # (B,T,d)
        return ctx, attn

    def forward(self, h):
        z = self.ln1(h)
        ctx, _ = self.attention(z)
        a = self.Wo(ctx)
        h = h + a
        z2 = self.ln2(h)
        m = self.fc_out(self.act(self.fc_in(z2)))
        h = h + m
        return h


class TinyTransformer(nn.Module):
    def __init__(self, vocab: int = 32, seq_len: int = 16, d_model: int = 64,
                 n_heads: int = 4, d_ff: int = 128, depth: int = 2,
                 num_classes: int = 2):
        super().__init__()
        self.vocab = vocab
        self.seq_len = seq_len
        self.d_model = d_model
        self.depth = depth

        self.embed = nn.Embedding(vocab, d_model)
        self.pos = nn.Parameter(torch.zeros(seq_len, d_model))
        nn.init.normal_(self.pos, std=0.1)
        self.blocks = nn.ModuleList([_Block(d_model, n_heads, d_ff) for _ in range(depth)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, tokens):
        h = self.embed(tokens) + self.pos.unsqueeze(0)
        for blk in self.blocks:
            h = blk(h)
        pooled = self.ln_f(h).mean(dim=1)
        return self.head(pooled)

    # ------------------------------------------------------------------
    def get_linear_layers_deepest_first(self) -> List[nn.Linear]:
        layers: List[nn.Linear] = [self.head]
        for blk in reversed(self.blocks):
            layers += [blk.fc_out, blk.fc_in, blk.Wo, blk.Wv, blk.Wk, blk.Wq]
        return layers

    def linear_layer_names_deepest_first(self) -> List[str]:
        names = ["head"]
        for bi in reversed(range(self.depth)):
            names += [f"b{bi}.fc_out", f"b{bi}.fc_in", f"b{bi}.Wo",
                      f"b{bi}.Wv", f"b{bi}.Wk", f"b{bi}.Wq"]
        return names

    # ------------------------------------------------------------------
    def forward_with_state(self, tokens) -> Dict[str, torch.Tensor]:
        """Forward returning every Linear's input (pre-activation INPUT) and
        output (pre-activation OUTPUT) needed for feature distillation, plus
        the pooled representation. All token-wise tensors are (B,T,d) (or
        (B,T,d_ff)); head tensors are (B,d)/(B,n_classes).

        Keys per block bi:
          bi.ln1_out      LN1(h)                input to Wq/Wk/Wv
          bi.q/k/v_pre    W{q,k,v}(ln1_out)     pre-activation outputs
          bi.ctx          attention context     input to Wo
          bi.o_pre        Wo(ctx)
          bi.ln2_out      LN2(h after attn)     input to fc_in
          bi.fcin_pre     fc_in(ln2_out)
          bi.fcin_post    ReLU(fcin_pre)        input to fc_out
          bi.fcout_pre    fc_out(fcin_post)
        Global:
          pooled_in       LN_f(h).mean(1)       input to head
          logits          head(pooled_in)
        """
        s: Dict[str, torch.Tensor] = {}
        h = self.embed(tokens) + self.pos.unsqueeze(0)
        for bi, blk in enumerate(self.blocks):
            z = blk.ln1(h)
            s[f"b{bi}.ln1_out"] = z
            B, T, d = z.shape
            q = blk.Wq(z); k = blk.Wk(z); v = blk.Wv(z)
            s[f"b{bi}.q_pre"] = q
            s[f"b{bi}.k_pre"] = k
            s[f"b{bi}.v_pre"] = v
            qh = q.view(B, T, blk.n_heads, blk.d_head).transpose(1, 2)
            kh = k.view(B, T, blk.n_heads, blk.d_head).transpose(1, 2)
            vh = v.view(B, T, blk.n_heads, blk.d_head).transpose(1, 2)
            scores = (qh @ kh.transpose(-2, -1)) / math.sqrt(blk.d_head)
            attn = scores.softmax(dim=-1)
            ctx = (attn @ vh).transpose(1, 2).reshape(B, T, d)
            s[f"b{bi}.ctx"] = ctx
            o = blk.Wo(ctx)
            s[f"b{bi}.o_pre"] = o
            h = h + o
            z2 = blk.ln2(h)
            s[f"b{bi}.ln2_out"] = z2
            fcin = blk.fc_in(z2)
            s[f"b{bi}.fcin_pre"] = fcin
            fcin_post = blk.act(fcin)
            s[f"b{bi}.fcin_post"] = fcin_post
            fcout = blk.fc_out(fcin_post)
            s[f"b{bi}.fcout_pre"] = fcout
            h = h + fcout
        pooled = self.ln_f(h).mean(dim=1)
        s["pooled_in"] = pooled
        s["logits"] = self.head(pooled)
        return s

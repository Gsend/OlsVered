"""
Model definitions for benchmarking.

Provides simple, reproducible models for comparing optimisers:
  - SmallTransformer: minimal GPT-style model (all nn.Linear layers)
  - SimpleResNet: small residual network for CIFAR-10
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Small GPT-style Transformer (all-Linear, K-FAC friendly)
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Single transformer block with multi-head self-attention + FFN."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        # Attention projections (all nn.Linear — K-FAC hooks apply)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # Self-attention
        h = self.ln1(x)
        q = self.q_proj(h).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(h).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(h).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        # Causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(mask, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)
        x = x + out

        # FFN
        x = x + self.ffn(self.ln2(x))
        return x


class SmallTransformer(nn.Module):
    """Minimal GPT-style language model for benchmarking.

    Parameters
    ----------
    vocab_size : int
        Vocabulary size. Default: 1000.
    d_model : int
        Model dimension. Default: 256.
    n_heads : int
        Number of attention heads. Default: 4.
    n_layers : int
        Number of transformer blocks. Default: 4.
    d_ff : int
        FFN inner dimension. Default: 512.
    max_seq_len : int
        Maximum sequence length. Default: 128.
    dropout : float
        Dropout rate. Default: 0.1.
    """

    def __init__(
        self,
        vocab_size: int = 1000,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        max_seq_len: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        tok = self.token_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = self.dropout(tok + pos)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def count_linear_layers(self) -> int:
        return sum(1 for m in self.modules() if isinstance(m, nn.Linear))


# ---------------------------------------------------------------------------
# Simple MLP for fast smoke tests
# ---------------------------------------------------------------------------

class SimpleMLP(nn.Module):
    """Small MLP for unit testing and smoke tests."""

    def __init__(self, input_dim: int = 784, hidden_dim: int = 256,
                 output_dim: int = 10, n_layers: int = 3):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.view(x.shape[0], -1))

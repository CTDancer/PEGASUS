"""Minimal neural components required by the scratch Uniform-LKF checkpoint.

These definitions intentionally preserve the parameter names and computation used
by the original TCFM peptide model so existing M8 checkpoints load strictly.
"""

from __future__ import annotations

import torch
import torch.nn as nn

Tensor = torch.Tensor


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, t: Tensor) -> Tensor:
        return self.mlp(t.unsqueeze(-1))


class DiTBlock(nn.Module):
    def __init__(self, hidden_size: int, n_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(c).chunk(6, dim=1)

        x_norm1 = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_output, _ = self.attn(x_norm1, x_norm1, x_norm1)
        x = x + gate_msa.unsqueeze(1) * attn_output

        x_norm2 = modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_output = self.mlp(x_norm2)
        x = x + gate_mlp.unsqueeze(1) * mlp_output
        return x


__all__ = ["modulate", "TimestepEmbedder", "DiTBlock"]

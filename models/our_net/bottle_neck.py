"""Recursive Hybrid Mamba Attention (RHMA) from SGMA-Net."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from mamba_ssm import Mamba


class LGFI(nn.Module):
    """Linear Global Feature Interaction with shared recursive projections."""

    def __init__(self, channels: int, recursion_depth: int = 2):
        super().__init__()
        self.recursion_depth = recursion_depth
        self.norm = nn.LayerNorm(channels)
        self.q = nn.Linear(channels, channels, bias=False)
        self.k = nn.Linear(channels, channels, bias=False)
        self.v = nn.Linear(channels, channels, bias=False)
        self.proj = nn.Linear(channels, channels, bias=False)
        self.ffn_norm = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, 2 * channels),
            nn.GELU(),
            nn.Linear(2 * channels, channels),
        )

    def _refine_once(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        normed = self.norm(tokens)
        q = self.q(normed).transpose(1, 2)
        k = self.k(normed).transpose(1, 2)
        v = self.v(normed).transpose(1, 2)
        affinity = torch.softmax(
            torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(h * w),
            dim=-1,
        )
        refined = torch.matmul(affinity, v).transpose(1, 2)
        tokens = tokens + self.proj(refined)
        tokens = tokens + self.ffn(self.ffn_norm(tokens))
        return tokens.transpose(1, 2).reshape(b, c, h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(self.recursion_depth):
            x = self._refine_once(x)
        return x


class RHMA(nn.Module):
    """LGFI -> official Mamba S6 -> two-head MHSA at the 8x8 bottleneck."""

    def __init__(
        self,
        channels: int = 64,
        d_state: int = 32,
        d_conv: int = 4,
        expand: int = 2,
        heads: int = 2,
        recursion_depth: int = 2,
    ):
        super().__init__()
        if channels % heads:
            raise ValueError("channels must be divisible by heads")

        self.lgfi = LGFI(channels, recursion_depth=recursion_depth)
        self.s6_norm = nn.RMSNorm(channels)
        self.s6_linear = nn.Linear(channels, channels)
        self.s6 = Mamba(
            d_model=channels,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.s6_ffn_norm = nn.RMSNorm(channels)
        self.s6_ffn = nn.Sequential(
            nn.Linear(channels, 2 * channels),
            nn.GELU(),
            nn.Linear(2 * channels, channels),
        )

        self.mhsa_norm = nn.RMSNorm(channels)
        self.mhsa = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=heads,
            dropout=0.0,
            batch_first=True,
        )
        self.mhsa_ffn_norm = nn.RMSNorm(channels)
        self.mhsa_ffn = nn.Sequential(
            nn.Linear(channels, 2 * channels),
            nn.GELU(),
            nn.Linear(2 * channels, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z1 = self.lgfi(x)
        b, c, h, w = z1.shape
        tokens = z1.flatten(2).transpose(1, 2)

        s6_input = self.s6_linear(self.s6_norm(tokens))
        u1 = tokens + self.s6(s6_input)
        z2 = u1 + self.s6_ffn(self.s6_ffn_norm(u1))

        mhsa_input = self.mhsa_norm(z2)
        mhsa_output, _ = self.mhsa(
            mhsa_input, mhsa_input, mhsa_input, need_weights=False
        )
        u2 = z2 + mhsa_output
        out = u2 + self.mhsa_ffn(self.mhsa_ffn_norm(u2))
        return out.transpose(1, 2).reshape(b, c, h, w)

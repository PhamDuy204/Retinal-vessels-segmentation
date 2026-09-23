"""SGMA-Net building blocks named as in the paper."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """3x3 depthwise convolution followed by 1x1 pointwise projection."""

    def __init__(self, in_channels: int, out_channels: int, dilation: int = 1, activate: bool = True):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding=dilation,
            dilation=dilation, groups=in_channels, bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True) if activate else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.pointwise(self.depthwise(x))))


class DSConvBlock(nn.Module):
    """Two depthwise-separable convolutional units used in encoder/decoder."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            DepthwiseSeparableConv(in_channels, out_channels),
            DepthwiseSeparableConv(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SAG(nn.Module):
    """Statistical Aggregation Gate (SAG), Eq. (4) / Appendix A."""

    def __init__(self, channels: int):
        super().__init__()
        self.x_interaction = DepthwiseSeparableConv(channels, channels)
        self.h_interaction = DepthwiseSeparableConv(channels, channels)
        self.feature_projection = nn.Conv2d(2 * channels, channels, 1, bias=False)
        self.alpha = nn.Conv2d(6 * channels, channels, 1, bias=True)
        self.beta = nn.Conv2d(2 * channels, channels, 1, bias=True)
        self.out = nn.Conv2d(channels, channels, 1, bias=False)

    @staticmethod
    def _gap(x: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(x, 1)

    @staticmethod
    def _gmp(x: torch.Tensor) -> torch.Tensor:
        return F.adaptive_max_pool2d(x, 1)

    def forward(self, x_i: torch.Tensor, h_i: torch.Tensor) -> torch.Tensor:
        interaction = self.x_interaction(x_i) + self.h_interaction(h_i)
        statistics = torch.cat(
            (
                self._gap(x_i), self._gmp(x_i),
                self._gap(interaction), self._gmp(interaction),
                self._gap(h_i), self._gmp(h_i),
            ),
            dim=1,
        )
        fused = self.feature_projection(torch.cat((x_i, h_i), dim=1))
        g_i = fused * torch.sigmoid(self.alpha(statistics))
        refinement_stats = torch.cat((self._gap(g_i), self._gmp(g_i)), dim=1)
        g_tilde = g_i * torch.sigmoid(self.beta(refinement_stats))
        return self.out(x_i + g_tilde)


class MDC(nn.Module):
    """Multi-Dilated Convolution with dilation rates {1, 3, 5}."""

    def __init__(self, channels: int):
        super().__init__()
        self.atrous = nn.ModuleList(
            [nn.Conv2d(channels, channels, 3, padding=d, dilation=d, bias=False) for d in (1, 3, 5)]
        )
        self.fuse = nn.Conv2d(3 * channels, channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fuse(torch.cat([conv(x) for conv in self.atrous], dim=1))


class MDSA(nn.Module):
    """Multi-Dimensional Statistical Attention (MDSA), Eq. (5)."""

    def __init__(self, channels: int, height: int, width: int):
        super().__init__()
        self.channel_view = MDC(channels)
        self.height_view = MDC(height)
        self.width_view = MDC(width)
        self.fuse_views = nn.Conv2d(3 * channels, channels, 1, bias=False)
        self.statistical_attention = nn.Sequential(
            nn.Conv2d(3, 3, 3, padding=1, groups=3, bias=False),
            nn.Conv2d(3, 1, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b_c = self.channel_view(x)
        x_h = x.permute(0, 2, 1, 3).contiguous()
        b_h = self.height_view(x_h).permute(0, 2, 1, 3).contiguous()
        x_w = x.permute(0, 3, 2, 1).contiguous()
        b_w = self.width_view(x_w).permute(0, 3, 2, 1).contiguous()
        f_md = self.fuse_views(torch.cat((b_c, b_h, b_w), dim=1))
        descriptors = torch.cat(
            (
                f_md.mean(dim=1, keepdim=True),
                f_md.amax(dim=1, keepdim=True),
                f_md.std(dim=1, keepdim=True, unbiased=False),
            ),
            dim=1,
        )
        a_sa = self.statistical_attention(descriptors)
        return x * a_sa


class SGWL(nn.Module):
    """Statistical Global Weight Learning for exactly three decoder maps."""

    def __init__(self, output_size: int = 64):
        super().__init__()
        self.output_size = output_size
        self.full_map = nn.Conv2d(3, 3, kernel_size=output_size, groups=3, bias=False)
        self.weight_logits = nn.Conv2d(3, 3, kernel_size=1, bias=True)

    def fusion_weights(self, stacked_predictions: torch.Tensor) -> torch.Tensor:
        if stacked_predictions.shape[1] != 3:
            raise ValueError("SGWL expects exactly three decoder predictions")
        if stacked_predictions.shape[-2:] != (self.output_size, self.output_size):
            raise ValueError(
                f"SGWL expects {self.output_size}x{self.output_size} maps, "
                f"got {tuple(stacked_predictions.shape[-2:])}"
            )
        gap = F.adaptive_avg_pool2d(stacked_predictions, 1)
        gmp = F.adaptive_max_pool2d(stacked_predictions, 1)
        full_map = self.full_map(stacked_predictions)
        return torch.softmax(self.weight_logits(gap + gmp + full_map), dim=1)

    def forward(self, s1: torch.Tensor, s2: torch.Tensor, s3: torch.Tensor) -> torch.Tensor:
        stacked = torch.cat((s1, s2, s3), dim=1)
        weights = self.fusion_weights(stacked)
        return (stacked * weights).sum(dim=1, keepdim=True)

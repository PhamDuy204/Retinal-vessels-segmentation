"""SGMA-Net implementation aligned with the paper module names."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .bottle_neck import RHMA
    from .modules import DSConvBlock, MDSA, SAG, SGWL
except ImportError:
    from bottle_neck import RHMA
    from modules import DSConvBlock, MDSA, SAG, SGWL


class EncoderStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.features = DSConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features(x)
        return features, self.pool(features)


class DecoderStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_size: int,
        use_sag: bool,
        use_mdsa: bool,
    ):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.sag = SAG(out_channels) if use_sag else None
        self.mdsa = MDSA(out_channels, spatial_size, spatial_size) if use_mdsa else None
        self.decode = DSConvBlock(2 * out_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        semantic = self.up(x)
        refined_skip = self.sag(skip, semantic) if self.sag is not None else skip
        if self.mdsa is not None:
            refined_skip = self.mdsa(refined_skip)
        return self.decode(torch.cat((semantic, refined_skip), dim=1))


class SGMANet(nn.Module):
    """Configurable core shared by the full network and paper ablations."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        width: int = 64,
        *,
        use_rhma: bool = True,
        use_sag: bool = True,
        use_mdsa: bool = True,
        use_sgwl: bool = True,
    ):
        super().__init__()
        if width < 16 or width % 4:
            raise ValueError("width must be a multiple of 4 and at least 16")

        c1, c2, c3, cb = width // 4, width // 2, width, width
        self.use_sgwl = use_sgwl

        self.encoder1 = EncoderStage(in_channels, c1)
        self.encoder2 = EncoderStage(c1, c2)
        self.encoder3 = EncoderStage(c2, c3)
        self.bottleneck_conv = DSConvBlock(c3, cb)
        self.rhma = RHMA(cb) if use_rhma else nn.Identity()
        self.bottleneck_mdsa = MDSA(cb, 8, 8) if use_mdsa else nn.Identity()

        self.decoder1 = DecoderStage(cb, c3, 16, use_sag, use_mdsa)
        self.decoder2 = DecoderStage(c3, c2, 32, use_sag, use_mdsa)
        self.decoder3 = DecoderStage(c2, c1, 64, use_sag, use_mdsa)

        self.head1 = nn.Conv2d(c3, out_channels, 1)
        self.head2 = nn.Conv2d(c2, out_channels, 1)
        self.head3 = nn.Conv2d(c1, out_channels, 1)
        self.sgwl = SGWL(output_size=64) if use_sgwl else None

    @staticmethod
    def _resize_prediction(prediction: torch.Tensor) -> torch.Tensor:
        if prediction.shape[-2:] == (64, 64):
            return prediction
        return F.interpolate(
            prediction, size=(64, 64), mode="bilinear", align_corners=False
        )

    def forward(self, x: torch.Tensor):
        e1, x = self.encoder1(x)
        e2, x = self.encoder2(x)
        e3, x = self.encoder3(x)

        x = self.bottleneck_conv(x)
        x = self.rhma(x)
        x = self.bottleneck_mdsa(x)

        d1 = self.decoder1(x, e3)
        d2 = self.decoder2(d1, e2)
        d3 = self.decoder3(d2, e1)

        s1 = self._resize_prediction(self.head1(d1))
        s2 = self._resize_prediction(self.head2(d2))
        s3 = self.head3(d3)

        final_logits = self.sgwl(s1, s2, s3) if self.sgwl is not None else s3
        if self.training:
            if self.sgwl is not None:
                return final_logits, s1, s2, s3
            return s3, s1, s2
        return torch.sigmoid(final_logits)


class SegModel(SGMANet):
    """Default full SGMA-Net used by --model our_net."""

    def __init__(self, in_channels: int, out_channels: int, width: int = 64):
        super().__init__(
            in_channels,
            out_channels,
            width,
            use_rhma=True,
            use_sag=True,
            use_mdsa=True,
            use_sgwl=True,
        )

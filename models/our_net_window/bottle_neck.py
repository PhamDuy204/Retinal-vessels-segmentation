import torch
import torch.nn as nn
from mambapy.mamba import MambaBlock, MambaConfig

from models.our_net.bottle_neck import CAB, CAB_1


class BottleNeck_2(nn.Module):
    """Windows-friendly counterpart of our_net BottleNeck_2.

    Uses mambapy's pure-PyTorch Mamba v1 block instead of Tri Dao Mamba2.
    Input/output layout and the surrounding residual/MLP path match our_net.
    """

    def __init__(self, dimension, d_state=32, d_conv=4):
        super().__init__()
        config = MambaConfig(
            d_model=dimension,
            n_layers=1,
            d_state=d_state,
            d_conv=d_conv,
            expand_factor=2,
            conv_bias=True,
            pscan=True,
            use_cuda=False,
        )
        self.mamba = MambaBlock(config)
        self.out = nn.Sequential(
            nn.Linear(dimension, dimension, bias=False),
            nn.GELU(),
            nn.Linear(dimension, dimension, bias=False),
        )

    def forward(self, x: torch.Tensor):
        b, c, h, w = x.shape
        seq = x.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)
        forward_states = self.mamba(seq) + seq
        merged = self.out(forward_states) + forward_states
        return merged.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()

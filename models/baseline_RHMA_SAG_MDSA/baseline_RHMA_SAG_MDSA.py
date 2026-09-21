"""Paper ablation wrapper using the shared SGMA-Net implementation."""

from models.our_net.our_net import SGMANet


class SegModel(SGMANet):
    def __init__(self, in_channels, out_channels, width=64):
        super().__init__(
            in_channels,
            out_channels,
            width,
            use_rhma=True,
            use_sag=True,
            use_mdsa=True,
            use_sgwl=False,
        )

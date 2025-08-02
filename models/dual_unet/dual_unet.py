import torch.nn as nn 
import torch.nn.functional as F
from .modules import EncoderBlock, DecoderBlock


class SegModel(nn.Module):
    """
    Dual-Branch U-Net Architecture for Image Segmentation
    """
    
    def __init__(self, in_channels=1, num_class=1): 
        """
        Initialize the Dual-Branch U-Net model.
        
        Args:
            in_channels (int): Number of input channels (default: 3 for RGB images)
            num_class (int): Base number of output classes
                               This value is scaled throughout the network:
                               - Encoder: 64 -> 128 -> 256 -> 512 -> 1024
                               - Decoder: 512 -> 256 -> 128 -> 64
        """
        super(SegModel, self).__init__()

        # Encoder Path - Dual-branch feature extraction with progressive downsampling
        # Each encoder block combines depthwise separable convolution and dynamic convolution
        self.down1 = EncoderBlock(in_channels, 64)          # 3 -> 64,   spatial: H/2 x W/2
        self.down2 = EncoderBlock(64, 128)     # 64 -> 128, spatial: H/4 x W/4
        self.down3 = EncoderBlock(128, 256) # 128 -> 256, spatial: H/8 x W/8
        self.down4 = EncoderBlock(256, 512) # 256 -> 512, spatial: H/16 x W/16

        # Bottleneck - Deepest feature representation
        self.bottle_neck = EncoderBlock(512, 1024) # 512 -> 1024, spatial: H/32 x W/32

        # Decoder Path - Attention-guided feature fusion with progressive upsampling
        # Each decoder block uses attention mechanism to fuse encoder features with upsampled features
        self.up1 = DecoderBlock(1024 + 512, 512)   # 512 + 1024 -> 512, spatial: H/16 x W/16
        self.up2 = DecoderBlock(512 + 256, 256)   # 256 + 512 -> 256,  spatial: H/8 x W/8
        self.up3 = DecoderBlock(256 + 128, 128)   # 128 + 256 -> 128,  spatial: H/4 x W/4  # Fixed: 256 not 512
        self.up4 = DecoderBlock(128 + 64, 64)           # 64 + 128 -> 64,    spatial: H/2 x W/2

        # Final Output Layer - Generate segmentation mask
        self.out = nn.Sequential(
            nn.Conv2d(64, num_class, kernel_size=1, padding=0) ,  # 64 -> 1 channel segmentation mask
            nn.Sigmoid()  # Sigmoid activation for binary segmentation (0-1 range)
        )        

    def forward(self, x):
        """
        Forward pass through the Dual-Branch U-Net.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
        
        Returns:
            torch.Tensor: Segmentation mask of shape (batch_size, 1, height, width)
        
        """
        # Store input size for final output scaling
        input_size = x.shape[2:]
        
        # Encoder Path - Progressive feature extraction and downsampling
        d1, f1 = self.down1(x)    # d1: downsampled (N, 64, H/2, W/2),   f1: skip features (N, 64, H, W)
        d2, f2 = self.down2(d1)   # d2: downsampled (N, 128, H/4, W/4),  f2: skip features (N, 128, H/2, W/2)
        d3, f3 = self.down3(d2)   # d3: downsampled (N, 256, H/8, W/8),  f3: skip features (N, 256, H/4, W/4)
        d4, f4 = self.down4(d3)   # d4: downsampled (N, 512, H/16, W/16), f4: skip features (N, 512, H/8, W/8)

        # Bottleneck - Process deepest features
        b1, _ = self.bottle_neck(d4)  # b1: bottleneck features (N, 1024, H/32, W/32)

        # Decoder Path - Attention-guided upsampling and feature fusion
        u1 = self.up1(f4, b1)    # Fuse f4 (512 ch) + b1 (1024 ch) -> u1 (512 ch, H/16, W/16)
        u2 = self.up2(f3, u1)    # Fuse f3 (256 ch) + u1 (512 ch)  -> u2 (256 ch, H/8, W/8)
        u3 = self.up3(f2, u2)    # Fuse f2 (128 ch) + u2 (256 ch)  -> u3 (128 ch, H/4, W/4)
        u4 = self.up4(f1, u3)    # Fuse f1 (64 ch)  + u3 (128 ch)  -> u4 (64 ch, H/2, W/2)

        # Final segmentation mask generation
        out = self.out(u4)        # Convert to single channel segmentation mask (N, 1, H', W')
        
        # Ensure output matches input size
        if out.shape[2:] != input_size:
            out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=False)

        return out


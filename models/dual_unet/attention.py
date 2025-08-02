"""
Attention Mechanism for Dual-Branch U-Net
"""

import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    """
    Attention mechanism for feature fusion in U-Net decoder.
    """
    
    def __init__(self, in_channels, out_channels):
        """
        Initialize the attention mechanism.
        Args:
            in_channels (int): Number of input channels (from encoder skip connection)
            out_channels (int): Number of output channels after attention processing
        """
        super(Attention, self).__init__()
        self.relu = nn.ReLU()
        self.upsampling = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.out_channels = out_channels
        self.conv1x1 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.flatten = nn.Flatten()
        self.sigmoid = nn.Sigmoid()
        
        # Pre-define alignment layers to avoid creating them during forward pass
        self.x1_align = None
        self.x2_align = None

    def forward(self, x1, x2):
        """
        Apply attention mechanism to fuse two feature maps.
        Args:
            x1 (torch.Tensor): Feature map from encoder (skip connection)
            x2 (torch.Tensor): Feature map from deeper decoder level
        Returns:
            torch.Tensor: Attention-weighted feature map
        """
        upsampled_x2 = self.upsampling(x2)
        if upsampled_x2.shape[2:] != x1.shape[2:]:
            upsampled_x2 = F.interpolate(x2, size=x1.shape[2:], mode='bilinear', align_corners=False)

        # Dynamically align channels if needed
        if x1.shape[1] != upsampled_x2.shape[1]:
            # Create alignment layers if they don't exist
            if self.x1_align is None or self.x1_align.in_channels != x1.shape[1]:
                self.x1_align = nn.Conv2d(x1.shape[1], self.out_channels, kernel_size=1, padding=0).to(x1.device)
            if self.x2_align is None or self.x2_align.in_channels != upsampled_x2.shape[1]:
                self.x2_align = nn.Conv2d(upsampled_x2.shape[1], self.out_channels, kernel_size=1, padding=0).to(upsampled_x2.device)
            
            x1_aligned = self.x1_align(x1)
            x2_aligned = self.x2_align(upsampled_x2)
        else:
            x1_aligned = x1
            x2_aligned = upsampled_x2

        fused = self.relu(x1_aligned + x2_aligned)
        fused = self.conv1x1(fused)
        batch_size, out_channels, H, W = fused.shape
        vector = self.flatten(fused)
        logits = self.sigmoid(vector)
        matrix = logits.view(batch_size, out_channels, H, W)
        out = x2_aligned * matrix
        return out
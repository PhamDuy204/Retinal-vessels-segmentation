"""
Attention Mechanism for Dual-Branch U-Net

This module implements an attention mechanism for selective feature fusion in the
decoder path of the U-Net architecture. The attention mechanism helps the model
focus on relevant features when combining skip connections from the encoder with
upsampled features from deeper decoder levels.

Key Features:
- Dynamic channel alignment for tensors with different channel dimensions
- Spatial dimension matching through interpolation
- Feature-wise attention weighting for improved fusion
"""

import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    """
    Attention mechanism for feature fusion in U-Net decoder.
    
    This attention module combines two feature maps (typically from encoder skip 
    connections and upsampled decoder features) by:
    1. Spatially and channel-wise aligning the input tensors
    2. Computing attention weights based on fused features
    3. Applying attention weights to select relevant features
    
    The attention mechanism is particularly useful when combining features from
    different network depths that may have varying channel dimensions and 
    semantic importance.
    """
    
    def __init__(self, in_channels, out_channels):
        """
        Initialize the attention mechanism.
        
        Args:
            in_channels (int): Number of input channels (from encoder skip connection)
            out_channels (int): Number of output channels after attention processing
            
        Components:
            - relu: ReLU activation for feature fusion
            - upsampling: 2x bilinear upsampling for spatial alignment
            - conv1x1: 1x1 convolution for channel processing
            - flatten: Flatten operation for attention weight computation
            - sigmoid: Sigmoid activation for attention weights (0-1 range)
        """
        super(Attention, self).__init__()

        # Core attention components
        self.relu = nn.ReLU()
        self.upsampling = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) 
        self.conv1x1 = nn.Conv2d(out_channels, out_channels, kernel_size=1) 
        self.flatten = nn.Flatten()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        """
        Apply attention mechanism to fuse two feature maps.
        
        Args:
            x1 (torch.Tensor): Feature map from encoder (skip connection)
                              Shape: (batch_size, channels_1, height, width)
            x2 (torch.Tensor): Feature map from deeper decoder level
                              Shape: (batch_size, channels_2, height//2, width//2)
                              
        Returns:
            torch.Tensor: Attention-weighted feature map
                         Shape: (batch_size, target_channels, height, width)
                         where target_channels = min(channels_1, channels_2)
                         
        Processing Flow:
            1. Upsample x2 to match x1's spatial dimensions
            2. Dynamically align channel dimensions by reducing to smaller dimension
            3. Element-wise addition and ReLU activation for feature fusion
            4. Generate attention weights via flatten -> sigmoid
            5. Reshape attention weights to feature map dimensions
            6. Apply attention weights to upsampled features
            
        Note: Channel alignment uses the smaller dimension to avoid information loss
              and ensure compatibility across different encoder-decoder level combinations.
        """

        # Step 1: Spatial upsampling and alignment
        upsampled_x2 = self.upsampling(x2)
        
        # Ensure exact spatial dimension matching through interpolation if needed
        if upsampled_x2.shape[2:] != x1.shape[2:]:
            upsampled_x2 = F.interpolate(x2, size=x1.shape[2:], mode='bilinear', align_corners=False)

        # Step 2: Dynamic channel alignment
        if x1.shape[1] != upsampled_x2.shape[1]:
            # Align to the smaller channel dimension to prevent information loss
            target_channels = min(x1.shape[1], upsampled_x2.shape[1])
            
            # Create temporary 1x1 conv layers for channel alignment
            if x1.shape[1] != target_channels:
                x1_align = nn.Conv2d(x1.shape[1], target_channels, kernel_size=1).to(x1.device)
                x1 = x1_align(x1)
            
            if upsampled_x2.shape[1] != target_channels:
                x2_align = nn.Conv2d(upsampled_x2.shape[1], target_channels, kernel_size=1).to(upsampled_x2.device)
                upsampled_x2 = x2_align(upsampled_x2)

        # Step 3: Feature fusion and attention weight generation
        fused = self.relu(x1 + upsampled_x2)    # Element-wise addition + activation
        fused = self.conv1x1(fused)             # Channel processing
        batch_size, out_channels, H, W = fused.shape 
        
        # Step 4: Attention weight computation
        vector = self.flatten(fused)            # Flatten to (batch_size, out_channels * H * W)
        logits = self.sigmoid(vector)           # Generate attention weights in [0, 1] range

        # Step 5: Reshape attention weights to feature map dimensions
        matrix = logits.view(batch_size, out_channels, H, W)

        # Step 6: Apply attention weights to upsampled features
        out = upsampled_x2 * matrix             # Element-wise multiplication
        
        return out 
"""
Dual-Branch U-Net Modules

This module contains the core building blocks for the Dual-Branch U-Net architecture:
- EncoderBlock: Dual-branch encoder with depthwise and dynamic convolutions
- DecoderBlock: Attention-guided decoder with dynamic feature fusion

The architecture combines efficient depthwise separable convolutions with adaptive 
dynamic convolutions in the encoder, and uses attention mechanisms in the decoder
for improved feature fusion and segmentation performance.
"""

import torch 
import numpy as np 
import torch.nn as nn 
import torch.nn.functional as F 

from .depthwise import DepthwiseConv
from .dyconv import DynamicConv
from .attention import Attention 

class EncoderBlock(nn.Module):
    """
    Dual-Branch Encoder Block for feature extraction and downsampling.
    
    This block implements a dual-branch architecture that processes input features
    through two parallel paths:
    1. Depthwise Separable Convolution: Efficient spatial feature extraction
    2. Dynamic Convolution: Adaptive kernel selection based on input content
    
    The outputs from both branches are concatenated and fused through a 1x1 convolution,
    followed by downsampling for the next encoder level.
    
    Architecture:
        Input -> [DepthwiseConv, DynamicConv] -> Concatenate -> 1x1 Conv -> Downsample
        
    Returns:
        - Downsampled features for next encoder level
        - Skip connection features for decoder fusion
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, padding='same'):
        """
        Initialize the dual-branch encoder block.
        
        Args:
            in_channels (int): Number of input feature channels
            out_channels (int): Number of output feature channels after dual-branch processing
            kernel_size (int): Convolution kernel size (default: 3)
            padding (str): Padding value - (default: 'same')
            
        Components:
            - depthwise: Depthwise separable convolution (efficient spatial processing)
            - dynamicconv: Dynamic convolution (adaptive kernel selection)
            - conv1x1: 1x1 convolution for channel fusion
            - down: Downsampling convolution (stride=2)
        """
        super(EncoderBlock, self).__init__() 

        # Dual-branch feature extraction
        self.depthwise = DepthwiseConv(in_channels, out_channels, kernel_size, padding=padding, stride=1)
        self.dynamicconv = DynamicConv(in_channels, out_channels, kernel_size, padding=padding)
        # Feature fusion and downsampling
        self.conv1x1 = nn.Conv2d(out_channels*2, out_channels, kernel_size=1, padding='same') # Fusion before downsampling and also is the output of skip connection
        self.down = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),  # decreaseed by 2 
                nn.BatchNorm2d(num_features=out_channels), 
                nn.ReLU()
        )


    def forward(self, X):
        """
        Forward pass through the dual-branch encoder.
        
        Args:
            X (torch.Tensor): Input feature tensor of shape (batch_size, in_channels, height, width)
            
        Returns:
            tuple: (downsampled_features, skip_features)
                - downsampled_features: Processed features for next encoder level
                  Shape: (batch_size, out_channels, height//2, width//2)
                - skip_features: Features for decoder skip connections
                  Shape: (batch_size, out_channels, height, width)
                  
        Processing Flow:
            1. Dual-branch processing:
               - Depthwise: Efficient spatial feature extraction
               - Dynamic: Content-adaptive kernel selection
            2. Spatial alignment if needed (via interpolation)
            3. Channel concatenation of both branches
            4. Feature fusion via 1x1 convolution
            5. Downsampling for next encoder level
        """
        # Dual-branch feature extraction
        depthwise_out = self.depthwise(X)     # Shape: (N, out_channels, H, W)  # Fixed comment
        dyconv_out = self.dynamicconv(X)      # Shape: (N, out_channels, H, W)

                # Ensure H, W of the two feature map 
        if depthwise_out.shape[2:] != dyconv_out.shape[2:]: 
            delta_height = np.abs(depthwise_out.size()[2] - dyconv_out.size()[2])
            delta_width = np.abs(depthwise_out.size()[3] - dyconv_out.size()[3])

            dyconv_out = F.pad(dyconv_out, [delta_width // 2, delta_width - delta_width // 2, 
                            delta_height // 2, delta_height - delta_height // 2])
            
        # Concatenate dual-branch outputs along channel dimension
        combined = torch.cat((depthwise_out, dyconv_out), dim=1)  # Shape: (N, out_channels*2, H, W)
        
        # Fuse concatenated features through 1x1 convolution
        fused = self.conv1x1(combined)        # Shape: (N, out_channels, H, W)

        # Generate downsampled features for next encoder level
        out = self.down(fused)                # Shape: (N, out_channels, H//2, W//2)
        
        return out, fused  # Return (downsampled features, skip connection features)
    


class DecoderBlock(nn.Module):
    
    def __init__(self, skip_channels, deeper_channels, out_channels):
        super(DecoderBlock, self).__init__()
        
        self.out_channels = out_channels 
        self.batchnorm = nn.BatchNorm2d(num_features=skip_channels)  # Fixed: should match skip connection channels
        self.relu = nn.ReLU()

        self.attention = Attention(skip_channels, deeper_channels, out_channels)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x1, x2):
        """
        Forward pass through the attention-guided decoder.
        
        Args:
            x1 (torch.Tensor): Encoder skip connection features
                              Shape: (batch_size, skip_channels, height, width)
            x2 (torch.Tensor): Features from deeper decoder level
                              Shape: (batch_size, deeper_channels, height//2, width//2)
            
        Returns:
            torch.Tensor: Fused and processed features
                         Shape: (batch_size, out_channels, height, width)
        """
        
        # Apply batch normalization to skip connection features
        x1_norm = self.batchnorm(x1)
        
        # Apply attention mechanism for feature selection and fusion
        attn_features = self.attention(x1_norm, x2)
        upsampled_x2 = self.upsample(x2)
        
        # Ensure spatial dimensions match before concatenation
        if upsampled_x2.shape[2:] != attn_features.shape[2:]:
            upsampled_x2 = F.interpolate(upsampled_x2, size=attn_features.shape[2:], 
                                        mode='bilinear', align_corners=False)

        concat = torch.cat([attn_features, upsampled_x2], dim=1)
        
        # Create convolution layers dynamically based on actual concatenated dimensions
        if not hasattr(self, 'conv_procedure') or self.conv_procedure[0].in_channels != concat.shape[1]:
            self.conv_procedure = nn.Sequential(
                nn.Conv2d(concat.shape[1], self.out_channels, kernel_size=3, padding=1), 
                nn.BatchNorm2d(num_features=self.out_channels), 
                nn.ReLU(inplace=True)
            ).to(concat.device)

        out = self.conv_procedure(concat)

        return out
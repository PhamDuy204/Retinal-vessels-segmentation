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
            padding (str): Padding type - 'same' or 'valid' (default: 'same')
            
        Components:
            - depthwise: Depthwise separable convolution (efficient spatial processing)
            - dynamicconv: Dynamic convolution (adaptive kernel selection)
            - conv1x1: 1x1 convolution for channel fusion
            - down: Downsampling convolution (stride=2)
        """
        super(EncoderBlock, self).__init__() 

        # Dual-branch feature extraction
        self.depthwise = DepthwiseConv(in_channels, kernel_size, padding='same', stride=1)
        self.dynamicconv = DynamicConv(in_channels, out_channels, kernel_size, padding='same')
        
        # Feature fusion and downsampling
        self.conv1x1 = nn.Conv2d(in_channels + out_channels, out_channels, kernel_size=1, padding=0)
        self.down = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

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
        depthwise_out = self.depthwise(X)     # Shape: (N, in_channels, H, W)
        dyconv_out = self.dynamicconv(X)      # Shape: (N, out_channels, H, W)
        
        # Ensure spatial dimensions match before concatenation
        if depthwise_out.shape[2:] != dyconv_out.shape[2:]:
            depthwise_out = F.interpolate(depthwise_out, size=dyconv_out.shape[2:], 
                                        mode='bilinear', align_corners=False)

        # Concatenate dual-branch outputs along channel dimension
        combined = torch.cat((depthwise_out, dyconv_out), dim=1)  # Shape: (N, in_channels + out_channels, H, W)
        
        # Fuse concatenated features through 1x1 convolution
        fused = self.conv1x1(combined)        # Shape: (N, out_channels, H, W)

        # Generate downsampled features for next encoder level
        out = self.down(fused)                # Shape: (N, out_channels, H//2, W//2)
        
        return out, fused  # (downsampled, skip_connection) 
    


class DecoderBlock(nn.Module):
    """
    Attention-Guided Decoder Block for feature upsampling and fusion.
    
    This block implements an attention-guided decoder that combines:
    1. Skip connection features from the corresponding encoder level
    2. Upsampled features from the deeper decoder level
    
    The attention mechanism helps the model focus on relevant features during
    the fusion process, improving segmentation accuracy.
    
    Architecture:
        encoder_features + upsampled_features -> Attention -> Concatenate -> Conv -> Output
        
    Key Features:
        - Attention-based feature selection
        - Dynamic convolution layer creation based on actual tensor dimensions
        - Spatial and channel dimension alignment
    """
    
    def __init__(self, in_channels, out_channels):
        """
        Initialize the attention-guided decoder block.
        
        Args:
            in_channels (int): Number of input channels from encoder skip connection
            out_channels (int): Number of output channels after processing
            
        Components:
            - batchnorm: Batch normalization for encoder features
            - attention: Attention mechanism for feature selection
            - upsample: Bilinear upsampling (scale_factor=2)
            - conv_procedure: Dynamically created based on concatenated feature dimensions
        """
        super(DecoderBlock, self).__init__()

        self.batchnorm = nn.BatchNorm2d(num_features=in_channels)
        self.attention = Attention(in_channels, out_channels)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.out_channels = out_channels

        # Note: conv_procedure is created dynamically in forward() based on actual tensor dimensions

    def forward(self, x1, x2):
        """
        Forward pass through the attention-guided decoder.
        
        Args:
            x1 (torch.Tensor): Encoder skip connection features
                              Shape: (batch_size, in_channels, height, width)
            x2 (torch.Tensor): Features from deeper decoder level
                              Shape: (batch_size, channels, height//2, width//2)
                              
        Returns:
            torch.Tensor: Fused and processed features
                         Shape: (batch_size, out_channels, height, width)
                         
        Processing Flow:
            1. Upsample x2 to match x1's spatial dimensions
            2. Apply attention mechanism to select relevant features
            3. Ensure spatial dimension alignment between attention output and upsampled features
            4. Concatenate attention-weighted features with upsampled features
            5. Process through dynamically created convolution layers
            
        Note: The convolution layers are created dynamically to handle varying 
              channel dimensions that arise from the attention mechanism and 
              different encoder-decoder level combinations.
        """

        # Upsample deeper features to match encoder feature spatial dimensions
        upsampled_x2 = self.upsample(x2)

        # Apply attention mechanism for feature selection and fusion
        # Attention takes original x2 and handles its own upsampling internally
        attn_map = self.attention(x1, x2)

        # Ensure spatial dimension alignment between attention output and upsampled features
        if upsampled_x2.shape[2:] != attn_map.shape[2:]:
            upsampled_x2 = F.interpolate(upsampled_x2, size=attn_map.shape[2:], 
                                        mode='bilinear', align_corners=False)

        # Concatenate attention-weighted features with upsampled features
        concat = torch.cat([attn_map, upsampled_x2], dim=1)

        # Create convolution layers dynamically based on actual concatenated dimensions
        # This flexibility is needed because attention mechanism may change channel dimensions
        if not hasattr(self, 'conv_procedure') or self.conv_procedure[0].in_channels != concat.shape[1]:
            self.conv_procedure = nn.Sequential(
                nn.Conv2d(concat.shape[1], self.out_channels, kernel_size=3, padding=1), 
                nn.BatchNorm2d(num_features=self.out_channels), 
                nn.ReLU(inplace=True)
            ).to(concat.device)

        # Process concatenated features through convolution layers
        out = self.conv_procedure(concat)

        return out




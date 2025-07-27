"""
Depthwise Separable Convolution Module

This module implements depthwise separable convolution, which is a computationally
efficient alternative to standard convolution. It performs spatial convolution
independently for each input channel, reducing parameters and computational cost
while maintaining feature extraction capability.

Depthwise separable convolution is particularly useful in mobile and efficient
architectures, and serves as one branch of the dual-branch encoder in this U-Net.
"""

import torch.nn as nn 


class DepthwiseConv(nn.Module):
    """
    Depthwise Separable Convolution Layer.
    
    Performs spatial convolution independently for each input channel using grouped
    convolution with groups=in_channels. This reduces parameters from:
    - Standard conv: in_channels × out_channels × kernel_size²
    - Depthwise conv: in_channels × kernel_size² (when out_channels = in_channels)
    
    This implementation maintains the input channel count (out_channels = in_channels)
    and focuses on spatial feature extraction rather than channel mixing.
    
    Benefits:
    - Reduced computational cost and memory usage
    - Fewer parameters, reducing overfitting risk
    - Efficient spatial feature extraction
    - Complementary to dynamic convolution in dual-branch architecture
    """
    
    def __init__(self, in_channels, kernel_size, padding='same', stride=1, bias=False):
        """
        Initialize depthwise separable convolution.
        
        Args:
            in_channels (int): Number of input channels (also determines output channels)
            kernel_size (int): Size of the convolution kernel
            padding (str or int): Padding strategy - 'same' for same output size, 
                                 'valid' for no padding, or integer for specific padding
            stride (int): Convolution stride (default: 1)
            bias (bool): Whether to include bias parameters (default: False)
            
        Architecture:
            Input -> Depthwise Conv (groups=in_channels) -> Output
            
        Note: Output channels equal input channels since groups=in_channels.
              For channel mixing, this would typically be followed by a pointwise
              (1x1) convolution, but in this dual-branch architecture, channel
              mixing is handled by the fusion layers in EncoderBlock.
        """
        super(DepthwiseConv, self).__init__()

        self.depthwise = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=in_channels,  # Same as input channels for pure depthwise operation
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            groups=in_channels,        # Key parameter: makes it depthwise
            bias=bias
        )

    def forward(self, X):
        """
        Forward pass through depthwise convolution.
        
        Args:
            X (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, in_channels, height', width')
                         where height' and width' depend on kernel_size, stride, and padding
                         
        Processing:
            Applies spatial convolution independently to each channel using grouped
            convolution. Each of the 'in_channels' groups processes one input channel
            with its own set of kernel weights.
        """
        return self.depthwise(X) 
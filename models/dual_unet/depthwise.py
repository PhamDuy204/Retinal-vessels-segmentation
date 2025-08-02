"""
Depthwise Separable Convolution Module
"""

import torch.nn as nn 

class DepthwiseConv(nn.Module):
    """
    Depthwise Separable Convolution Layer.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, padding='same', stride=1, bias=False):
        """
        Initialize depthwise separable convolution.
        
        Args:
            in_channels (int): Number of input channels (also determines output channels of depthwise)
            out_channels (int): Number of output channels of block 
            kernel_size (int): Size of the convolution kernel
            padding (str or int): Padding strategy - 'same' for same output size, 
                                 'valid' for no padding, or integer for specific padding
            stride (int): Convolution stride (default: 1)
            bias (bool): Whether to include bias parameters (default: False)
        """
        super(DepthwiseConv, self).__init__()

        self.depthwise = nn.Conv2d(
            in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, 
            stride=stride, padding=kernel_size//2, groups=in_channels, bias=bias
        )

        self.pointwise = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, 
            kernel_size=1, padding=0, bias=bias 
        )

    def forward(self, X):
        """
        Forward pass through depthwise convolution.
        Args:
            X (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height', width')
                         where height' and width' depend on kernel_size, stride, and padding
        """
        out = self.depthwise(X) 
        out = self.pointwise(out)  # Fix: Apply pointwise to depthwise output

        return out 

import torch 
import numpy as np 
import torch.nn as nn 
import torch.nn.functional as F 

from .depthwise import DepthwiseConv
from .dyconv import DynamicConv
from .attention import Attention 

class EncoderBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size=3, padding='same'):
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
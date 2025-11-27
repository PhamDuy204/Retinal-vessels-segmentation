import torch 
import torch.nn as nn 

import torch.nn.functional as F
from models.depthunet.depthwise import DepthwiseConv


class DoubleConv(nn.Module): 
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__() 
        if not mid_channels: 
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            DepthwiseConv(in_channels, mid_channels, kernel_size=3, padding='same'), 
            nn.BatchNorm2d(mid_channels), 
            nn.ReLU(inplace=True), 
            DepthwiseConv(mid_channels, out_channels, kernel_size=3, padding='same'), 
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(inplace=True)
        ) 
    
    def forward(self, X): 
        return self.double_conv(X) 


class DownScaling(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.downscaling = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channel, out_channel)
        )

    def forward(self, x):
        return self.downscaling(x)

class UpScaling(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.upscaling = nn.ConvTranspose2d(in_channel, in_channel // 2, kernel_size=2, stride=2)
        self.double_conv = DoubleConv(in_channel, out_channel)

    def forward(self, x1, x2): # x1 from ConvTransposed, x2 from Encoder
        x1 = self.upscaling(x1)

        delta_height = x2.size()[2] - x1.size()[2]
        delta_width = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [delta_width // 2, delta_width - delta_width // 2,
                        delta_height // 2, delta_height - delta_height // 2])

        x = torch.cat([x2, x1], dim=1)

        return self.double_conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
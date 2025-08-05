import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from .depthwise import Depthwise
from .garbor import GaborConv


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels): 
        super(DoubleConv, self).__init__() 
        self.depthwise1 = Depthwise(in_channels=in_channels, out_channels=out_channels, kernel_size=3)
        self.depthwise2 = Depthwise(in_channels=out_channels, out_channels=out_channels, kernel_size=3)
        self.batch_norm = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU() 

        self.garbor = GaborConv(in_channels=out_channels, out_channels=out_channels, kernel_size=3)
    
    def forward(self, X): 
        x = self.depthwise1(X) 
        x = self.depthwise2(x) 
        x = self.batch_norm(x) 
        skip = self.relu(x) 
        out = self.garbor(skip)

        return skip, out 


class DownSampling(nn.Module): 
    def __init__(self, in_channels, out_channels):
        super(DownSampling, self).__init__() 
        self.doubleconv = DoubleConv(in_channels=in_channels,out_channels=out_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, X): 
        skip, out = self.doubleconv(X)
        out = self.maxpool(out) 

        return skip, out 


class UpSampling(nn.Module): 
    def __init__(self, in_channels, out_channels,): 
        super(UpSampling, self).__init__() 
        self.upsample = nn.Upsample(scale_factor=2)
        self.doubleconv = DoubleConv(in_channels=in_channels, out_channels=out_channels)
    
    def forward(self, skip, X): 
        X = self.upsample(X)
        if skip.shape[2:] != X.shape[2:]:
            # Resize X to match skip connection size
            X = F.interpolate(X, size=skip.shape[2:], mode="bilinear", align_corners=False)

        fuse = torch.cat([skip, X], dim=1)
        _, out = self.doubleconv(fuse)

        return out
    
    
class OutConv(nn.Module): 
    def __init__(self, in_channels, out_channels): 
        super(OutConv, self).__init__() 
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
    
    def forward(self, X): 
        return self.conv(X)
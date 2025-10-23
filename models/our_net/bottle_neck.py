import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
#from mamba_ssm import Mamba2
from modules import SA, CA
# --- helpers ---
def _same_padding(kernel_size, dilation=1):
    k = kernel_size
    return (dilation * (k - 1)) // 2

def safe_group(channels: int, preferred: int = 8) -> int:
    pref = min(preferred, channels)
    for g in range(pref, 0, -1):
        if channels % g == 0:
            return g
    return 1

def get_rmsnorm(dim: int):
    if hasattr(nn, "RMSNorm"):
        return nn.RMSNorm(dim)
    else:
        return nn.LayerNorm(dim)



class TSB(nn.Module): 
    def __init__(self, in_channels, out_channels): 
        super(TSB, self).__init__() 
        
        # Layer 1 
        self.conv11 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, 
                               groups=in_channels, kernel_size=3, padding='same', dilation=1, bias=False)
        self.conv12 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, 
                                groups=in_channels, kernel_size=3, padding='same', dilation=2, bias=False)
        self.conv13 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, 
                                groups=in_channels, kernel_size=3, padding='same', dilation=3, bias=False)
        
        # Layer 2
        self.conv21 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                                kernel_size=1, padding='same', bias=False)
        self.conv22 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                                kernel_size=1, padding='same', bias=False)
        self.conv23 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=1, padding='same', bias=False)
        
        # Layer 3
        self.conv31 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                                kernel_size=1, padding='same', bias=False)
        self.conv32 = nn.Conv2d(in_channels=out_channels*3, out_channels=out_channels, 
                                kernel_size=1, padding='same', bias=False)
           
    def forward(self, X): 
        x11 = self.conv11(X) 
        x12 = self.conv12(X) 
        x13 = self.conv13(X) 

        x21 = self.conv21(x11)
        x22 = self.conv22(x12) 
        x23 = self.conv23(x13) 

        x22 = x22 + x21 
        x23 = x23 + x22 

        x31 = self.conv31(X) 
        x32 = torch.concat([x21, x22, x23], dim=1)
        x32 = self.conv32(x32)

        out = x31 + x32

        return out  

class VGG(nn.Module): 
    def __init__(self, in_channels, out_channels): 
        super(VGG, self).__init__() 
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, 
                               kernel_size=3, padding='same', bias=False, dilation=1)
        self.gelu1 = nn.GELU()

        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                               kernel_size=3, padding='same', bias=False, dilation=2) 
        self.gelu2 = nn.ReLU() 

    def forward(self, X): 
        x = self.conv1(X) 
        x = self.gelu1(x) 
        x = self.conv2(x) 
        x = self.gelu2(x) 
        return x

class ResNet(nn.Module): 
    def __init__(self, in_channels, out_channels): 
        super(ResNet, self).__init__() 
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, 
                               kernel_size=3, padding='same', dilation=1, bias=False) 
        
        self.gelu1 = nn.GELU()
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                               kernel_size=3, padding='same', dilation=2, bias=False) 
                               
        self.gelu2 = nn.ReLU() 

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding='same', bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, X): 
        x = self.conv1(X) 
        x = self.gelu1(x) 
        x = self.conv2(x) 

        shortcut = self.shortcut(X)
        x = x + shortcut

        x = self.gelu2(x) 
        return x

class BottleNeck(nn.Module): 
    def __init__(self, in_channels, out_channels): 
        super(BottleNeck, self).__init__() 
        self.branch_1 = nn.Sequential(
            VGG(in_channels, in_channels*2), 
            SA(), 
        )

        self.branch_2 = nn.Sequential(
            ResNet(in_channels, in_channels*2), 
            CA(in_channels*2), 
        )

        self.tsb = TSB(in_channels*2, out_channels) 
        self.gn = nn.GroupNorm(num_channels=out_channels, 
                               num_groups=out_channels, affine=False)

    def forward(self, X): 
        x_1 = self.branch_1(X)
        x_2 = self.branch_2(X) 

        fusion = x_1 + x_2 - x_1 * x_2 

        out = self.tsb(fusion) 
        out = self.gn(out) 

        return out  
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

class EFB(nn.Module):
    def __init__(self, in_channels,out_channels):
        super().__init__()
        self.norm_x=nn.GroupNorm(1,in_channels)
        self.dw_0=nn.Conv2d(in_channels,in_channels,3,bias=False,groups=in_channels)
        self.dw_1=nn.Conv2d(in_channels,in_channels,3,bias=False,groups=in_channels)
        self.conv=nn.Conv2d(in_channels,in_channels,3,bias=False)
        self.ff=nn.Sequential(nn.Conv2d(in_channels,2*in_channels,1,bias=False),
                              nn.GroupNorm(1,2*in_channels),
                              nn.GELU(),
                              nn.Conv2d(2*in_channels,in_channels,1,bias=False))
        self.out=nn.Conv2d(in_channels,out_channels,1,bias=False)
    def forward(self,x):
        x=self.norm_x(x)
        dw_0=self.dw_0(x)
        dw_1=self.dw_1(x)
        conv=self.conv(x)
        attn=F.sigmoid(dw_0)*dw_1
        out = conv+attn
        return self.out(self.ff(out)+out)
        
class TSB(nn.Module): 
    def __init__(self, in_channels, out_channels): 
        super(TSB, self).__init__() 
        
        # Layer 1 
        self.conv11 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, 
                               groups=in_channels, kernel_size=3, padding='same')
        self.conv12 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, 
                                groups=in_channels, kernel_size=5, padding='same')
        self.conv13 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, 
                                groups=in_channels, kernel_size=7, padding='same')
        
        # Layer 2
        self.conv21 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                                kernel_size=1, padding='same')
        self.conv22 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                                kernel_size=1, padding='same')
        self.conv23 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=1, padding='same')
        
        # Layer 3
        self.conv31 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                                kernel_size=1, padding='same')
        self.conv32 = nn.Conv2d(in_channels=out_channels*3, out_channels=out_channels, 
                                kernel_size=1, padding='same')
        

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



class CustomBottleNeck(nn.Module): 
    def __init__(self, in_channels, out_channels):
        super(CustomBottleNeck, self).__init__() 

        self.conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, 
                                groups=in_channels, kernel_size=3, padding='same')
        self.sigmoid_1 = nn.Sigmoid() 

        self.conv_2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, 
                                groups=in_channels, kernel_size=3, padding='same')
        self.sigmoid_2 = nn.Sigmoid() 

        self.conv_mid = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, 
                                  groups=in_channels, kernel_size=5, padding='same')
        
        self.groupnorm = nn.GroupNorm(num_groups=in_channels, num_channels=in_channels, 
                                      affine=False)
        
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, 
                              groups=in_channels, kernel_size=3, padding='same') 
        
        self.tsb = TSB(in_channels=in_channels, out_channels=out_channels)

    def forward(self, X): 
        x_1 = self.conv_1(X)
        x_1 = self.sigmoid_1(x_1) 

        x_2 = self.conv_2(X) 
        x_2 = self.sigmoid_2(x_2)

        x_mid = self.conv_mid(X) 

        x = x_1 + x_mid + x_2
        x = self.groupnorm(x) 
        x = self.conv(x) 

        x = x + X 

        out = self.tsb(x) 

        return out 

class Depthwise(nn.Module): 
    def __init__(self, in_channels, out_channels, kernel_size): 
        super(Depthwise, self).__init__() 
        self.depthwise = nn.Conv2d( in_channels=in_channels, out_channels=in_channels, 
            groups=in_channels, kernel_size=kernel_size, padding='same'
        )
        self.group_norm = nn.GroupNorm(num_groups=in_channels, num_channels=in_channels, affine=False)
        self.pointwise = nn.Conv2d( in_channels=in_channels, out_channels=out_channels, 
            kernel_size=1, padding='same'
        )
        
    def forward(self, X): 
        x = self.depthwise(X) 
        x = self.group_norm(x) 
        x = self.pointwise(x) 
        return x 


class VGG(nn.Module): 
    def __init__(self, in_channels, out_channels): 
        super(VGG, self).__init__() 
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3)
        self.gelu1 = nn.GELU()
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=5)
        self.gelu2 = nn.GELU() 

    def forward(self, X): 
        x = self.conv1(X) 
        x = self.gelu1(x) 
        x = self.conv2(x) 
        x = self.gelu2(x) 
        return x 


class ResNet(nn.Module): 
    def __init__(self, in_channels, out_channels): 
        super(ResNet, self).__init__() 
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3) 
        self.gelu1 = nn.GELU()
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=5)
        self.gelu2 = nn.GELU() 

        # projection layer nếu số kênh thay đổi
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding='same')
        else:
            self.shortcut = nn.Identity()

    def forward(self, X): 
        x = self.conv1(X) 
        x = self.gelu1(x) 
        x = self.conv2(x) 

        shortcut = self.shortcut(X)   # đảm bảo cùng số kênh
        x = x + shortcut

        x = self.gelu2(x) 
        return x 


class CustomBottleNeck1(nn.Module): 
    def __init__(self, in_channels, out_channels): 
        super(CustomBottleNeck1, self).__init__() 
        self.vgg = VGG(in_channels=in_channels, out_channels=in_channels*2)
        self.resnet = ResNet(in_channels=in_channels, out_channels=in_channels*2)
        self.gelu = nn.GELU() 
        self.conv1 = nn.Conv2d(in_channels=in_channels*2, out_channels=out_channels, kernel_size=1, padding='same')

    def forward(self, X): 
        x1 = self.vgg(X) 
        x2 = self.resnet(X) 
        x = self.gelu(x1 + x2 - x1 * x2)
        out = self.conv1(x) 
        return out
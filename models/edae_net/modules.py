import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,bias=False)
        self.bn = nn.BatchNorm2d(out_channels,affine=False)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            ConvBlock(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            ConvBlock(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
    def forward(self, x):
        x = self.conv(x)
        return x

class downsampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(downsampling, self).__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
    def forward(self, x):
        x = self.conv(x)
        return x,self.max_pool(x)


class AtrousConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super(AtrousConv, self).__init__()
        self.atrous_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation,bias=False)
        self.bn = nn.BatchNorm2d(out_channels,affine=False)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class MAC_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MAC_Block, self).__init__()
        self.atrous_conv1 = AtrousConv(in_channels, out_channels, dilation=1)
        self.atrous_conv2 = AtrousConv(in_channels, out_channels, dilation=3)
        self.atrous_conv3 = AtrousConv(in_channels, out_channels, dilation=5)
    def forward(self, x):
        x1 = self.atrous_conv1(x)
        x2 = self.atrous_conv2(x)
        x3 = self.atrous_conv3(x)
        x = x1 + x2 + x3
        return x


class DAC_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DAC_Block, self).__init__()
        self.mac_1 = MAC_Block(in_channels, out_channels)
        self.mac_2 = MAC_Block(in_channels, out_channels)
        self.mac_3 = MAC_Block(in_channels, out_channels)
    def forward(self, x):
        x = self.mac_1(x)+x
        x = self.mac_2(x)+x
        x = self.mac_3(x)+x
        return x

class CPSE_Block(nn.Module):
    def __init__(self, in_channels):
        super(CPSE_Block, self).__init__()
        self.h_pool = nn.MaxPool1d(3,1,1)
        self.v_pool = nn.MaxPool1d(3,1,1)
        self.dac = DAC_Block(in_channels, in_channels)
    def forward(self, x):
        # x : B C H W
        b,c,h,w = x.size()
        x_h = self.h_pool(x.flatten(1,-2)).view(b,c,h,w)  # B CH W
        x_v = self.v_pool(x.transpose(-1,-2).flatten(1,-2)).view(b,c,w,h).transpose(-1,-2).contiguous()   # B CW H 
        m = torch.where(x_h>x_v,x_h,x_v)
        return self.dac(m)

class MDAE_Block(nn.Module):
    def __init__(self, in_channels,in_size=(64,64)):
        super(MDAE_Block, self).__init__()
        h,w = in_size
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv0 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(h, h, kernel_size=1, stride=1, padding=0,bias=False),
            nn.Sigmoid())
        self.conv1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(w, w, kernel_size=1, stride=1, padding=0,bias=False),
            nn.Sigmoid())
        self.conv2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0,bias=False),
            nn.Sigmoid())
    def forward(self, x):
        x_0 =  x.permute(0,2,1,3).contiguous()  # B H C W
        x_1 =  x.permute(0,3,2,1).contiguous() # B W H C

        x_0 = (x_0*self.conv0(x_0)).permute(0,2,1,3).contiguous()
        x_1 = (x_1*self.conv1(x_1)).permute(0,3,2,1).contiguous()
        x = x*self.conv2(x)
        return x + x_0 + x_1

class DGF(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DGF, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2)
        self.conv_0 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1,bias=False)
        self.conv_1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1,bias=False)
        self.compute_w = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2*out_channels, out_channels, kernel_size=1, stride=1, padding=0,bias=False),
            nn.Sigmoid())
        self.out = nn.Conv2d(2*out_channels, out_channels, kernel_size=1, stride=1, padding=0,bias=False)
        
    def forward(self, hfm, lfm):
        hfm=self.upsample(hfm)
        hfm=self.conv_0(hfm)
        lfm=self.conv_1(lfm)
        w = self.compute_w(torch.cat([hfm,lfm],dim=1))*lfm+lfm
        return self.out(torch.cat([hfm,w],dim=1))

class upsampling(nn.Module):
    def __init__(self, in_channels, out_channels,up_final=1):
        super(upsampling, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2)
        self.conv = DoubleConv(in_channels, out_channels)
        self.upfinal = nn.Sequential(
            nn.Upsample(scale_factor=up_final),
            nn.Conv2d(out_channels, 1, 1,bias=False),
            )
    def forward(self, x,mdae):
        x = self.upsample(x)
        m = torch.cat([x,mdae],dim=1)
        c =self.conv(m)
        return c,self.upfinal(c)

class awl(nn.Module):
    def __init__(self,in_channels):
        super(awl, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0,bias=False),
            nn.Sigmoid())
    def forward(self, x):
        gap = self.gap(x)
        gmp = self.gmp(x)
        sum_ = gap + gmp
        y = self.conv(sum_)
        return y * x
    
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from bottle_neck import *

class conv_func(nn.Module):
    def __init__(self,in_channels, out_channels,kernel_size,stride=1,padding:str|int=0,dilation=1,with_activation=True):
        super().__init__()
        self.out= nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,dilation,bias=False,padding_mode='zeros'),
            nn.GroupNorm(out_channels,out_channels),
            nn.ReLU() if with_activation else nn.Identity(),
            )
    def forward(self,x):
        return self.out(x)
    
class residual(nn.Module):
    def __init__(self,in_channels, out_channels,kernel_size,stride=1,padding:str|int=0,dilation=1) -> None:
        super().__init__()
        self.id_=nn.Conv2d(in_channels,out_channels,1,bias=False)
        self.attn=nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels,max(in_channels//2,1),1,bias=False),
            nn.ReLU(),
            nn.Conv2d(max(in_channels//2,1),out_channels,1,bias=False),
            nn.Sigmoid()
        )
        self.convs= nn.Sequential(
            conv_func(in_channels,out_channels,kernel_size,stride,padding,dilation,with_activation=False),
            conv_func(out_channels,out_channels,kernel_size,stride,padding,dilation)
            )
    def forward(self,x):
        id_=self.id_(x)
        attn=self.attn(x)
        return (self.convs(x)+id_)*attn

class down_sampling(nn.Module):
    def __init__(self,in_channels, out_channels):
        super().__init__()
        self.res=nn.Sequential(
            conv_func(in_channels,out_channels,3,padding='same',with_activation=False),
            residual(out_channels, out_channels,3,padding='same'),)
        self.pool=nn.MaxPool2d(2)
    def forward(self,x):
        res = self.res(x)
        return res,self.pool(res)
    
class MSB(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.b_0=conv_func(in_channels,in_channels,1,padding='same',with_activation=False)
        self.b_1=conv_func(in_channels,in_channels,3,padding='same',with_activation=False)
        self.b_2=conv_func(in_channels,in_channels,5,padding='same',with_activation=False)

        self.b_3=conv_func(in_channels,in_channels,3,dilation=2,padding='same',with_activation=False)
        self.b_4=conv_func(in_channels,in_channels,3,dilation=5,padding='same',with_activation=False)
        self.b_5=conv_func(in_channels,in_channels,3,dilation=7,padding='same',with_activation=False)
        self.merge=nn.Sequential(
            nn.Conv2d(7*in_channels,in_channels,1,bias=False),
        )
    def forward(self,x):
        b_0=self.b_0(x)
        b_1=self.b_1(x)
        b_2=self.b_2(x)
        b_3=self.b_3(x)
        b_4=self.b_4(x)
        b_5=self.b_5(x)
        res = self.merge(torch.cat((x,b_0,b_1,b_2,b_3,b_4,b_5),1))
        return res
class up_sampling(nn.Module):
    def __init__(self,in_channels, out_channels):
        super().__init__()
        self.msbs=nn.Sequential(
            MSB(out_channels)
        )
            # BottleNeck(out_channels),
            # residual(out_channels,out_channels,3,padding='same'),)
        self.horizontal_conv=nn.Conv2d(out_channels,out_channels,(1,3),padding='same',bias=False)
        self.vertical_conv=nn.Conv2d(out_channels,out_channels,(3,1),padding='same',bias=False)
        self.affine_shape = nn.Conv2d(in_channels,out_channels,1,bias=False)
        self.up_conv=nn.ConvTranspose2d(out_channels,out_channels,2,2,bias=False)
        self.affine=nn.Sequential(
            nn.Conv2d(out_channels,out_channels,3,padding='same',bias=False,groups=out_channels),
            nn.Conv2d(out_channels,out_channels,1,bias=False))
        # self.msb=MSB(out_channels)
        self.res =nn.Sequential(
            nn.Conv2d(2*out_channels,2*out_channels,3,padding='same',bias=False,groups=2*out_channels),
            nn.Conv2d(2*out_channels,out_channels,1,bias=False))
        self.GAP=nn.AdaptiveAvgPool2d(1)
        self.attn=nn.Sequential(
            nn.Conv2d(2*out_channels,out_channels,1,bias=False),
            nn.GELU(),
            nn.Conv2d(out_channels,out_channels,1,bias=False),
            nn.Sigmoid()
        )
        # self.up_sample=nn.Upsample(scale_factor=2,align_corners=True,mode='bilinear')
    def forward(self,predown,down,cur):
        '''
        predown: out_channels,h,w
        down: out_channels,h/2,w/2
        cur: in_channels,h/2,w/2
        '''
        cur=self.horizontal_conv(self.affine_shape(cur))
        down=self.vertical_conv(down)
        high_f=self.up_conv(self.msbs(cur+down))
        low_f=self.affine(predown)
        cat_f=torch.cat((high_f,low_f),1)
        attn=self.attn(self.GAP(cat_f))*low_f+low_f
        return  self.res(torch.cat((attn,high_f),1))


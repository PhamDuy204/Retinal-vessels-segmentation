import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from bottle_neck import *
from typing import Optional


def _same_padding(kernel_size, dilation=1):
    k = kernel_size
    return (dilation * (k - 1)) // 2

def safe_group(channels: int, preferred: int = 8) -> int:
    pref = min(preferred, channels)
    for g in range(pref, 0, -1):
        if channels % g == 0:
            return g
    return 1
    

class ConvFunc(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=1, padding: Optional[int]='same', dilation=1):
        super().__init__()
        if padding == 'same':
            pad = _same_padding(kernel_size, dilation)
        else:
            pad = padding
        groups = in_channels

        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride, pad, dilation, bias=False)

        self.gn = nn.GroupNorm(8,in_channels, affine=False)
        self.act = nn.ReLU()
        self.merge=nn.Conv2d(2*in_channels, in_channels, 3,padding='same',bias=False)
    def forward(self, x):
        y = self.conv(x)
        y = self.gn(y)
        return self.act(self.merge(torch.cat((x,y),1)))  

class MKIR(nn.Module):
    def __init__(self, in_channels, out_channels, in_size=(64,64)): # them nhieu6 kernel size 
        super().__init__()
        # project in -> out (1x1)
        self.first_conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)

        # three dilation branches (depthwise separable inside ConvFunc)
        self.b1 = nn.Sequential(
            ConvFunc(out_channels, kernel_size=3, dilation=1),
            nn.GroupNorm(8,out_channels, affine=False),
            nn.ReLU()
        )
        self.b2 = nn.Sequential(
            ConvFunc(out_channels, kernel_size=3, dilation=2),
            nn.GroupNorm(8,out_channels, affine=False),
            nn.ReLU()
        )
        self.b3 = nn.Sequential(
            ConvFunc(out_channels, kernel_size=3, dilation=3),
            nn.GroupNorm(8,out_channels, affine=False),
            nn.ReLU()
        )
        self.conv_1 = nn.Sequential(
            nn.Conv2d(4*out_channels, 4*out_channels, 3, padding=_same_padding(3),groups=4*out_channels, bias=False),
            nn.Conv2d(4*out_channels, out_channels, 1, bias=False),
            nn.ReLU()
        )
        # fusion
        self.out = nn.Sequential(
            nn.Conv2d(2 * out_channels, out_channels, 3, padding=_same_padding(3), bias=False),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.first_conv(x)
        # print(x.shape)
        b1 = self.b1(x)
        b2 = self.b2(x)
        b3 = self.b3(x)
        merged = self.conv_1(torch.cat((b1,b2,b3,x),1)) 
        # print(merged.shape)
        return self.out(torch.cat((merged*2, x), dim=1))
        

class CA(nn.Module):
    def __init__(self, channels, reduction_rate=4):
        super().__init__()
        hidden = max(channels // reduction_rate, 4)
        self.squeeze = nn.ModuleList([
            nn.AdaptiveAvgPool2d(1),
            nn.AdaptiveMaxPool2d(1)
        ])
        self.excitation = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_feat = self.squeeze[0](x)
        max_feat = self.squeeze[1](x)
        avg_out = self.excitation(avg_feat)
        max_out = self.excitation(max_feat)
        attention = self.sigmoid(avg_out + max_out)
        return attention * x
    
class SA(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        pad = _same_padding(kernel_size)
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=pad, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_feat = torch.mean(x, dim=1, keepdim=True)
        max_feat, _ = torch.max(x, dim=1, keepdim=True)
        feat = torch.cat([avg_feat, max_feat], dim=1)
        out_feat = self.conv(feat)
        attention = self.sigmoid(out_feat)
        return attention * x


class AG(nn.Module):
    def __init__(self,in_channels,in_size=(64,64)):
        super().__init__()
        self.gconv_x_u = nn.Sequential(
            nn.Conv2d(in_channels,in_channels,3,groups=in_channels,padding='same',bias=False),
            nn.ReLU(),)
        self.gconv_x_e = nn.Sequential(nn.Conv2d(in_channels,in_channels,3,groups=in_channels,padding='same',bias=False),
                                       nn.ReLU(),)
        self.out =  nn.Sequential(
            nn.Conv2d(2*in_channels,in_channels,1,bias=False),
            nn.ReLU(),
            )
        self.compute_weigh=nn.Sequential(
            nn.Conv1d(1,1,6,dilation=in_channels,bias=False),
            nn.ReLU(),
        )
        self.compute_weigh_1=nn.Sequential(
            nn.Conv2d(in_channels,in_channels*16,1,bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels*16,in_channels,1,bias=False),
            nn.Sigmoid()
        )
        self.change_dim=nn.Sequential(
            nn.Identity()
        )
        self.merge=nn.Sequential(
            nn.Conv2d(2*in_channels,in_channels,3,padding='same',bias=False),
            nn.GroupNorm(8,in_channels, affine=False),
            nn.ReLU(),
        )
        self.gap=nn.AdaptiveAvgPool2d(1)
        self.map=nn.AdaptiveMaxPool2d(1)
    def forward(self,x_e,x_u):
        '''
        x_e : in_channels,h,w
        x_u: in_channels,h,w
        '''
        b,c,h,w=x_e.shape
        gconv_x_u=self.gconv_x_u(x_u)
        gconv_x_e=self.gconv_x_e(x_e)
        merge= self.out(torch.cat((gconv_x_u,gconv_x_e),1))
        gap_x_m=self.gap(merge)
        # print(gap_x.shape)
        map_m=self.map(merge)

        gap_x_e=self.gap(x_e)
        # print(gap_x.shape)
        gap_x_u=self.gap(x_u)
        map_e=self.map(x_e)
        map_x_u=self.map(x_u)
        w=torch.cat((gap_x_e,map_e,gap_x_u,map_x_u,gap_x_m,map_m),1).flatten(-3).unsqueeze(1)
        # print(w.shape)
        # return
        weigh= self.compute_weigh(w)
        weigh=self.compute_weigh_1(weigh.view(b,-1,1,1))
        # print(weigh.shape)
        merge=merge*weigh

        return self.merge(torch.cat((x_e,self.change_dim(merge)),1))+merge

class MAB(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        g = safe_group(in_channels, preferred=8)
        self.branch = nn.Sequential(
            CA(in_channels),
            SA(),
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.Mish()
        )

        self.branch_t = nn.Sequential(
            CA(in_channels),
            SA(),
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.Mish()
        )
        self.cat = nn.Sequential(
            nn.Conv2d(2 * in_channels, in_channels, 3, padding=_same_padding(3), bias=False),
            nn.GroupNorm(1,in_channels,affine=False),
        )
        self.act=nn.ReLU()

    def forward(self, x):

        b1 = x * self.branch(x)

        t_x = x.transpose(-2, -1).contiguous()     # (B, C, W, H)
        b2_t = t_x * self.branch_t(t_x)            # (B, C, W, H)
        b2 = b2_t.transpose(-2, -1).contiguous()   # (B, C, H, W)

        fusion = b1 + b2 - b1*b2 
        d_fusion = torch.cat([fusion, fusion], dim=1)
        out = self.cat(torch.cat([b1, b2], dim=1)) + x - d_fusion 

        return self.act(out)
    

class down_sampling(nn.Module):
    def __init__(self, in_channels, out_channels, in_size):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.ReLU()
        )
        self.proj_2 = nn.Sequential(
            nn.Conv2d(2*out_channels, out_channels, 1, bias=False),
            nn.ReLU()
        )
        self.mkir = nn.Sequential(MKIR(out_channels, out_channels, in_size=in_size),MAB(out_channels))

        self.down_1 = nn.Conv2d(out_channels, out_channels, kernel_size=2, stride=2, bias=False)
        self.down_2 = nn.MaxPool2d(kernel_size=2, stride=2) 

    def forward(self, x):
        x0 = self.proj(x)

        x1 = self.mkir(x0)
        out = self.proj_2(torch.cat((x1,x0),1)) 
        
        down1 = self.down_1(out) 
        down2 = self.down_2(out) 

        down = (down1 + down2) / 2
        return out, down

class UpFunc(nn.Module):
    def __init__(self, in_channels,out_channels,scale_factor=2):
        super().__init__()
        self.up_sampling=nn.Upsample(scale_factor=scale_factor,mode='bicubic', align_corners=False)
        self.up_conv=nn.ConvTranspose2d(in_channels,in_channels,scale_factor,stride=scale_factor,bias=False)
        self.conv_1=nn.Conv2d(2*in_channels,out_channels,3,padding='same',bias=False)
        self.grnorm = nn.GroupNorm(num_groups=in_channels, num_channels=in_channels*2, affine=False)
        self.gelu=nn.GELU()
    def forward(self,x):
        up_s=self.up_sampling(x)
        up_c=self.up_conv(x)
        cat_u = torch.cat((2*up_s,up_c),1)
        cat_u = self.grnorm(cat_u)

        return self.gelu(self.conv_1(cat_u)) + x 
        

       
class up_sampling(nn.Module):
    def __init__(self,in_channels,in_channels_t,out_channels,in_size):
        super().__init__()
        self.ag=AG(out_channels,in_size)
        self.up_t=UpFunc(in_channels_t,out_channels,2)
        self.up=UpFunc(in_channels,out_channels,2)
        self.mab=MAB(out_channels)
        self.cat=nn.Conv2d(2*out_channels,out_channels,3,padding='same',bias=False)
        self.merge=MKIR(3*out_channels,out_channels)
    def forward(self,x_u,x_e,x_u_t):
        '''
        x_u : inc,h/2,w/2
        x_u_t : inc_t,h/2,w/2
        x_e : out,h,w
        '''
        x_u_t=self.up_t(x_u_t)
        x_u_t=self.ag(x_e,x_u_t)
        x_u=self.up(x_u)
        mab=self.mab(self.cat(torch.cat((x_e,x_u),1)))
        m=self.merge(torch.cat((x_u,mab,x_u_t),1))
        return m,x_u_t

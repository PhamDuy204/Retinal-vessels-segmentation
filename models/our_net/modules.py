import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from bottle_neck import *

class GRN(nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma=nn.Parameter(torch.tensor(1.))
    def forward(self,x):
        x=x.permute(0,2,3,1).contiguous()
        gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True)+1e-6)
        return (self.gamma * (x * nx) + x).permute(0,3,1,2).contiguous()
    

class ConvFunc(nn.Module):
    def __init__(self, in_channels,kernel_size=3,stride=1,padding:int|str='same',dilation=1,in_size=(64,64)):
        super().__init__()
        self.out=nn.Sequential(
            nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,bias=False),
            nn.LayerNorm([in_channels,in_size[0],in_size[1]],bias=False),
            nn.Conv2d(in_channels,in_channels*4,1,bias=False),
            nn.LeakyReLU(0.001),
            GRN(),
            nn.Conv2d(in_channels*4,in_channels,1,bias=False),
        )
    def forward(self,x):
        return x+self.out(x)
    
class MKIR(nn.Module):
    def __init__(self, in_channels,out_channels,in_size=(64,64)):
        super().__init__()
        self.first_conv=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,1,bias=False),
            nn.LayerNorm([out_channels,in_size[0],in_size[1]],bias=False),
            nn.LeakyReLU(0.001),
        )
        self.b1=nn.Sequential(
            ConvFunc(out_channels,kernel_size=3,padding='same',in_size=in_size),
            nn.LayerNorm([out_channels,in_size[0],in_size[1]],bias=False),
            nn.LeakyReLU(0.001))
        self.b2=nn.Sequential(ConvFunc(out_channels,kernel_size=3,padding='same',in_size=in_size),
                            nn.LayerNorm([out_channels,in_size[0],in_size[1]],bias=False),
                            nn.LeakyReLU(0.001),
                            ConvFunc(out_channels,kernel_size=5,padding='same',in_size=in_size),
                            nn.LayerNorm([out_channels,in_size[0],in_size[1]],bias=False),
                            nn.LeakyReLU(0.001),)
        self.b3=nn.Sequential(ConvFunc(out_channels,kernel_size=3,padding='same',in_size=in_size),
                            nn.LayerNorm([out_channels,in_size[0],in_size[1]],bias=False),
                            nn.LeakyReLU(0.001),
                            ConvFunc(out_channels,kernel_size=5,padding='same',in_size=in_size),
                            nn.LayerNorm([out_channels,in_size[0],in_size[1]],bias=False),
                            nn.LeakyReLU(0.001),
                            ConvFunc(out_channels,kernel_size=7,padding='same',in_size=in_size),
                            nn.LayerNorm([out_channels,in_size[0],in_size[1]],bias=False),
                            nn.LeakyReLU(0.001),)
        self.conv_1=nn.Sequential(nn.Conv2d(out_channels,out_channels*4,1,bias=False),
                                  nn.LayerNorm([out_channels*4,in_size[0],in_size[1]],bias=False),
                                    nn.Conv2d(out_channels*4,out_channels,1,bias=False),)
    def forward(self,x):
        x=self.first_conv(x)
        b1=self.b1(x)
        b2=self.b2(x)
        b3=self.b3(x)
        merge = self.conv_1(b1+b2+b3)
        return merge+x
        

class CA(nn.Module):
    def __init__(self, channels, reduction_rate=4):
        super().__init__()
        self.squeeze = nn.ModuleList([
            nn.AdaptiveAvgPool2d(1),
            nn.AdaptiveMaxPool2d(1)
        ])
        self.excitation = nn.Sequential(
            nn.Conv2d(in_channels=channels,
                      out_channels=max(channels // reduction_rate,1),
                      kernel_size=1,bias=False),
            nn.LeakyReLU(0.001),
            nn.Conv2d(in_channels=max(channels // reduction_rate,1),
                      out_channels=channels,
                      kernel_size=1,bias=False)
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
    def __init__(self, kernel_size=5):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding='same',
            bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # mean on spatial dim
        avg_feat    = torch.mean(x, dim=1, keepdim=True)
        # max on spatial dim
        max_feat, _ = torch.max(x, dim=1, keepdim=True)
        feat = torch.cat([avg_feat, max_feat], dim=1)
        out_feat = self.conv(feat)
        attention = self.sigmoid(out_feat)
        return attention * x


class AG(nn.Module):
    def __init__(self,in_channel,in_size=(64,64)):
        super().__init__()
        self.gconv_x_u = nn.Sequential(
            nn.Conv2d(in_channel,in_channel,3,groups=in_channel,padding='same',bias=False),
            nn.LayerNorm([in_channel,in_size[0],in_size[1]],bias=False),)
        self.gconv_x_e = nn.Sequential(nn.Conv2d(in_channel,in_channel,3,groups=in_channel,padding='same',bias=False),
                                        nn.LayerNorm([in_channel,in_size[0],in_size[1]],bias=False),)
        self.out =  nn.Sequential(
            nn.LeakyReLU(0.001),
            nn.Conv2d(in_channel,in_channel,1,bias=False),
            nn.LayerNorm([in_channel,in_size[0],in_size[1]],bias=False),
            )
        self.compute_weigh=nn.Sequential(
            nn.Conv1d(1,1,6,dilation=in_channel,bias=False),
            nn.LeakyReLU(0.001),
        )
        self.compute_weigh_1=nn.Sequential(
            nn.Conv2d(in_channel,4*in_channel,1,bias=False),
            nn.GroupNorm(1,4*in_channel),
            nn.Conv2d(4*in_channel,in_channel,1,bias=False),
            nn.Sigmoid()
        )
        self.change_dim=nn.Sequential(
            nn.LayerNorm([1,in_size[0],in_size[1]],bias=False),
            nn.Sigmoid())
        self.gap=nn.AdaptiveAvgPool2d(1)
        self.map=nn.AdaptiveMaxPool2d(1)
    def forward(self,x_e,x_u):
        '''
        x_e : in_channel,h,w
        x_u: in_channel,h,w
        '''
        b,c,h,w=x_e.shape
        gconv_x_u=self.gconv_x_u(x_u)
        gconv_x_e=self.gconv_x_e(x_e)
        merge= self.out(gconv_x_u+gconv_x_e)
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
        merge=torch.sum(merge*weigh,1,keepdim=True)

        return x_e*self.change_dim(merge)
class MAB(nn.Module):
    def __init__(self,in_channel):
        super().__init__()
        self.b1=nn.Sequential(
            CA(in_channel),
            SA(),
            nn.Conv2d(in_channel,in_channel,1,bias=False),
            nn.Sigmoid())
        self.b2=nn.Sequential(
            CA(in_channel),
            SA(),
            nn.Conv2d(in_channel,in_channel,1,bias=False),
            nn.Sigmoid())
    def forward(self,x):
        b1=x*self.b1(x)
        t_x=x.transpose(-2,-1).contiguous()
        b2=t_x*self.b2(t_x)
        return b1+b2
    

class down_sampling(nn.Module):
    def __init__(self,in_channels,out_channels,in_size):
        super().__init__()
        self.mkir=MKIR(in_channels,out_channels,in_size)
        self.down=nn.MaxPool2d(2)
    def forward(self,x):
        mkir=self.mkir(x)
        return mkir,self.down(mkir)
class UpFunc(nn.Module):
    def __init__(self, in_channels,out_channels,scale_factor=2):
        super().__init__()
        self.up=nn.UpsamplingBilinear2d(scale_factor=scale_factor)
        self.conv_1=nn.Conv2d(in_channels,out_channels,1,bias=False)
    def forward(self,x):
        return  self.conv_1(self.up(x))
       
class up_sampling(nn.Module):
    def __init__(self,in_channels,in_channels_t,out_channels,in_size):
        super().__init__()
        self.ag=AG(out_channels,in_size)
        self.up_t=UpFunc(in_channels_t,out_channels,2)
        self.up=UpFunc(in_channels,out_channels,2)
        self.down=nn.MaxPool2d(2)
        self.mab=MAB(out_channels)
        self.merge=nn.Sequential(
            nn.Conv2d(2*out_channels,out_channels,3,padding='same',bias=False),
            nn.LayerNorm([out_channels,in_size[0],in_size[1]],bias=False),
            nn.LeakyReLU(0.001),
            nn.Conv2d(out_channels,out_channels,3,padding='same',bias=False),
            nn.LayerNorm([out_channels,in_size[0],in_size[1]],bias=False),
            nn.LeakyReLU(0.001),
        )
    def forward(self,x_u,x_e,x_u_t):
        '''
        x_u : inc,h/2,w/2
        x_u_t : inc_t,h/2,w/2
        x_e : out,h,w
        '''
        x_u_t=self.up_t(x_u_t)
        x_u_t=self.ag(x_e,x_u_t) # out,h,w
        x_u=self.up(x_u)
        mab=self.mab(x_u)
        return self.merge(torch.cat((mab,x_u_t),1)),x_u_t
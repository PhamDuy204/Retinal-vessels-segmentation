import torch
import numpy as np
import torch.nn as nn
import math
import torch.nn.functional as F
from timm.models.swin_transformer import window_partition,window_reverse
from einops.layers.torch import Rearrange
from mamba_ssm import Mamba
from timm.models.layers import DropPath
from timm.models.swin_transformer_v2 import SwinTransformerV2Block

class MFSB(nn.Module):
    def __init__(self, in_channels,out_channels,with_activation=True):
        super().__init__()
        self.sc1=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,3,padding='same',bias=False),
            nn.GroupNorm(out_channels,out_channels),
            nn.ReLU()
        )
        self.sc2=nn.Sequential(
            nn.Conv2d(in_channels,in_channels,3,padding='same',bias=False),
            nn.GroupNorm(in_channels,in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels,out_channels,5,padding='same',bias=False),
            nn.GroupNorm(out_channels,out_channels),
            nn.ReLU()
        )
        self.sc3=nn.Sequential(
            nn.Conv2d(in_channels,in_channels,3,padding='same',bias=False),
            nn.GroupNorm(in_channels,in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels,in_channels,5,padding='same',bias=False),
            nn.GroupNorm(in_channels,in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels,out_channels,7,padding='same',bias=False),
            nn.GroupNorm(out_channels,out_channels),
            nn.ReLU()
        )
        self.norm=nn.Sequential(
            nn.GroupNorm(out_channels,out_channels),
            nn.ReLU() if with_activation else nn.Identity()
        )
    def forward(self,x):
        sc1=self.sc1(x)
        sc2=self.sc2(x)
        sc3=self.sc3(x)
        return self.norm(sc1+sc2+sc3)
    
class EFB(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.norm_x=nn.GroupNorm(1,in_channels)
        self.dw_0=nn.Conv2d(in_channels,in_channels,3,padding='same',bias=False,groups=in_channels)
        self.dw_1=nn.Conv2d(in_channels,in_channels,3,padding='same',bias=False,groups=in_channels)
        self.conv=nn.Conv2d(in_channels,in_channels,3,padding='same',bias=False)
        self.ff=nn.Sequential(nn.Conv2d(in_channels,2*in_channels,1,bias=False),
                              nn.GroupNorm(1,2*in_channels),
                              nn.ReLU(),
                              nn.Conv2d(2*in_channels,in_channels,1,bias=False))
        self.norm=nn.GroupNorm(1,in_channels)
    def forward(self,x):
        x=self.norm_x(x)
        dw_0=self.dw_0(x)
        dw_1=self.dw_1(x)
        conv=self.conv(x)
        attn=F.sigmoid(dw_0)*dw_1
        out = self.norm(conv+attn)
        return nn.ReLU()(self.ff(out)+out)
    
class ChoiseSample(nn.Module):
    def __init__(self,in_channels,kernel_size,num_sample):
        super().__init__()
        self.params = nn.ParameterList()
        fan_in = in_channels * kernel_size * kernel_size
        fan_out = fan_in  # nếu bạn muốn, có thể chỉnh cho phù hợp với output
        a = (6.0 / (fan_in + fan_out)) ** 0.5  # công thức Xavier Uniform

        for _ in range(num_sample):
            w = torch.empty(fan_in)
            nn.init.uniform_(w, -a, a)  # Xavier uniform init
            self.params.append(nn.Parameter(w))
        self.num_sample=num_sample
        self.kernel_size=kernel_size
    def forward(self,x):
        b,c,h,w=x.shape
        s=max(self.kernel_size//2,1)
        unfold_x=F.unfold(x,self.kernel_size,1,0,stride=s).permute(0,2,1)
        ones_=F.fold(torch.ones_like(unfold_x).permute(0,2,1),(h,w),self.kernel_size,1,0,s)
        out=[]

        for param in self.params:
            tanh_p=F.tanh(param)+1
            # if (np.random.rand()>0.3) and (self.train==True):
            #     noise=torch.randn_like(unfold_x)*0.03
            # else:noise=0
            unfold_x_1=unfold_x*tanh_p+tanh_p-1
            unfold_x_1=F.fold(unfold_x_1.permute(0,2,1),(h,w),self.kernel_size,1,0,s)/ones_
            unfold_x_1=unfold_x_1.mean(1,keepdim=True)
            if (np.random.rand()>0.3) and (self.train==True):
                noise=torch.randn_like(unfold_x_1)
            else:
                noise=0
            unfold_x_1+=noise
            out.append((unfold_x_1+x)/2)
        return torch.cat(out,1)
# class deepwide_block(nn.Module):
#     def __init__(self,in_channel,out_channel,kernel=3,stride=1,dilation=1):
#         super().__init__()
#         self.out = nn.Sequential(
#             nn.Conv2d(in_channel,out_channel,1,bias=False),
#             nn.ReLU(),
#             nn.Conv2d(out_channel,out_channel,kernel,padding='same',stride=stride,groups=out_channel,dilation=dilation,bias=False)
#         )
#         self.change = nn.Conv2d(in_channel,out_channel,1)
#     def forward(self,x):
#         return self.change(x)+self.out(x)

# class multi_scope(nn.Module):
#     def __init__(self,in_channel,out_channel,kernel=3):
#         super().__init__()
#         self.b1 = deepwide_block(in_channel,out_channel,kernel,dilation=1)
#         self.b2 = deepwide_block(in_channel,out_channel,kernel,dilation=3)
#         self.b3 = deepwide_block(in_channel,out_channel,kernel,dilation=5)

#         self.change = nn.Conv2d(3*out_channel,out_channel,1,bias=False)
#     def forward(self,x):
#         return self.change(torch.cat((self.b1(x),self.b2(x),self.b3(x)),1))
    
# class multi_scope_block(nn.Module):
#     def __init__(self,in_channel,out_channel,kernel=3):
#         super().__init__()
        
#         self.branch = multi_scope(in_channel,out_channel,kernel)
#         self.change_feature = nn.Conv2d(in_channel,out_channel,1,bias=False)
#         self.change = nn.Conv2d(in_channel*2,in_channel,1,bias=False)
#     def forward(self,x):
#         # new_x = self.change(torch.cat((x,x.permute(0,1,3,2)),1))
        
#         return self.change_feature(x)+self.branch(x)
class Residual_net(nn.Module):
    def __init__(self, in_channels,out_channels):
        super().__init__()
        self.shortcut=MFSB(in_channels,out_channels,False)
        self.msbfs=nn.ModuleList([
                MFSB(in_channels,in_channels),
                MFSB(in_channels,in_channels)
        ])
        self.conv=nn.Conv2d(in_channels,out_channels,1,bias=False)
        self.eh=EFB(out_channels)
        self.activation=nn.ReLU()
    def forward(self,x):

        shortcut=self.shortcut(x)
        for block in self.msbfs:
            x=x+block(x)
        return self.eh(self.activation(self.conv(x)+shortcut))
    
class aloalo(nn.Module):
    def __init__(self,in_channel):
        super().__init__()
        self.b1_0 = nn.AdaptiveAvgPool2d(1)
        self.b1_1 = nn.AdaptiveMaxPool2d(1)
        
        self.conv_0 = nn.Conv2d(in_channel,in_channel,1,bias=False)
        self.conv_1 = nn.Conv2d(in_channel,in_channel,1,bias=False)
        self.conv_2 = nn.Conv2d(in_channel,in_channel,1,bias=False)
        self.conv_3 = nn.Conv2d(in_channel,in_channel,1,bias=False)

        self.out = nn.Conv2d(2*in_channel,in_channel,1,bias=False)
    def forward(self,x):
        b,c,h,w = x.shape
        new_x = window_partition(x.permute(0,2,3,1),[2,2]).permute(0,3,1,2)
        avg_new_x = self.conv_0(self.b1_0(new_x)*new_x)
        max_new_x = self.conv_1(self.b1_1(new_x)*new_x)
        sum_new_x = window_reverse((avg_new_x+max_new_x).permute(0,2,3,1),[2,2],h,w).permute(0,3,1,2)
        avg_x = self.conv_2(self.b1_0(x)*x)
        max_x = self.conv_3(self.b1_1(x)*x)
        sum_x = avg_x+max_x
        return x+self.out(torch.cat((sum_new_x,sum_x),1))


class ulaula(nn.Module):
    def __init__(self,in_channel):
        super().__init__()
        self.out = nn.Sequential(
            # multi_scope_block(in_channel,out_channel,3),
            aloalo(in_channel)
        )
    def forward(self,x):
        return self.out(x)


class Conv_func(nn.Module):
    def __init__(self,in_channel,out_channel,kernel=3):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(in_channel,out_channel,kernel_size=kernel,padding='same',bias=False),
            nn.GroupNorm(out_channel,out_channel,affine=False),
            nn.ReLU()
            )
    def forward(self,x):
        return self.feature(x)

class Residual(nn.Module):
    def __init__(self,in_channel,out_channel,kernel=3):
        super().__init__()
        self.feature = nn.Sequential(
            Conv_func(in_channel,out_channel,kernel = kernel),
            Conv_func(out_channel,out_channel,kernel = kernel)
            )
        self.change_feature = nn.Conv2d(in_channel,out_channel,1,bias=False)
    def forward(self,x):
        return self.change_feature(x)+self.feature(x)


class Unpooling_func(nn.Module):
    def __init__(self,in_channel,out_channel,scale_factor = 2):
        super().__init__()
        self.out = nn.ConvTranspose2d(in_channel,out_channel,kernel_size= scale_factor,stride = scale_factor,bias=False)
    def forward(self,x):
        return self.out(x)
    

class down_sampling(nn.Module):
    def __init__(self,in_channel,out_channel):
        super().__init__()
        self.out =  nn.Sequential(
            Residual_net(in_channel,out_channel),
            )
        # self.down = Conv_func_1(out_channel,out_channel,kernel=2,padding=0,stride=2,activation=False)
        self.down = nn.Conv2d(out_channel,out_channel,kernel_size=2,stride=2,bias=False)
    def forward(self,x):
        out = self.out(x)
        return out,self.down(out)
class Up_sampling(nn.Module):
    def __init__(self,in_channel,out_channel,scale_factor = 2):
        super().__init__()
        self.up = Unpooling_func(in_channel,in_channel,scale_factor=scale_factor)

        self.out =  nn.Sequential(
            Residual(in_channel*2,out_channel,3),
            EFB(out_channel),
            )
        self.compute_weigh=nn.Sequential(
            nn.Conv1d(1,1,4,dilation=in_channel),
            nn.Sigmoid()
        )
        self.change_dim=nn.Conv2d(in_channel,out_channel,1,bias=False)
        self.gap=nn.AdaptiveAvgPool2d(1)
        self.map=nn.AdaptiveMaxPool2d(1)
    def forward(self,x,x_encode):
        b,c,h,w=x.shape
        gap_x=self.gap(x)
        # print(gap_x.shape)
        gap_x_encode=self.gap(x_encode)
        map_x=self.map(x)
        map_x_encode=self.map(x_encode)
        w=torch.cat((gap_x,map_x_encode,gap_x_encode,map_x),1).flatten(-3).unsqueeze(1)
        # print(w.shape)
        # return
        weigh= self.compute_weigh(w)
        weigh=weigh.view(b,-1,1,1)
        # print(weigh.shape)
        x=x+x*weigh
        x_encode=x_encode+x_encode*weigh
        up = self.up(x)
        cat_x=self.out(torch.cat((up,x_encode),1))
        return cat_x+self.change_dim(x_encode)
    
class model_exchange_feature(nn.Module):
    def __init__(self,in_channel):
        super().__init__()
        self.rs1 = Residual_net(in_channel,in_channel,3)
        self.rs2 = Residual_net(in_channel,in_channel,3)

    def forward(self,x1,x2):
        x1_1,x1_2 = x1.chunk(2,1)
        x2_1,x2_2 = x2.chunk(2,1)
        new_x1 = torch.cat((x1_1,x2_2),1)
        new_x2 = torch.cat((x1_2,x2_1),1)
        return self.rs1(new_x1),self.rs2(new_x2)

class kame_func(nn.Module):
    def __init__(self,in_channel,out_channel):
        super().__init__()
        self.conv_tran = nn.ConvTranspose2d(in_channel,out_channel,2,2,bias=False)
        self.change = nn.Conv2d(out_channel*2,out_channel,1,bias=False)

    def forward(self,low_size,high_size):
        return self.change(torch.cat((self.conv_tran(low_size),high_size),1))


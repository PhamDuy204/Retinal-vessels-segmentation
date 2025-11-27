import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# from bottle_neck import *
from typing import Optional
# from mamba_ssm import Mamba2 

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

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, pad, dilation,groups=in_channels, bias=False),
            nn.Conv2d(in_channels,in_channels,1,bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, pad, dilation,groups=in_channels, bias=False),
            nn.Conv2d(in_channels,in_channels,1,bias=False),nn.ReLU())
        self.gn=nn.GroupNorm(safe_group(in_channels,16),in_channels)
        self.act = nn.ReLU()
        self.merge=nn.Conv2d(2*in_channels, in_channels, 1,padding='same',bias=False)
    def forward(self, x):
        y = self.conv(x)
        return self.act(self.gn(self.merge(torch.cat((x,y),1))))

class MKIR(nn.Module):
    def __init__(self, in_channels, out_channels, in_size=(64,64)):
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
        return self.out(torch.cat((merged, x), dim=1))
        
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
    def __init__(self, in_channels, in_size=(64, 64), reduction_ratio=4, spatial_kernel_size=7):
        super().__init__()
        
        self.gconv_x_u = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, groups=in_channels, padding='same', bias=False),
            nn.ReLU(),
        )
        self.gconv_x_e = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, groups=in_channels, padding='same', bias=False),
            nn.ReLU(),
        )
        self.fuse_gconv = nn.Sequential(
            nn.Conv2d(2 * in_channels, in_channels, 1, bias=False),
            nn.ReLU(),
        )
        self.channel_attention = nn.Sequential(
            nn.Conv2d(6 * in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

        self.spatial_attention = nn.Sequential(
            # Input là 2 kênh (từ AvgPool và MaxPool)
            nn.Conv2d(2, 1, kernel_size=spatial_kernel_size,
                      padding=spatial_kernel_size // 2, bias=False),
            nn.Sigmoid()
        )
        self.final_merge = nn.Sequential(
            nn.Conv2d(2 * in_channels, in_channels, 3, padding='same', bias=False),
            nn.GroupNorm(8, in_channels, affine=False), # Giữ nguyên GroupNorm
            nn.ReLU(),
        )
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.map = nn.AdaptiveMaxPool2d(1)

    def forward(self, x_e, x_u):
        '''
        x_e : tensor (b, c, h, w) từ encoder (skip connection)
        x_u: tensor (b, c, h, w) từ decoder (upsampled)
        '''
        b, c, h, w = x_e.shape


        gconv_x_u = self.gconv_x_u(x_u)
        gconv_x_e = self.gconv_x_e(x_e)

        merge = self.fuse_gconv(torch.cat((gconv_x_u, gconv_x_e), 1)) # (b, c, h, w)

        gap_x_m = self.gap(merge)
        map_m = self.map(merge)
        gap_x_e = self.gap(x_e)
        gap_x_u = self.gap(x_u)
        map_e = self.map(x_e)
        map_x_u = self.map(x_u)


        global_context = torch.cat((gap_x_e, map_e, gap_x_u, map_x_u, gap_x_m, map_m), 1)

        channel_weights = self.channel_attention(global_context)
        

        merge_ca = merge * channel_weights # (b, c, h, w)



        avg_pool = torch.mean(merge_ca, dim=1, keepdim=True) # (b, 1, h, w)
        max_pool = torch.max(merge_ca, dim=1, keepdim=True)[0] # (b, 1, h, w)
        
        # (b, 2, h, w) -> (b, 1, h, w)
        spatial_weights = self.spatial_attention(torch.cat([avg_pool, max_pool], dim=1))
        

        attended_merge = merge_ca * spatial_weights # (b, c, h, w)


        fused_output = self.final_merge(torch.cat((x_e, attended_merge), 1))
        
        return fused_output + attended_merge

class MAB(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        g = safe_group(in_channels, preferred=8)
        self.b_1= nn.Conv2d(in_channels, in_channels, 3, bias=False,padding='same',groups=in_channels)
        self.b_2= nn.Conv2d(in_channels, in_channels, 3,dilation=2, bias=False,padding='same',groups=in_channels)

        self.branch_0 = nn.Sequential(
            CA(in_channels),
            SA(),
            nn.Conv2d(in_channels, in_channels, 3,dilation=2,padding='same', bias=False),
            nn.Sigmoid()
        )
      
        self.branch_1 = nn.Sequential(
            CA(in_channels),
            SA(),
            nn.Conv2d(in_channels, in_channels, 3,dilation=1,padding='same', bias=False),
            nn.Sigmoid()
        )
       
        self.branch_2 = nn.Sequential(
            CA(in_channels),
            SA(),
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.cat=nn.Conv2d(4 * in_channels, in_channels,3,padding='same', bias=False)

        self.act=nn.ReLU()

    def forward(self, x):
        x_1=self.b_1(x)
        x_2=self.b_2(x)

        b0= x*self.branch_0(x)
        # t_x = x.transpose(-2, -1).contiguous()     # (B, C, W, H)
        # b0_t = t_x * self.branch_0_1(t_x)            # (B, C, W, H)
        # b0_1 = b0_t.transpose(-2, -1).contiguous()
        # out_0 = self.cat_0(torch.cat([b0, b0_1], dim=1))

        b1= x*self.branch_1(x_1)
        # t_x_1 = x_1.transpose(-2, -1).contiguous()     # (B, C, W, H)
        # b1_t = t_x_1 * self.branch_1_1(t_x_1)            # (B, C, W, H)
        # b1_1 = b1_t.transpose(-2, -1).contiguous()
        # out_1 = self.cat_1(torch.cat([b1, b1_1], dim=1))

        b2= x*self.branch_2(x_2)
        # t_x_2 = x_2.transpose(-2, -1).contiguous()     # (B, C, W, H)
        # b2_t = t_x_2 * self.branch_2_1(t_x_2)            # (B, C, W, H)
        # b2_1 = b2_t.transpose(-2, -1).contiguous()
        # out_2 = self.cat_2(torch.cat([b2, b2_1], dim=1))
        return self.act(self.cat(torch.cat((b0,b1,b2,x),1)))
    

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

        self.down = nn.Conv2d(out_channels, out_channels, kernel_size=2, stride=2, bias=False)

    def forward(self, x):
        x0 = self.proj(x)

        x1 = self.mkir(x0)
        out = self.proj_2(torch.cat((x1,x0),1)) 
        return out, self.down(out)


# class down_sampling(nn.Module): 
#     def __init__(self, in_channels, out_channels, in_size): 
#         super().__init__() 
        
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, 3, padding='same', bias=False), 
#             nn.LeakyReLU() 
#         )

#         self.mamba = Mamba2(out_channels, 64, conv_bias=False, d_conv=4, expand=2, headdim=4)
#         self.mamba_rev = Mamba2(out_channels, 64, conv_bias=False, d_conv=4, expand=2, headdim=4)

#         self.fusion = nn.Sequential(
#             SA(), 
#             MKIR(out_channels * 2, out_channels)
#         )

#         self.proj_out = nn.Sequential(
#             nn.Conv2d(out_channels * 2, out_channels, 1, bias=False),
#             nn.LeakyReLU()
#         )

#         self.down = nn.Conv2d(out_channels, out_channels, kernel_size=2, stride=2, bias=False)

    
#     def forward(self, x): 
#         x0 = self.conv(x)
#         b, c, h, w = x0.shape 

#         seq = x0.permute(0, 2, 3, 1).contiguous().view(b, h * w, c) 

#         forward_states = self.mamba(seq)

#         rev_seq = torch.flip(seq, dims=[1])
#         backward_states_rev = self.mamba_rev(rev_seq)
#         backward_states = torch.flip(backward_states_rev, dims=[1])

#         mamba_out_seq = torch.cat((forward_states, backward_states), dim=-1)
#         mamba_out_img = mamba_out_seq.view(b, h, w, c * 2).permute(0, 3, 1, 2).contiguous()
#         x1 = self.fusion(mamba_out_img)

#         out = self.proj_out(torch.cat((x1, x0), dim=1))

#         return out, self.down(out)


class UpFunc(nn.Module):
    def __init__(self, in_channels,out_channels,scale_factor=2):
        super().__init__()
        self.up_sampling=nn.Upsample(scale_factor=scale_factor,mode='bicubic', align_corners=False)
        self.up_conv=nn.ConvTranspose2d(in_channels,in_channels,scale_factor,stride=scale_factor,bias=False)
        self.conv_1=nn.Conv2d(2*in_channels,out_channels,3,padding='same',bias=False)
        self.gelu=nn.GELU()
    def forward(self,x):
        up_s=self.up_sampling(x)
        up_c=self.up_conv(x)
        cat_u=torch.cat((up_s,up_c),1)
        return self.gelu(self.conv_1(cat_u))

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
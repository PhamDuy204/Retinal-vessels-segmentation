import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import SA, CA
from mamba_ssm import Mamba2

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


class CAB(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.q = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.k = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.v = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.pj = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.out_norm = nn.GroupNorm(1,in_channels)
        mid = max(in_channels * 2, 16)
        self.ff = nn.Sequential(
            nn.Conv2d(in_channels, mid, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(mid, in_channels, 1, bias=False),
        )

    def forward(self, x):
        """
        x: (B, C, H, W)
        returns: (B, C, H, W) with residual
        """
        b, c, h, w = x.shape      
        q = self.q(x)                      
        k = self.k(x)
        v = self.v(x)


        N = h * w
        q_flat = q.permute(0, 2, 3, 1).contiguous().view(b, N, c) 
        k_flat = k.permute(0, 2, 3, 1).contiguous().view(b, N, c)
        v_flat = v.permute(0, 2, 3, 1).contiguous().view(b, N, c)


        scale = torch.sqrt(torch.tensor(c, dtype=q.dtype, device=q.device))
        attn_logits = torch.matmul(q_flat, k_flat.transpose(-1, -2)) / scale
        attn = torch.softmax(attn_logits, dim=-1)  
        out_flat = torch.matmul(attn, v_flat) 
        out = out_flat.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()  
        out =  self.out_norm(self.pj(out)+x)
        ff_out = self.ff(out)+out

        return ff_out


class CAB_1(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.q = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.k = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.v = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.pj = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.out_norm = nn.GroupNorm(1,in_channels)
        mid = max(in_channels * 2, 16)
        self.ff = nn.Sequential(
            nn.Conv2d(in_channels, mid, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(mid, in_channels, 1, bias=False),
        )

    def forward(self, x):
        """
        x: (B, C, H, W)
        returns: (B, C, H, W) with residual
        """
        b, c, h, w = x.shape      
        q = self.q(x)                      
        k = self.k(x)
        v = self.v(x)

        scale = torch.sqrt(torch.tensor(h, dtype=q.dtype, device=q.device))
        attn_logits = F.sigmoid(torch.matmul(q, k.transpose(-1, -2))/ scale)

        out = attn_logits*v

        q = self.q(out)                      
        k = self.k(out)
        v = self.v(out)

        attn_logits = F.sigmoid(torch.matmul(q, k.transpose(-1, -2)))

        out = attn_logits*v

        out =  self.out_norm(self.pj(out)+x)
        ff_out = self.ff(out)+out

        return ff_out 

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
        self.gelu1 = nn.ReLU()

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
        
        self.gelu1 = nn.ReLU()
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


class BottleNeck_2(nn.Module):
    def __init__(self, dimension, d_state=16,d_conv=4,norm_before_gate=True):
        super().__init__()
        self.rev_pre_norm = get_rmsnorm(dimension)
        self.post_norm = get_rmsnorm(dimension)

        self.mamba = Mamba2(dimension,32, conv_bias=True, d_conv=4, expand=2,norm_before_gate=True)

        self.out = nn.Sequential(
            nn.Linear(dimension, dimension, bias=False),
            nn.GELU(),
            nn.Linear(dimension, dimension, bias=False),
        )

    def forward(self, x: torch.Tensor):
        """
        x: (B, C, H, W)
        returns: (B, C, H, W)
        """
        b, c, h, w = x.shape


        seq = x.permute(0, 2, 3, 1).contiguous().view(b, h * w, c) 
        forward_states = self.mamba(seq) + seq    
        merged = self.out(forward_states)+forward_states
        out = merged.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        return out
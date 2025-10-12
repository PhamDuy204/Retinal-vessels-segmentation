import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba2

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

class CAB(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.pre_norm = nn.GroupNorm(safe_group(in_channels, 8), in_channels, affine=True)

        self.norm_qk = nn.LayerNorm(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.k = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.v = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.pj = nn.Conv2d(in_channels, in_channels, 1, bias=False)
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
        norm_x = self.pre_norm(x)              
        q = self.q(norm_x)                      
        k = self.k(norm_x)
        v = self.v(norm_x)


        N = h * w
        q_flat = q.permute(0, 2, 3, 1).contiguous().view(b, N, c) 
        k_flat = k.permute(0, 2, 3, 1).contiguous().view(b, N, c)
        v_flat = v.permute(0, 2, 3, 1).contiguous().view(b, N, c)


        qn = self.norm_qk(q_flat)  
        kn = self.norm_qk(k_flat)


        scale = torch.sqrt(torch.tensor(c, dtype=qn.dtype, device=qn.device))
        attn_logits = torch.matmul(qn, kn.transpose(-1, -2)) / scale
        attn = torch.softmax(attn_logits, dim=-1)  


        out_flat = torch.matmul(attn, v_flat) 

        out = out_flat.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()  
        out = self.pj(out)  
        ff_out = self.ff(out) 

        return x + ff_out 



class BottleNeck(nn.Module):
    def __init__(self, dimension, d_state=16):
        super().__init__()
        self.pre_norm = get_rmsnorm(dimension)
        self.rev_pre_norm = get_rmsnorm(dimension)
        self.post_norm = get_rmsnorm(dimension)

        self.mamba = Mamba2(dimension, 64, conv_bias=False, d_conv=4, expand=2)
        self.mamba2 = Mamba2(dimension, 64, conv_bias=False, d_conv=4, expand=2)

        self.merge = nn.Sequential(
            nn.Linear(2 * dimension, max(dimension // 2, 8), bias=False),
            nn.GELU(),
            nn.Linear(max(dimension // 2, 8), dimension, bias=False),
        )

    def forward(self, x: torch.Tensor):
        """
        x: (B, C, H, W)
        returns: (B, C, H, W)
        """
        b, c, h, w = x.shape


        seq = x.permute(0, 2, 3, 1).contiguous().view(b, h * w, c) 


        norm_fwd = self.pre_norm(seq) 
        forward_states = self.mamba(norm_fwd) + seq

        rev_seq = torch.flip(seq, dims=[1])
        norm_rev = self.rev_pre_norm(rev_seq)
        backward_states = self.mamba2(norm_rev) + rev_seq

        backward_states = torch.flip(backward_states, dims=[1])

       
        merged = self.merge(torch.cat((forward_states, backward_states), dim=-1)) 

        merged = self.post_norm(merged)


        out = merged.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        return out

import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from mamba_ssm import Mamba2

class FeedForward(nn.Module):
    def __init__(self, dimension,dropout=0.0):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(dimension, dimension * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dimension * 4,dimension),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
class SoftMoe(nn.Module):
    def __init__(self, dimension,n_experts=8,slots_per_expert=2,dropout=0.0):
        super().__init__()
        self.experts = nn.ModuleList([FeedForward(dimension,dropout) for _ in range(n_experts)])
        self.phi = nn.Parameter(torch.randn(dimension, n_experts * slots_per_expert))
        self.slots_per_expert=slots_per_expert
    def forward(self, x: torch.Tensor):
        logits = torch.matmul(x, self.phi) # (batch_size, seq_len, slots)
        dispatch_weights = F.softmax(logits, dim=-1)
        combine_weights = F.softmax(logits, dim=1)
        xs = torch.bmm(dispatch_weights.transpose(1, 2), x)
        ys = torch.cat(
            [expert(xs[:, i * self.slots_per_expert : (i + 1) * self.slots_per_expert, :]) 
                          for i, expert in enumerate(self.experts)],
            dim=1
            )
        y = torch.bmm(combine_weights, ys)
        return y

class MultiHeadAttention(nn.Module):
    def __init__(self, dimension,n_heads=1,dropout=0.0):
        super().__init__()
        self.head_dimension = dimension //n_heads
        self.n_heads = n_heads
        self.q_projection = nn.Conv2d(dimension, dimension,1, bias=False)
        self.k_projection = nn.Conv2d(dimension, dimension,1, bias=False)
        self.v_projection = nn.Conv2d(dimension, dimension,1, bias=False)

        self.q_norm=nn.LayerNorm(self.head_dimension,bias=False)
        self.k_norm=nn.LayerNorm(self.head_dimension,bias=False)
        self.v_norm=nn.LayerNorm(self.head_dimension,bias=False)

        self.linear = nn.Linear(dimension, dimension, bias=False)
        self.out_norm=nn.LayerNorm(self.head_dimension,bias=False)
        self.dropout = nn.Dropout(dropout)
        self.dropout_ratio=dropout
    def forward(self, x: torch.Tensor):
        b,c,h, w = x.shape
        Q = self.q_norm(self.q_projection(x).permute(0,2,3,1).contiguous().flatten(1,2).view(b, h*w, self.n_heads, self.head_dimension).transpose(1, 2))
        K = self.k_norm(self.k_projection(x).permute(0,2,3,1).contiguous().flatten(1,2).view(b, h*w, self.n_heads, self.head_dimension).transpose(1, 2))
        V = self.v_norm(self.v_projection(x).permute(0,2,3,1).contiguous().flatten(1,2).view(b, h*w, self.n_heads, self.head_dimension).transpose(1, 2))
        heads = F.scaled_dot_product_attention(Q, K, V, dropout_p=self.dropout_ratio, is_causal=False)
        concat = heads.transpose(1, 2).contiguous().view(b, h*w, c)
        linear = self.linear(concat)
        return self.out_norm((self.dropout(linear)+concat)).view(b,h,w,c).permute(0,3,1,2).contiguous()

class CAB(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.pre_norm=nn.GroupNorm(1,in_channels)
        self.q=nn.Conv2d(in_channels,in_channels,1,bias=False)
        self.k=nn.Conv2d(in_channels,in_channels,1,bias=False)
        self.v=nn.Conv2d(in_channels,in_channels,1,bias=False)
        self.pj=nn.Conv2d(in_channels,in_channels,1,bias=False)
        self.ff=nn.Sequential(
            nn.GroupNorm(1,in_channels),
            nn.Conv2d(in_channels,in_channels,1,bias=False,groups=in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels,in_channels,1,bias=False,groups=in_channels),
        )
    def forward(self, x):
        b,c,h,w=x.shape
        norm_x=self.pre_norm(x)
        q=self.q(norm_x)
        k=self.k(norm_x)
        v=self.v(norm_x)
        attn=F.sigmoid((q@k.transpose(-1,-2))/(h*w))
        out=self.pj(v*attn)
        t_stage=self.ff(out+x)
        return t_stage+x
class BottleNeck(nn.Module):
    def __init__(self, dimension,n_heads=1,dropout=0.0,n_experts=4,slots_per_expert=2,d_state=16):
        super().__init__()
        self.pre_norm=nn.RMSNorm(dimension,0.00001)
        self.rev_pre_norm=nn.RMSNorm(dimension,0.00001)
        self.post_norm=nn.RMSNorm(dimension,0.00001)
        self.mamba=Mamba2(dimension,d_state,conv_bias=False)
        self.mamba2=Mamba2(dimension,d_state,conv_bias=False)
        self.merge=nn.Sequential(
            nn.Linear(2*dimension,dimension//2,bias=False),
            nn.GELU(),
            nn.Linear(dimension//2,dimension,bias=False),)
        # self.ff_0=SoftMoe(dimension,n_experts,slots_per_expert,dropout)
        # self.mha=MultiHeadAttention(dimension,n_heads,dropout)
        # self.ff_1=SoftMoe(dimension,n_experts,slots_per_expert,dropout)

    def forward(self, x: torch.Tensor):
        b,c,h, w = x.shape
        x=x.permute(0,2,3,1).contiguous().flatten(1,2)
        rev_x= torch.flip(x, dims=[1])
        forward_states=self.mamba(self.pre_norm(x))+x
        backward_states = self.mamba2(self.rev_pre_norm(rev_x))+rev_x
        mamba=self.merge(torch.cat((forward_states,backward_states),-1))
        # first_stage=(self.ff_0(self.post_norm(mamba))+mamba).view(b,h,w,c).permute(0,3,1,2).contiguous()
        # second_stage=self.mha(first_stage)
        # second_stage=second_stage.permute(0,2,3,1).contiguous().flatten(1,2)
        return mamba.view(b,h,w,c).permute(0,3,1,2).contiguous()


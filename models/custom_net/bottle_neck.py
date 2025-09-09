import torch 
import torch.nn as nn 
import torch.nn.functional as F 
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

# class MambaVisionMixer(nn.Module):
#     def __init__(self, dim, d_state=16, kernel_size=3):
#         super().__init__()
#         self.d_state = d_state
#         self.dt_rank = math.ceil(dim / 16)
#         self.in_proj = nn.Linear(dim, dim,bias=False)
#         self.x_proj = nn.Linear(dim//2, self.dt_rank + self.d_state *2,bias=False)
#         self.conv1d_x = nn.Conv1d(dim//2, dim//2, kernel_size=kernel_size, padding='same', groups=dim//2,bias=False)
#         self.conv1d_z = nn.Conv1d(dim//2, dim//2, kernel_size=kernel_size, padding='same', groups=dim//2,bias=False)
#         self.dt_proj = nn.Linear(self.dt_rank, dim//2)
#         # dt = torch.exp(torch.rand(self.dim//2) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
#         A_log = torch.log(repeat(torch.arange(1, self.d_state + 1), 'n -> d n', d=dim//2))
#         self.A_log = nn.Parameter(A_log)
#         self.D = nn.Parameter(torch.ones(dim//2))
#         self.out_proj = nn.Linear(dim, dim,bias=False)
#     def forward(self, hidden_states):
#         xz = rearrange(self.in_proj(hidden_states), 'b l d -> b d l')
#         x, z = xz.chunk(2, dim=1)
#         A = -torch.exp(self.A_log)
#         x = F.silu(self.conv1d_x(x))
#         z = F.silu(self.conv1d_z(z))
#         seqlen = hidden_states.shape[1]
#         x_dbl = self.x_proj(rearrange(x, 'b d l -> (b l) d'))
#         dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
#         dt = rearrange(self.dt_proj(dt), '(b l) d -> b d l', l=seqlen)
#         B = rearrange(B, '(b l) dstate -> b dstate l', l=seqlen)
#         C = rearrange(C, '(b l) dstate -> b dstate l', l=seqlen)
#         x_ssm = selective_scan_fn(x, dt, A, B, C, self.D)
#         hidden_states = rearrange(torch.cat([x_ssm, z], dim=1), 'b d l -> b l d')
#         return self.out_proj(hidden_states)



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




# import torch
# import math
# import torch.nn as nn
# import torch.nn.functional as F
# from einops import rearrange, repeat
# from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

# class FeedForward(nn.Module):
#     def __init__(self, dimension,dropout=0.0):
#         super().__init__()
#         self.network = nn.Sequential(
#             nn.Linear(dimension, dimension * 4),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(dimension * 4,dimension),
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.network(x)
# class SoftMoe(nn.Module):
#     def __init__(self, dimension,n_experts=8,slots_per_expert=2,dropout=0.0):
#         super().__init__()
#         self.experts = nn.ModuleList([FeedForward(dimension,dropout) for _ in range(n_experts)])
#         self.phi = nn.Parameter(torch.randn(dimension, n_experts * slots_per_expert))
#         self.slots_per_expert=slots_per_expert
#     def forward(self, x: torch.Tensor):
#         logits = torch.matmul(x, self.phi) # (batch_size, seq_len, slots)
#         dispatch_weights = F.softmax(logits, dim=-1)
#         combine_weights = F.softmax(logits, dim=1)
#         xs = torch.bmm(dispatch_weights.transpose(1, 2), x)
#         ys = torch.cat(
#             [expert(xs[:, i * self.slots_per_expert : (i + 1) * self.slots_per_expert, :]) 
#                           for i, expert in enumerate(self.experts)],
#             dim=1
#             )
#         y = torch.bmm(combine_weights, ys)
#         return y

# class MambaVisionMixer(nn.Module):
#     def __init__(self, dim, d_state=16, kernel_size=3):
#         super().__init__()
#         self.d_state = d_state
#         self.dt_rank = math.ceil(dim / 16)
#         self.in_proj = nn.Linear(dim, dim,bias=False)
#         self.x_proj = nn.Linear(dim//2, self.dt_rank + self.d_state *2,bias=False)
#         self.conv1d_x = nn.Conv1d(dim//2, dim//2, kernel_size=kernel_size, padding='same', groups=dim//2,bias=False)
#         self.conv1d_z = nn.Conv1d(dim//2, dim//2, kernel_size=kernel_size, padding='same', groups=dim//2,bias=False)
#         self.dt_proj = nn.Linear(self.dt_rank, dim//2)
#         # dt = torch.exp(torch.rand(self.dim//2) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
#         A_log = torch.log(repeat(torch.arange(1, self.d_state + 1), 'n -> d n', d=dim//2))
#         self.A_log = nn.Parameter(A_log)
#         self.D = nn.Parameter(torch.ones(dim//2))
#         self.out_proj = nn.Linear(dim, dim,bias=False)
#     def forward(self, hidden_states):
#         xz = rearrange(self.in_proj(hidden_states), 'b l d -> b d l')
#         x, z = xz.chunk(2, dim=1)
#         A = -torch.exp(self.A_log)
#         x = F.silu(self.conv1d_x(x))
#         z = F.silu(self.conv1d_z(z))
#         seqlen = hidden_states.shape[1]
#         x_dbl = self.x_proj(rearrange(x, 'b d l -> (b l) d'))
#         dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
#         dt = rearrange(self.dt_proj(dt), '(b l) d -> b d l', l=seqlen)
#         B = rearrange(B, '(b l) dstate -> b dstate l', l=seqlen)
#         C = rearrange(C, '(b l) dstate -> b dstate l', l=seqlen)
#         x_ssm = selective_scan_fn(x, dt, A, B, C, self.D)
#         hidden_states = rearrange(torch.cat([x_ssm, z], dim=1), 'b d l -> b l d')
#         return self.out_proj(hidden_states)


# class MultiHeadAttention(nn.Module):
#     def __init__(self, dimension,n_heads=1,dropout=0.0):
#         super().__init__()
#         self.head_dimension = dimension //n_heads
#         self.n_heads = n_heads
#         self.q_projection = nn.Conv2d(dimension, dimension,1, bias=False)
#         self.k_projection = nn.Conv2d(dimension, dimension,1, bias=False)
#         self.v_projection = nn.Conv2d(dimension, dimension,1, bias=False)

#         self.q_norm=nn.LayerNorm(self.head_dimension,bias=False)
#         self.k_norm=nn.LayerNorm(self.head_dimension,bias=False)
#         self.v_norm=nn.LayerNorm(self.head_dimension,bias=False)

#         self.linear = nn.Linear(dimension, dimension, bias=False)
#         self.out_norm=nn.LayerNorm(self.head_dimension,bias=False)
#         self.dropout = nn.Dropout(dropout)
#         self.dropout_ratio=dropout
#     def forward(self, x: torch.Tensor):
#         b,c,h, w = x.shape
#         Q = self.q_norm(self.q_projection(x).permute(0,2,3,1).contiguous().flatten(1,2).view(b, h*w, self.n_heads, self.head_dimension).transpose(1, 2))
#         K = self.k_norm(self.k_projection(x).permute(0,2,3,1).contiguous().flatten(1,2).view(b, h*w, self.n_heads, self.head_dimension).transpose(1, 2))
#         V = self.v_norm(self.v_projection(x).permute(0,2,3,1).contiguous().flatten(1,2).view(b, h*w, self.n_heads, self.head_dimension).transpose(1, 2))
#         heads = F.scaled_dot_product_attention(Q, K, V, dropout_p=self.dropout_ratio, is_causal=False)
#         concat = heads.transpose(1, 2).contiguous().view(b, h*w, c)
#         linear = self.linear(concat)
#         return self.out_norm((self.dropout(linear)+concat)).view(b,h,w,c).permute(0,3,1,2).contiguous()


# class BottleNeck(nn.Module):
#     def __init__(self, dimension,n_heads=1,dropout=0.0,n_experts=8,slots_per_expert=2,d_state=16, kernel_size=3):
#         super().__init__()
#         self.pre_norm=nn.LayerNorm(dimension,bias=False)
#         self.post_norm=nn.LayerNorm(dimension,bias=False)
#         self.mamba=MambaVisionMixer(dimension,d_state,kernel_size)
#         self.ff_0=SoftMoe(dimension,n_experts,slots_per_expert,dropout)
#         self.mha=MultiHeadAttention(dimension,n_heads,dropout)
#         self.ff_1=SoftMoe(dimension,n_experts,slots_per_expert,dropout)

#     def forward(self, x: torch.Tensor):
#         b,c,h, w = x.shape
#         x=x.permute(0,2,3,1).contiguous().flatten(1,2)
#         mamba=self.mamba(self.pre_norm(x))+x
#         first_stage=(self.ff_0(self.post_norm(mamba))+mamba).view(b,h,w,c).permute(0,3,1,2).contiguous()
#         second_stage=self.mha(first_stage)
#         second_stage=second_stage.permute(0,2,3,1).contiguous().flatten(1,2)
#         return (second_stage+self.ff_1(second_stage)).view(b,h,w,c).permute(0,3,1,2).contiguous()
class BottleNeck(nn.Module):
    def __init__(self, dimension,n_heads=1,dropout=0.0,n_experts=8,slots_per_expert=2,d_state=16):
        super().__init__()
        self.pre_norm=nn.LayerNorm(dimension,bias=False)
        self.post_norm=nn.LayerNorm(dimension,bias=False)
        self.mamba=Mamba2(dimension,d_state,conv_bias=False)
        self.ff_0=SoftMoe(dimension,n_experts,slots_per_expert,dropout)
        self.mha=MultiHeadAttention(dimension,n_heads,dropout)
        self.ff_1=SoftMoe(dimension,n_experts,slots_per_expert,dropout)

    def forward(self, x: torch.Tensor):
        b,c,h, w = x.shape
        x=x.permute(0,2,3,1).contiguous().flatten(1,2)
        mamba=self.mamba(self.pre_norm(x))+x
        # print(mamba)
        first_stage=(self.ff_0(self.post_norm(mamba))+mamba).view(b,h,w,c).permute(0,3,1,2).contiguous()
        second_stage=self.mha(first_stage)
        second_stage=second_stage.permute(0,2,3,1).contiguous().flatten(1,2)
        return (second_stage+self.ff_1(second_stage)).view(b,h,w,c).permute(0,3,1,2).contiguous()


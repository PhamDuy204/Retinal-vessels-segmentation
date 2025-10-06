import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba2
# from timm.models.swin_transformer import window_partition,window_reverse

class CAB(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.pre_norm=nn.GroupNorm(1,in_channels)
        self.norm_qk=nn.GroupNorm(in_channels,in_channels)
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
        attn=F.sigmoid(self.norm_qk((q@k.transpose(-1,-2))/torch.sqrt(torch.tensor(h))))
        out=self.pj(v*attn)
        t_stage=self.ff(out+x)
        return t_stage+x
class BottleNeck(nn.Module):
    def __init__(self, dimension,d_state=16):
        super().__init__()
        self.pre_norm=nn.RMSNorm(dimension,0.001)
        self.rev_pre_norm=nn.RMSNorm(dimension,0.001)
        self.post_norm=nn.RMSNorm(dimension,0.001)
        self.mamba=Mamba2(dimension,d_state,conv_bias=False)
        self.mamba2=Mamba2(dimension,d_state,conv_bias=False)
        self.merge=nn.Sequential(
            nn.Linear(2*dimension,dimension//2,bias=False),
            nn.GELU(),
            nn.Linear(dimension//2,dimension,bias=False),)


    def forward(self, x: torch.Tensor):
        b,c,h, w = x.shape
        x=x.permute(0,2,3,1).contiguous().flatten(1,2)
        rev_x= torch.flip(x, dims=[1])
        forward_states=self.mamba(self.pre_norm(x))+x
        backward_states = self.mamba2(self.rev_pre_norm(rev_x))+rev_x
        mamba=self.merge(torch.cat((forward_states,backward_states),-1))
        return mamba.view(b,h,w,c).permute(0,3,1,2).contiguous()
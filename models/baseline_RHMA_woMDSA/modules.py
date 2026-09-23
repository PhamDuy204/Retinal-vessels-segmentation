import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# from bottle_neck import *
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

def init_module_weights(module: nn.Module):
    """
    Init trọng số theo quy tắc chung:
      - Conv2d, ConvTranspose2d: Kaiming Uniform (fan_in, relu) trừ kernel=1 -> Xavier
      - Linear: Kaiming Uniform
      - GroupNorm/BatchNorm/LayerNorm: weight=1 bias=0 (nếu affine)
      - Embedding: normal std=0.02
    Gọi trong mỗi __init__() của class: init_module_weights(self)
    """
    for m in module.modules():
        # Conv2d
        if isinstance(m, nn.Conv2d):
            # m.kernel_size có thể là tuple
            k = m.kernel_size if hasattr(m, 'kernel_size') else (1,1)
            # nếu 1x1 conv thường dùng xavier; các conv khác dùng kaiming (ReLU-ish)
            if isinstance(k, tuple) and k[0] == 1 and k[1] == 1:
                nn.init.xavier_uniform_(m.weight)
            else:
                # depthwise conv (groups == in_channels) vẫn ok với kaiming
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        # ConvTranspose2d
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        # Linear
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        # Normalization layers
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            # nếu layer có tham số affine (weight/bias), set weight=1 bias=0
            if hasattr(m, 'weight') and m.weight is not None:
                try:
                    nn.init.constant_(m.weight, 1)
                except Exception:
                    pass
            if hasattr(m, 'bias') and m.bias is not None:
                try:
                    nn.init.constant_(m.bias, 0)
                except Exception:
                    pass

        # Embedding
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

# -------------------------
# Các lớp (giữ cấu trúc của bạn)
# -------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convs = nn.Sequential(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3,padding='same', bias=False),
            nn.GroupNorm(safe_group(out_channels),out_channels,affine=False),
            nn.ReLU()),nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3,padding='same', bias=False),
            nn.GroupNorm(safe_group(out_channels),out_channels,affine=False),
            nn.ReLU()))
    def forward(self,x):
        return self.convs(x)
class ConvFunc(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=1, padding: Optional[int]='same', dilation=1,bias=False,with_activate=True):
        super().__init__()
        # lưu các param nếu cần debug
        self._in_ch = in_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, bias=bias),
            nn.GroupNorm(1,in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size,padding='same', bias=bias),
            nn.ReLU())
        self.gn=nn.GroupNorm(safe_group(in_channels,16),in_channels)
        self.act = nn.ReLU()
        self.merge=nn.Conv2d(2*in_channels, in_channels, 1,bias=False)
        self.with_activate=with_activate

        # gọi init trong class
        init_module_weights(self)

    def forward(self, x):
        y = self.conv(x)
        out = self.merge(torch.cat((x,y),1))
        if self.with_activate:
            return self.act(self.gn(out))
        return out
class MKIR(nn.Module):
    def __init__(self, in_channels, out_channels, in_size=(64,64)):
        super().__init__()
        # project in -> out (1x1)
        self.first_conv = nn.Conv2d(in_channels, out_channels, 3,padding='same', bias=False)

        # three dilation branches (depthwise separable inside ConvFunc)
        self.b1 = ConvFunc(out_channels,3)
        self.b2 = ConvFunc(out_channels,3,dilation=3)
        self.b3 = ConvFunc(out_channels,3,dilation=5)
        # fusion
        self.out =nn.Sequential(
            nn.Conv2d(3*out_channels, out_channels, 1, bias=False),
            nn.ReLU())
        self.norm = nn.GroupNorm(1,out_channels)
        self.act=nn.ReLU()

        init_module_weights(self)

    def forward(self, x):
        x = self.first_conv(x)
        b1 = self.b1(x)
        b2 = self.b2(x)
        b3 = self.b3(x)
        return self.act(self.norm(self.out(torch.cat((b1,b2,b3),1))+x))
class down_sampling(nn.Module):
    def __init__(self, in_channels, out_channels, in_size):
        super().__init__()
        self.proj =nn.Sequential(nn.Conv2d(in_channels, out_channels, 3,padding='same', bias=False),ConvFunc(out_channels))
        self.down = nn.Conv2d(out_channels,out_channels,2,2,bias=False)

        init_module_weights(self)

    def forward(self, x):
        out = self.proj(x) 
        return out, self.down(out)
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

        init_module_weights(self)

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

        init_module_weights(self)

    def forward(self, x):
        avg_feat = torch.mean(x, dim=1, keepdim=True)
        max_feat, _ = torch.max(x, dim=1, keepdim=True)
        feat = torch.cat([avg_feat, max_feat], dim=1)
        out_feat = self.conv(feat)
        attention = self.sigmoid(out_feat)
        return attention * x
class UpFunc(nn.Module):
    def __init__(self, in_channels,out_channels,scale_factor=2):
        super().__init__()
        self.up_conv=nn.ConvTranspose2d(in_channels,out_channels,scale_factor,stride=scale_factor,bias=True)

        init_module_weights(self)

    def forward(self,x):
        return self.up_conv(x)

class up_sampling(nn.Module):
    def __init__(self,in_channels,in_channels_t,out_channels,in_size):
        super().__init__()
        self.up_t=UpFunc(in_channels_t,out_channels,2)
        self.up=UpFunc(in_channels,out_channels,2)
        self.merge=MKIR(out_channels,out_channels)
        self.conv=ConvFunc(out_channels)
        self.mab=ConvFunc(out_channels)
        self.cat=nn.Sequential(
            nn.Conv2d(3*out_channels,out_channels,3,padding='same',bias=False),
            nn.GroupNorm(out_channels,out_channels),
            nn.ReLU(),
            CA(out_channels),
            SA()
        )
        
        init_module_weights(self)

    def forward(self,x_u,x_e,x_u_t):
        '''
        x_u : inc,h/2,w/2
        x_u_t : inc_t,h/2,w/2
        x_e : out,h,w
        '''
        x_u_t=self.up_t(x_u_t)
        x_u=self.up(x_u)

        return self.merge(self.cat(torch.cat((x_u,x_u_t,x_e),1))),self.conv(self.mab(x_u_t)+x_u_t)
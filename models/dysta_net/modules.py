import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def safe_group(channels: int, preferred: int = 8) -> int:
    pref = min(preferred, channels)
    for g in range(pref, 0, -1):
        if channels % g == 0:
            return g
    return 1

class DynamicConv(nn.Module):
    def __init__(self, in_channels,out_channels,kernel_size=3,padding='same',stride=1,dilation=1,k=4):
        super().__init__()
        self.kernel_size=kernel_size
        self.padding=padding
        self.stride=stride
        self.dilation=dilation
        self.k = k
        self.in_channels = in_channels
        self.out_channels = out_channels

        # k bộ depthwise kernel: (k, in_channels, 1, kH, kW)
        self.k_conv = nn.Parameter(
            torch.empty(k, in_channels, 1, kernel_size, kernel_size)
        )

        # k bộ pointwise kernel: (k, out_channels, in_channels, 1, 1)
        self.out_conv = nn.Parameter(
            torch.empty(k, out_channels, in_channels, 1, 1)
        )

        self.attn = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(in_channels, max(in_channels//4,1), 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(max(in_channels//4,1), k, 1, bias=False),
            nn.Flatten()
        )

        self.norm = nn.GroupNorm(safe_group(out_channels),out_channels, affine=False)
        self.activation = nn.ReLU()

        self._initialize()

    def _initialize(self):
        # Depthwise kernels
        nn.init.kaiming_uniform_(self.k_conv, mode='fan_in', nonlinearity='relu')

        # Pointwise kernels
        nn.init.xavier_uniform_(self.out_conv)

        # Attention conv layers
        for m in self.attn.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        c = float(C)
        attn_x = F.softmax(self.attn(x) / torch.sqrt(torch.tensor(c)), -1)  # B,k

        # reshape input
        x = x.flatten(0,1).unsqueeze(0)   # 1, B*C, H, W

        # Compute depthwise weights
        w_dw = torch.mm(attn_x, self.k_conv.flatten(1)) \
                    .view(-1, 1, self.kernel_size, self.kernel_size)

        # Compute pointwise weights
        w_pw = torch.mm(attn_x, self.out_conv.flatten(1)) \
                    .view(-1, C, 1, 1)

        # DW conv
        dw_x = F.conv2d(
            x, w_dw, groups=B*C,
            padding=self.padding, stride=self.stride, dilation=self.dilation
        )

        # PW conv
        out = F.conv2d(
            dw_x, w_pw, groups=B,
            padding=self.padding, stride=self.stride, dilation=self.dilation
        ).view(B, -1, H, W)

        return self.activation(self.norm(out))


class DWConv(nn.Module):
  def __init__(self,in_channels,out_channels):
    super().__init__()
    self.conv = nn.Sequential(
        nn.Conv2d(in_channels,in_channels,3,padding='same',groups= in_channels,bias=False),
        nn.Conv2d(in_channels,out_channels,1,bias=False)
    )
    self.norm = nn.GroupNorm(safe_group(out_channels),out_channels, affine=False)
    self.activation = nn.ReLU()
  def forward(self,x):
    x=self.conv(x)
    return self.activation(self.norm(x))
  
class DSConv(nn.Module):
  def __init__(self,in_channels,out_channels):
    super().__init__()
    self.first_conv=nn.Conv2d(in_channels,out_channels,1,bias=False)
    self.convs = nn.Sequential(
        DWConv(out_channels,out_channels),
        DWConv(out_channels,out_channels),
    )
  def forward(self,x):
    x= self.first_conv(x)
    x0=self.convs(x)
    return x+x0
  
class AttnBlock(nn.Module):
  def __init__(self,in_channels):
    super().__init__()
    self.first = DSConv(in_channels,in_channels)
    self.b0 = DynamicConv(in_channels,in_channels,3,dilation=1)
    self.b1 = DynamicConv(in_channels,in_channels,3,dilation=3)
    self.b2 = DynamicConv(in_channels,in_channels,3,dilation=5)
    self.conv1x1 = nn.Conv2d(3*in_channels,in_channels,1,bias=False)

    self.attn=nn.Sequential(
        DSConv(3,3),
        nn.Conv2d(3,in_channels,1,bias=False)
    )
  def forward(self,x):
    x_f = self.first(x)
    b0 = self.b0(x_f)
    b1 = self.b1(x_f)
    b2 = self.b2(x_f)
    m = self.conv1x1(torch.cat([b0,b1,b2],1))
    x=x*m
    avg_x = x.mean(1,keepdim=True)
    max_x = x.max(1,keepdim=True).values
    std_x = x.std(1,keepdim=True)


    s_m = torch.cat([avg_x,max_x,std_x],1)
    return x*self.attn(s_m)

class downsampling(nn.Module):
  def __init__(self,in_channels,out_channels):
    super().__init__()
    self.conv = DSConv(in_channels,out_channels)
    self.down = nn.Conv2d(out_channels,out_channels,2,2,groups=out_channels,bias=False)
  def forward(self,x):
    x=self.conv(x)
    return x, self.down(x)

class upsampling(nn.Module):
  def __init__(self,in_channels,out_channels):
    super().__init__()
    self.up = nn.Sequential(
        nn.ConvTranspose2d(in_channels,in_channels,2,2,bias=False,groups =in_channels),
        nn.Conv2d(in_channels,out_channels,1,bias=False))
    
    self.conv = DSConv(2*out_channels,out_channels)
    self.attn = AttnBlock(out_channels)
  def forward(self,hfm,lfm):
    hfm=self.up(hfm)
    m = torch.cat([hfm,lfm],1)
    return  self.attn(self.conv(m))  
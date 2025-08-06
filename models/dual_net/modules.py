import torch.nn as nn
import torch
import torch.nn.functional as F


class DynamicConv(nn.Module):
    def __init__(self, in_channels,out_channels,kernel_size=3,padding='same',stride=1,dilation=1,k=4):
        super().__init__()
        self.kernel_size=kernel_size
        self.padding=padding
        self.stride=stride
        self.dilation=dilation
        self.k_conv=nn.Parameter(torch.rand(k,in_channels,1,kernel_size,kernel_size))
        self.out_conv=nn.Parameter(torch.rand(k,out_channels,in_channels,1,1))
        self.attn=nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(in_channels,in_channels//2,1,1,bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels//2,k,1,1,bias=False),
            nn.Flatten()
        )
        self.norm = nn.BatchNorm2d(out_channels,affine=False)
        self.activation=nn.ReLU()
        self._initialize()
    def _initialize(self):
        for i in range(len(self.k_conv)):
            nn.init.kaiming_uniform_(self.k_conv[i])
        for i in range(len(self.out_conv)):
            nn.init.kaiming_uniform_(self.out_conv[i])
    def forward(self,x):
        B,C,H,W=x.shape
        c=torch.tensor(float(C))
        attn_x=F.softmax(self.attn(x)/torch.sqrt(c),-1) #B,k

        x=x.flatten(0,1).unsqueeze(0)  #1,b*k,h,w
        weigh_dw=torch.mm(attn_x,self.k_conv.flatten(1)).view(-1,1,self.kernel_size,self.kernel_size) # B,in_channelsx1xkernel_sizexkernel_size
        weigh_out = torch.mm(attn_x,self.out_conv.flatten(1)).view(-1,C,1,1)
        dw_x=F.conv2d(x,weigh_dw,groups=B*C,padding=self.padding,stride=self.stride,dilation=self.dilation)
        out=F.conv2d(dw_x,weigh_out,groups=B,padding=self.padding,stride=self.stride,dilation=self.dilation).view(B,-1,H,W)
        return self.activation(self.norm(out))
    
class DWConv(nn.Module):
    def __init__(self, in_channels,out_channels,kernel_size=3,padding='same',stride=1,dilation=1):
        super().__init__()
        self.out = nn.Sequential(
            nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation=dilation,groups=in_channels,bias=False),
            nn.Conv2d(in_channels,out_channels,1,bias=False),
            nn.BatchNorm2d(out_channels,affine=False),
            nn.ReLU()
        )
    def forward(self,x):
        return self.out(x)

class DownSampling(nn.Module):
    def __init__(self, in_channels,out_channels):
        super().__init__()
        self.dynamic=DynamicConv(in_channels,out_channels)
        self.dw=DWConv(in_channels,out_channels)
        self.down=nn.Conv2d(2*out_channels,out_channels,1,bias=False)
    def forward(self,x):
        dynamic=self.dynamic(x)
        dw=self.dw(x)
        return self.down(torch.cat((dynamic,dw),1))
    
class Attn(nn.Module):
    def __init__(self, in_channels_1,in_channels_2):
        super().__init__()
        self.up=nn.ConvTranspose2d(in_channels_1,in_channels_1,4,2,1,groups=in_channels_2,bias=False)
        self.b_1=nn.Conv2d(in_channels_1,in_channels_2,1,bias=False)
        self.b_2=nn.Conv2d(in_channels_2,in_channels_2,1,bias=False)
        self.w=nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(2*in_channels_2,1,1,bias=False),
            nn.Sigmoid()
        )
    def forward(self,x_1,x_2):
        up =self.up(x_1)
        dynamic_1=self.b_1(up)
        dynamic_2=self.b_2(x_2)
        w= self.w(torch.cat((dynamic_1,dynamic_2),1))
        return up*w
    
class UpSampling(nn.Module):
    def __init__(self, in_channels,out_channels):
        super().__init__()
        self.attn=Attn(in_channels,out_channels)
        self.up=nn.ConvTranspose2d(in_channels,in_channels,4,2,1,groups=in_channels,bias=False)
        self.out = DWConv(2*in_channels,out_channels)

    def forward(self,x_1,x_2):
        attn=self.attn(x_1,x_2)
        up = self.up(x_1)
        return self.out(torch.cat((attn,up),1))



import torch
import torch.nn as nn
import timm 
from timm.models.swin_transformer import window_partition,window_reverse
"""# MDAE"""

class GAP_conv(nn.Module):
    def __init__(self,in_channel):
        super().__init__()
        self.feature = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channel,in_channel,1,bias=False),
            nn.Sigmoid()
        )
    def forward(self,x):
        return x*self.feature(x)




import torch
import torch.nn as nn

class MDAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self._initialized = False
        self.h_shape = None
        self.w_shape = None
        self.c_shape = None


    def _initialize(self, x):
        _, c, h, w = x.shape
        if (not hasattr(self, 'conv1')) or (self.h_shape != h):
            self.conv1 = nn.Sequential(
                nn.Conv2d(h, h, 1, bias=False).to(x.device),
                nn.Sigmoid()
            )
            self.h_shape = h

            self.add_module("conv1",self.conv1)
        if (not hasattr(self, 'conv2'))  or (self.w_shape != w):
            self.conv2 = nn.Sequential(
                nn.Conv2d(w, w, 1, bias=False).to(x.device),
                nn.Sigmoid()
            )
            self.w_shape = w
            self.add_module("conv2",self.conv2)

        if (not hasattr(self, 'conv3')) or (self.c_shape != c):
            self.conv3 = nn.Sequential(
                nn.Conv2d(c, c, 1, bias=False).to(x.device),
                nn.Sigmoid()
            )
            self.add_module("conv3",self.conv3)

        self._initialized = True

    def forward(self, x):
        if not self._initialized:
            self._initialize(x)
        x_1 = self.conv1(self.gap(x.permute(0, 2, 1, 3)))
        x_3 = self.conv2(self.gap(x.permute(0, 3, 2, 1)))
        x_2 = self.conv3(self.gap(x))
        return x+x_2 + x_3.permute(0, 3, 2, 1) + x_1.permute(0, 2, 1, 3)


# class self_attention(nn.Module):
#   def __init__(self,in_channel):
#     super().__init__()
#     self.bn1 = nn.LayerNorm(in_channel)
#     self.bn2 = nn.LayerNorm(in_channel)
#     self.bn3 = nn.LayerNorm(in_channel)
#     self.lr = nn.Linear(in_channel,3*in_channel,bias=False)
#     self.out = nn.Linear(in_channel,in_channel,bias=False)
#   def forward(self,x):
#     b,c,h,w = x.shape
#     x = x.view(b,c,h*w).permute(0,2,1)
#     q,k,v = self.lr(x).chunk(3,-1)
#     softmax = nn.Softmax(2)(self.bn1(torch.matmul(q.permute(0,2,1),k)))
#     rs = self.bn2(torch.matmul(v,softmax))
#     return self.bn3(self.out(rs)).permute(0,2,1).view(b,c,h,w)
# class maxxvit(nn.Module):
#   def __init__(self,in_channel):
#     super().__init__()
#     self.attn = self_attention(in_channel)
#   def forward(self,x):
#     b,c,h,w = x.shape
#     x = x.permute(0,2,3,1).contiguous()
#     x = self.attn(window_partition(x,(4,4)).permute(0,3,1,2).contiguous()).permute(0,2,3,1).contiguous()
#     x = window_reverse(x,(4,4),h,w)
#     return x.permute(0,3,1,2).contiguous()


class MAC(nn.Module):
    def __init__(self,in_channel):
        super().__init__()
        self.b1 = nn.Conv2d(in_channel,in_channel,3,dilation=1,padding='same',bias=False,groups = in_channel)
        self.b2 = nn.Conv2d(in_channel,in_channel,3,dilation=3,padding='same',bias=False,groups = in_channel)
        self.b3 = nn.Conv2d(in_channel,in_channel,3,dilation=5,padding='same',bias=False,groups = in_channel)
        self.norm =   nn.GroupNorm(in_channel,in_channel,affine=False)

    def forward(self,x):
        return self.norm(x + self.b1(x)+self.b2(x)+self.b3(x))


class CPSE(nn.Module):
    def __init__(self,in_channel):
        super().__init__()
        self.max_pool1d = nn.MaxPool1d(3,padding=1,stride=1)
        self.MAC_3 = nn.Sequential(
            MAC(in_channel=in_channel),
            MAC(in_channel=in_channel),
            MAC(in_channel=in_channel),
        )
        self.norm = nn.GroupNorm(in_channel,in_channel,affine=False)

    def forward(self,x):
        b,c,h,w =  x.shape
        x_hori = self.max_pool1d(x.reshape(b,c*h,w)).reshape(b,c,h,w)
        x_ver = self.max_pool1d(x.permute(0,1,3,2).reshape(b,c*w,h)).reshape(b,c,w,h).permute(0,1,3,2)
        x_max = torch.max(x_hori,x_ver)
        return self.norm(self.MAC_3(x_max))



"""# DGF"""

class DGF(nn.Module):
    def __init__(self,in_channel_high,in_channel_low,out_channel = 64):
        super().__init__()

        self.low_feature = nn.Conv2d(in_channel_low,out_channel,3,padding='same',bias=False) # b,64,h,w
        self.high_feature = nn.Sequential(
            nn.ConvTranspose2d(in_channel_high,in_channel_high,2,stride=2,bias=False), # b,64,h,w
            nn.Conv2d(in_channel_high,out_channel,3,padding='same',bias=False)
        )
        self.feature_concat = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channel*2,out_channel,1),
            nn.Sigmoid(),

        )
        self.norm =  nn.GroupNorm(out_channel*2,out_channel*2,affine=False)


    def forward(self,x_high_level,x_low_level):
        '''
        in_channel
        low_level: b,in_channel,h,w
        high_level : b,2*in_channel,h/2,w/2
        '''
        x_high = self.high_feature(x_high_level) # b,64,h,w
        x_low = self.low_feature(x_low_level) # b,64,h,w
        x_low_concat = x_low + x_low*self.feature_concat(torch.concat((x_low,x_high),1)) #b,64,h,w* b,64,1,1
        return  self.norm(torch.cat((x_low_concat,x_high),1))


"""# AWL"""

class AWL(nn.Module):
    def __init__(self,in_channel=3):
        super().__init__()
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.GMP = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel,in_channel,kernel_size=1),
            nn.Sigmoid(),
            )
        self.change_feature = nn.Conv2d(3,3,1)
        self.out = nn.Conv2d(3,1,1)
    def forward(self,x1,x2,x3):
        x = self.change_feature(torch.cat((x1,x2,x3),1))
        GAP = self.GAP(x)
        GMP = self.GMP(x)

        sum = self.conv(GMP+GAP)
        alpha,beta,gamma=torch.chunk(sum,3,dim=1)
        return self.out(torch.cat((x1+alpha*x1,x2+beta*x2,x3+gamma*x3),1))


"""# MODEL

## CONVOLUTION
"""

class convolution(nn.Module):
    def __init__(self,in_channel,out_channel):
        super().__init__()
        self.out = nn.Sequential(
            nn.Conv2d(in_channel,out_channel,3,padding='same',bias=False),
            nn.GroupNorm(out_channel,out_channel,affine=False),
            nn.ReLU(),

            nn.Conv2d(out_channel,out_channel,3,padding='same',bias=False),
            nn.GroupNorm(out_channel,out_channel,affine=False),
            nn.ReLU(),
        )
    def forward(self,x):
        return self.out(x)

class change_feature_size_x4(nn.Module):
    def __init__(self,in_channel,out_channel=1):
        super().__init__()
        self.out = nn.Sequential(
            nn.ConvTranspose2d(in_channel,64,4,stride=2,padding=1,bias=False),
            nn.ReLU(),
            nn.ConvTranspose2d(64,1,4,stride=2,padding=1,bias=False),
        )
        self.norm = nn.GroupNorm(1,1,affine=False)

    def forward(self,x):
        return self.norm(self.out(x))

class change_feature_size_x2(nn.Module):
    def __init__(self,in_channel,out_channel=1):
        super().__init__()
        self.out = nn.Sequential(
            nn.ConvTranspose2d(in_channel,64,4,stride=2,padding=1,bias=False),
            nn.ReLU(),
            nn.Conv2d(64,1,3,padding='same',bias=False),

            # nn.ConvTranspose2d(64,1,4,stride=2,padding=1,bias=False),
        )
        self.norm = nn.GroupNorm(1,1,affine=False)

    def forward(self,x):
        return self.norm(self.out(x))

class change_feature_size_x1(nn.Module):
    def __init__(self,in_channel,out_channel=1):
        super().__init__()
        self.out = nn.Sequential(
            nn.Conv2d(in_channel,64,3,padding='same',bias=False),
            nn.ReLU(),
            nn.Conv2d(64,1,1),

            # nn.ConvTranspose2d(64,1,4,stride=2,padding=1,bias=False),
        )
        self.norm = nn.GroupNorm(1,1,affine=False)

    def forward(self,x):
        return self.norm(self.out(x))


class change_feature_size(nn.Module):
    def __init__(self,in_channel,out_channel,scale_factor):
        super().__init__()
        self.out = nn.Sequential(
            nn.ConvTranspose2d(in_channel,in_channel,scale_factor,stride=scale_factor,bias=False),
            nn.Conv2d(in_channel,out_channel,1,bias=False)
        )
    def forward(self,x):
        return self.out(x)



import torch
import torchvision.ops
from torch import nn
from mamba_ssm import Mamba2
import math
import torch.nn.functional as F
import os
import sys
sys.path.extend(['/'.join(os.path.dirname(__file__).split('/')[:])])
sys.path.extend(['/'.join(os.path.dirname(__file__).split('/')[:-2])])
from utils import *
class DeformableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 bias=False):
        super().__init__()

        kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        self.dilation = dilation

        self.offset_conv = nn.Conv2d(in_channels,
                                     2 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=self.padding,
                                     dilation=self.dilation,
                                     bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        if self.offset_conv.bias is not None:
            nn.init.constant_(self.offset_conv.bias, 0.)

        self.modulator_conv = nn.Conv2d(in_channels,
                                        1 * kernel_size[0] * kernel_size[1],
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=self.padding,
                                        dilation=self.dilation,
                                        bias=True)

        nn.init.constant_(self.modulator_conv.weight, 0.)
        if self.modulator_conv.bias is not None:
            nn.init.constant_(self.modulator_conv.bias, 0.)

        self.regular_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      dilation=self.dilation,
                                      bias=bias)

    def forward(self, x):

        offset = self.offset_conv(x) 
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))

        x = torchvision.ops.deform_conv2d(input=x,
                                          offset=offset,
                                          weight=self.regular_conv.weight,
                                          bias=self.regular_conv.bias,
                                          padding=self.padding,
                                          mask=modulator,
                                          stride=self.stride,
                                          dilation=self.dilation)
        return x




class DepthwiseConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 bias=False):
        super().__init__()
        self.out = nn.Sequential(
            nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias),
            nn.Conv2d(in_channels,out_channels,1,bias=bias)
        )
    def forward(self,x):
        return self.out(x)

class LeakyBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 bias=False):
        super().__init__()
        self.identity=nn.Conv2d(in_channels,out_channels,1,bias=bias)
        self.out = nn.Sequential(
            DepthwiseConv(in_channels,out_channels,kernel_size,padding=1,bias=bias),
            nn.GroupNorm(out_channels,out_channels,affine=bias),
            nn.LeakyReLU(),
            DepthwiseConv(out_channels,out_channels,kernel_size,padding=1,bias=bias),
            nn.GroupNorm(out_channels,out_channels,affine=bias),
            nn.LeakyReLU(),
        )
    def forward(self,x):
        return self.out(x)+self.identity(x)

class SFFA(nn.Module):
    def __init__(self,
                 in_channels,
                 mode='same_pad',
                 bias=False):
        super().__init__()
        self.mamba_lst=nn.ModuleList(
            [
                Mamba2(d_model=in_channels,norm_before_gate=True,bias=bias,conv_bias=bias),
                Mamba2(d_model=in_channels,norm_before_gate=True,bias=bias,conv_bias=bias),
                Mamba2(d_model=in_channels,norm_before_gate=True,bias=bias,conv_bias=bias)
            ]
        )
        self.phrase_2=nn.ModuleList([
            nn.Sequential(
                DepthwiseConv(in_channels,in_channels),
                nn.GroupNorm(in_channels,in_channels,affine=False),
                nn.SiLU()
            ),
            nn.Sequential(
                DepthwiseConv(in_channels,in_channels),
                nn.GroupNorm(in_channels,in_channels,affine=False),
                nn.SiLU()
            )
        ])
        self.mode = mode
    def forward(self,x):
        output=torch.zeros_like(x)
        b,c,h,w=x.shape
        for m_i,i in enumerate([1,2,4]):
            if self.mode=='same_pad':
                pad_width = ((w-1) - w + i)/2
                pad_height =((h-1) - h + i)/2
                pad=(math.floor(pad_width),math.ceil(pad_width),math.floor(pad_height),math.ceil(pad_height))
                stride=(1,1)
                pad_x=F.pad(x,pad,mode='reflect')
                pool_x=nn.MaxPool2d(i,stride=stride)(pad_x)
            else:
                pool_x=nn.MaxPool2d(i)(x)
            flatten_x=pool_x.permute(0,2,3,1).contiguous().flatten(1,2) #b,n_seq,c
            mamba_x=self.mamba_lst[m_i](flatten_x).view(b,h//i if self.mode!='same_pad' else h,w//i if self.mode!='same_pad' else w,c).permute(0,3,1,2)
            mamba_x=(mirror_padding(mamba_x) if self.mode=='same_pad' else mirror_padding(F.interpolate(mamba_x,scale_factor=i)))[:,:,:h,:w]
            
            h_m,w_m=mamba_x.shape[-2:]
            d_h,d_w=abs(h_m-h)/2,abs(w_m-w)/2

            output+=F.pad(mamba_x,(math.floor(d_w),math.ceil(d_w),math.floor(d_h),math.ceil(d_h)),mode='reflect')

        f = torch.fft.fft2(output)
        mag = torch.abs(f)
        phase = torch.angle(f)

        mag= self.phrase_2[0](mag)
        phase=self.phrase_2[1](phase)

        real = mag * torch.cos(phase)
        imag = mag * torch.sin(phase)

        X_rec = torch.complex(real, imag)

        return torch.fft.ifft(X_rec).abs()+output

class CDCA(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                stride=1,
                padding='same',
                 bias=False):
        super().__init__()
        self.dilated_convs=nn.ModuleList(
            [nn.Sequential(DepthwiseConv(in_channels,out_channels,kernel_size,stride,padding,1,bias),nn.GroupNorm(out_channels,out_channels,affine=bias),nn.ReLU()),
             nn.Sequential(DepthwiseConv(in_channels,out_channels,kernel_size,stride,padding,2,bias),nn.GroupNorm(out_channels,out_channels,affine=bias),nn.ReLU()),
             nn.Sequential(DepthwiseConv(in_channels,out_channels,kernel_size,stride,padding,5,bias),nn.GroupNorm(out_channels,out_channels,affine=bias),nn.ReLU())]
        )
        self.ln=nn.Sequential(nn.Conv2d(in_channels,out_channels,1,bias=bias),nn.Sigmoid())
        self.gap=nn.AdaptiveMaxPool2d(1)
        self.dconv=DeformableConv2d(3*out_channels,out_channels)
    def forward(self,x):
        out=[]
        for i in range(len(self.dilated_convs)):
            out.append(self.dilated_convs[i](x))
        cat_x=torch.cat(out,dim=1)
        cat_x=self.dconv(cat_x)
        gap_x=self.gap(x)
        return self.ln(gap_x)*cat_x
    
class CFA(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=False):
        super().__init__()
        self.h_branch=nn.Sequential(
            DepthwiseConv(in_channels,in_channels,bias=bias),
            nn.ReLU(),
            nn.Sigmoid()
        )
        self.l_branch=nn.Sequential(
            DepthwiseConv(in_channels,in_channels,bias=bias),
            nn.GELU(),
            nn.Sigmoid()
        )
        self.out =  nn.Conv2d(2*in_channels,out_channels,1,bias=bias)

    def forward(self,flow,fhigh):
        f_flow=fhigh*self.l_branch(flow)
        f_fhigh=flow*self.h_branch(fhigh)
        cat_f=torch.cat((f_fhigh,f_flow),1)
        return self.out(cat_f)


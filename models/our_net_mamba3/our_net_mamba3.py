import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from modules import *
from bottle_neck import *
import random

class SegModel(nn.Module):
    def __init__(self, in_channels,out_channels, width=64):
        super().__init__()
        self.out_channels=out_channels
        c = width
        # self.eps = 1e-7
        self.down_0=down_sampling(in_channels,c,(64,64)) #B,64,32,32
        self.down_1=down_sampling(c,c,(32,32)) #B,128,16,16
        self.down_2=down_sampling(c,c,(16,16))#B,256,8,8
        self.bneck=nn.Sequential(CAB_1(c),BottleNeck_2(c),MAB(c,(8,8)),CAB(c))
        self.up_0=up_sampling(c,c,c,(16,16)) #B,64,32,32
        self.up_1=up_sampling(c,c,c,(32,32)) #B,128,16,16
        
        self.up_2=up_sampling(c,c,c,(64,64))#B,256,8,8
        self.sig=nn.Identity()
        self.final_refine=VesselRefineBlock(c, gamma_init=0.05)
        self.out_fut=nn.Sequential(
            ConvFunc(c,with_activate=True),
            nn.Conv2d(c,out_channels,1,bias=False),
        )
        self.out=nn.Sequential(
            ConvFunc(c,with_activate=True),
            nn.Conv2d(c,out_channels,1,bias=False),
        )
        self.up_f_0=nn.Sequential(
            UpFunc(c,c,4),
            ConvFunc(c,with_activate=True),
            nn.Conv2d(c,out_channels,1,bias=False)
            # nn.Sigmoid()
        )
        self.up_f_1=nn.Sequential(
            UpFunc(c,c,2),
            ConvFunc(c,with_activate=True),
            nn.Conv2d(c,out_channels,1,bias=False)
            # nn.Sigmoid()
        )
        self.swl=swl((64,64))
        self._init_weights()    
    def _init_weights(self):
        # init conv đi qua Sigmoid
        for layer in [self.out_fut, self.out, self.up_f_0, self.up_f_1]:
            last_conv = layer[-1]  # conv cuối
            if isinstance(last_conv, nn.Conv2d):
                nn.init.xavier_uniform_(last_conv.weight)
                if last_conv.bias is not None:
                    nn.init.constant_(last_conv.bias, 0)
    def forward(self, x):
        b,c,h,width=x.shape
        x_0,down_0=self.down_0(x)
        x_1,down_1=self.down_1(down_0)
        x_2,down_2=self.down_2(down_1)
        bneck=self.bneck(down_2)
        x_up_0,x_up_0_t=self.up_0(bneck,x_2,bneck) #B,256,16,16
        x_up_1,x_up_1_t=self.up_1(x_up_0,x_1,x_up_0_t) #B,128,32,32
        x_up_2,fut=self.up_2(x_up_1,x_0,x_up_1_t) #B,64,64,64
        x_up_2=self.final_refine(x_up_2)

        fut=self.out_fut(fut)
        out=self.out(x_up_2)
        out_1=self.up_f_1(x_up_1)
        out_0=self.up_f_0(x_up_0)
        # print(merge_out.shape)
        # print(computed_w.shape)
        # final_out=((computed_w*merge_out).view(b,4,self.out_channels,h,width)).sum(dim=1)
        final_out=self.swl(out,out_0,out_1,fut)
        final_out=self.sig(final_out)
        if self.training:
           return final_out,self.sig(out),self.sig(out_0),self.sig(out_1),self.sig(fut)
        return F.sigmoid(final_out)
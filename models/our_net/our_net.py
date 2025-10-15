import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from modules import *
from bottle_neck import *

class SegModel(nn.Module):
    def __init__(self, in_channels,out_channels):
        super().__init__()
        self.out_channels=out_channels
        # self.eps = 1e-7
        self.first_conv=nn.Conv2d(in_channels,4,1,bias=False)
        self.down_0=down_sampling(4,16,(64,64)) #B,64,32,32
        self.down_1=down_sampling(16,32,(32,32)) #B,128,16,16
        self.down_2=down_sampling(32,64,(16,16))#B,256,8,8
        self.bneck=nn.Sequential(MAB(64),BottleNeck(64),CAB(64))
        self.up_0=up_sampling(64,64,64,(16,16)) #B,64,32,32
        self.up_1=up_sampling(64,64,32,(32,32)) #B,128,16,16
        
        self.up_2=up_sampling(32,32,16,(64,64))#B,256,8,8
        self.sig=nn.Identity()
        self.out=nn.Sequential(
            nn.Conv2d(16,out_channels,1,bias=False),
        )
        self.up_f_0=nn.Sequential(
            UpFunc(64,32*out_channels,4),
            MKIR(32*out_channels,32*out_channels),
            nn.Conv2d(32*out_channels,32*out_channels,3,padding='same',groups=32*out_channels,bias=False),
            MAB(32*out_channels),
            nn.Conv2d(32*out_channels,out_channels,1,bias=False)
            # nn.Sigmoid()
        )
        self.up_f_1=nn.Sequential(
            UpFunc(32,32*out_channels,2),
            MKIR(32*out_channels,32*out_channels),
            MAB(32*out_channels),
            nn.Conv2d(32*out_channels,32*out_channels,3,padding='same',groups=32*out_channels,bias=False),
            nn.Conv2d(32*out_channels,out_channels,1,bias=False)
            # nn.Sigmoid()
        )
        self.map=nn.AdaptiveAvgPool2d(1)
        self.gap=nn.AdaptiveMaxPool2d(1)
        self.awl=nn.Sequential(
            nn.Conv2d(3*out_channels,48*out_channels,1,bias=False),
            nn.GELU(),
            nn.Conv2d(48*out_channels,3*out_channels,1,bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b,c,h,width=x.shape
        x=self.first_conv(x)
        x_0,down_0=self.down_0(x)
        # print(down_0.shape)
        x_1,down_1=self.down_1(down_0)
        # print(x_1)
        x_2,down_2=self.down_2(down_1)
        bneck=self.bneck(down_2)
        # print(bneck)
        x_up_0,x_up_0_t=self.up_0(bneck,x_2,bneck) #B,256,16,16
        x_up_1,x_up_1_t=self.up_1(x_up_0,x_1,x_up_0_t) #B,128,32,32
        x_up_2,_=self.up_2(x_up_1,x_0,x_up_1_t) #B,64,64,64

        out=self.out(x_up_2)
        out_1=self.up_f_1(x_up_1)
        out_0=self.up_f_0(x_up_0)

        w=self.map(out)+self.gap(out)
        w1=self.map(out_1)+self.gap(out_1)
        w0=self.map(out_0)+self.gap(out_0)
        cat_w=torch.cat((w,w1,w0),1)
        computed_w=self.awl(cat_w)

        merge_out=torch.cat((out,out_1,out_0),1)
        # print(merge_out.shape)
        # print(computed_w.shape)
        final_out=((computed_w*merge_out).view(b,3,self.out_channels,h,width)).sum(dim=1)
        
        final_out=self.sig(final_out)
        if self.training:
           return final_out,self.sig(out),self.sig(out_1),self.sig(out_0)
        return F.sigmoid(final_out)
# Kaiming init 
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from modules import *
from bottle_neck import *

class SegModel(nn.Module):
    def __init__(self, in_channels,out_channels):
        super().__init__()
        # self.eps = 1e-7
        self.down_0=down_sampling(in_channels,16,(64,64)) #B,64,32,32
        self.down_1=down_sampling(16,32,(32,32)) #B,128,16,16
        self.down_2=down_sampling(32,64,(16,16))#B,256,8,8
        self.bneck=nn.Sequential(nn.Conv2d(64,64,1,bias=False),CAB(64),BottleNeck(64))
        self.up_0=up_sampling(64,64,64,(16,16)) #B,64,32,32
        self.up_1=up_sampling(64,64,32,(32,32)) #B,128,16,16
        self.up_2=up_sampling(32,32,16,(64,64))#B,256,8,8
        self.sig=nn.Identity()
        self.out=nn.Sequential(
            nn.Conv2d(16,out_channels,1,bias=False),
            # nn.Sigmoid()
        )
        self.up_f_0=nn.Sequential(
            UpFunc(64,out_channels,4),
            # nn.Sigmoid()
        )
        self.up_f_1=nn.Sequential(
            UpFunc(32,out_channels,2),
            # nn.Sigmoid()
        )
        self.map=nn.AdaptiveAvgPool2d(1)
        self.gap=nn.AdaptiveMaxPool2d(1)
        self.awl=nn.Sequential(
            nn.Conv2d(2*3*out_channels,3*out_channels,1,bias=False),
            nn.GroupNorm(1,3*out_channels,affine=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        x_0,down_0=self.down_0(x)
        x_1,down_1=self.down_1(down_0)
        x_2,down_2=self.down_2(down_1)
        bneck=self.bneck(down_2)
        x_up_0,x_up_0_t=self.up_0(bneck,x_2,bneck) #B,256,16,16
        x_up_1,x_up_1_t=self.up_1(x_up_0,x_1,x_up_0_t) #B,128,32,32
        x_up_2,_=self.up_2(x_up_1,x_0,x_up_1_t) #B,64,64,64

        out=self.out(x_up_2)
        out_1=self.up_f_1(x_up_1)
        out_0=self.up_f_0(x_up_0)

        merge_out=torch.cat((out,out_1,out_0),1)
        map_out=self.map(merge_out)
        gap_out=self.gap(merge_out)
        w=self.awl(torch.cat((map_out,gap_out),1))

        chunk_w=w.chunk(3,dim=1)
        chunk_out =merge_out.chunk(3,dim=1)
        # print(chunk_w[0].shape)
        final_out=self.sig(chunk_w[0]*chunk_out[0]+chunk_w[1]*chunk_out[1]+chunk_w[2]*chunk_out[2])
        if self.training:
           return final_out,self.sig(out),self.sig(out_1),self.sig(out_0)
        return F.sigmoid(final_out)

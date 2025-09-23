import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from modules import *
from bottle_neck import *

class SegModel(nn.Module):
    def __init__(self, in_channels,out_channels):
        super().__init__()
        self.down_0=down_sampling(in_channels,32) #64,64,64->64,32,32
        self.down_1=down_sampling(32,64) #128,32,32->128,16,16
        self.bneck= nn.Sequential(
            residual(64,128,3,padding='same')
        ) #128,16,16
        self.up_0=up_sampling(128,64)  #128,32,32
        self.up_1=up_sampling(64,32) #128,32,32
        self.out=nn.Sequential(
            nn.Conv2d(32,out_channels,1,bias=False),
            nn.Sigmoid()
        )
    def forward(self,x):
        predown_0,down_0=self.down_0(x)
        predown_1,down_1=self.down_1(down_0)
        bneck=self.bneck(down_1)
        up_0=self.up_0(predown_1,down_1,bneck)
        up_1=self.up_1(predown_0,down_0,up_0)
        return self.out(up_1)
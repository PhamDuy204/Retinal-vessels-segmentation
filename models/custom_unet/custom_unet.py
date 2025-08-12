import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from modules import *

import sys
import torch.nn.functional as F

class SegModel(nn.Module):
    def __init__(self,in_channel,out_channel):
        super().__init__()
        self.encode_0 = down_sampling(in_channel,64) #b,64,64,64/b,64,32,32
        self.encode_1 = down_sampling(64,128)   #b,128,32,32/b,128,16,16


        self.before_bneck = Conv_func(128,256)
        self.bottle_neck = nn.Sequential(
            Residual_block(256,256),
            Residual_block(256,256)
        )  #b,256,16,16

        self.decode_1_0 = Up_sampling(256,128,1)   #b,128,32,32
        self.decode_1_1 = Up_sampling(256,128,1/2) #b,128,32,32
        self.merge_1_feature = nn.Conv2d(128*2,128,1,bias=False)

        self.decode_0_0 = Up_sampling(128,64,1) #b,64,64,64
        self.decode_0_1 = Up_sampling(128,64,1/2) #b,64,64,64
        self.up = Unpooling_conv(256,128)

        self.out = nn.Sequential(
            nn.Conv2d(2*64,out_channel,1,bias=False),
            nn.Sigmoid()
        )


    def forward(self,x):
        conv_0,down_0 = self.encode_0(x) #b,64,64,64/b,64,32,32
        conv_1,down_1 = self.encode_1(down_0)  #b,128,32,32/b,128,16,16

        before_neck = self.before_bneck(down_1)
        b_neck = before_neck + self.bottle_neck(before_neck)  #b,256,16,16

        up_1_0 = self.decode_1_0(b_neck,down_1) #b,128,32,32
        up_1_1 = self.decode_1_1(b_neck,conv_1)

        merge_1 = self.merge_1_feature(torch.cat((up_1_0,up_1_1),1))

        up_0_0 = self.decode_0_0(merge_1,down_0)
        up_0_1 = self.decode_0_1(self.up(b_neck),conv_0)

        merge = torch.cat((up_0_0,up_0_1),1)

        return self.out(merge)
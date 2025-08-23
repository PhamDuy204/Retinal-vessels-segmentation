import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from modules import *


class SegModel(nn.Module):
    def __init__(self,in_channel,out_channel):
        super().__init__()
        self.encode_0 = down_sampling(in_channel,32) #b,64,64,64|b,64,32,32
        
        self.encode_1 = down_sampling(32,64)#b,128,32,32|b,128,16,16
        self.encode_2 = down_sampling(64,128) #b,256,16,16|b,256,8,8

        self.bottle_neck = nn.Sequential(
            Residual_net(128,256),  #b,512,8,8
            Residual_net(256,256)
            )

        self.decode_2_0 = Up_sampling(256,128) #b,256,16,16
        self.decode_2_1 = Up_sampling(256,128)#b,256,16,16


        self.decode_1_0  = Up_sampling(128,64)#b,128,32,32
        self.decode_1_1  = Up_sampling(128,64)#b,128,32,32
        self.up_encode_2_0 = nn.ConvTranspose2d(128,64,2,2,bias=False)
        self.change_feature_decode_1 = nn.Conv2d(64*2,64,1,bias=False) 


        self.decode_0_0  = Up_sampling(64,32)#b,128,32,32
        self.decode_0_1  = Up_sampling(64,32)#b,128,32,32
        self.up_encode_1_0 = nn.ConvTranspose2d(64,32,2,2,bias=False)
        self.up_up_encode_2_0 = nn.ConvTranspose2d(64,32,2,2,bias=False)
        self.change_feature_0 = nn.Conv2d(32*3,32,1,bias=False)

        # self.change = nn.Conv2d(2*64,64,1,bias=False)


        self.out = nn.Sequential(
            nn.Conv2d(32*2,32,1,bias=False),
            # Unpooling_func(64,out_channel,2),
            nn.Conv2d(32,out_channel,3,padding='same',bias=False),
            nn.Sigmoid()
        )
        
        # self.down = nn.Conv2d(64,128,2,2,bias=False)
        # self.change_de_1_0 = nn.Conv2d(128*2,128,1,bias=False)
    def forward(self,x):

        conv_0,down_0 = self.encode_0(x)  #b,64,64,64|b,64,32,32
        
        conv_1,down_1 = self.encode_1(down_0)#b,128,32,32|b,128,16,16

        conv_2,down_2 = self.encode_2(down_1) #b,256,16,16|b,256,8,8

        b_neck = self.bottle_neck(down_2)  #b,512,8,8


        up_2_0 = self.decode_2_0(b_neck,conv_2)#b,256,16,16
        up_2_1 = self.decode_2_1(b_neck,conv_2)#b,256,16,16



        up_encode_2_0 = self.up_encode_2_0(conv_2)
        up_1_0 = self.decode_1_0(up_2_0,self.change_feature_decode_1(torch.cat((up_encode_2_0,conv_1),1)))
        up_1_1 = self.decode_1_1(up_2_1,conv_1)
        

        up_0_0 = self.decode_0_0(up_1_0,self.change_feature_0(torch.cat((conv_0,self.up_up_encode_2_0(up_encode_2_0),self.up_encode_1_0(conv_1)),1)))
        up_0_1 = self.decode_0_1(up_1_1,conv_0)

        merge = torch.cat((up_0_0,up_0_1),1)
        out = self.out(merge)
        return out
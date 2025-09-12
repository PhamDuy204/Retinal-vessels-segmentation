import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from modules import *
from bottle_neck import BottleNeck
from bottle_neck_1 import CustomBottleNeck1
# import sys
# import torch.nn.functional as F

class SegModel(nn.Module):
    def __init__(self,in_channel,out_channel):
        super().__init__()
        self.down_image = nn.Conv2d(in_channel,64,2,2,bias=False) #b,3,256,256
        
        self.encode_0 = down_sampling(64,128) #b,8,256,256/b,8,128,128
        self.encode_1 = down_sampling(128,256) #b,16,128,128/b,16,64,64

        self.bottle_neck =CustomBottleNeck1(256,256)


        self.decode_1_0 = Up_sampling(256,128) #b,16,128,128
        self.decode_1_1 = Up_sampling(256,128)#b,16,128,128

        self.up_encode_1 =  nn.ConvTranspose2d(256,128,kernel_size=2,stride=2,bias=False)

        self.decode_0_0  = Up_sampling(128,64)#b,8,256,256
        self.decode_0_1  = Up_sampling(128,64)#b,8,256,256

        self.out = nn.Sequential(
            nn.Conv2d(64*2,64,1,bias=False),
            Unpooling_func(64,out_channel,2),
            nn.Sigmoid()
        )

        self.down = nn.Conv2d(128,256,2,2,bias=False)
        self.change_down = nn.Conv2d(256*2,256,1,bias=False)
        self.up = nn.ConvTranspose2d(128,64,2,2,bias=False)
        self.change_up = nn.Conv2d(64*2,64,1,bias=False)
        

    def forward(self,x):

        # dwt = DWTForward(J=1, wave='haar')  
        # J = số level decomposition, wave = loại wavelet

        # Thực hiện DWT
        # Yl, Yh = dwt(x)
        # frequency = torch.mean(Yh[0],2)


        down_image = self.down_image(x) #b,3,256,256
        # print(down_image.shape)
        conv_0,down_0 = self.encode_0(down_image) #b,8,256,256/b,8,128,128
        conv_1,down_1 = self.encode_1(down_0) #b,16,128,128/b,16,64,64

        b_neck = self.bottle_neck(down_1) #b,32,64,64
        # print(b_neck.shape)
        up_1_0 = self.decode_1_0(b_neck,self.change_down(torch.cat((conv_1,self.down(conv_0)),1)))#b,16,128,128
        # print(up_1_0.shape)
        up_1_1 = self.decode_1_1(b_neck,conv_1)#b,16,128,128

        
        up_0_0 = self.decode_0_0(up_1_0,self.up_encode_1(conv_1))
        up_0_1 = self.decode_0_1(up_1_1,conv_0)

        merge = torch.cat((up_0_0,up_0_1),1)
        out = self.out(merge)
        return out
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from modules import *


class SegModel(nn.Module):
    def __init__(self,in_channel,out_channel):
        super().__init__()
        self.down_image = nn.Conv2d(in_channel,3,kernel_size=2,stride=2,bias=False) #b,3,256,256
        
        self.encode_0_0 = down_sampling(3,64) #b,8,256,256/b,8,128,128
        self.encode_0_1 = down_sampling(3,64) #b,8,256,256/b,8,128,128

        self.encode_1_0 = down_sampling(64,128) #b,16,128,128/b,16,64,64
        self.encode_1_1 = down_sampling(64,128) #b,16,128,128/b,16,64,64
        
        self.bottle_neck = nn.Sequential(
            Residual_net(256,256,3),  #b,32,64,64)
            Residual_net(256,256,3)
            )
        
        self.decode_1_0 = Up_sampling(256,128) #b,16,128,128
        self.decode_1_1 = Up_sampling(256,128)#b,16,128,128

        self.up_encode_1 =  nn.ConvTranspose2d(128,64,kernel_size=2,stride=2,bias=False)

        self.decode_0_0  = Up_sampling(128,64)#b,8,256,256
        self.decode_0_1  = Up_sampling(128,64)#b,8,256,256

        self.out = nn.Sequential(
            nn.Conv2d(64*2,64,1,bias=False),
            Unpooling_func(64,out_channel,2),
            nn.Sigmoid()
        )
        self.up_encode_1 = nn.ConvTranspose2d(128,64,2,2,bias=False)
    def forward(self,x):
        x = self.down_image(x)
        b,c,h,w = x.shape
        x_reverse = window_partition(x.permute(0,2,3,1),[8,8]).permute(0,3,1,2)

        en_0_0,en_0_0_down = self.encode_0_0(x)
        en_0_1,en_0_1_down = self.encode_0_1(x_reverse)
        
        en_1_0,en_1_0_down = self.encode_1_0(en_0_0_down)
        en_1_1,en_1_1_down = self.encode_1_1(en_0_1_down)

        x_reverse = window_reverse(en_1_1_down.permute(0,2,3,1),[8,8],int(h/4),int(w/4)).permute(0,3,1,2)

        # print(x_reverse.shape)
        # print(en_1_0_down.shape)
        b_neck = self.bottle_neck(torch.cat((x_reverse,en_1_0_down),1))
        
        up_1_0 = self.decode_1_0(b_neck,en_1_0)#b,16,128,128
        up_1_1 = self.decode_1_1(b_neck,en_1_0)#b,16,128,128

        up_0_0 = self.decode_0_0(up_1_0,self.up_encode_1(en_1_0))
        up_0_1 = self.decode_0_1(up_1_1,en_0_0)
        
        merge = torch.cat((up_0_0,up_0_1),1)
        out = self.out(merge)
        
        return out


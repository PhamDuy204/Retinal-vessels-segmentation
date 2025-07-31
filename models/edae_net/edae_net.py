import os
import sys
import torch.nn.functional as F
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from modules import *

class Model(nn.Module):
    def __init__(self,in_channel):
        super().__init__()
        self.conv1 = convolution(in_channel,64) #b,64,64,64
        
        self.down1 = nn.Conv2d(64,64,kernel_size=2,stride=2,bias=False) #b,64,32,32

        self.conv2 = convolution(64,128)    #b,128,32,32
        self.down2 = nn.Conv2d(128,128,kernel_size=2,stride=2,bias=False) #b,128,16,16

        self.conv3 = convolution(128,256)   #b,256,16,16
        self.down3 = nn.Conv2d(256,256,kernel_size=2,stride=2,bias=False) #b,256,8,8

        self.bottle_neck = CPSE(256) #b,256,8,8

        self.DGF_3 = DGF(256,256) # b,128,16,16
        self.MDAE_3 = MDAE()

        self.DGF_2 = DGF(128,128) #b,128,128
        self.MDAE_2 = MDAE()

        self.DGF_1 = DGF(128,64)
        self.MDAE_1 = MDAE()
 
        # self.DGF_2 = DGF(,128)
        self.b3 = change_feature_size(128,1,4)
        self.b2 = change_feature_size(128,1,2)
        self.b1 = change_feature_size(128,1,1)

        self.AWL_func = AWL(3)

    def forward(self,x):

        x_conv1 = self.conv1(x)  #b,64,606,700
        x_down1 = self.down1(x_conv1) #b,64,304,350

        x_conv2 = self.conv2(x_down1)  #b,128,304,350
        x_down2 = self.down2(x_conv2)    #b,128,158,176

        x_conv3 = self.conv3(x_down2) #b,256,158,176
        x_down3 = self.down3(x_conv3) #b,256,80,88

        bottle_neck = self.bottle_neck(x_down3) #b,256,88,88
        
        x_DGF_3 = self.DGF_3(bottle_neck,x_conv3)  #b,256,158,176
        x_MDAE_3 =  self.MDAE_3(x_DGF_3)  #b,256,158,176

        x_DGF_2 = self.DGF_2(x_DGF_3,x_conv2)
        x_MDAE_2 = self.MDAE_2(x_DGF_2)

        x_DGF_1 = self.DGF_1(x_DGF_2,x_conv1)
        x_MDAE_1 = self.MDAE_1(x_DGF_1)

        return nn.Sigmoid()(self.AWL_func(self.b3(x_MDAE_3),self.b2(x_MDAE_2),self.b1(x_MDAE_1)))


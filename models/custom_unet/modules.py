import torch
import torch.nn as nn

class Conv_func(nn.Module):
    def __init__(self,in_channel,out_channel,kernel=3):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(in_channel,out_channel,kernel_size=kernel,padding='same',bias=False),
            nn.BatchNorm2d(out_channel,affine=False),
            nn.ReLU(),
            )
    def forward(self,x):
        return self.feature(x)


class down_sampling(nn.Module):
    def __init__(self,in_channel,out_channel):
        super().__init__()
        self.out =  Conv_func(in_channel,out_channel,3)
        self.down = nn.MaxPool2d(2)
    def forward(self,x):
        out = self.out(x)
        return out,self.down(out)
    

class Residual_block(nn.Module):
    def __init__(self,in_channel,out_channel,kernel=3):
        super().__init__()
        self.feature = nn.Sequential(
            Conv_func(in_channel,out_channel,kernel = kernel),
            Conv_func(out_channel,out_channel,kernel = kernel))
        self.change_feature = nn.Conv2d(in_channel,out_channel,3,bias=False)
    def forward(self,x):
        return self.change_feature(x)+self.feature(x)


class Unpooling_conv(nn.Module):
    def __init__(self,in_channel,out_channel,scale_factor = 2):
        super().__init__()
        self.out = nn.ConvTranspose2d(in_channel,out_channel,kernel_size= scale_factor,stride = scale_factor,bias=False)
    def forward(self,x):
        return self.out(x)
    
class Up_sampling(nn.Module):
    def __init__(self,in_channel,out_channel,scale_factor):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor)
        self.out = Unpooling_conv(in_channel=out_channel*2,out_channel=out_channel)
        self.change_feature  = nn.Conv2d(in_channel,out_channel,1,bias=False)
    def forward(self,x,x_encode):
        return self.out((torch.cat((self.change_feature(x),self.upsample(x_encode)),1)))

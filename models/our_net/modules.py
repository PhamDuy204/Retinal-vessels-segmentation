import torch
import torch.nn as nn
import torch.nn.functional as F

class conv_func(nn.Module):
    def __init__(self,in_channels, out_channels,kernel_size,stride=1,padding:str|int=0,dilation=1,with_activation=True):
        super().__init__()
        self.out= nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,dilation,bias=False,padding_mode='zeros'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if with_activation else nn.Identity(),
            )
    def forward(self,x):
        return self.out(x)
    
class residual(nn.Module):
    def __init__(self,in_channels, out_channels,kernel_size,stride=1,padding:str|int=0,dilation=1) -> None:
        super().__init__()
        self.identity=nn.Conv2d(in_channels,out_channels,1,bias=False)
        self.convs= nn.Sequential(
            conv_func(in_channels,out_channels,kernel_size,stride,padding,dilation),
            conv_func(out_channels,out_channels,kernel_size,stride,padding,dilation,with_activation=False)
            )
        self.activation=nn.ReLU()
    def forward(self,x):
        id=self.identity(x)
        return self.activation(self.convs(x)+id)

class down_sampling(nn.Module):
    def __init__(self,in_channels, out_channels):
        super().__init__()
        self.res=residual(in_channels, out_channels,3,padding='same')
        self.pool=nn.MaxPool2d(2)
    def forward(self,x):
        res = self.res(x)
        return res,self.pool(res)

class up_sampling(nn.Module):
    def __init__(self,in_channels, out_channels):
        super().__init__()
        self.res = residual(2*out_channels,out_channels,3,padding='same')
        self.affine_shape = nn.Conv2d(out_channels+in_channels,out_channels,1,bias=False)
        self.up_conv=nn.ConvTranspose2d(out_channels+in_channels,out_channels,2,2,bias=False)
        self.up_sample=nn.Upsample(scale_factor=2)
    def forward(self,predown,down,cur):
        '''
        predown: out_channels,h,w
        down: out_channels,h/2,w/2
        cur: in_channels,h/2,w/2
        '''
        up_sample_cur=self.up_sample(cur)
        cat_upsample_cur_down=torch.cat((up_sample_cur,predown),1)
        cat_upsample_cur_down=self.affine_shape(cat_upsample_cur_down)
        cat_conv_cur_down=torch.cat((cur,down),1)
        cat_conv_cur_down= self.up_conv(cat_conv_cur_down)
        res = self.res(torch.cat((cat_upsample_cur_down,cat_conv_cur_down),1))
        return res


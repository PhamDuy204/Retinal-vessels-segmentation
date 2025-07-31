import torch
import torch.nn as nn

class MDAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
    def forward(self,x):
        _,c,h,w = x.shape
        if not hasattr(self, 'conv1'):
            self.conv1=nn.Sequential(
                nn.Conv2d(h,h,1,bias=False),
                nn.Sigmoid()
                )
        x_1 =self.conv1((self.gap(x.permute(0,2,1,3))))

        # fc3 = GAP_conv1(w)
        if not hasattr(self, 'conv2'):
            self.conv2=nn.Sequential(
                nn.Conv2d(w,w,1,bias=False),
                nn.Sigmoid(),
                )
        x_3 = self.conv2(self.gap(x.permute(0,3,2,1)))

        if not hasattr(self, 'conv3'):
            self.conv3=nn.Sequential(
                nn.Conv2d(c,c,1,bias=False),
                nn.Sigmoid(),
                )
        x_2 = self.conv3(self.gap(x))
        return x_2+x_3.permute(0,3,2,1)+x_1.permute(0,2,1,3)
        

class MAC(nn.Module):
    def __init__(self,in_channel):
        super().__init__()
        self.b1 = nn.Conv2d(in_channel,in_channel,3,dilation=1,padding='same',bias=False,groups=in_channel)
        self.b2 = nn.Conv2d(in_channel,in_channel,3,dilation=3,padding='same',bias=False,groups=in_channel)
        self.b3 = nn.Conv2d(in_channel,in_channel,3,dilation=5,padding='same',bias=False,groups=in_channel)
    def forward(self,x):
        return x + self.b1(x)+self.b2(x)+self.b3(x)
    

class CPSE(nn.Module):
    def __init__(self,in_channel):
        super().__init__()
        self.max_pool1d = nn.MaxPool1d(3,padding=1,stride=1)
        self.MAC = nn.Sequential(
            MAC(in_channel=in_channel),
            MAC(in_channel=in_channel),
            MAC(in_channel=in_channel),
   
        )
    def forward(self,x):
        b,c,h,w =  x.shape
        x_hori = self.max_pool1d(x.reshape(b,c*h,w)).reshape(b,c,h,w)
        x_ver = self.max_pool1d(x.permute(0,1,3,2).reshape(b,c*w,h)).reshape(b,c,w,h).permute(0,1,3,2)
        x_max = torch.max(x_hori,x_ver)
        return self.MAC(x_max) 

class DGF(nn.Module):
    def __init__(self,in_channel_high,in_channel_low,out_channel = 64):
        super().__init__()
        
        self.low_feature = nn.Conv2d(in_channel_low,out_channel,3,padding='same',bias=False) # b,64,h,w
        self.high_feature = nn.Sequential(
            nn.ConvTranspose2d(in_channel_high,in_channel_high,2,stride=2,bias=False), # b,64,h,w
            nn.Conv2d(in_channel_high,out_channel,3,padding='same',bias=False)
        )
        self.feature_concat = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channel*2,out_channel,1,bias=False),
            nn.Sigmoid(),

        )
    def forward(self,x_high_level,x_low_level):
        '''
        in_channel
        low_level: b,in_channel,h,w
        high_level : b,2*in_channel,h/2,w/2
        ''' 
        x_high = self.high_feature(x_high_level) # b,64,h,w
        x_low = self.low_feature(x_low_level) # b,64,h,w
        x_low_concat = x_low + x_low*self.feature_concat(torch.concat((x_low,x_high),1)) #b,64,h,w* b,64,1,1
        return torch.cat((x_low_concat,x_high),1)

class AWL(nn.Module):
    def __init__(self,in_channel=3):
        super().__init__()
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.GMP = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel,in_channel,kernel_size=1,bias = False),
            nn.Sigmoid(),
            )
    def forward(self,x1,x2,x3):
        x = torch.cat((x1,x2,x3),1)
        GAP = self.GAP(x)
        GMP = self.GMP(x)
        
        sum = self.conv(GMP+GAP)
        alpha,beta,gamma=torch.chunk(sum,3,dim=1)
        return alpha*x1+beta*x2+gamma*x3
    

class convolution(nn.Module):
    def __init__(self,in_channel,out_channel):
        super().__init__()
        self.out = nn.Sequential(
            nn.Conv2d(in_channel,out_channel,3,padding='same',bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),

            nn.Conv2d(out_channel,out_channel,3,padding='same',bias=False,groups=out_channel),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )
    def forward(self,x):
        return self.out(x)
    

class change_feature_size(nn.Module):
    def __init__(self,in_channel,out_channel,scale_factor):
        super().__init__()
        self.out = nn.Sequential(
            nn.ConvTranspose2d(in_channel,in_channel,scale_factor,stride=scale_factor,bias=False),
            nn.Conv2d(in_channel,out_channel,1,bias=False)
        )
    def forward(self,x):
        return self.out(x)
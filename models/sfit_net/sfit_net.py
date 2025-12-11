import sys 
import os 
import torch 
import torch.nn as nn 
import torch.nn.functional as F  
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


from modules import * 
from bottle_neck import * 



class SegModel(nn.Module):
    def __init__(self, in_channels=1, num_classes=1):
        super().__init__()
        
        self.enc1 = ConvBNReLU(in_channels=in_channels, out_channels=64)
        self.sru1 = SRU(channels=64)
        self.pool1 = nn.MaxPool2d(2)


        self.enc2 = ConvBNReLU(in_channels=64, out_channels=128)
        self.sru2 = SRU(128)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBNReLU(in_channels=128, out_channels=256)
        
        self.fit = FIT(256)
        

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.tfa1 = TFA(x_channels=256, t_channels=128, out_channels=128)
        self.up_conv1 = UpConv(128, 128) 
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.tfa2 = TFA(x_channels=128, t_channels=64, out_channels=64)
        self.up_conv2 = UpConv(64, 64)  
        
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x) 
        e1_sru = self.sru1(e1) 
        
        x = self.pool1(e1_sru)
        e2 = self.enc2(x)
        e2_sru = self.sru2(e2) 
        
        x = self.pool2(e2_sru)
        e3 = self.enc3(x)      
        
        f3 = self.fit(e3)   
        
        d1_up = self.up1(f3)   
        d1_tfa = self.tfa1(x=d1_up, t=e2_sru) 
        d1 = self.up_conv1(d1_tfa) 

        d2_up = self.up2(d1)   
        d2_tfa = self.tfa2(x=d2_up, t=e1_sru)
        d2 = self.up_conv2(d2_tfa) 
        out = self.final_conv(d2)
        
        return out
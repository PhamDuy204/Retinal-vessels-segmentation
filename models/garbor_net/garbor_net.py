import torch.nn as nn 

from .modules import DownSampling, UpSampling, OutConv

class SegModel(nn.Module): 
    def __init__(self, in_channels, num_classes): 
        super(SegModel, self).__init__() 

        self.down1 = DownSampling(in_channels=in_channels, out_channels=64) 
        self.down2 = DownSampling(in_channels=64, out_channels=128)
        self.down3 = DownSampling(in_channels=128, out_channels=256)

        self.bottle_neck = DownSampling(in_channels=256, out_channels=512)

        self.up1 = UpSampling(256 + 512, 256)  # s3 (256 ch) + bn (512 ch) = 768 ch input
        self.up2 = UpSampling(128 + 256, 128)  # s2 (128 ch) + u1 (256 ch) = 384 ch input  
        self.up3 = UpSampling(64 + 128, 64)    # s1 (64 ch) + u2 (128 ch) = 192 ch input 

        self.out_conv = OutConv(64, num_classes)

        self.sigmoid = nn.Sigmoid() 

    def forward(self, X): 

        s1, d1 = self.down1(X)
        s2, d2 = self.down2(d1)
        s3, d3 = self.down3(d2)

        _, bn = self.bottle_neck(d3)

        u1 = self.up1(s3, bn)
        u2 = self.up2(s2, u1)
        u3 = self.up3(s1, u2)

        out = self.sigmoid(self.out_conv(u3)) 

        return out 

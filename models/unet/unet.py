import os
import sys
import torch.nn.functional as F
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from modules import *
class SegModel(nn.Module):
    def __init__(self, n_channel, n_class):
        super().__init__()
        self.n_channel = n_channel
        self.n_class = n_class

        self.first_conv = DoubleConv(n_channel, 64)
        
        self.down1 = DownScaling(64, 128)
        self.down2 = DownScaling(128, 256)
        self.down3 = DownScaling(256, 512)
        self.down4 = DownScaling(512, 1024)

        self.up1 = UpScaling(1024, 512)
        self.up2 = UpScaling(512, 256)
        self.up3 = UpScaling(256, 128)
        self.up4 = UpScaling(128, 64)

        self.final_conv = OutConv(64, n_class)

    def forward(self, x):
        x1 = self.first_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return F.sigmoid(self.final_conv(x))
import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from .modules import DoubleConv, PWConv
from .modules import FeatureExtraction, FeatureFusion



class SegModel(nn.Module): 
    def __init__(self, in_channels, num_classes): 
        super(SegModel, self).__init__() 

        # Feature Extraction 
        self.fe1 = FeatureExtraction(in_channels=in_channels, out_channels=64)
        self.fe2 = FeatureExtraction(in_channels=64, out_channels=128)
        self.fe3 = FeatureExtraction(in_channels=128, out_channels=256)

        # Bottle Neck and Concatenation 
        self.maxpool1 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.maxpool12 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.bottle_neck1 = DoubleConv(in_channels=64 + 256, out_channels=256, kernel_size=3)
        self.bottle_neck2 = DoubleConv(in_channels=128 + 256, out_channels=256, kernel_size=3)
        self.pointwise = PWConv(in_channels=256 + 256, out_channels=256)

        # Feature Fusion 
        self.fs1 = FeatureFusion(in_channels=512, out_channels=256)
        # CORRECTED: The in_channels should be the sum of the channels from the skip connection (sk_fe2: 128) 
        # and the previous fusion block (out_fs1: 256), which is 128 + 256 = 384.
        self.fs2 = FeatureFusion(in_channels=384, out_channels=128)
        # CORRECTED: The in_channels should be the sum of the channels from the skip connection (sk_fe1: 64)
        # and the previous fusion block (out_fs2: 128), which is 64 + 128 = 192.
        self.fs3 = FeatureFusion(in_channels=192, out_channels=64)                                 

        # Out 
        self.out_conv = nn.Sequential(
            PWConv(in_channels=64, out_channels=num_classes), 
            nn.Sigmoid()
        )

    def forward(self, X): 

        # Feature Extraction 
        sk_fe1, out_fe1 = self.fe1(X) 
        sk_fe2, out_fe2 = self.fe2(out_fe1)
        sk_fe3, out_fe3 = self.fe3(out_fe2)

        # Bottle Neck 
        btn_1 = torch.cat([self.maxpool1(out_fe1), out_fe3], dim=1)
        btn_1 = self.bottle_neck1(btn_1)

        btn_2 = torch.cat([self.maxpool12(out_fe2), out_fe3], dim=1)
        btn_2 = self.bottle_neck2(btn_2)

        btn = torch.cat([btn_1, btn_2], dim=1)
        btn = self.pointwise(btn)


        # Feature Fusion 
        out_fs1 = self.fs1(sk_fe3, btn)
        out_fs2 = self.fs2(sk_fe2, out_fs1)
        out_fs3 = self.fs3(sk_fe1, out_fs2)

        # Out Conv 
        out = self.out_conv(out_fs3)

        return out
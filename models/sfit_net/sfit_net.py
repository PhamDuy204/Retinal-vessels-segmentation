import sys 
import os 
import torch 
import torch.nn as nn 
import torch.nn.functional as F  
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


from modules import * 
from bottle_neck import * 


def init_module_weights(module: nn.Module):
    """
    Init trọng số theo quy tắc chung:
      - Conv2d, ConvTranspose2d: Kaiming Uniform (fan_in, relu) trừ kernel=1 -> Xavier
      - Linear: Kaiming Uniform
      - GroupNorm/BatchNorm/LayerNorm: weight=1 bias=0 (nếu affine)
      - Embedding: normal std=0.02
    Gọi trong mỗi __init__() của class: init_module_weights(self)
    """
    for m in module.modules():
        # Conv2d
        if isinstance(m, nn.Conv2d):
            # m.kernel_size có thể là tuple
            k = m.kernel_size if hasattr(m, 'kernel_size') else (1,1)
            # nếu 1x1 conv thường dùng xavier; các conv khác dùng kaiming (ReLU-ish)
            if isinstance(k, tuple) and k[0] == 1 and k[1] == 1:
                nn.init.xavier_uniform_(m.weight)
            else:
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        # ConvTranspose2d
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        # Linear
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        # Normalization layers
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            # nếu layer có tham số affine (weight/bias), set weight=1 bias=0
            if hasattr(m, 'weight') and m.weight is not None:
                try:
                    nn.init.constant_(m.weight, 1)
                except Exception:
                    pass
            if hasattr(m, 'bias') and m.bias is not None:
                try:
                    nn.init.constant_(m.bias, 0)
                except Exception:
                    pass

        # Embedding
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

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
        self.tfa = TFA(x_channels=256, t_channels=256, out_channels=256)

        self.up1_0 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.up_conv1 = UpConv(128, 128) 
        self.tfa1 = TFA(x_channels=256+128, t_channels=256+128, out_channels=128)

        self.up1_1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)


        self.up2_0 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.up_conv2 = UpConv(64, 64) 
        self.tfa2 = TFA(x_channels=64+128, t_channels=64+256, out_channels=64)
        self.up2_1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1,bias=False)
        self.apply(init_module_weights)
    def forward(self, x):
        e1 = self.enc1(x) 
        e1_sru = self.sru1(e1) 
        
        mp1 = self.pool1(e1_sru)
        e2 = self.enc2(mp1)
        e2_sru = self.sru2(e2) 
        
        mp2 = self.pool2(e2_sru)
        e3 = self.enc3(mp2)      
        
        f3 = self.fit(e3)
        f3=  self.tfa(f3,f3)

        d1_0_up = self.up1_0(f3)
        e1_0_up = torch.cat((self.up_conv1(mp2),d1_0_up),1)
        d1_1_up = self.up1_1(f3)
        e1_1_up =torch.cat((e2_sru,d1_1_up),1)
        tfa1=self.tfa1(e1_0_up,e1_1_up)


        d2_0_up = self.up2_0(tfa1)
        e2_0_up = torch.cat((self.up_conv2(mp1),d2_0_up),1)

        d2_1_up = self.up2_1(f3)
        e2_1_up =torch.cat((e1_sru,d2_1_up),1)
        tfa2=self.tfa2(e2_0_up,e2_1_up)

        out = self.final_conv(tfa2)
        if self.training:
          return out
        return F.sigmoid(out)
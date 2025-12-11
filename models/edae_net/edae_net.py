import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch.nn.functional as F
from modules import *
import torch.nn.init as init

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
    def __init__(self, in_channels=1, out_channels=1):
        super(SegModel, self).__init__()
        self.down0 = downsampling(in_channels, 64)
        self.down1 = downsampling(64, 128)
        self.down2 = downsampling(128, 256)
        self.dgf22 = DGF(256,256)
        self.dgf21 = DGF(256,128)
        self.dgf10 = DGF(128,64)

        self.mdae22 = MDAE_Block(256,(16,16))
        self.mdae21 =  MDAE_Block(128,(32,32))
        self.mdae10 =  MDAE_Block(64,(64,64))

        self.cpse = CPSE_Block(256)

        self.up_0 = upsampling(256+256, 128,4)
        self.up_1 = upsampling(128+128, 64,2)
        self.up_f = upsampling(64+64, out_channels,1)

        self.awl  = awl(out_channels)
        self.apply(init_module_weights)

    def forward(self, x):
        x0,mp0 = self.down0(x)  # 64x64,32x32
        x1,mp1 = self.down1(mp0) # 32x32,16x16
        x2,mp2 = self.down2(mp1) # 16x16 ,8x8
        bneck = self.cpse(mp2)  # 8x8
        dgf22 = self.dgf22(bneck, x2)
        dgf21 = self.dgf21(dgf22, x1)
        dgf10 = self.dgf10(dgf21, x0)
        mdae22 = self.mdae22(dgf22)
        mdae21 = self.mdae21(dgf21)
        mdae10 = self.mdae10(dgf10)

        up0, out0 = self.up_0(bneck, mdae22)
        up1, out1 = self.up_1(up0, mdae21)
        _, out2 = self.up_f(up1, mdae10)
        final = self.awl(out2) + self.awl(out1) + self.awl(out0)
        if self.training:
            return out2, out1, out0,final
        return F.sigmoid(final)
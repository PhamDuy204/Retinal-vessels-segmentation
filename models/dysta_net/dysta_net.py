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
  def __init__(self,in_channels,out_channels):
    super().__init__()
    self.conv_inc = nn.Conv2d(in_channels,8,7,2,3,bias=False)
    self.down_0 = downsampling(8,8)
    self.down_1 = downsampling(8,16)
    self.bneck = DSConv(16,32)

    self.up0_0= upsampling(32,16)
    self.up0_1= upsampling(32,16)

    self.tup = nn.Sequential(
        nn.ConvTranspose2d(16,16,2,2,bias=False,groups =16),
        nn.Conv2d(16,8,1,bias=False))
    self.up1_0= upsampling(16,8)
    self.up1_1= upsampling(16,8)

    self.up_final =nn.Sequential(
        DSConv(16,8),
        nn.ConvTranspose2d(8,8,2,2,bias=False,groups=8),
        nn.Conv2d(8,out_channels,1,bias=False)
    )
    self.apply(init_module_weights)
  def forward(self,x):
    x=self.conv_inc(x)
    x_0,d0 = self.down_0(x)
    x_1,d1 = self.down_1(d0)
    bneck = self.bneck(d1)

    up0_0 = self.up0_0(bneck,x_1)
    up0_1 = self.up0_1(bneck,x_1)

    up1_0 = self.up1_0(up0_0, self.tup(x_1))
    up1_1 = self.up1_1(up0_1, x_0)

    up_m = torch.cat([up1_0,up1_1],1)
    out =self.up_final(up_m)

    if self.training:
      return out
    return F.sigmoid(out)
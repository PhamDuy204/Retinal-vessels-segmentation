from torch import nn
from torch import nn 
from torch.nn import functional as F 
import torch 

class DoubleConv(nn.Module):
    def __init__(self, in_channel, out_channel, mid_channel=None):
        super().__init__()
        if not mid_channel:
            mid_channel = out_channel

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
class EFB(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.ln_1 = nn.Linear(in_channels//2,in_channels//2,bias=False)
        self.ln_2 = nn.Linear(in_channels//2,3*in_channels//2,bias=False)
        self.out_1 = nn.Sequential(
            nn.Linear(in_channels//2,in_channels//2,bias=False),
            nn.LayerNorm(in_channels//2),
        )
        self.out_2 = nn.Sequential(
            nn.Linear(in_channels,in_channels//2,bias=False),
            nn.LayerNorm(in_channels//2),
        )
        self.out = nn.Sequential(
            nn.Linear(in_channels,in_channels,bias=False),
            nn.LayerNorm(in_channels),
            nn.ReLU(),
        )
    def forward(self,x):
        x1,x2 = torch.chunk(x,2,1)
        b,half_c,h,w = x1.shape

        new_domain_x2 = self.ln_2(x2.permute(0,2,3,1).contiguous()).permute(0,3,1,2).contiguous()
        new_domain_x1 = self.ln_1(x1.permute(0,2,3,1).contiguous()).permute(0,3,1,2).contiguous()

        new_domain_x1=new_domain_x1.flatten(2)
        new_domain_x2=new_domain_x2.flatten(2)

        q_x2,k_x2,v_x2 = new_domain_x2.chunk(3,1)

        attn_x2 = F.softmax((q_x2@(k_x2.transpose(1,2)))/(h*w*half_c),-1)@v_x2

        x2 = self.out_2(torch.cat((x2,attn_x2.view(b,half_c,h,w)),1).permute(0,2,3,1).contiguous())\
                                                                    .permute(0,3,1,2).contiguous().flatten(2)

        x1 = x1.flatten(2)
        
        corr = nn.Sigmoid()(F.softmax((new_domain_x1@(x2.transpose(1,2)))/(h*w*half_c),-1)@x2)
        x1 = x1*corr+x1

        x1 = self.out_1(x1.view(b,half_c,h,w).permute(0,2,3,1).contiguous()).permute(0,3,1,2).contiguous()
        x2 = x2.view(b,half_c,h,w)

        merge = torch.cat((x1,x2),1)
        return self.out(merge.permute(0,2,3,1).contiguous()).permute(0,3,1,2).contiguous()+x

class DownScaling(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.downscaling = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channel, out_channel),
            EFB(out_channel)
        )

    def forward(self, x):
        return self.downscaling(x)

class UpScaling(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.upscaling = nn.ConvTranspose2d(in_channel, in_channel // 2, kernel_size=2, stride=2)
        self.double_conv = nn.Sequential(DoubleConv(in_channel, out_channel),EFB(out_channel))

    def forward(self, x1, x2): # x1 from ConvTransposed, x2 from Encoder
        x1 = self.upscaling(x1)

        delta_height = x2.size()[2] - x1.size()[2]
        delta_width = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [delta_width // 2, delta_width - delta_width // 2,
                        delta_height // 2, delta_height - delta_height // 2])

        x = torch.cat([x2, x1], dim=1)

        return self.double_conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
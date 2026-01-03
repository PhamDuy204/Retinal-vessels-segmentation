import torch
import torch.nn as nn
import torch.nn.functional as F

class SRU(nn.Module):
    def __init__(self, channels, num_groups=8):
        super(SRU, self).__init__()
        
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=channels)
        self.sigmoid = nn.Sigmoid()
        self.threshold = 0.5

    def forward(self, x):
        b, c, h, w = x.shape
        
        x_out_gn = self.gn(x)
        
        gn_gamma = self.gn.weight  # Shape: (C,)
        
        w_gamma = gn_gamma / (torch.sum(gn_gamma) + 1e-6)
        w_gamma = w_gamma.view(1, c, 1, 1) 
        weights = self.sigmoid(x_out_gn * w_gamma)
        
        mask1 = (weights > self.threshold).float()
        mask2 = (weights <= self.threshold).float()
        
        x1_w = x * mask1
        x2_w = x * mask2
        
        x11, x12 = torch.chunk(x1_w, 2, dim=1)
        x21, x22 = torch.chunk(x2_w, 2, dim=1)
        
        out1 = x11 + x22 
        out2 = x21 + x12 
        out = torch.cat([out1, out2], dim=1)
        
        return out


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBNReLU, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBN, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.block(x)

class TFA(nn.Module):

    def __init__(self, x_channels, t_channels, out_channels):
        super(TFA, self).__init__()
        
        self.tau_gate = ConvBNReLU(t_channels, out_channels)
    
        self.tau_x = ConvBNReLU(x_channels, out_channels)
        self.tau_res = ConvBNReLU(t_channels, out_channels)

        self.theta_1 = ConvBN(out_channels, out_channels) 
        self.project_x_mul = nn.Identity()
        if x_channels != out_channels:
            self.project_x_mul = nn.Conv2d(x_channels, out_channels, kernel_size=1, bias=False)

        self.theta_2 = ConvBN(out_channels, out_channels)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, t):

        t_gate = self.tau_gate(t)         
        att_map = self.sigmoid(t_gate)
        
        x_proj = self.project_x_mul(x)    
        x_gated = x_proj * att_map         
        
        feat_1 = self.theta_1(x_gated)
        
        x_res = self.tau_x(x)              # Shape: (B, out, H, W)
        
        sum_1 = feat_1 + x_res
        
        feat_2 = self.theta_2(sum_1)
        
        t_res = self.tau_res(t)            
        out = feat_2 + t_res
        
        return out


class ConvFunc(nn.Module): 
    def __init__(self, in_channels, out_channels): 
        super(ConvFunc, self).__init__() 

        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, 
                                    kernel_size=3, padding='same'), 
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        ) 
    
    def forward(self, X): 
        return self.block(X)
    


class UpConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)
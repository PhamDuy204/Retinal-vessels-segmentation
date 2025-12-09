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




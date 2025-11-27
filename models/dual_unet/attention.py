import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, C_i, C_j, out_channels):
        super(Attention, self).__init__()

        self.W_i = nn.Sequential(
            nn.Conv2d(C_i, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels)
        )

        self.up_wj = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # Changed to bilinear for better results
        self.W_j = nn.Sequential(
            nn.Conv2d(C_j, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels)
        )

        self.relu = nn.ReLU()
        self.conv1x1 = nn.Conv2d(out_channels, out_channels, kernel_size=1)  # Changed to out_channels to maintain feature richness
        self.sigmoid = nn.Sigmoid()

    def forward(self, X_i, X_j):
        # Process skip connection features
        x_i = self.W_i(X_i)
        
        # Upsample features from deeper level
        x_j_raw = self.up_wj(X_j)
        x_j = self.W_j(x_j_raw)

        # Handle spatial dimension mismatch
        if x_i.shape[2:] != x_j.shape[2:]:
            # Interpolate to the larger dimensions for better feature preservation
            if x_i.shape[2] * x_i.shape[3] > x_j.shape[2] * x_j.shape[3]:
                x_j = F.interpolate(x_j, size=x_i.shape[2:], mode='bilinear', align_corners=False)
                x_j_raw = F.interpolate(x_j_raw, size=x_i.shape[2:], mode='bilinear', align_corners=False)
            else:
                x_i = F.interpolate(x_i, size=x_j.shape[2:], mode='bilinear', align_corners=False)

        # Generate attention map
        psi = self.relu(x_i + x_j)
        psi = self.conv1x1(psi)
        psi = self.sigmoid(psi)

        # Apply attention weights to processed features (not raw features)
        # This ensures channel dimensions match: psi and x_j both have out_channels
        out = psi * x_j
        
        return out  
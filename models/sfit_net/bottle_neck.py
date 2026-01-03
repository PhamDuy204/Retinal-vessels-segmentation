import torch 
import torch.nn as nn 
import torch.nn.functional as F 


class LGFI(nn.Module):
    def __init__(self, channels):
        super(LGFI, self).__init__()
        self.channels = channels
        
        self.norm1 = nn.GroupNorm(1, channels) 
        
        self.proj_q = nn.Conv2d(channels, channels, kernel_size=1,bias=False)
        self.proj_k = nn.Conv2d(channels, channels, kernel_size=1,bias=False)
        self.proj_v = nn.Conv2d(channels, channels, kernel_size=1,bias=False)
        
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1,bias=False)
        
        self.norm2 = nn.GroupNorm(1, channels)
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1,bias=False),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1,bias=False)
        )

    def forward(self, x):
        """
        Input: (B, C, H, W)
        """
        b, c, h, w = x.shape
        residual = x
        
        x_norm = self.norm1(x)
        
        q = self.proj_q(x_norm)
        k = self.proj_k(x_norm)
        v = self.proj_v(x_norm)
        
        q = q.view(b, c, -1)     # (B, C, N)
        k = k.view(b, c, -1)     # (B, C, N)
        v = v.view(b, c, -1)     # (B, C, N)
        
        attn_logits = torch.bmm(q, k.transpose(1, 2))
        
        attn_map = F.softmax(attn_logits, dim=-1) # (B, C, C)
        
        x_attn = torch.bmm(attn_map, v)
        
        x_attn = x_attn.view(b, c, h, w)
        x = residual + self.proj_out(x_attn)
        
        residual = x
        x_norm = self.norm2(x)
        x_ffn = self.ffn(x_norm)
        
        out = residual + x_ffn
        return out

class MHSA(nn.Module):

    def __init__(self, channels, img_size, num_heads=8, dim_feedforward=512):
        super(MHSA, self).__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.scale = (channels // num_heads) ** -0.5
        
        # Layer Norm
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        
        # MHSA Projections
        self.qkv = nn.Linear(channels, channels * 3,bias=False)
        self.proj = nn.Linear(channels, channels,bias=False)
        
        num_patches = img_size[0] * img_size[1]
        self.pos_bias = nn.Parameter(torch.zeros(1, num_heads, num_patches, num_patches))
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(channels, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, channels)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        n = h * w
        
        x_flat = x.flatten(2).transpose(1, 2) 
        residual = x_flat
        x_norm = self.norm1(x_flat)
        qkv = self.qkv(x_norm).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] 
        attn = (q @ k.transpose(-2, -1)) * self.scale 
        
        if self.pos_bias.shape[-1] == n:
            attn = attn + self.pos_bias
        else:
            pass 
        
        attn = attn.softmax(dim=-1)
        x_attn = (attn @ v).transpose(1, 2).reshape(b, n, c) # (B, N, C)
        x_attn = self.proj(x_attn)
        
        # Residual 1
        x = residual + x_attn
        
        ## FFN 
        residual = x
        x_norm = self.norm2(x)
        x_ffn = self.ffn(x_norm)
        
        # Residual 2
        out = residual + x_ffn
        out = out.transpose(1, 2).reshape(b, c, h, w)
        return out


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x


class FIT(nn.Module):
    def __init__(self, channels, img_size=(16, 16), num_heads=8):
        super(FIT, self).__init__()
        
        self.norm1 = LayerNorm2d(channels) 
        self.lgfi = LGFI(channels) 

        self.norm2 = LayerNorm2d(channels)
        self.mhsa = MHSA(channels, img_size, num_heads) 
        self.norm3 = LayerNorm2d(channels)
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, channels * 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels * 4, channels, kernel_size=1)
        )

    def forward(self, x):
        residual = x
        
        x_norm = self.norm1(x)
        x_lgfi = self.lgfi(x_norm)
        
        x = residual + x_lgfi
        
        residual = x
        
        x_norm = self.norm2(x)
        x_mhsa = self.mhsa(x_norm) 
        
        x = residual + x_mhsa
        residual = x
        x_norm = self.norm3(x)
        x_ffn = self.ffn(x_norm)
        x = residual + x_ffn
        
        return x
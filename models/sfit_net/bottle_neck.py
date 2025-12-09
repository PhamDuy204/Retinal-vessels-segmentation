import torch 
import torch.nn as nn 
import torch.nn.functional as F 


class LGFI(nn.Module):
    def __init__(self, channels):
        super(LGFI, self).__init__()
        self.channels = channels
        
        self.norm1 = nn.GroupNorm(1, channels) 
        
        self.proj_q = nn.Conv2d(channels, channels, kernel_size=1)
        self.proj_k = nn.Conv2d(channels, channels, kernel_size=1)
        self.proj_v = nn.Conv2d(channels, channels, kernel_size=1)
        
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)
        
        self.norm2 = nn.GroupNorm(1, channels)
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1)
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
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)
        
        # Learnable positional bias (epsilon) 
        # Kích thước bias phụ thuộc vào số lượng tokens (N x N), với N = H*W
        num_patches = img_size[0] * img_size[1]
        self.pos_bias = nn.Parameter(torch.zeros(1, num_heads, num_patches, num_patches))
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(channels, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, channels)
        )

    def forward(self, x):
        """
        Input: (B, C, H, W)
        """
        b, c, h, w = x.shape
        n = h * w
        
        # Reshape và Permute để phù hợp với MHSA tiêu chuẩn: (B, N, C) 
        x_flat = x.flatten(2).transpose(1, 2) # (B, N, C)
        
        # --- MHSA Part ---
        residual = x_flat
        x_norm = self.norm1(x_flat)
        
        # Q, K, V computation
        qkv = self.qkv(x_norm).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # (B, Heads, N, Head_Dim)
        
        # Attention: Softmax(Q * K.T + epsilon) 
        attn = (q @ k.transpose(-2, -1)) * self.scale # (B, Heads, N, N)
        
        # Cộng thêm bias vị trí (epsilon)
        if self.pos_bias.shape[-1] == n:
            attn = attn + self.pos_bias
        else:
             # Interpolate nếu kích thước ảnh đầu vào thay đổi (an toàn)
             pass 
        
        attn = attn.softmax(dim=-1)
        x_attn = (attn @ v).transpose(1, 2).reshape(b, n, c) # (B, N, C)
        x_attn = self.proj(x_attn)
        
        # Residual 1
        x = residual + x_attn
        
        # --- FFN Part ---
        residual = x
        x_norm = self.norm2(x)
        x_ffn = self.ffn(x_norm)
        
        # Residual 2
        out = residual + x_ffn
        
        # Reshape lại về (B, C, H, W) để nối vào mạng U-Net
        out = out.transpose(1, 2).reshape(b, c, h, w)
        return out

class FIT_Module(nn.Module):
    """
    Feature Interaction Transformer (FIT) Module
    Kết hợp LGFI và MHSA theo trình tự.
    
    Tham khảo: Fig. 1 và Section 2.3
    """
    def __init__(self, channels, img_size=(16, 16), num_heads=8):
        super(FIT_Module, self).__init__()
        
        # Step 1: Local-Global Feature Interaction
        self.lgfi = LGFI(channels)
        
        # Step 2: Multi-headed Self Attention (với global context)
        self.mhsa = MHSA_Block(channels, img_size, num_heads=num_heads)

    def forward(self, x):
        # Step-by-step extraction 
        # Đầu tiên qua LGFI để tương tác local-global
        x = self.lgfi(x)
        
        # Sau đó qua MHSA để nắm bắt long-range dependencies
        x = self.mhsa(x)
        
        return x

# --- Example Usage ---
if __name__ == "__main__":
    # Cấu hình giả định ở tầng đáy (bottleneck) của U-Net
    # Input size thường nhỏ (ví dụ 16x16) nhưng channels lớn (ví dụ 256)
    c = 256
    h, w = 16, 16
    input_tensor = torch.randn(2, c, h, w)
    
    # Khởi tạo FIT Module
    fit_module = FIT_Module(channels=c, img_size=(h, w), num_heads=8)
    
    # Forward pass
    output_tensor = fit_module(input_tensor)
    
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output_tensor.shape}")
    
    assert input_tensor.shape == output_tensor.shape
    print("FIT module executed successfully.")
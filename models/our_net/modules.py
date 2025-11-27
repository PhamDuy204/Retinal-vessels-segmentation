import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# from bottle_neck import *
from typing import Optional
# from mamba_ssm import Mamba2 

def _same_padding(kernel_size, dilation=1):
    k = kernel_size
    return (dilation * (k - 1)) // 2

def safe_group(channels: int, preferred: int = 8) -> int:
    pref = min(preferred, channels)
    for g in range(pref, 0, -1):
        if channels % g == 0:
            return g
    return 1

# -------------------------
# Helper: khởi tạo mặc định
# -------------------------
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
                # depthwise conv (groups == in_channels) vẫn ok với kaiming
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

# -------------------------
# Các lớp (giữ cấu trúc của bạn)
# -------------------------
class ConvFunc(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=1, padding: Optional[int]='same', dilation=1,bias=False,with_activate=True):
        super().__init__()
        # lưu các param nếu cần debug
        self._in_ch = in_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, bias=bias),
            nn.GroupNorm(1,in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size,padding='same', bias=bias),
            nn.ReLU())
        self.gn=nn.GroupNorm(safe_group(in_channels,16),in_channels)
        self.act = nn.ReLU()
        self.merge=nn.Conv2d(2*in_channels, in_channels, 1,bias=False)
        self.with_activate=with_activate

        # gọi init trong class
        init_module_weights(self)

    def forward(self, x):
        y = self.conv(x)
        out = self.merge(torch.cat((x,y),1))
        if self.with_activate:
            return self.act(self.gn(out))
        return out


class MKIR(nn.Module):
    def __init__(self, in_channels, out_channels, in_size=(64,64)):
        super().__init__()
        # project in -> out (1x1)
        self.first_conv = nn.Conv2d(in_channels, out_channels, 3,padding='same', bias=False)

        # three dilation branches (depthwise separable inside ConvFunc)
        self.b1 = ConvFunc(out_channels,3)
        self.b2 = ConvFunc(out_channels,3,dilation=3)
        self.b3 = ConvFunc(out_channels,3,dilation=5)
        # fusion
        self.out =nn.Sequential(
            nn.Conv2d(3*out_channels, out_channels, 1, bias=False),
            nn.ReLU())
        self.norm = nn.GroupNorm(1,out_channels)
        self.act=nn.ReLU()

        init_module_weights(self)

    def forward(self, x):
        x = self.first_conv(x)
        b1 = self.b1(x)
        b2 = self.b2(x)
        b3 = self.b3(x)
        return self.act(self.norm(self.out(torch.cat((b1,b2,b3),1))+x))
        
class CA(nn.Module):
    def __init__(self, channels, reduction_rate=4):
        super().__init__()
        hidden = max(channels // reduction_rate, 4)
        self.squeeze = nn.ModuleList([
            nn.AdaptiveAvgPool2d(1),
            nn.AdaptiveMaxPool2d(1)
        ])
        self.excitation = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

        init_module_weights(self)

    def forward(self, x):
        avg_feat = self.squeeze[0](x)
        max_feat = self.squeeze[1](x)
        avg_out = self.excitation(avg_feat)
        max_out = self.excitation(max_feat)
        attention = self.sigmoid(avg_out + max_out)
        return attention * x

class SA(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        pad = _same_padding(kernel_size)
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=pad, bias=False)
        self.sigmoid = nn.Sigmoid()

        init_module_weights(self)

    def forward(self, x):
        avg_feat = torch.mean(x, dim=1, keepdim=True)
        max_feat, _ = torch.max(x, dim=1, keepdim=True)
        feat = torch.cat([avg_feat, max_feat], dim=1)
        out_feat = self.conv(feat)
        attention = self.sigmoid(out_feat)
        return attention * x


class AG(nn.Module):
    def __init__(self, in_channels, in_size=(64, 64), reduction_ratio=4, spatial_kernel_size=7):
        super().__init__()
        
        self.gconv_x_u = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, groups=in_channels, padding='same', bias=False),
            nn.ReLU(),
        )
        self.gconv_x_e = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, groups=in_channels, padding='same', bias=False),
            nn.ReLU(),
        )
        self.fuse_gconv = nn.Sequential(
            nn.Conv2d(2 * in_channels, in_channels, 1, bias=False),
            nn.ReLU(),
        )
        self.channel_attention = nn.Sequential(
            nn.Conv2d(6 * in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=spatial_kernel_size,
                      padding=spatial_kernel_size // 2, bias=False),
            nn.Sigmoid()
        )
        self.final_merge = nn.Sequential(
            nn.Conv2d(2 * in_channels, in_channels, 3, padding='same', bias=False),
            nn.GroupNorm(8, in_channels, affine=False), # giữ nguyên như bạn đặt
            nn.ReLU(),
        )
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.map = nn.AdaptiveMaxPool2d(1)

        init_module_weights(self)

    def forward(self, x_e, x_u):
        '''
        x_e : tensor (b, c, h, w) từ encoder (skip connection)
        x_u: tensor (b, c, h, w) từ decoder (upsampled)
        '''
        b, c, h, w = x_e.shape

        gconv_x_u = self.gconv_x_u(x_u)
        gconv_x_e = self.gconv_x_e(x_e)

        merge = self.fuse_gconv(torch.cat((gconv_x_u, gconv_x_e), 1)) # (b, c, h, w)

        gap_x_m = self.gap(merge)
        map_m = self.map(merge)
        gap_x_e = self.gap(x_e)
        gap_x_u = self.gap(x_u)
        map_e = self.map(x_e)
        map_x_u = self.map(x_u)

        global_context = torch.cat((gap_x_e, map_e, gap_x_u, map_x_u, gap_x_m, map_m), 1)

        channel_weights = self.channel_attention(global_context)
        
        merge_ca = merge * channel_weights # (b, c, h, w)

        avg_pool = torch.mean(merge_ca, dim=1, keepdim=True) # (b, 1, h, w)
        max_pool = torch.max(merge_ca, dim=1, keepdim=True)[0] # (b, 1, h, w)
        
        spatial_weights = self.spatial_attention(torch.cat([avg_pool, max_pool], dim=1))
        
        attended_merge = merge_ca * spatial_weights # (b, c, h, w)

        fused_output = self.final_merge(torch.cat((x_e, attended_merge), 1))
        
        return fused_output + attended_merge

class DWConv(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=1, padding: Optional[int]='same', dilation=1,bias=False,with_activate=True):
        super().__init__()
        # lưu các param nếu cần debug
        self._in_ch = in_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation,groups = in_channels, bias=bias),
            nn.Conv2d(in_channels,in_channels,1,bias=False),
            nn.GroupNorm(1,in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding,groups = in_channels, bias=bias),
            nn.Conv2d(in_channels,in_channels,1,bias=False),
            )
        self.gn=nn.GroupNorm(safe_group(in_channels,16),in_channels)
        self.act = nn.ReLU()
        self.merge=nn.Conv2d(2*in_channels, in_channels, 1,bias=False)
        self.with_activate=with_activate
        # gọi init trong class
        init_module_weights(self)

    def forward(self, x):
        x_0=self.conv(x)

        return  self.act(self.gn(self.merge(torch.cat((x_0,x),1))))

class MAB(nn.Module):
    def __init__(self, in_channels,in_size=(64, 64)):
        super().__init__()
        self.first_conv= DWConv(in_channels)
        h,w= in_size
        self.branch_0_0 = DWConv(h,dilation=1)
        self.branch_0_1 = DWConv(h,dilation=3)
        self.branch_0_2 = DWConv(h,dilation=5)

        self.branch_1_0 = DWConv(w,dilation=1)
        self.branch_1_1 = DWConv(w,dilation=3)
        self.branch_1_2 = DWConv(w,dilation=5)

        self.branch_2_0 = DWConv(in_channels,dilation=1)
        self.branch_2_1 = DWConv(in_channels,dilation=3)
        self.branch_2_2 = DWConv(in_channels,dilation=5)
       
        self.merge=nn.Conv2d(9*in_channels,in_channels,1,bias=False)

        self.transform_statistic=nn.Sequential(
            DWConv(3),
            nn.Conv2d(3,in_channels,1,bias=False))
        init_module_weights(self)

    def forward(self, x):

        x_f=self.first_conv(x)

        x_0=x_f.permute(0,2,1,3).contiguous() # B,H,C,W
        x_1=x_f.permute(0,3,2,1).contiguous() # B,W,H,C

        b0_0= self.branch_0_0(x_0).permute(0,2,1,3).contiguous()
        b0_1= self.branch_0_1(x_0).permute(0,2,1,3).contiguous()
        b0_2= self.branch_0_2(x_0).permute(0,2,1,3).contiguous()
        
        b1_0= self.branch_1_0(x_1).permute(0,3,2,1).contiguous()
        b1_1= self.branch_1_1(x_1).permute(0,3,2,1).contiguous()
        b1_2= self.branch_1_2(x_1).permute(0,3,2,1).contiguous()

        b2_0= self.branch_2_0(x_f)
        b2_1= self.branch_2_1(x_f)
        b2_2= self.branch_2_2(x_f)

        m=self.merge(torch.cat((b0_0,b0_1,b0_2,b1_0,b1_1,b1_2,b2_0,b2_1,b2_2),1))*x

        avg_m=m.mean(1,keepdim=True)
        max_m=m.max(1,keepdim=True).values
        std_m=m.std(1,keepdim=True)

        w=self.transform_statistic(torch.cat((avg_m,max_m,std_m),1))
        return w*m
    

class down_sampling(nn.Module):
    def __init__(self, in_channels, out_channels, in_size):
        super().__init__()
        self.proj =nn.Sequential(nn.Conv2d(in_channels, out_channels, 3,padding='same', bias=False),ConvFunc(out_channels))
        self.down = nn.Conv2d(out_channels,out_channels,2,2,bias=False)

        init_module_weights(self)

    def forward(self, x):
        out = self.proj(x) 
        return out, self.down(out)


# class down_sampling(nn.Module): 
#     def __init__(self, in_channels, out_channels, in_size): 
#         super().__init__() 
        
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, 3, padding='same', bias=False), 
#             nn.LeakyReLU() 
#         )

#         self.mamba = Mamba2(out_channels, 64, conv_bias=False, d_conv=4, expand=2, headdim=4)
#         self.mamba_rev = Mamba2(out_channels, 64, conv_bias=False, d_conv=4, expand=2, headdim=4)

#         self.fusion = nn.Sequential(
#             SA(), 
#             MKIR(out_channels * 2, out_channels)
#         )

#         self.proj_out = nn.Sequential(
#             nn.Conv2d(out_channels * 2, out_channels, 1, bias=False),
#             nn.LeakyReLU()
#         )

#         self.down = nn.Conv2d(out_channels, out_channels, kernel_size=2, stride=2, bias=False)

    
#     def forward(self, x): 
#         x0 = self.conv(x)
#         b, c, h, w = x0.shape 

#         seq = x0.permute(0, 2, 3, 1).contiguous().view(b, h * w, c) 

#         forward_states = self.mamba(seq)

#         rev_seq = torch.flip(seq, dims=[1])
#         backward_states_rev = self.mamba_rev(rev_seq)
#         backward_states = torch.flip(backward_states_rev, dims=[1])

#         mamba_out_seq = torch.cat((forward_states, backward_states), dim=-1)
#         mamba_out_img = mamba_out_seq.view(b, h, w, c * 2).permute(0, 3, 1, 2).contiguous()
#         x1 = self.fusion(mamba_out_img)

#         out = self.proj_out(torch.cat((x1, x0), dim=1))

#         return out, self.down(out)


class UpFunc(nn.Module):
    def __init__(self, in_channels,out_channels,scale_factor=2):
        super().__init__()
        self.up_conv=nn.ConvTranspose2d(in_channels,out_channels,scale_factor,stride=scale_factor,bias=True)

        init_module_weights(self)

    def forward(self,x):
        return self.up_conv(x)

class up_sampling(nn.Module):
    def __init__(self,in_channels,in_channels_t,out_channels,in_size):
        super().__init__()
        self.ag=AG(out_channels,in_size)
        self.up_t=UpFunc(in_channels_t,out_channels,2)
        self.up=UpFunc(in_channels,out_channels,2)
        self.merge=MKIR(out_channels,out_channels)
        self.conv=ConvFunc(out_channels)
        self.mab=MAB(out_channels,in_size)
        self.cat=nn.Sequential(
            nn.Conv2d(3*out_channels,out_channels,3,padding='same',bias=False),
            nn.GroupNorm(out_channels,out_channels),
            nn.ReLU(),
            CA(out_channels),
            SA()
        )
        
        init_module_weights(self)

    def forward(self,x_u,x_e,x_u_t):
        '''
        x_u : inc,h/2,w/2
        x_u_t : inc_t,h/2,w/2
        x_e : out,h,w
        '''
        x_u_t=self.up_t(x_u_t)
        x_u_t=self.ag(x_e,x_u_t)
        x_u=self.up(x_u)

        return self.merge(self.cat(torch.cat((x_u,x_u_t,x_e),1))),self.conv(self.mab(x_u_t)+x_u_t)


class swl(nn.Module):
    def __init__(self, in_size):
        super().__init__()

        # Sửa lỗi cú pháp: thêm dấu phẩy, dùng nn.ReLU(), thêm dim cho Softmax
        self.gap=nn.AdaptiveAvgPool2d(1)
        self.gmp=nn.AdaptiveMaxPool2d(1)
        self.wl=nn.Sequential(
            nn.Conv2d(4, 4, in_size,groups=4, bias=False),)
        self.wl_1=nn.Sequential(
            nn.Conv2d(12,64,1,bias=False),
            nn.ReLU(),
            nn.Conv2d(64,4,1,bias=False),
            nn.Softmax()
        )
        self._init_weights()

    def _init_weights(self):
        # Init chung
        init_module_weights(self)

        # Tùy chỉnh: tìm conv cuối trước softmax và init nhỏ để softmax ban đầu gần đều
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m.out_channels == 4:
                # giả định: conv có out_channels==4 là logits trước softmax trong module này
                # khởi tạo rất nhỏ
                try:
                    nn.init.xavier_uniform_(m.weight, gain=0.01)
                except Exception:
                    pass
                if getattr(m, 'bias', None) is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self,x_0,x_1,x_2,x_3):
        cat_x=torch.cat((x_0,x_1,x_2,x_3),1)
        return (cat_x*self.wl_1(nn.Softmax()(torch.cat((self.gmp(cat_x),self.gap(cat_x),self.wl(cat_x)),1)))).sum(1,keepdim=True)
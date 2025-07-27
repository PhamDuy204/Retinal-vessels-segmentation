from collections import Iterable
import itertools

import torch
import math
import torch.nn.functional as F
from torch.nn import init
from torch.nn.modules.utils import _pair
from torch import nn


class AttentionBlock(nn.Module): 
    def __init__(self, in_channels, hidden_dim, nof_kernels): 
        """
        Args: 
            - in_channels: Number of channels of input image
            - hidden_dim: Hidden dimension for attention computation
            - nof_kernels: Number of kernel filters used
        Key Features:
            - Performing attention in dynamic convolution 
        """
        super(AttentionBlock, self).__init__()
        # Apply max pooling and flatten input matrix 
        self.global_pooling = nn.Sequential(
            nn.AdaptiveMaxPool2d(1), # Use adaptive max pooling for consistent output size
            nn.Flatten() 
        )
      
        self.attn_scores = nn.Sequential(
            nn.Linear(in_features=in_channels, out_features=hidden_dim), 
            nn.ReLU(inplace=True), 
            nn.Linear(hidden_dim, nof_kernels)
        )

        self.softmax = nn.Softmax(dim=1) 

    def forward(self, x, temperature=1): 
        out = self.global_pooling(x) 
        score = self.attn_scores(out)  # Use pooled output, not original x

        return self.softmax(score / temperature)

    


class DynamicConv(nn.Module):
    def __init__(self, nof_kernels, reduce, in_channels, out_channels, kernel_size, 
                 stride=1, padding=1, dilation=1, groups=1, bias=True): 
        """
        Args: 
            - nof_kernels: Number of kernels used
            - reduce: Refers to hidden_dim in attention. hidden_dim = in_channels / reduce
            - in_channels: Number of channels of input image. 
            - out_channels: Number of output channels. 
            - kernel_size: Size of filter matrix 
            - stride: stride gap 
            - padding: number of padding cell and column 
            - dilation: 
            - groups: 
            - bias: Adjust coeficient. If True, convolution has learnable params.

        Key Features: 
            - Performing Dynamic Convolution.  
        """
        super(DynamicConv, self).__init__() 
        self.nof_kernels = nof_kernels
        self.groups = groups
        self.conv_args = {'stride': stride, 'padding': padding, 'dilation': dilation}
        self.attention = AttentionBlock(in_channels, max(1, in_channels // reduce), nof_kernels)

        self.kernel_weights = nn.Parameter(
                                    torch.Tensor(
                                        nof_kernels, out_channels, 
                                        in_channels // groups, *_pair(kernel_size)
                                    ),
                                requires_grad=True)
        
        if bias: 
            self.kernel_bias = nn.Parameter(
                                    torch.Tensor(
                                        nof_kernels, out_channels
                                    ), requires_grad=True
            )
        else: 
            self.register_parameter('kernel_bias', None)
        
        self.initialize_parameters() 


    def initialize_parameters(self):
        """
        Key Features:
            - Initialize the params. 
            - Follows kaiming_uniform method because of ReLU usage in models. 
        """

        for i_kernel in range(self.nof_kernels):
            init.kaiming_uniform_(self.kernel_weights[i_kernel], a=math.sqrt(5))
        if self.kernel_bias is not None:
            bound = 1 / math.sqrt(self.kernel_weights[0, 0].numel())
            nn.init.uniform_(self.kernel_bias, -bound, bound)

    
    def forward(self, x, temperature=1): 
        batch_size = x.shape[0]

        alphas = self.attention(x, temperature)
        agg_weights = torch.sum(
            torch.mul(self.kernel_weights.unsqueeze(0), alphas.view(batch_size, -1, 1, 1, 1)), dim=1
        )
        agg_weights = agg_weights.view(-1, *agg_weights.shape[-3:]) 

        if self.kernel_bias is not None: 
            agg_bias = torch.sum(
                torch.mul(self.kernel_bias.unsqueeze(0), alphas.view(batch_size, -1, 1)), dim=1
            )
            agg_bias = agg_bias.view(-1) 
        else: 
            agg_bias = None 
        
        x_grouped = x.view(1, -1, *x.shape[-2:]) 

        out = F.conv2d(x_grouped, agg_weights, agg_bias, groups=self.groups * batch_size, **self.conv_args)
        out = out.view(batch_size, -1, *out.shape[-2:]) # batch_size, out_channels, H, W 
        
        return out
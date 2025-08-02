"""
Dynamic Convolution Module
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 


class AttentionBlock(nn.Module):
    """
    Attention mechanism for dynamic kernel selection.
    """
    
    def __init__(self, in_channels, ratios, K, temperature, init_weight=False):
        """
        Initialize the attention block for kernel selection.
        
        Args:
            in_channels (int): Number of input channels
            ratios (float): Compression ratio for hidden dimension calculation
            K (int): Number of convolution kernels to select from
            temperature (int): Temperature parameter for softmax sharpening
                              Must satisfy: temperature % 3 == 1
            init_weight (bool): Whether to initialize weights using custom scheme
        """
        super(AttentionBlock, self).__init__()
        assert temperature % 3 == 1, "Temperature must satisfy: temperature % 3 == 1"
        
        # Global context extraction
        self.avgpool = nn.AdaptiveMaxPool2d(1)
        
        # Hidden dimension calculation
        if in_channels != 3:
            hidden_dim = int(in_channels * ratios) + 1
        else:
            hidden_dim = K
            
        # Attention weight generation network
        self.fc1 = nn.Conv2d(in_channels, hidden_dim, 1, bias=False)
        self.fc2 = nn.Conv2d(hidden_dim, K, 1, bias=True)
        self.temperature = temperature
        
        if init_weight:
            self._initialize_weights()


    def _initialize_weights(self):
        """
        Initialize network weights using Kaiming normal initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def updata_temperature(self):
        """
        Update temperature parameter during training for curriculum learning.
        """
        if self.temperature != 1:
            self.temperature -= 3
            print('Change temperature to:', str(self.temperature))

    def forward(self, x):
        """
        Generate attention weights for kernel selection.
        Args:
            x (torch.Tensor): Input feature tensor (batch_size, in_channels, height, width)
            
        Returns:
            torch.Tensor: Attention weights of shape (batch_size, K)
                         where K is the number of kernels, values sum to 1.0
                         
        """
        # Extract global context
        x = self.avgpool(x)                    # Shape: (batch_size, in_channels, 1, 1)
        
        # Generate attention weights
        x = self.fc1(x)                        # Shape: (batch_size, hidden_dim, 1, 1)
        x = F.relu(x)                          # ReLU activation
        x = self.fc2(x).view(x.size(0), -1)    # Shape: (batch_size, K)
        
        # Temperature-scaled softmax for attention distribution
        return F.softmax(x / self.temperature, 1)  # Shape: (batch_size, K)


class DynamicConv(nn.Module):
    """
    Dynamic Convolution Layer with Adaptive Kernel Selection.
    Architecture:
        Input -> AttentionBlock -> Weighted Kernel Aggregation -> Grouped Conv -> Output
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, ratio=0.25, stride=1, 
                 padding=1, dilation=1, groups=1, bias=True, K=4, temperature=1, init_weight=True):
        """
        Initialize dynamic convolution layer.
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int): Size of convolution kernels (default: 3)
            ratio (float): Compression ratio for attention mechanism (default: 0.25)
            stride (int): Convolution stride (default: 1)
            padding (int): Padding size (default: 1)
            dilation (int): Dilation factor (default: 1)
            groups (int): Number of groups for grouped convolution (default: 1)
            bias (bool): Whether to use bias parameters (default: True)
            K (int): Number of parallel kernels (default: 4)
            temperature (int): Temperature parameter for attention softmax (default: 1)
            init_weight (bool): Whether to initialize weights (default: True)
        """
        super(DynamicConv, self).__init__()
        assert in_channels % groups == 0, "in_channels must be divisible by groups"
        
        # Store convolution parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        
        # Attention mechanism for kernel selection
        self.attention = AttentionBlock(in_channels, ratio, K, temperature)

        # Learnable parameters: K parallel convolution kernels
        self.weight = nn.Parameter(
            torch.randn(K, out_channels, in_channels // groups, kernel_size, kernel_size), 
            requires_grad=True
        )
        
        # Optional bias parameters
        if bias:
            self.bias = nn.Parameter(torch.zeros(K, out_channels))
        else:
            self.bias = None
            
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize convolution kernel weights using Kaiming uniform initialization.
        
        Each of the K kernels is initialized independently to ensure diverse
        initial representations before attention-based selection takes effect.
        """
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])

    def update_temperature(self):
        """
        Update temperature parameter in the attention mechanism.
        
        This enables curriculum learning by gradually making attention more focused.
        """
        self.attention.updata_temperature()

    def forward(self, x):
        """
        Forward pass through dynamic convolution.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height', width')
                         where spatial dimensions depend on stride, padding, and kernel_size
        """
        # Step 1: Generate attention weights for kernel selection
        softmax_attention = self.attention(x)  # Shape: (batch_size, K)
        batch_size, in_channels, height, width = x.size()
        
        # Step 2: Prepare input for grouped convolution
        x = x.view(1, -1, height, width)  # Combine batch and channel dims: (1, batch_size * in_channels, H, W)
        weight = self.weight.view(self.K, -1)  # Flatten kernel weights: (K, out_channels * in_channels_per_group * kernel_size²)

        # Step 3: Aggregate kernels using attention weights
        aggregate_weight = torch.mm(softmax_attention, weight).view(
            batch_size * self.out_channels, 
            self.in_channels // self.groups, 
            self.kernel_size, 
            self.kernel_size
        )
        
        # Step 4: Handle bias aggregation if present
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
            output = F.conv2d(
                x, weight=aggregate_weight, bias=aggregate_bias, 
                stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=self.groups * batch_size
            )
        else:
            output = F.conv2d(
                x, weight=aggregate_weight, bias=None, 
                stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=self.groups * batch_size
            )

        # Step 5: Reshape output to proper batch format
        output = output.view(batch_size, self.out_channels, output.size(-2), output.size(-1))
        return output


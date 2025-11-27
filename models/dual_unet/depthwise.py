import torch.nn as nn 

class DepthwiseConv(nn.Module):    
    def __init__(self, in_channels, out_channels, kernel_size, padding='same', stride=1, bias=False):
        super(DepthwiseConv, self).__init__()

        self.depthwise = nn.Conv2d(
            in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, 
            stride=stride, padding=kernel_size//2, groups=in_channels, bias=bias
        )

        self.pointwise = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, 
            kernel_size=1, padding=0, bias=bias 
        )

    def forward(self, X):
        out = self.depthwise(X) 
        out = self.pointwise(out) 

        return out 
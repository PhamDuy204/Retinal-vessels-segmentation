import torch 
import torch.nn as nn 

class DepthwiseConv(nn.Module): 
    def __init__(self, in_channels, out_channels, kernel_size=3, padding='same'): 
        super(DepthwiseConv, self).__init__() 
        self.in_channels = in_channels
        self.out_channels = out_channels 

        self.depthwise = nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, 
                                    kernel_size=3, groups=self.in_channels, padding='same')
        
        self.pointwise = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1, padding='same')
    
    
    def forward(self, X): 
        out = self.depthwise(X) 
        out = self.pointwise(X) 

        return out





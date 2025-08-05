import torch.nn as nn 


class Depthwise(nn.Module): 
    def __init__(self, in_channels, out_channels, kernel_size): 
        super(Depthwise, self).__init__() 

        self.depthwise = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, 
                                   groups=in_channels, padding=kernel_size // 2)
        
        self.pointwise = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0)

    
    def forward(self, X): 
        out = self.depthwise(X) 
        out = self.pointwise(out)

        return out 
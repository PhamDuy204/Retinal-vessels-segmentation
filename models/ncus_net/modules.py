import torch 
import torch.nn as nn 
import torch.nn.functional as F 


class ResNet(nn.Module): 
    def __init__(self, in_channels, out_channels): 
        super(ResNet, self).__init__() 
        pass

    def forward(self): 
        pass 


class VGG(nn.Module): 
    def __init__(self, in_channels, out_channels): 
        super(VGG, self).__init__() 

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding='same')
        self.group_norm1 = nn.GroupNorm(num_groups=out_channels, num_channels=out_channels, affine=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding='same')
        self.relu2 = nn.ReLU(inplace=True)
 

    def forwward(self, X): 
        X = self.conv1(X) 
        X = self.group_norm1(X) 
        X = self.relu1(X) 
        X = self.conv2(X) 
        X = self.relu2(X) 

        
        

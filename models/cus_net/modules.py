import torch 
import torch.nn as nn 
import torch.nn.functional as F 

class PWConv(nn.Module): 
    def __init__(self, in_channels, out_channels): 
        super(PWConv, self).__init__() 
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.batchnorm = nn.GroupNorm(num_groups=out_channels, num_channels=out_channels, affine=False)

    def forward(self, X): 
        return self.batchnorm(self.conv(X))


class DWConv(nn.Module): 
    def __init__(self, in_channels, out_channels, kernel_size): 
        super(DWConv, self).__init__() 

        self.depthwise = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, padding='same', groups=in_channels)
        self.groupnorm1 = nn.GroupNorm(num_groups=in_channels, num_channels=in_channels, affine=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.pointwise = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding='same')
        self.groupnorm2 = nn.GroupNorm(num_groups=out_channels, num_channels=out_channels, affine=False)
        self.relu2 = nn.ReLU(inplace=True)
    
    def forward(self, X): 

        X = self.depthwise(X) 
        X = self.groupnorm1(X) 
        X = self.relu1(X) 
        X = self.pointwise(X) 
        X = self.groupnorm2(X) 
        X = self.relu2(X) 

        return X 


class DoubleConv(nn.Module): 
    def __init__(self, in_channels, out_channels, kernel_size, hidden_dim = None): 
        super(DoubleConv, self).__init__() 
        if not hidden_dim: 
            hidden_dim = out_channels

        self.dwconv1 = DWConv(in_channels=in_channels, out_channels=hidden_dim, kernel_size = kernel_size)
        self.batchnorm1 = nn.GroupNorm(num_groups=hidden_dim, num_channels=hidden_dim, affine=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.dwconv2 = DWConv(in_channels=hidden_dim, out_channels=out_channels, kernel_size = kernel_size)
        self.batchnorm2 = nn.GroupNorm(num_groups=out_channels, num_channels=out_channels, affine=False)   
        self.relu2 = nn.ReLU(inplace=True)
    

    def forward(self, X): 

        X = self.dwconv1(X) 
        X = self.batchnorm1(X) 
        X = self.relu1(X) 
        X = self.dwconv2(X) 
        X = self.batchnorm2(X) 
        X = self.relu2(X)

        return X 
    

class FeatureExtraction(nn.Module): 
    def __init__(self, in_channels, out_channels):
        super(FeatureExtraction, self).__init__() 

        self.conv = DoubleConv(in_channels=in_channels, out_channels=out_channels, kernel_size=3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
    
    def forward(self, X): 
        X = self.conv(X) 
        out = self.maxpool(X)

        return X, out 


class FeatureFusion(nn.Module): 
    def __init__(self, in_channels, out_channels): 
        super(FeatureFusion, self).__init__() 

        self.conv = DoubleConv(in_channels=in_channels, out_channels=out_channels, kernel_size=3) 
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    
    def forward(self, skip, X): 
        X = self.upsample(X)

        delta_height = skip.size()[2] - X.size()[2]
        delta_width = skip.size()[3] - X.size()[3]

        X = F.pad(X, [delta_width // 2, delta_width - delta_width // 2,
                        delta_height // 2, delta_height - delta_height // 2])

        x = torch.cat([skip, X], dim=1)

        return self.conv(x)


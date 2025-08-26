import torch 
import torch.nn as nn 
import torch.nn.functional as F 



class TSB(nn.Module): 
    def __init__(self, in_channels, out_channels): 
        super(TSB, self).__init__() 
        
        # Layer 1 
        self.conv11= nn.Conv2d(in_channels=in_channels, out_channels=in_channels, 
                               groups=in_channels, kernel_size=3, padding='same')
        self.conv12 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, 
                                groups=in_channels, kernel_size=5, padding='same')
        self.conv13 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, 
                                groups=in_channels, kernel_size=7, padding='same')
        
        # Layer 2
        self.conv21 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                                kernel_size=1, padding='same')
        self.conv22 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                                kernel_size=1, padding='same')
        self.conv23 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=1, padding='same')
        
        # Layer 3
        self.conv31 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                                kernel_size=1, padding='same')
        self.conv32 = nn.Conv2d(in_channels=out_channels*3, out_channels=out_channels, 
                                kernel_size=1, padding='same')
        

    def forward(self, X): 
        x11 = self.conv11(X) 
        x12 = self.conv12(X) 
        x13 = self.conv13(X) 

        x21 = self.conv21(x11)
        x22 = self.conv22(x12) 
        x23 = self.conv23(x13) 

        x22 = x22 + x21 
        x23 = x23 + x22 

        x31 = self.conv31(X) 
        x32 = torch.concat([x21, x22, x23], dim=1)
        x32 = self.conv32(x32)

        out = x31 + x32

        return out  



class CustomBottleNeck(nn.Module): 
    def __init__(self, in_channels, out_channels):
        super(CustomBottleNeck, self).__init__() 

        self.conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, 
                                groups=in_channels, kernel_size=3, padding='same')
        self.sigmoid_1 = nn.Sigmoid() 

        self.conv_2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, 
                                groups=in_channels, kernel_size=3, padding='same')
        self.sigmoid_2 = nn.Sigmoid() 

        self.conv_mid = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, 
                                  groups=in_channels, kernel_size=5, padding='same')
        
        self.groupnorm = nn.GroupNorm(num_groups=in_channels, num_channels=in_channels, 
                                      affine=False)
        
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, 
                              groups=in_channels, kernel_size=3, padding='same') 
        
        self.tsb = TSB(in_channels=in_channels, out_channels=out_channels)

    def forward(self, X): 
        x_1 = self.conv_1(X)
        x_1 = self.sigmoid_1(x_1) 

        x_2 = self.conv_2(X) 
        x_2 = self.sigmoid_2(x_2)

        x_mid = self.conv_mid(X) 

        x = x_1 + x_mid + x_2
        x = self.groupnorm(x) 
        x = self.conv(x) 

        x = x + X 

        out = self.tsb(x) 

        return out 





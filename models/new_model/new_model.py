import torch
import torch.nn as nn 




class SegModel(nn.Module): 
    def __init__(self, in_channels, out_channels): 
        super(SegModel, self).__init__() 
        
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import *
from pytorch_wavelets import DWTForward
sys.path.append('/'.join(os.path.dirname(__file__).split('/')[:-3]))
from utils import *
class SegModel(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()
        # low branch
        self.conv_1_1_0=nn.Conv2d(in_channels,32,1,bias=False)
        self.block_0_0=nn.Sequential(
            SFFA(32,'default'),
            LeakyBlock(32,64)
        )
        self.block_0_1=nn.Sequential(
            SFFA(64,'default'),
            LeakyBlock(64,128)
        )
        self.ffa=SFFA(128,'default')

        # low branch
        self.conv_1_1_1=nn.Conv2d(in_channels,32,1,bias=False)
        self.block_1_0=nn.Sequential(
            CDCA(32,64),
            LeakyBlock(64,64)
        )
        self.block_1_1=nn.Sequential(
            CDCA(64,128),
            LeakyBlock(128,128)
        )
        self.cdca=CDCA(128,128)
        # merge
        self.cfa=CFA(128,256)
        self.conv_3_3=LeakyBlock(512,128)

        #out
        self.out = nn.Sequential(
            LeakyBlock(256,256),
            nn.ConvTranspose2d(256,out_channels,4,2,1),
            nn.Sigmoid()
        )
        self.dwt=DWTForward(J=1, mode='zero', wave='haar')
        for param in self.dwt.parameters():
            param.requires_grad_(False)
    def forward(self,x):
        xl, xh_lst = self.dwt(x)
        xh=xh_lst[0].sum(-3)
        xl=mirror_padding(xl)
        xh=mirror_padding(xh)
        # low
        xl= self.conv_1_1_0(xl)
        block_0_0=self.block_0_0(xl)
        block_0_1=self.block_0_1(block_0_0)
        sffa=self.ffa(block_0_1)

        # high
        xh= self.conv_1_1_1(xh)
        block_1_0=self.block_1_0(xh)
        block_1_1=self.block_1_1(block_1_0)
        cdca=self.cdca(block_1_1)

        #merge
        cfa=self.cfa(sffa,cdca)
        cfa= self.conv_3_3(torch.cat((block_0_1,block_1_1,cfa),1))
        out= self.out(torch.cat((block_0_0,block_1_0,cfa),1))
        return out

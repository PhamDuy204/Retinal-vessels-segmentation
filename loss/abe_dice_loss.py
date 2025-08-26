import torch.nn as nn
import torch
import sys
import os
sys.path.append(os.path.dirname(__file__))
from lovasz_loss import LovaszSoftmax
from hd_loss import HausdorffERLoss
class AbeDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        pass
    def forward(self,pred,truth):
        lovasz=LovaszSoftmax()(pred,truth)
        hd95_loss=HausdorffERLoss()(pred,truth.unsqueeze(1))
        pred=pred.squeeze().float()
        truth=truth.squeeze().float()
        # print(truth.shape)
        # hd95_loss=HausdorffERLoss()(pred,truth)
        # lovasz=LovaszSoftmax()(pred,truth)
        # Abe Dice Loss
        erc = torch.pow(pred,2*(1-(pred**2)))
        abe_diceloss_all = 1-(torch.sum(2*erc*truth)/torch.sum(erc**2 +truth))
        # diceloss = 1-(torch.sum(2*pred*truth)/torch.sum(pred +truth)) 
        focal_loss = torch.mean(-((1-pred)**2)*truth*torch.log(pred+1e-7)-(pred**2)*(1-truth)*torch.log((1-pred)+1e-7))
        return nn.L1Loss()(pred.flatten(),truth.flatten())+20*focal_loss+abe_diceloss_all+nn.BCELoss()(pred,truth)+lovasz+0.2*hd95_loss
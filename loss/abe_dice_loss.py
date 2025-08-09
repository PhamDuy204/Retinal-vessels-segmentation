import torch.nn as nn
import torch

class AbeDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        pass
    def forward(self,pred,truth):
        pred=pred.squeeze().float()
        truth=truth.squeeze().float()
        # Abe Dice Loss
        # erc = torch.pow(pred,2*(1-(pred**2)))
        abe_diceloss_all = 1-(torch.sum(2*pred*truth)/torch.sum(pred +truth)) 
        return 0.7*abe_diceloss_all + 0.3*nn.BCELoss()(pred.flatten(),truth.flatten())
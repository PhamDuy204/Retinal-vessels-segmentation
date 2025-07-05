import torch.nn as nn
import torch

class AbeDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        pass
    def forward(self,pred,truth):
        pred=pred.squeeze()
        truth=truth.squeeze()
        # Abe Dice Loss

        erc = torch.pow(pred,2*(1-(pred**2)))
        abe_diceloss_all = 1-(torch.sum(2*erc*truth)/torch.sum(erc**2 +truth))

        return abe_diceloss_all
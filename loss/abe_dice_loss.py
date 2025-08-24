import torch.nn as nn
import torch

class AbeDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        pass
    def forward(self,pred,truth):
        pred=pred.squeeze().float().flatten()
        truth=truth.squeeze().float().flatten()
        # Abe Dice Loss
        erc = torch.pow(pred,2*(1-(pred**2)))
        abe_diceloss_all = 1-(torch.sum(2*erc*truth)/torch.sum(erc**2 +truth))
        # diceloss = 1-(torch.sum(2*pred*truth)/torch.sum(pred +truth)) 
        focal_loss = torch.mean(-((1-pred)**2)*truth*torch.log(pred+1e-7)-(pred**2)*(1-truth)*torch.log((1-pred)+1e-7))
        return nn.L1Loss()(pred.flatten(),truth.flatten())+20*focal_loss+abe_diceloss_all+nn.BCELoss()(pred,truth)
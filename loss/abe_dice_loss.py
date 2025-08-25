import torch.nn as nn
import torch
import torch.nn.functional as F

class AbeDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        pass
    def forward(self,pred,truth):
        # pred=pred.squeeze().float().flatten()
        # truth=truth.squeeze().float().flatten()
        # # Abe Dice Loss
        # erc = torch.pow(pred,2*(1-(pred**2)))
        # abe_diceloss_all = 1-(torch.sum(2*erc*truth)/torch.sum(erc**2 +truth))
        # # diceloss = 1-(torch.sum(2*pred*truth)/torch.sum(pred +truth)) 
        # focal_loss = torch.mean(-((1-pred)**2)*truth*torch.log(pred+1e-7)-(pred**2)*(1-truth)*torch.log((1-pred)+1e-7))
        # return nn.L1Loss()(pred.flatten(),truth.flatten())+20*focal_loss+abe_diceloss_all+nn.BCELoss()(pred,truth)

        '''
        pred : b,c,h,w
        target : b,c,h,w
        '''
        pred = pred.squeeze()
        truth = truth.squeeze()
        b,h,w = truth.shape
        
        unfold_pred=F.unfold(pred,8,stride=4).reshape(b,8,8,-1).permute(0,3,1,2)
        unfold_target = F.unfold(truth,8,stride=4).reshape(b,8,8,-1).permute(0,3,1,2)

        # print(unfold_pred.shape)
        sum_=torch.sum(unfold_target,(-1,-2))
        dice_loss = 1-(torch.sum(2*unfold_pred*unfold_target,(-1,-2))/torch.sum(unfold_pred+unfold_target,(-1,-2)))
        print(dice_loss)
        out = (sum_ > 8).long()*dice_loss
        w = nn.Softmax(1)(sum_/-64)
        
        return torch.mean(torch.sum(w*out,-1))
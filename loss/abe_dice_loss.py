import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

class MultiScopeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        pass
    def forward(self,pred,truth):
        b = truth.shape[0]
        pred = pred.squeeze(1).float()
        truth = truth.squeeze(1).float()
        loss=torch.zeros(1).to(pred.device)
        for i,wl in [(4,0.5),(8,0.2),(16,0.3)]:
            unfold_pred=F.unfold(pred,i,stride=max(i//2,1)).reshape(b,i,i,-1).permute(0,3,1,2)
            unfold_target = F.unfold(truth,i,stride=max(i//2,1)).reshape(b,i,i,-1).permute(0,3,1,2)
            sum_=torch.sum(unfold_target,(-1,-2))
            dice_loss = 1-(torch.sum(2*unfold_pred*unfold_target,(-1,-2))/torch.sum(unfold_pred+unfold_target,(-1,-2)))
            f=torch.where(sum_ > (i*i)//3,1,1e-6)
            w = F.sigmoid(sum_/-sum_.shape[-1])
            loss+=torch.mean(torch.sum(w*dice_loss*f,-1))*wl
        return loss

class AbeDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        pass
    def forward(self,pred,truth):

        '''
        pred : b,c,h,w
        target : b,c,h,w
        '''
        b = truth.shape[0]
        pred = pred.squeeze(1).float()
        truth = truth.squeeze(1).float()
        # loss=torch.zeros(1).to(pred.device)
        # for i,wl in [(4,0.5),(8,0.2),(16,0.3)]:
        #     unfold_pred=F.unfold(pred,i,stride=max(i//2,1)).reshape(b,i,i,-1).permute(0,3,1,2)
        #     unfold_target = F.unfold(truth,i,stride=max(i//2,1)).reshape(b,i,i,-1).permute(0,3,1,2)
        #     sum_=unfold_target.sum(dim=[-2,-1])
        #     fil_=torch.where(sum_>i*i//3)
        #     unfold_pred=unfold_pred[fil_[0],fil_[1]].flatten()
        #     unfold_target=unfold_target[fil_[0],fil_[1]].flatten()
        #     focal_loss = torch.mean(-((1-unfold_pred)**2)*unfold_target*torch.log(unfold_pred)-(unfold_pred**2)*(1-unfold_target)*torch.log((1-unfold_pred)))
        diceloss = 1-(torch.sum(2*pred*truth)/(torch.sum(pred +truth)+1e-6))
        mse_loss = nn.MSELoss()(pred,truth)
        bce_loss = nn.BCELoss()(pred,truth)
        js_loss=1-(torch.sum(pred*truth)/(torch.sum(pred +truth -pred*truth)+1e-12))
            # loss+=(10*focal_loss+diceloss+mse_loss+bce_loss)*wl
        focal_loss = torch.mean(-((1-pred)**2)*truth*torch.log(pred+1e-6)-(pred**3)*(1-truth)*torch.log((1-pred)+1e-6))
        all_loss= diceloss+5*focal_loss+mse_loss+bce_loss+js_loss
        # print(all_loss)
        return all_loss
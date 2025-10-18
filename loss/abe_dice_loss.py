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
        for i,wl in [(3,1),(5,0),(7,0)]:
            unfold_pred=F.unfold(pred,i,stride=max(i//2,1)).reshape(b,i,i,-1).permute(0,3,1,2)
            unfold_target = F.unfold(truth,i,stride=max(i//2,1)).reshape(b,i,i,-1).permute(0,3,1,2)
            sum_=torch.sum(unfold_target,(-1,-2))
            dice_loss = 1-((torch.sum(2*unfold_pred*unfold_target,(-1,-2))+1e-6)/(torch.sum(unfold_pred+unfold_target,(-1,-2))+1e-6))
            out = torch.where(sum_ > (i*i)//3,dice_loss,0)
            # c=sum_.max(-1).values.unsqueeze(-1)
            # c=torch.where(c<1,1,c)
            w = nn.Softmax(1)(sum_/-sum_.shape[-1])
            loss+=torch.mean(torch.sum(w*out,-1))*wl
        return loss
class AbeDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        pass
    
    def compute_loss(self,pred,truth):
        # print(pred)
        pred = pred.squeeze(1).float()
        truth = truth.squeeze(1).float()
        pred_flat = pred.flatten()
        truth_flat = truth.flatten()
        sig_pred=F.sigmoid(pred)
        # jaccard_loss=1-((torch.sum(sig_pred*truth)+1e-6)/(torch.sum(sig_pred+truth-sig_pred*truth)+1e-6))
        # print(jaccard_loss)
        focal_loss = torch.mean(-((1 - pred_flat)**2) * truth_flat * torch.log(pred_flat + 1e-6) - (pred_flat**2) * (1 - truth_flat) * torch.log((1 - pred_flat) + 1e-6))
        diceloss = 1-(torch.sum(2*sig_pred*truth)/(torch.sum(sig_pred+truth)+1e-6))
        bce_loss = nn.BCEWithLogitsLoss()(pred,truth)
        return diceloss+bce_loss+5*focal_loss+nn.MSELoss()(sig_pred,truth)+5*MultiScopeLoss()(sig_pred,truth)        
    def forward(self,preds,truth):

        '''
        pred : b,c,h,w
        target : b,c,h,w
        '''
        # print(len(preds))
        t_truth=[]
        if not isinstance(preds,tuple):
            preds=(preds)
        for _ in range(len(preds)):
            t_truth.append(truth)
        cat_truth=torch.cat(t_truth,0)
        cat_pred=torch.cat([preds], 0)

        loss=self.compute_loss(cat_pred,cat_truth)
        return loss

        
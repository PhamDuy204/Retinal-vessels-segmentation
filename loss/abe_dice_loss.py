import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

# class MultiScopeLoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#         pass
#     def forward(self,pred,truth):
#         b = truth.shape[0]
#         pred = pred.squeeze(1).float()
#         truth = truth.squeeze(1).float()
#         loss=torch.zeros(1).to(pred.device)
#         for i,wl in [(4,0.5),(8,0.2),(16,0.3)]:
#             unfold_pred=F.unfold(pred,i,stride=max(i//2,1)).reshape(b,i,i,-1).permute(0,3,1,2)
#             unfold_target = F.unfold(truth,i,stride=max(i//2,1)).reshape(b,i,i,-1).permute(0,3,1,2)
#             sum_=torch.sum(unfold_target,(-1,-2))
#             dice_loss = 1-(torch.sum(2*unfold_pred*unfold_target,(-1,-2))/torch.sum(unfold_pred+unfold_target,(-1,-2)))
#             f=torch.where(sum_ > (i*i)//3,1,1e-6)
#             w = F.sigmoid(sum_/-sum_.shape[-1])
#             loss+=torch.mean(torch.sum(w*dice_loss*f,-1))*wl
#         return loss
class AbeDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        pass
    def compute_loss(self,pred,truth):
        # print(pred)
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
        focal_loss = torch.mean(-((1-F.sigmoid(pred))**2)*truth*torch.log(F.sigmoid(pred)+1e-6)-(F.sigmoid(pred)**2)*(1-truth)*torch.log((1-F.sigmoid(pred))+1e-6))
        diceloss = 1-(torch.sum(2*F.sigmoid(pred)*truth)/(torch.sum(F.sigmoid(pred)+truth)+1e-6))
        bce_loss = nn.BCEWithLogitsLoss()(pred,truth)
        return diceloss*0.5+bce_loss*0.5+2*focal_loss
        
    def forward(self,preds,truth):

        '''
        pred : b,c,h,w
        target : b,c,h,w
        '''
        
        if not isinstance(preds,tuple):
            preds=(preds)
        loss=self.compute_loss(preds[0],truth)*(1/(len(preds)+1e-6))
        for i in range(1,len(preds)):
            loss+=self.compute_loss(preds[i],truth)*((1/len(preds)+1e-6))
        # print(loss)
        return loss
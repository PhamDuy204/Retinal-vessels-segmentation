import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

class SimpleLoss(nn.Module):
    def __init__(self):
        super().__init__()
        pass
    def compute_loss(self,pred,truth):
        # print(pred)
        pred = pred.squeeze(1).float()
        truth = truth.squeeze(1).float()

        sig_pred=F.sigmoid(pred)
        # jaccard_loss=1-((torch.sum(sig_pred*truth)+1e-6)/(torch.sum(sig_pred+truth-sig_pred*truth)+1e-6))
        # print(jaccard_loss)
        focal_loss = torch.mean(-((1-sig_pred)**2)*truth*torch.log(sig_pred+1e-6)-(sig_pred**2)*(1-truth)*torch.log((1-sig_pred)+1e-6))
        diceloss = 1-(torch.sum(2*sig_pred*truth)/(torch.sum(sig_pred+truth)+1e-6))
        bce_loss = nn.BCEWithLogitsLoss()(pred,truth)
        return diceloss+bce_loss+5*focal_loss+nn.MSELoss()(sig_pred,truth)
            
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
        cat_pred=torch.cat(preds,0)

        loss=self.compute_loss(cat_pred,cat_truth)
        return loss
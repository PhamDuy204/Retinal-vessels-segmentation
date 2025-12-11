<<<<<<< HEAD
import torch.nn as nn 

class BceLoss(nn.Module):
    def __init__(self): 
        super(BceLoss, self).__init__() 
        self.loss = nn.BCELoss() 
        pass 

    def forward(self, pred, truth): 
        pred = pred.float() 
        truth = truth.float() 
        
        pred = pred.squeeze()
        truth = truth.squeeze()

        return self.loss(pred, truth)
    
=======
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

class BceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        pass
    def compute_loss(self,pred,truth):
        # print(pred)
        pred = pred.squeeze(1).float()
        truth = truth.squeeze(1).float()
        bce_loss = nn.BCEWithLogitsLoss()(pred,truth)
        return bce_loss
            
    def forward(self,preds,truth):

        '''
        pred : b,c,h,w
        target : b,c,h,w
        '''
        # print(len(preds))
        t_truth=[]
        if not isinstance(preds,tuple):
            preds=(preds,)
        for _ in range(len(preds)):
            t_truth.append(truth)
        if len(preds)>1:
            cat_truth=torch.cat(t_truth,0)
            cat_pred=torch.cat(preds,0)

            loss=self.compute_loss(cat_pred,cat_truth)
        else:
            loss=self.compute_loss(preds[0],truth)
        return loss
>>>>>>> main

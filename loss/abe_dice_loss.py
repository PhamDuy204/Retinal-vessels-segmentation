import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

class WindowLoss(nn.Module):
    def __init__(self,ws=3):
        super().__init__()
        self.ws=ws
        pass
    def forward(self,pred,truth):
        b = truth.shape[0]
        pred = pred.squeeze(1).float()
        truth = truth.squeeze(1).float()
        unfold_pred=F.unfold(pred,self.ws,stride=2).reshape(b,self.ws,self.ws,-1).permute(0,3,1,2)
        unfold_target = F.unfold(truth,self.ws,stride=2).reshape(b,self.ws,self.ws,-1).permute(0,3,1,2)
        sum_=torch.sum(unfold_target,(-1,-2))
        dice_loss = 1-(torch.sum(2*unfold_pred*unfold_target,(-1,-2))/torch.sum(unfold_pred+unfold_target,(-1,-2)))
        # print(dice_loss.shape)
        # out = torch.where(sum_ >= (self.ws**2)//3-1,dice_loss,0)
        sum_=sum_.type(torch.float32)
        NEG_INF = -1e9
        sum_=torch.where(sum_ >= (self.ws**2)//3-1,-sum_/(self.ws**2),NEG_INF)
        w = nn.Softmax(1)(sum_)
        # print(w.shape)
        loss= torch.mean(torch.sum(w*dice_loss,-1))
        # print(loss)
        return loss

class AbeDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        pass
    def forward(self,pred,truth):
        # pred=pred.squeeze().float().flatten()
        # truth=truth.squeeze().float().flatten()
        # # Abe Dice Loss
        # # erc = torch.pow(pred,2*(1-(pred**2)))
        # # abe_diceloss_all = 1-(torch.sum(2*erc*truth)/torch.sum(erc**2 +truth))
        # focal_loss = torch.mean(-((1-pred)**2)*truth*torch.log(pred+1e-7)-(pred**2)*(1-truth)*torch.log((1-pred)+1e-7))
        # # return nn.L1Loss()(pred.flatten(),truth.flatten())+20*focal_loss+abe_diceloss_all+nn.BCELoss()(pred,truth)
        # return diceloss

        '''
        pred : b,c,h,w
        target : b,c,h,w
        '''
        b = truth.shape[0]
        pred = pred.squeeze().float()
        truth = truth.squeeze().float()
        
        
        # unfold_pred=F.unfold(pred,8,stride=2).reshape(b,8,8,-1).permute(0,3,1,2)
        # unfold_target = F.unfold(truth,8,stride=2).reshape(b,8,8,-1).permute(0,3,1,2)

        # print(unfold_pred.shape)
        # sum_=torch.sum(unfold_target,(-1,-2))
        # dice_loss = 1-(torch.sum(2*unfold_pred*unfold_target,(-1,-2))/torch.sum(unfold_pred+unfold_target,(-1,-2)))
        # print(dice_loss)
        # out = torch.where(sum_ > 8,1,0)*dice_loss
        # w = nn.Softmax(1)(sum_/-64)
        focal_loss = torch.mean(-((1-pred)**2)*truth*torch.log(pred+1e-6)-(pred**2)*(1-truth)*torch.log((1-pred)+1e-6))
        # diceloss = 1-(torch.sum(2*pred*truth)/torch.sum(pred +truth))

        return 5*WindowLoss()(pred,truth)+10*focal_loss+nn.BCELoss()(pred,truth)+nn.L1Loss()(pred.flatten(),truth.flatten())
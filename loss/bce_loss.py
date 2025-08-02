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
    

import torch.nn as nn
import torch.nn.functional as F

class BceDiceLoss(nn.Module):
    def __init__(self, alpha=0.5, eps=1e-6):
        super(BceDiceLoss, self).__init__()
        self.alpha = alpha
        self.eps = eps

    def forward(self, pred, target):
        pred = pred.float()
        target = target.float()
        
        bce = F.binary_cross_entropy(pred, target)

        # Flatten
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + self.eps) / (pred_flat.sum() + target_flat.sum() + self.eps)
        dice_loss = 1 - dice

        return self.alpha * bce + (1 - self.alpha) * dice_loss

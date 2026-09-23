"""Exact-safe copy of abe_dice_loss with zero-weight work removed.

The original loss/abe_dice_loss.py is intentionally left unchanged.  This
variant evaluates the same non-zero mathematical terms and can be selected with
``--loss abe_dice_loss_optimized``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScopeLoss(nn.Module):
    SCOPES = ((3, 1), (5, 0), (7, 0))

    def forward(self, pred, truth):
        batch_size = truth.shape[0]
        pred = pred.squeeze(1).float()
        truth = truth.squeeze(1).float()
        loss = pred.new_zeros(1)
        for kernel_size, weight in self.SCOPES:
            if weight == 0:
                continue
            stride = max(kernel_size // 2, 1)
            unfold_pred = (
                F.unfold(pred, kernel_size, stride=stride)
                .reshape(batch_size, kernel_size, kernel_size, -1)
                .permute(0, 3, 1, 2)
            )
            unfold_target = (
                F.unfold(truth, kernel_size, stride=stride)
                .reshape(batch_size, kernel_size, kernel_size, -1)
                .permute(0, 3, 1, 2)
            )
            target_sum = torch.sum(unfold_target, (-1, -2))
            dice_loss = 1 - (
                (torch.sum(2 * unfold_pred * unfold_target, (-1, -2)) + 1e-6)
                / (torch.sum(unfold_pred + unfold_target, (-1, -2)) + 1e-6)
            )
            selected = torch.where(
                target_sum > (kernel_size * kernel_size) // 3,
                dice_loss,
                0,
            )
            scope_weights = torch.softmax(
                target_sum / -target_sum.shape[-1], dim=1
            )
            loss += torch.mean(torch.sum(scope_weights * selected, -1)) * weight
        return loss


class AbeDiceLossOptimized(nn.Module):
    def __init__(self):
        super().__init__()
        self.multi_scope = MultiScopeLoss()

    def compute_loss(self, pred, truth):
        pred = pred.squeeze(1).float()
        truth = truth.squeeze(1).float()
        sigmoid_pred = torch.sigmoid(pred)
        focal_loss = torch.mean(
            -((1 - sigmoid_pred) ** 2)
            * truth
            * torch.log(sigmoid_pred + 1e-6)
            - (sigmoid_pred**2)
            * (1 - truth)
            * torch.log((1 - sigmoid_pred) + 1e-6)
        )
        dice_loss = 1 - (
            torch.sum(2 * sigmoid_pred * truth)
            / (torch.sum(sigmoid_pred + truth) + 1e-6)
        )
        bce_loss = F.binary_cross_entropy_with_logits(pred, truth)
        mse_loss = F.mse_loss(sigmoid_pred, truth)
        return (
            dice_loss
            + bce_loss
            + 5 * focal_loss
            + mse_loss
            + 5 * self.multi_scope(sigmoid_pred, truth)
        )

    def forward(self, preds, truth):
        if not isinstance(preds, tuple):
            preds = (preds,)
        repeated_truth = [truth for _ in range(len(preds))]
        return self.compute_loss(torch.cat(preds, 0), torch.cat(repeated_truth, 0))

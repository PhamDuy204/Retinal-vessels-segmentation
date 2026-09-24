"""Default optimized loss used by SGMA-Net training.

This keeps the validated Abe-Dice objective while replacing materialized
``unfold`` windows with pooling-based window sums to reduce training overhead.
The numerically sensitive loss terms are accumulated in FP32.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScopeLoss(nn.Module):
    SCOPES = ((8, 0.2), (16, 0.3), (32, 0.5))

    def forward(self, pred, truth):
        pred = pred.squeeze(1).float().unsqueeze(1)
        truth = truth.squeeze(1).float().unsqueeze(1)
        loss = pred.new_zeros(1)

        for kernel_size, weight in self.SCOPES:
            stride = max(kernel_size // 2, 1)

            def window_sum(value):
                return F.avg_pool2d(
                    value,
                    kernel_size,
                    stride=stride,
                    divisor_override=1,
                ).flatten(1)

            target_sum = window_sum(truth)
            dice_loss = 1 - (
                (window_sum(2 * pred * truth) + 1e-6)
                / (window_sum(pred + truth) + 1e-6)
            )
            selected = torch.where(
                target_sum > (kernel_size * kernel_size) // 3,
                dice_loss,
                0,
            )
            scope_weights = torch.softmax(
                target_sum / -target_sum.shape[-1],
                dim=1,
            )
            loss += torch.mean(torch.sum(scope_weights * selected, -1)) * weight

        return loss


class MainLoss(nn.Module):
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
        return self.compute_loss(
            torch.cat(preds, 0),
            torch.cat(repeated_truth, 0),
        )

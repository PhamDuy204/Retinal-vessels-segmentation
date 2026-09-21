"""Global BCE + Dice supervision used for architecture ablations."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _target_like(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = target.float()
    while target.ndim < logits.ndim:
        target = target.unsqueeze(1)
    return target


def _global_bce_dice(
    logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    target = _target_like(logits, target)
    logits = logits.float()
    probability = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, target)
    dims = tuple(range(1, probability.ndim))
    intersection = (probability * target).sum(dim=dims)
    denominator = probability.sum(dim=dims) + target.sum(dim=dims)
    dice = 1.0 - ((2.0 * intersection + eps) / (denominator + eps))
    return bce + dice.mean()


class BceDiceLoss(nn.Module):
    def forward(self, predictions, target: torch.Tensor) -> torch.Tensor:
        if not isinstance(predictions, (tuple, list)):
            predictions = (predictions,)
        return sum(_global_bce_dice(prediction, target) for prediction in predictions)

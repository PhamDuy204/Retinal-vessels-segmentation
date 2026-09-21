"""Paper-aligned SGMA-Net loss: BCE + Dice + MSPG."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from loss.bce_dice_loss import _global_bce_dice, _target_like


class MSPGLoss(nn.Module):
    """Multi-Scope Patch-Guided loss, Algorithm 1 / Eq. (7)."""

    def __init__(
        self,
        scopes=(8, 16, 32),
        scale_weights=(0.5, 0.3, 0.2),
        rho: float = 0.1,
        eps: float = 1e-6,
    ):
        super().__init__()
        if len(scopes) != len(scale_weights):
            raise ValueError("scopes and scale_weights must have equal length")
        self.scopes = tuple(scopes)
        self.scale_weights = tuple(scale_weights)
        self.rho = float(rho)
        self.eps = float(eps)

    def forward(self, probability: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = _target_like(probability, target)
        probability = probability.float()
        target = target.float()
        batch_size = probability.shape[0]
        total = probability.new_zeros(())

        for kernel_size, scale_weight in zip(self.scopes, self.scale_weights):
            stride = kernel_size // 2
            pred_patches = F.unfold(
                probability, kernel_size=kernel_size, stride=stride
            )
            target_patches = F.unfold(
                target, kernel_size=kernel_size, stride=stride
            )

            foreground = target_patches.sum(dim=1)
            valid = foreground > self.rho * (kernel_size ** 2)

            intersection = (pred_patches * target_patches).sum(dim=1)
            denominator = pred_patches.sum(dim=1) + foreground
            local_dice = 1.0 - (
                (2.0 * intersection + self.eps) / (denominator + self.eps)
            )

            occupancy_weight = torch.exp(
                -foreground / float(kernel_size ** 2)
            ) * valid
            normalizer = occupancy_weight.sum(dim=1, keepdim=True)
            normalized_weight = torch.where(
                normalizer > 0,
                occupancy_weight / normalizer.clamp_min(self.eps),
                torch.zeros_like(occupancy_weight),
            )
            per_image = (normalized_weight * local_dice).sum(dim=1)
            scale_loss = per_image.sum() / batch_size
            total = total + float(scale_weight) * scale_loss

        return total


class OurLoss(nn.Module):
    """Unit-weight BCE + Dice + MSPG on final and deep-supervision outputs."""

    def __init__(self):
        super().__init__()
        self.mspg = MSPGLoss()

    def _one_output(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_like = _target_like(logits, target)
        probability = torch.sigmoid(logits.float())
        return _global_bce_dice(logits, target_like) + self.mspg(
            probability, target_like
        )

    def forward(self, predictions, target: torch.Tensor) -> torch.Tensor:
        if not isinstance(predictions, (tuple, list)):
            predictions = (predictions,)
        return sum(self._one_output(prediction, target) for prediction in predictions)

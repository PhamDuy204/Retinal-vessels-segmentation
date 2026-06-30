from __future__ import annotations

from contextlib import nullcontext

import torch
from skimage.morphology import skeletonize
from torch.profiler import record_function
from torchmetrics.classification import AUROC
from tqdm import tqdm

from utils import (
    check_model_forward_args,
    extract_patches_with_target_count,
    mirror_padding_v2,
    reverse_to_original_image,
)


EVALUATION_THRESHOLD = 0.487


def _profile_region(enabled: bool, name: str):
    return record_function(name) if enabled else nullcontext()


def metrics_from_confusion(
    true_positive: torch.Tensor,
    true_negative: torch.Tensor,
    false_positive: torch.Tensor,
    false_negative: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Match the existing binary torchmetrics formulas from one confusion matrix."""

    def divide(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
        return torch.where(
            denominator > 0,
            numerator.float() / denominator.float(),
            torch.zeros((), dtype=torch.float32, device=denominator.device),
        )

    total = true_positive + true_negative + false_positive + false_negative
    accuracy = divide(true_positive + true_negative, total)
    f1 = divide(2 * true_positive, 2 * true_positive + false_positive + false_negative)
    iou = divide(true_positive, true_positive + false_positive + false_negative)
    recall = divide(true_positive, true_positive + false_negative)
    specificity = divide(true_negative, true_negative + false_positive)
    return accuracy, f1, iou, recall, specificity


def centerline_dice(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> float:
    """Compute hard centerline Dice (clDice/cDice) for one binary image.

    The score is the harmonic mean of topology precision and topology
    sensitivity, using morphological skeletons of the prediction and target.
    """
    prediction_mask = prediction.detach().bool().cpu().numpy()
    target_mask = target.detach().bool().cpu().numpy()
    prediction_skeleton = skeletonize(prediction_mask)
    target_skeleton = skeletonize(target_mask)

    prediction_centerline_pixels = int(prediction_skeleton.sum())
    target_centerline_pixels = int(target_skeleton.sum())
    if prediction_centerline_pixels == 0 and target_centerline_pixels == 0:
        return 1.0
    if prediction_centerline_pixels == 0 or target_centerline_pixels == 0:
        return 0.0

    topology_precision = float(
        (prediction_skeleton & target_mask).sum() / prediction_centerline_pixels
    )
    topology_sensitivity = float(
        (target_skeleton & prediction_mask).sum() / target_centerline_pixels
    )
    denominator = topology_precision + topology_sensitivity
    if denominator == 0:
        return 0.0
    return 2.0 * topology_precision * topology_sensitivity / denominator


def _forward_in_batches(
    model,
    image: torch.Tensor,
    edge: torch.Tensor | None,
    batch_size: int,
    forward_args: int,
    amp: bool,
    profile: bool,
    pad_final_batch: bool,
) -> torch.Tensor:
    if batch_size <= 0:
        chunk_count = max(image.shape[0] // 128, 1)
        image_batches = torch.chunk(image, chunk_count, dim=0)
        edge_batches = (
            torch.chunk(edge, chunk_count, dim=0)
            if edge is not None
            else (None,) * len(image_batches)
        )
        outputs = []
        for image_batch, edge_batch in zip(image_batches, edge_batches):
            with _profile_region(profile, "evaluation_model_forward"):
                with torch.amp.autocast("cuda", enabled=amp):
                    if forward_args == 2:
                        output = model(image_batch, edge_batch)
                    else:
                        output = model(image_batch)
            outputs.append(output)
        return torch.cat(outputs, dim=0)

    outputs = []
    for start in range(0, image.shape[0], batch_size):
        stop = min(start + batch_size, image.shape[0])
        image_batch = image[start:stop]
        edge_batch = edge[start:stop] if edge is not None else None
        valid_count = stop - start
        if pad_final_batch and valid_count < batch_size:
            pad_count = batch_size - valid_count
            image_padding = image_batch[-1:].expand(pad_count, -1, -1, -1)
            image_batch = torch.cat((image_batch, image_padding), dim=0)
            if edge_batch is not None:
                edge_padding = edge_batch[-1:].expand(pad_count, -1, -1, -1)
                edge_batch = torch.cat((edge_batch, edge_padding), dim=0)

        with _profile_region(profile, "evaluation_model_forward"):
            with torch.amp.autocast("cuda", enabled=amp):
                if forward_args == 2:
                    output = model(image_batch, edge_batch)
                else:
                    output = model(image_batch)
        outputs.append(output[:valid_count])
    return torch.cat(outputs, dim=0)


def eval_for_seg(
    model,
    val_loader,
    gpu_id,
    patch=False,
    patch_size=64,
    type_split="random",
    threshold=EVALUATION_THRESHOLD,
    non_blocking=True,
    amp=False,
    profile=False,
    batch_size=64,
):
    torch.cuda.set_device(gpu_id)
    device = torch.device("cuda", gpu_id)

    confusion = torch.zeros(4, dtype=torch.long, device=device)
    image_dice_scores: list[torch.Tensor] = []
    image_cdice_scores: list[float] = []
    auroc_metric = AUROC(task="binary").to(device)
    forward_args = check_model_forward_args(model)
    model.eval()

    with torch.inference_mode():
        iterator = iter(val_loader)
        for _ in tqdm(range(len(val_loader))):
            with _profile_region(profile, "evaluation_data_loading"):
                sample = next(iterator)
            image, mask, edge = sample.values()
            image = mirror_padding_v2(image)
            if forward_args == 2:
                edge = mirror_padding_v2(edge)
            else:
                edge = None
            image_count, channels, height, width = image.shape
            image = image.to(device, non_blocking=non_blocking)
            mask = mask.to(device, non_blocking=non_blocking)
            if edge is not None:
                edge = edge.to(device, non_blocking=non_blocking)

            stride = None
            patch_inference = patch and type_split != "random"
            if patch_inference:
                patch_grid = (
                    (height - patch_size) // 32 + 1,
                    (width - patch_size) // 8 + 1,
                )
                image, stride = extract_patches_with_target_count(
                    image, patch_size, patch_grid
                )
                if edge is not None:
                    edge, _ = extract_patches_with_target_count(
                        edge, patch_size, patch_grid
                    )
                if len(image.shape) > 4:
                    image = image.flatten(0, 1)
                    if edge is not None:
                        edge = edge.flatten(0, 1)

            probability_map = _forward_in_batches(
                model,
                image,
                edge,
                batch_size,
                forward_args,
                amp,
                profile,
                pad_final_batch=patch_inference,
            )

            # Reconstruct once; the old prob/prob_1 paths were identical.
            if stride is not None:
                probability_map = probability_map.view(
                    image_count, -1, 1, patch_size, patch_size
                )
                probability_map = reverse_to_original_image(
                    probability_map,
                    (height, width),
                    patch_size,
                    stride,
                )

            target_height, target_width = mask.shape[-2:]
            cropped_probability_map = probability_map[
                :, :, :target_height, :target_width
            ]
            prediction_maps = cropped_probability_map >= threshold
            target_maps = mask[:, :target_height, :target_width] >= 0.5
            for prediction_map, target_map in zip(
                prediction_maps[:, 0], target_maps
            ):
                image_cdice_scores.append(
                    centerline_dice(prediction_map, target_map)
                )

            probabilities = cropped_probability_map.flatten()
            target = target_maps.flatten().long()
            prediction = prediction_maps.flatten().long()

            prediction_positive = prediction == 1
            target_positive = target == 1
            true_positive = (prediction_positive & target_positive).sum()
            true_negative = ((~prediction_positive) & (~target_positive)).sum()
            false_positive = (prediction_positive & (~target_positive)).sum()
            false_negative = ((~prediction_positive) & target_positive).sum()
            confusion += torch.stack(
                (true_positive, true_negative, false_positive, false_negative)
            )

            dice_denominator = 2 * true_positive + false_positive + false_negative
            image_dice_scores.append(
                torch.where(
                    dice_denominator > 0,
                    (2 * true_positive).float() / dice_denominator.float(),
                    torch.full((), torch.nan, device=device),
                )
            )
            auroc_metric.update(probabilities, target)

    accuracy, f1, iou, recall, specificity = metrics_from_confusion(*confusion)
    auc = auroc_metric.compute()
    dice = torch.stack(image_dice_scores).nanmean()
    cdice = sum(image_cdice_scores) / len(image_cdice_scores)
    return (
        accuracy.item(),
        f1.item(),
        iou.item(),
        recall.item(),
        specificity.item(),
        auc.item(),
        dice.item(),
        cdice,
        float(threshold),
    )

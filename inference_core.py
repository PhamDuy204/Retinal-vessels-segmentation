"""DRIVE inference with the same preprocessing, patch grid and reconstruction as eval.py."""
from __future__ import annotations

from contextlib import nullcontext
from io import BytesIO
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageOps
import torch

from eval import EVALUATION_THRESHOLD
from load_model import load_model_class
from transforms import get_test_patch_transforms
from utils import (
    extract_patches_with_target_count,
    mirror_padding_v2,
    preprocessing_img,
    reverse_to_original_image,
)

PATCH_SIZE = 64
EVAL_BATCH_SIZE = 256
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".ppm", ".bmp", ".webp"}


def read_image(source: str | Path | BytesIO) -> np.ndarray:
    with Image.open(source) as image:
        return np.asarray(ImageOps.exif_transpose(image).convert("RGB"))


def load_checkpoint(path: str | Path, device: torch.device) -> torch.nn.Module:
    """Load state dicts only. Pickled model objects are intentionally unsupported."""
    if Path(path).suffix.lower() == ".safetensors":
        from safetensors import safe_open
        with safe_open(path, framework="pt", device="cpu") as file:
            metadata = file.metadata() or {}
            checkpoint = {"model": metadata.get("model", "our_net"),
                          "model_state_dict": {key: file.get_tensor(key) for key in file.keys()}}
    else:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Expected a training checkpoint containing model_state_dict.")
    name = checkpoint.get("model", "our_net")
    if name != "our_net":
        raise ValueError(f"Checkpoint model {name!r} is unsupported by this DRIVE infer view.")
    model = load_model_class(name)(1, 1)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if device.type == "cpu":
        from types import MethodType
        from cpu_mamba2 import mamba2_cpu_forward
        mamba = model.bneck[1].mamba
        mamba.forward = MethodType(mamba2_cpu_forward, mamba)
    return model.to(device).eval()


def prepare_patches(rgb: np.ndarray, device: torch.device):
    gray = preprocessing_img(rgb)
    image = get_test_patch_transforms()(image=gray)["image"]
    height, width = image.shape[-2:]
    image = mirror_padding_v2(image.unsqueeze(0).to(device))
    padded_size = image.shape[-2:]
    # The 32×32 inference grid preserves overlap and reduces DRIVE patches
    # from 1,387 to 361; evaluated separately from the paper's dense grid.
    grid = ((padded_size[0] - PATCH_SIZE) // 32 + 1,
            (padded_size[1] - PATCH_SIZE) // 32 + 1)
    patches, stride = extract_patches_with_target_count(image, PATCH_SIZE, grid)
    return patches, padded_size, stride, (height, width)


def reconstruct(patches: torch.Tensor, padded_size, stride, size):
    probability = reverse_to_original_image(
        patches.reshape(1, -1, 1, PATCH_SIZE, PATCH_SIZE),
        padded_size, PATCH_SIZE, stride,
    )[0, 0, :size[0], :size[1]]
    if not torch.isfinite(probability).all().item():
        raise FloatingPointError("Inference produced NaN/Inf; mask was not generated.")
    return probability


def predict(model: torch.nn.Module, rgb: np.ndarray, device: torch.device):
    started = time.perf_counter()
    patches, padded_size, stride, size = prepare_patches(rgb, device)
    outputs = []
    autocast = (torch.amp.autocast("cuda", dtype=torch.bfloat16)
                if device.type == "cuda" else nullcontext())
    with torch.inference_mode(), autocast:
        for batch in patches.split(EVAL_BATCH_SIZE):
            outputs.append(model(batch))
        probability = reconstruct(torch.cat(outputs), padded_size, stride, size)
        mask = (probability >= EVALUATION_THRESHOLD).to(torch.uint8).cpu().numpy()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return mask, time.perf_counter() - started


def gradcam(model: torch.nn.Module, rgb: np.ndarray, device: torch.device):
    """Grad-CAM of predicted vessel probability on the final 32-channel decoder map."""
    patches, padded_size, stride, size = prepare_patches(rgb, device)
    cams = []
    activations = []
    hook = model.out[-2].register_forward_hook(
        lambda _module, _inputs, output: activations.append(output)
    )
    try:
        autocast = (torch.amp.autocast("cuda", dtype=torch.bfloat16)
                    if device.type == "cuda" else nullcontext())
        with torch.enable_grad(), autocast:
            for batch in patches.split(32):
                activations.clear()
                prediction = model(batch.detach().requires_grad_(True))
                features = activations[0]
                gradient = torch.autograd.grad(prediction.sum(), features)[0]
                weights = gradient.float().mean(dim=(-2, -1), keepdim=True)
                cam = torch.relu((weights * features.float()).sum(dim=1, keepdim=True))
                cams.append(cam.detach())
        heat = reconstruct(torch.cat(cams), padded_size, stride, size).float()
        heat -= heat.min()
        heat /= heat.max().clamp_min(1e-8)
        values = heat.cpu().numpy()
        red = np.clip(2 * values, 0, 1)
        green = np.clip(2 - 2 * np.abs(values - 0.5), 0, 1)
        blue = np.clip(1 - 2 * values, 0, 1)
        color = np.stack((red, green, blue), axis=-1)
        return (0.55 * rgb + 0.45 * (color * 255)).clip(0, 255).astype(np.uint8)
    finally:
        hook.remove()

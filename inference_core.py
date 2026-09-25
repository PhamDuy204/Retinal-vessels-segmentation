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


def select_device() -> torch.device:
    """Use CUDA when an available GPU can initialize; otherwise fall back to CPU."""
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        try:
            torch.cuda.init()
            return torch.device("cuda:0")
        except (RuntimeError, AssertionError):
            pass
    return torch.device("cpu")


def read_image(source: str | Path | BytesIO) -> np.ndarray:
    with Image.open(source) as image:
        return np.asarray(ImageOps.exif_transpose(image).convert("RGB"))


def load_checkpoint(path: str | Path | None, device: torch.device,
                    model_name: str = "our_net", data: bytes | None = None) -> torch.nn.Module:
    """Load a matching checkpoint, or initialize the selected model without weights."""
    if model_name not in ("our_net", "our_net_window"):
        raise ValueError(f"Unsupported model: {model_name!r}.")
    if path is not None:
        if Path(path).suffix.lower() == ".safetensors":
            from safetensors import safe_open
            from tempfile import TemporaryDirectory
            with TemporaryDirectory() if data is not None else nullcontext() as directory:
                source = Path(directory) / "upload.safetensors" if directory else path
                if data is not None:
                    source.write_bytes(data)
                with safe_open(source, framework="pt", device="cpu") as file:
                    metadata = file.metadata() or {}
                    checkpoint = {"model": metadata.get("model", model_name),
                                  "model_state_dict": {key: file.get_tensor(key) for key in file.keys()}}
        elif Path(path).suffix.lower() == ".pt":
            checkpoint = torch.load(BytesIO(data) if data is not None else path,
                                    map_location="cpu", weights_only=True)
        else:
            raise ValueError("Checkpoint must be a .pt or .safetensors file.")
        if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model_state_dict"), dict):
            raise ValueError("Expected a training checkpoint containing model_state_dict.")
        name = checkpoint.get("model", model_name)
        if name != model_name:
            raise ValueError(f"Checkpoint uses {name!r}; --os selected {model_name!r}.")
    model = load_model_class(model_name)(1, 1)
    if path is not None:
        try:
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        except RuntimeError as exc:
            raise ValueError(f"Checkpoint weights do not match {model_name}: {exc}") from exc
    if device.type == "cpu" and model_name == "our_net":
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
    # Favor a 48-pixel stride on DRIVE; for other padded sizes select the
    # largest divisor <=48 so the final patch covers the far image edges.
    def grid_count(length):
        span = length - PATCH_SIZE
        stride = next(step for step in range(48, 0, -1) if span % step == 0)
        return span // stride + 1

    grid = (grid_count(padded_size[0]), grid_count(padded_size[1]))
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
    def isolate_cam_features(_module, _inputs, output):
        # Stop gradients at the decoder feature map: only the head is needed
        # for Grad-CAM, so upstream Mamba/encoder graphs need no retention.
        features = output.detach().requires_grad_(True)
        activations.append(features)
        return features

    hook = model.out[-2].register_forward_hook(isolate_cam_features)
    try:
        autocast = (torch.amp.autocast("cuda", dtype=torch.bfloat16)
                    if device.type == "cuda" else nullcontext())
        with torch.enable_grad(), autocast:
            for batch in patches.split(32):
                activations.clear()
                prediction = model(batch)
                features = activations[0]
                vessels = (prediction.detach() >= EVALUATION_THRESHOLD)
                gradient = torch.autograd.grad(
                    (prediction * vessels).sum(), features
                )[0]
                weights = gradient.float().mean(dim=(-2, -1), keepdim=True)
                cam = torch.relu((weights * features.float()).sum(dim=1, keepdim=True))
                cams.append(cam.detach())
        heat = reconstruct(torch.cat(cams), padded_size, stride, size).float()
        values = heat.cpu().numpy()
        fundus = rgb.mean(axis=-1) > 20
        region = values[fundus] if fundus.any() else values.ravel()
        low, high = np.percentile(region, (65, 99.5))
        strength = np.clip((values - low) / max(high - low, 1e-8), 0, 1)
        alpha = (0.82 * strength ** 1.2 * fundus)[..., None]
        accent = np.array([35, 230, 90], dtype=np.float32)
        return ((1 - alpha) * rgb + alpha * accent).clip(0, 255).astype(np.uint8)
    finally:
        hook.remove()

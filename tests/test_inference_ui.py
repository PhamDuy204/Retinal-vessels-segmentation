"""A DRIVE-shaped overlap regression check for the interactive inference grid."""
import numpy as np
import torch

from inference_core import prepare_patches, reconstruct


def test_fast_grid_reconstructs_drive_shape_without_uncovered_pixels():
    rgb = np.full((584, 565, 3), 128, dtype=np.uint8)
    patches, padded, stride, size = prepare_patches(rgb, torch.device("cpu"))
    assert patches.shape == (361, 1, 64, 64)
    assert stride == (32, 32)
    merged = reconstruct(torch.ones_like(patches), padded, stride, size)
    assert merged.shape == (584, 565)
    assert torch.all(merged == 1)

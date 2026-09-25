"""A DRIVE-shaped overlap regression check for the interactive inference grid."""
from unittest.mock import patch
from io import BytesIO

import numpy as np
import pytest
import torch

from inference_core import load_checkpoint, prepare_patches, reconstruct, select_device


@pytest.mark.parametrize(
    ("available", "count", "init_failure", "expected"),
    [(True, 1, False, "cuda:0"), (False, 0, False, "cpu"),
     (True, 0, False, "cpu"), (True, 1, True, "cpu")],
)
def test_select_device(available, count, init_failure, expected):
    with patch("torch.cuda.is_available", return_value=available), \
         patch("torch.cuda.device_count", return_value=count), \
         patch("torch.cuda.init", side_effect=RuntimeError("CUDA unavailable") if init_failure else None):
        assert str(select_device()) == expected


def test_fast_grid_reconstructs_drive_shape_without_uncovered_pixels():
    rgb = np.full((584, 565, 3), 128, dtype=np.uint8)
    patches, padded, stride, size = prepare_patches(rgb, torch.device("cpu"))
    assert patches.shape == (169, 1, 64, 64)
    assert stride == (48, 48)
    merged = reconstruct(torch.ones_like(patches), padded, stride, size)
    assert merged.shape == (584, 565)
    assert torch.all(merged == 1)


def test_fast_grid_covers_non_drive_image_edges():
    rgb = np.full((500, 500, 3), 128, dtype=np.uint8)
    patches, padded, stride, size = prepare_patches(rgb, torch.device("cpu"))
    assert padded == (512, 512)
    assert stride == (32, 32)
    merged = reconstruct(torch.ones_like(patches), padded, stride, size)
    assert torch.all(merged == 1)


@pytest.mark.parametrize("model_name", ["our_net", "our_net_window"])
def test_random_weight_models_run_without_a_checkpoint(model_name):
    with torch.inference_mode():
        model = load_checkpoint(None, torch.device("cpu"), model_name)
        result = model(torch.zeros(1, 1, 64, 64))
    assert result.shape == (1, 1, 64, 64)
    assert torch.isfinite(result).all()


def test_uploaded_checkpoint_for_wrong_architecture_is_rejected():
    buffer = BytesIO()
    torch.save({"model": "our_net_window", "model_state_dict": {}}, buffer)
    with pytest.raises(ValueError, match="Checkpoint uses 'our_net_window'"):
        load_checkpoint("upload.pt", torch.device("cpu"), "our_net", buffer.getvalue())

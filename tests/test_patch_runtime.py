from __future__ import annotations

import torch

import utils


def test_grid_patch_extract_matches_unfold_layout():
    image = torch.arange(200, dtype=torch.float32).reshape(1, 2, 10, 10)

    patches, stride = utils.extract_patches_with_target_count(
        image, patch_size=4, target_patches_per_dim=(4, 4)
    )

    expected = (
        torch.nn.functional.unfold(image, kernel_size=(4, 4), stride=(2, 2))
        .transpose(1, 2)
        .reshape(-1, 2, 4, 4)
    )
    assert stride == (2, 2)
    torch.testing.assert_close(patches, expected, rtol=0, atol=0)


def test_grid_patch_reconstruction_uses_native_torch_ops(monkeypatch):
    def kornia_should_not_run(*args, **kwargs):
        raise AssertionError("patch reconstruction should not call Kornia")

    monkeypatch.setattr(
        utils.kornia.contrib, "combine_tensor_patches", kornia_should_not_run
    )
    image = torch.arange(200, dtype=torch.float32).reshape(1, 2, 10, 10)
    patches = (
        torch.nn.functional.unfold(image, kernel_size=(4, 4), stride=(2, 2))
        .transpose(1, 2)
        .reshape(1, -1, 2, 4, 4)
    )

    reconstructed = utils.reverse_to_original_image(
        patches,
        original_size=(10, 10),
        patch_size=4,
        stride=(2, 2),
    )

    torch.testing.assert_close(reconstructed, image, rtol=0, atol=0)


def test_patch_grid_from_stride_matches_paper_protocol():
    assert utils.patch_grid_from_stride(640, 640, patch_size=64, stride=32) == (19, 19)
    assert utils.patch_grid_from_stride(512, 512, patch_size=64, stride=32) == (15, 15)

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

import dataset as dataset_module
from dataset import CustomTrainDataset


class _Tensorize:
    def __call__(self, *, image, mask):
        image_t = torch.from_numpy(np.asarray(image)).float().unsqueeze(0)
        mask_t = torch.from_numpy(np.asarray(mask)).long()
        return {"image": image_t, "mask": mask_t}


def test_our_net_dataset_skips_unused_edge_computation(tmp_path, monkeypatch):
    root = tmp_path / "DRIVE" / "training"
    (root / "images").mkdir(parents=True)
    (root / "mask").mkdir(parents=True)
    Image.fromarray(np.zeros((96, 96, 3), dtype=np.uint8)).save(root / "images" / "01.jpg")
    Image.fromarray(np.zeros((96, 96), dtype=np.uint8)).save(root / "mask" / "01.png")

    def edge_should_not_run(*args, **kwargs):
        raise AssertionError("our_net does not consume edge input")

    monkeypatch.setattr(dataset_module, "sobel_transform", edge_should_not_run)
    ds = CustomTrainDataset(
        str(root),
        img_transforms=_Tensorize(),
        with_patches=False,
        model_name="our_net",
    )

    sample = ds[0]
    assert sample["edge"].numel() <= 1

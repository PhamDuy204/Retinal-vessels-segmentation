import warnings
import sys
from pathlib import Path

import torch

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "our_net"
sys.path.insert(0, str(MODEL_DIR))

from modules import swl  # noqa: E402


def test_swl_does_not_use_implicit_softmax_dimension():
    module = swl((64, 64)).eval()
    tensors = [torch.randn(2, 1, 64, 64) for _ in range(4)]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        module(*tensors)
    assert not any(
        "Implicit dimension choice for softmax" in str(item.message)
        for item in caught
    )

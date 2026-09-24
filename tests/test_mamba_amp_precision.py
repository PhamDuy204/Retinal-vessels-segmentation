import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "our_net"
sys.path.insert(0, str(MODEL_DIR))

from bottle_neck import BottleNeck_2  # noqa: E402


class _DtypeRecorder(nn.Module):
    def __init__(self):
        super().__init__()
        self.seen_dtype = None

    def forward(self, x):
        self.seen_dtype = x.dtype
        return x


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA AMP required")
def test_bottleneck2_runs_only_fp16_mamba_region_in_fp32():
    block = BottleNeck_2(32).cuda().eval()
    recorder = _DtypeRecorder().cuda()
    block.mamba = recorder
    x = torch.randn(2, 32, 8, 8, device="cuda", dtype=torch.float16)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        output = block(x)

    assert recorder.seen_dtype == torch.float32
    assert output.dtype == torch.float16
    assert torch.isfinite(output).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA AMP required")
def test_bottleneck2_runs_bf16_mamba_region_in_fp32():
    block = BottleNeck_2(32).cuda().eval()
    recorder = _DtypeRecorder().cuda()
    block.mamba = recorder
    x = torch.randn(2, 32, 8, 8, device="cuda", dtype=torch.bfloat16)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        output = block(x)

    assert recorder.seen_dtype == torch.float32
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output).all()

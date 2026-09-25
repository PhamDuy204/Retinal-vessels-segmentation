import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "our_net"
sys.path.insert(0, str(MODEL_DIR))

from modules import MAB  # noqa: E402


class _ConstantMerge(nn.Module):
    def forward(self, x):
        b, _, h, w = x.shape
        return torch.full((b, 32, h, w), 20.0, device=x.device, dtype=x.dtype)


class _ScaledGate(nn.Module):
    def __init__(self, scale=0.01):
        super().__init__()
        self.scale = scale
        self.seen_dtype = None

    def forward(self, x):
        self.seen_dtype = x.dtype
        b, _, h, w = x.shape
        return torch.full((b, 32, h, w), self.scale, device=x.device, dtype=x.dtype)


def _make_lightweight_mab(*, bf16_fp32=False):
    block = MAB(32, (8, 8), bf16_fp32=bf16_fp32).cuda().eval()
    block.first_conv = nn.Identity()
    for name in (
        "branch_0_0", "branch_0_1", "branch_0_2",
        "branch_1_0", "branch_1_1", "branch_1_2",
        "branch_2_0", "branch_2_1", "branch_2_2",
    ):
        setattr(block, name, nn.Identity())
    block.merge = _ConstantMerge().cuda()
    gate = _ScaledGate().cuda()
    block.transform_statistic = gate
    return block, gate


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA AMP required")
def test_mab_promotes_overflow_sensitive_fp16_statistics_to_fp32():
    block, gate = _make_lightweight_mab()
    x = torch.full((2, 32, 8, 8), 5000.0, device="cuda", dtype=torch.float16)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        output = block(x)

    # 20 * 5000 = 100000, which would overflow if the MAB product were FP16.
    assert gate.seen_dtype == torch.float32
    assert output.dtype == torch.float16
    assert torch.isfinite(output).all()
    torch.testing.assert_close(
        output.float(),
        torch.full_like(output.float(), 1000.0),
        rtol=2e-3,
        atol=2.0,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA AMP required")
def test_mab_promotes_bf16_sensitive_statistics_to_fp32_when_enabled():
    block, gate = _make_lightweight_mab(bf16_fp32=True)
    x = torch.full((2, 32, 8, 8), 5000.0, device="cuda", dtype=torch.bfloat16)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        output = block(x)

    assert gate.seen_dtype == torch.float32
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output).all()

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA AMP required")
def test_mab_keeps_decoder_bf16_statistics_native_by_default():
    block, gate = _make_lightweight_mab()
    x = torch.full((2, 32, 8, 8), 5.0, device="cuda", dtype=torch.bfloat16)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        output = block(x)

    assert gate.seen_dtype == torch.bfloat16
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output).all()

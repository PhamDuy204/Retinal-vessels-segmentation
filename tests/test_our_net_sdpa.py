from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "our_net"
sys.path.insert(0, str(MODEL_DIR))

from bottle_neck import CAB  # noqa: E402


def _manual_cab_attention(module: CAB, x: torch.Tensor) -> torch.Tensor:
    b, c, h, w = x.shape
    q = module.q(x).permute(0, 2, 3, 1).contiguous().view(b, h * w, c)
    k = module.k(x).permute(0, 2, 3, 1).contiguous().view(b, h * w, c)
    v = module.v(x).permute(0, 2, 3, 1).contiguous().view(b, h * w, c)
    attn = torch.softmax((q @ k.transpose(-1, -2)) / math.sqrt(c), dim=-1)
    out = (attn @ v).view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
    out = module.out_norm(module.pj(out) + x)
    return module.ff(out) + out


def test_cab_delegates_attention_to_pytorch_sdpa(monkeypatch):
    called = False
    original = F.scaled_dot_product_attention

    def tracked(*args, **kwargs):
        nonlocal called
        called = True
        return original(*args, **kwargs)

    monkeypatch.setattr(F, "scaled_dot_product_attention", tracked)
    module = CAB(32).eval()
    module(torch.randn(2, 32, 8, 8))
    assert called


def test_cab_sdpa_preserves_manual_attention_math():
    torch.manual_seed(7)
    module = CAB(32).eval()
    x = torch.randn(2, 32, 8, 8)
    expected = _manual_cab_attention(module, x)
    actual = module(x)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=3e-6)

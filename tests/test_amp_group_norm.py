import sys
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models" / "our_net"))
import modules


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA AMP required")
def test_group_norm_retains_feature_dtype_and_fp32_parameter_gradients():
    assert hasattr(modules, "FeatureGroupNorm"), "Missing dtype-preserving native GroupNorm"
    layer = modules.FeatureGroupNorm(4, 8).cuda()
    layer.native_amp = True
    x = torch.randn(3, 8, 9, 9, device="cuda", dtype=torch.float16, requires_grad=True)
    reference = F.group_norm(x.float(), 4, layer.weight, layer.bias, layer.eps).half()
    with torch.autocast("cuda", dtype=torch.float16):
        actual = layer(x)
    assert actual.dtype == torch.float16
    torch.testing.assert_close(actual, reference, rtol=1e-3, atol=2e-3)
    actual.float().square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert layer.weight.grad.dtype == torch.float32
    assert torch.isfinite(layer.weight.grad).all()
    with torch.autocast("cuda", enabled=False):
        fp32 = x.detach().float()
        expected = F.group_norm(fp32, 4, layer.weight, layer.bias, layer.eps)
        torch.testing.assert_close(layer(fp32), expected, rtol=0, atol=0)

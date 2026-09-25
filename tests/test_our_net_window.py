import torch

from load_model import load_model_class
from mambapy.mamba import MambaBlock


def test_our_net_window_uses_pure_pytorch_mamba_backend():
    model_class = load_model_class("our_net_window")
    model = model_class(1, 1)

    assert isinstance(model.bneck[1].mamba, MambaBlock)
    assert model.bneck[1].mamba.config.use_cuda is False


def test_our_net_window_forward_is_finite_fp32():
    model_class = load_model_class("our_net_window")
    model = model_class(1, 1).eval()
    x = torch.randn(1, 1, 64, 64)

    with torch.inference_mode():
        output = model(x)

    assert output.dtype == torch.float32
    assert output.shape == (1, 1, 64, 64)
    assert torch.isfinite(output).all()

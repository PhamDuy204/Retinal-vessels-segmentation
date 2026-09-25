import pytest
import torch
from eval import _forward_in_batches, eval_for_seg

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA AMP required")


@pytest.mark.parametrize("batch_size", [0, 2])
def test_default_eval_amp_preserves_values_outside_fp16_range(batch_size):
    model = torch.nn.Conv2d(1, 1, 1, bias=False).cuda().eval()
    with torch.no_grad():
        model.weight.fill_(100000.0)
    image = torch.ones(3, 1, 2, 2, device="cuda")
    with torch.inference_mode():
        output = _forward_in_batches(model, image, None, batch_size, 1, True, False, False)
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output.float(), torch.full_like(image, 100000.0), rtol=0.004, atol=0)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_eval_rejects_nonfinite_predictions_before_metrics(bad_value):
    class InvalidModel(torch.nn.Module):
        def forward(self, image):
            return torch.full_like(image, bad_value)

    sample = {
        "image": torch.zeros(1, 1, 448, 448),
        "mask": torch.zeros(1, 448, 448),
        "edge": torch.zeros(1, 1, 448, 448),
    }
    with pytest.raises(FloatingPointError, match="Non-finite evaluation output"):
        eval_for_seg(InvalidModel().cuda(), [sample], 0, amp=True, batch_size=1)

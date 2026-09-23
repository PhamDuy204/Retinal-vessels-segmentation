import pytest
import torch
import torch.nn.functional as F
from loss.main_loss import MultiScopeLoss


def unfolded_reference(pred, truth):
    batch = truth.shape[0]
    pred, truth = pred.squeeze(1).float(), truth.squeeze(1).float()
    loss = pred.new_zeros(1)
    for kernel, weight in MultiScopeLoss.SCOPES:
        if weight == 0:
            continue
        p = F.unfold(pred, kernel, stride=kernel // 2).reshape(batch, kernel, kernel, -1).permute(0, 3, 1, 2)
        t = F.unfold(truth, kernel, stride=kernel // 2).reshape(batch, kernel, kernel, -1).permute(0, 3, 1, 2)
        count = t.sum((-1, -2))
        dice = 1 - ((2 * p * t).sum((-1, -2)) + 1e-6) / ((p + t).sum((-1, -2)) + 1e-6)
        selected = torch.where(count > kernel * kernel // 3, dice, 0)
        weights = torch.softmax(-count / count.shape[-1], dim=1)
        loss += (weights * selected).sum(-1).mean() * weight
    return loss


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA memory contract")
def test_scope_reduction_matches_values_gradients_with_less_memory():
    torch.manual_seed(9)
    pred = torch.rand(32, 1, 64, 64, device="cuda", requires_grad=True)
    truth = (torch.rand_like(pred) > 0.5).float()
    reference = unfolded_reference(pred, truth)
    expected_grad, = torch.autograd.grad(reference, pred)
    actual = MultiScopeLoss()(pred, truth)
    actual_grad, = torch.autograd.grad(actual, pred)
    torch.testing.assert_close(actual, reference, rtol=2e-6, atol=1e-7)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=2e-5, atol=1e-9)

    def peak(fn):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        loss = fn(pred, truth)
        torch.autograd.grad(loss, pred)
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() - before

    reference_bytes = peak(unfolded_reference)
    actual_bytes = peak(MultiScopeLoss())
    assert actual_bytes < reference_bytes * 0.5, (actual_bytes, reference_bytes)

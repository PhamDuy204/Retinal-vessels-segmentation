"""The new stopping rule selects training loss even when validation disagrees."""

import json

import pytest
import torch

import train


class FixedLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.steps = 0

    def forward(self, predicted, target):
        epoch = self.steps // 2
        self.steps += 1
        # Epoch means: 3, 1, then 2 forever. The unequal first two
        # micro-batch losses also verify that the logged loss is a mean.
        value = (2, 4) if epoch == 0 else (1, 1) if epoch == 1 else (2, 2)
        return predicted.mean() * 1e-4 + value[(self.steps - 1) % 2]


class RunLog:
    id = None
    url = None

    def __init__(self):
        self.summary = {}

    def log(self, values):
        pass


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Trainer currently requires CUDA")
@pytest.mark.parametrize(
    ("eval_start", "patience", "last_epoch", "eval_calls"),
    [(50, 20, 22, 1), (1, 2, 4, 4), (50, 0, 30, 1)],
)
def test_minimum_epoch_mean_loss_selects_checkpoint_even_without_scheduled_eval(
    monkeypatch, tmp_path, eval_start, patience, last_epoch, eval_calls
):
    model = torch.nn.Conv2d(1, 1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1)
    criterion = FixedLoss()
    batch = {
        "image": torch.ones(2, 1, 64, 64),
        "mask": torch.zeros(2, 1, 64, 64),
        "edge": torch.zeros(2, 1, 64, 64),
    }
    args = train.parse_args([
        "--experiment-id", "early-stop-test", "--epochs", "30",
        "--eval-start-epoch", str(eval_start),
        "--early-stopping-patience", str(patience),
        "--micro-batch-size", "1", "--no-amp", "--no-eval-amp",
        "--amp-dtype", "fp32", "--wandb-mode", "disabled",
    ])
    calls = []

    def fake_eval(model, *args, **kwargs):
        calls.append(model)
        score = len(calls) / 10  # Higher scores favor later epochs.
        return (score,) * 8 + (train.EVALUATION_THRESHOLD,)

    monkeypatch.setattr(train, "eval_for_seg", fake_eval)
    trainer = train.Trainer(
        model=model, train_loader=[batch], val_loader=[], criterion=criterion,
        optimizer=optimizer, scheduler=scheduler, gpu_id=0,
        dataset_name="test", run_dir=tmp_path, patch=False, args=args,
        wandb_run=RunLog(), model_forward_args=1, model_class_name="Conv2d",
    )
    selected = trainer.train()
    saved = torch.load(tmp_path / "best.pt", weights_only=True)
    assert criterion.steps // 2 == last_epoch
    assert len(calls) == eval_calls
    assert selected["selected_epoch"] == saved["selected_epoch"] == 2
    assert selected["loss"] == pytest.approx(1.0, abs=1e-3)
    assert selected["selection_key"] == [selected["loss"]]
    assert not torch.equal(saved["model_state_dict"]["weight"], model.weight.detach())
    assert json.loads((tmp_path / "selected_result.json").read_text())["selected_epoch"] == 2
    assert trainer.wandb_run.summary["best_selection_rule"] == "min(epoch_mean_training_loss)"

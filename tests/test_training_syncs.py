from pathlib import Path


def test_training_loss_stays_on_gpu_inside_microbatch_loop():
    source = (Path(__file__).resolve().parents[1] / "train.py").read_text()
    microbatch_loop = source.split(
        "for next_image, next_mask, next_edge in zip(", 1
    )[1].split("torch.cuda.synchronize()", 1)[0]

    assert "float(loss.detach())" not in microbatch_loop

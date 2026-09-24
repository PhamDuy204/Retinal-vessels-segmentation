from pathlib import Path


def test_patch_evaluation_does_not_pad_partial_inference_batch():
    source = (Path(__file__).resolve().parents[1] / "eval.py").read_text()
    call = source.split("probability_map = _forward_with_tta(", 1)[1].split(
        "# Reconstruct once", 1
    )[0]

    assert "profile,\n                False,\n                active_tta_mode" in call

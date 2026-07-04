from __future__ import annotations

from pathlib import Path

import modal


LOCAL_ROOT = Path(__file__).resolve().parent
CODE_ROOT = "/root/retinal-vessels"

ignore = modal.FilePatternMatcher(
    "data/**",
    "outputs/**",
    "checkpoints/**",
    ".git/**",
    ".venv/**",
    "venv/**",
    "env/**",
    "wandb/**",
    "benchmark_results/**",
    "full-sandbox-terminal-mcp/**",
    "__pycache__/**",
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install_from_requirements(str(LOCAL_ROOT / "requirements-modal.txt"))
    .pip_install(
        (
            "https://github.com/Dao-AILab/causal-conv1d/releases/download/"
            "v1.6.2.post1/causal_conv1d-1.6.2.post1%2Bcu12torch2.7"
            "cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
        ),
        (
            "https://github.com/state-spaces/mamba/releases/download/"
            "v2.2.5/mamba_ssm-2.2.5%2Bcu12torch2.7cxx11abiTRUE-"
            "cp311-cp311-linux_x86_64.whl"
        ),
        extra_options="--no-deps",
    )
    .pip_install("einops", "ninja")
    .add_local_dir(str(LOCAL_ROOT), remote_path=CODE_ROOT, copy=True, ignore=ignore)
)

app = modal.App("retinal-ournet-param-count")


@app.function(image=image, gpu="L4", timeout=600)
def count_params(widths: list[int]) -> list[tuple[int, str]]:
    import sys

    sys.path.insert(0, CODE_ROOT)
    from load_model import load_model_class
    from utils import count_trainable_params

    model_class = load_model_class("our_net")
    results = []
    for width in widths:
        try:
            model = model_class(1, 1, width=width)
            results.append((width, f"{count_trainable_params(model):,}"))
        except Exception as exc:
            results.append((width, f"INVALID: {type(exc).__name__}: {exc}"))
    return results


@app.local_entrypoint()
def main() -> None:
    for width, params in count_params.remote([64, 80, 88, 96, 104, 112, 128]):
        print(f"width={width} params={params}")

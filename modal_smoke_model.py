
from __future__ import annotations

from pathlib import Path
import modal

APP_NAME = "retinal-ournet-mamba3-smoke"
CODE_ROOT = "/root/retinal-vessels"
LOCAL_ROOT = Path(__file__).resolve().parent

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
    .pip_install("einops", "ninja", "mamba3-ssm==0.2.1")
    .add_local_dir(str(LOCAL_ROOT), remote_path=CODE_ROOT, copy=True, ignore=ignore)
)

app = modal.App(APP_NAME)


@app.function(image=image, gpu="L4", timeout=1800, memory=16384, cpu=4)
def smoke(model_name: str = "our_net_mamba3", width: int = 64) -> str:
    import sys
    import json
    import torch

    sys.path.insert(0, CODE_ROOT)
    from load_model import load_model_class
    from utils import count_trainable_params

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model_class = load_model_class(model_name)
    model = model_class(1, 1, width=width).cuda().train()
    params = count_trainable_params(model)
    x = torch.randn(1, 1, 64, 64, device="cuda")
    y = model(x)
    if isinstance(y, (tuple, list)):
        loss = sum(t.float().mean() for t in y)
        out_shapes = [tuple(t.shape) for t in y]
    else:
        loss = y.float().mean()
        out_shapes = [tuple(y.shape)]
    loss.backward()
    torch.cuda.synchronize()
    return json.dumps({
        "model": model_name,
        "width": int(width),
        "params": int(params),
        "loss": float(loss.detach().cpu()),
        "out_shapes": [list(shape) for shape in out_shapes],
        "cuda": str(torch.version.cuda),
        "torch": str(torch.__version__),
    }, sort_keys=True)


@app.local_entrypoint()
def main(model: str = "our_net_mamba3", width: int = 64) -> None:
    print(smoke.remote(model, width))

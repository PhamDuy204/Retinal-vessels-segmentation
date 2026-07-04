from __future__ import annotations

import os
import shlex

from beam import Image, Pod, Volume

CONFIG = os.environ.get("BEAM_BASELINE_CONFIG", "configs/baseline_beam1_pending_eval30_e60.json")
PENDING = os.environ.get("BEAM_BASELINE_PENDING", "configs/baseline_beam1_pending_eval30_e60.pending_tasks.json")
TASK_OFFSET = int(os.environ.get("BEAM_TASK_OFFSET", "0") or 0)
TASK_LIMIT = int(os.environ.get("BEAM_TASK_LIMIT", "0") or 0)
SMOKE = os.environ.get("BEAM_SMOKE", "0") == "1"
HOLD_AFTER = os.environ.get("BEAM_HOLD_AFTER", "0") == "1"
HOLD_SECONDS = int(os.environ.get("BEAM_HOLD_SECONDS", "3600") or 3600)

image = Image(
    python_version="python3.11",
    commands=[
        "apt-get update -y && apt-get install -y libgl1 libglib2.0-0",
        "python -m pip install --upgrade pip",
        "python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 torch==2.7.1+cu128 torchvision==0.22.1+cu128",
        "python -m pip install 'numpy<2.3' stringzilla==3.10.4 albumentations==2.0.8 scikit-image opencv-python-headless tqdm wandb torchmetrics ml-collections kornia einops ninja huggingface_hub packaging",
        "python -m pip install --no-deps https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.2.post1/causal_conv1d-1.6.2.post1%2Bcu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl",
        "python -m pip install --no-deps https://github.com/state-spaces/mamba/releases/download/v2.2.5/mamba_ssm-2.2.5%2Bcu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl",
    ],
)

outputs = Volume(name="retinal-baseline-outputs", mount_path="/outputs")

args = [
    "python", "-u", "beam_baseline_runtime.py",
    "--config", CONFIG,
    "--pending", PENDING,
    "--task-offset", str(TASK_OFFSET),
]
if TASK_LIMIT > 0:
    args += ["--task-limit", str(TASK_LIMIT)]
if SMOKE:
    args.append("--smoke")
if HOLD_AFTER:
    args += ["--hold-after", "--hold-seconds", str(HOLD_SECONDS)]

entrypoint = "\n".join([
    "set -uo pipefail",
    "cd /mnt/code",
    "mkdir -p /outputs/beam-debug",
    "exec > >(tee -a /outputs/beam-debug/baseline_runtime.log) 2>&1",
    "date -u",
    " ".join(shlex.quote(part) for part in args),
])

pod = Pod(
    app="retinal-baseline-pending",
    name="retinal-baseline-beam-runtime-a10g",
    gpu="A10G",
    cpu=8,
    memory=49152,
    image=image,
    volumes=[outputs],
    secrets=["WANDB_API_KEY"],
    keep_warm_seconds=0,
    entrypoint=["bash", "-lc", entrypoint],
)

from __future__ import annotations

import json
import shlex
from pathlib import Path

from beam import Image, Pod, Volume

CONFIG = "configs/baseline_beam1b_pending_eval35_e60.json"
PENDING = "configs/baseline_beam1b_pending_eval35_e60.pending_tasks.json"

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


FLAG_NAMES = {
    "batch_size": "--batch_size",
    "learning_rate": "--learning_rate",
    "patch_size": "--patch_size",
    "train_type": "--train_type",
    "type_split": "--type_split",
    "wandb_mode": "--wandb-mode",
    "num_workers": "--num-workers",
    "pin_memory": "--pin-memory",
    "persistent_workers": "--persistent-workers",
    "compile_model": "--compile-model",
    "eval_amp": "--eval-amp",
    "eval_batch_size": "--eval-batch-size",
    "eval_every": "--eval-every",
    "eval_start_epoch": "--eval-start-epoch",
    "fast_nondeterministic": "--fast-nondeterministic",
}


def flag(name: str, value):
    if value is None or value is False:
        return []
    cli = FLAG_NAMES.get(name, "--" + name.replace("_", "-"))
    if value is True:
        return [cli]
    return [cli, str(value)]


payload = json.loads(Path(PENDING).read_text(encoding="utf-8"))["tasks"]
config = json.loads(Path(CONFIG).read_text(encoding="utf-8"))
common = dict(config["common_train_args"])
for key in ("data_root", "output_root"):
    common.pop(key, None)

commands = [
    "set -euo pipefail",
    "cd /mnt/code",
    "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader",
    "python3 -m py_compile train.py",
    "python3 - <<'CHECK'\nfrom pathlib import Path\nfor d in ['data/CHASEDB_1','data/DRIVE','data/STARE_F1','data/STARE_F5']:\n    print(d, Path(d).exists())\nCHECK",
]
for task in payload:
    cmd = [
        "python", "-u", "train.py",
        "--model", str(task["model"]),
        "--seed", str(task["seed"]),
        "--experiment-id", str(task.get("experiment_id", config["experiment_id"])),
        "--datasets", str(task["dataset"]),
        "--data-root", "data",
        "--output-root", "/outputs",
        "--wandb-entity", str(config["wandb_entity"]),
        "--wandb-project", str(config["wandb_project"]),
    ]
    for key, value in common.items():
        cmd.extend(flag(key, value))
    cmd.append("--resume")
    commands.append("echo RUN " + shlex.quote(f"{task['model']} seed={task['seed']}"))
    commands.append(" ".join(shlex.quote(part) for part in cmd))

train_cmd = "\n".join(commands)

pod = Pod(
    app="retinal-baseline-pending",
    name="retinal-baseline-beam1b-eval35-e60-a10g",
    gpu="A10G",
    cpu=4,
    memory="32Gi",
    image=image,
    volumes=[outputs],
    secrets=["WANDB_API_KEY"],
    keep_warm_seconds=0,
    entrypoint=["bash", "-lc", train_cmd],
)

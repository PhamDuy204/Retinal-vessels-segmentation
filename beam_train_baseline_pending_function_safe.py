from __future__ import annotations

import json
import os
import shlex
from pathlib import Path

from beam import Image, Volume, function

CONFIG = os.environ.get("BEAM_BASELINE_CONFIG", "configs/baseline_beam1b_pending_eval35_e60.json")
PENDING = os.environ.get("BEAM_BASELINE_PENDING", "configs/baseline_beam1b_pending_eval35_e60.pending_tasks.json")
TASK_OFFSET = int(os.environ.get("BEAM_TASK_OFFSET", "0") or 0)
TASK_LIMIT = int(os.environ.get("BEAM_TASK_LIMIT", "0") or 0)
SMOKE = os.environ.get("BEAM_SMOKE", "0") == "1"

image = Image(
    python_version="python3.11",
    commands=[
        "apt-get update -y && apt-get install -y libgl1 libglib2.0-0 coreutils",
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
    "eval_auroc_device": "--eval-auroc-device",
    "eval_every": "--eval-every",
    "eval_start_epoch": "--eval-start-epoch",
    "fast_nondeterministic": "--fast-nondeterministic",
    "log_artifacts": "--log-artifacts",
}


def flag(name: str, value):
    if value is None or value is False:
        return []
    cli = FLAG_NAMES.get(name, "--" + name.replace("_", "-"))
    if value is True:
        return [cli]
    return [cli, str(value)]


def csv_values(value: str):
    return [part.strip() for part in str(value).split(",") if part.strip()]


pending_payload = json.loads(Path(PENDING).read_text(encoding="utf-8"))["tasks"]
expanded_tasks = []
for task in pending_payload:
    for dataset in csv_values(task["dataset"]):
        expanded_tasks.append({**task, "dataset": dataset})

if SMOKE:
    expanded_tasks = expanded_tasks[:1]
elif TASK_LIMIT > 0:
    expanded_tasks = expanded_tasks[TASK_OFFSET : TASK_OFFSET + TASK_LIMIT]
elif TASK_OFFSET:
    expanded_tasks = expanded_tasks[TASK_OFFSET:]

config = json.loads(Path(CONFIG).read_text(encoding="utf-8"))
common = dict(config["common_train_args"])
for key in ("data_root", "output_root"):
    common.pop(key, None)
common.update({
    "num_workers": 0,
    "pin_memory": False,
    "persistent_workers": False,
    "eval_batch_size": 1,
    "eval_auroc_device": "cpu",
})
if SMOKE:
    common.update({"epochs": 1, "eval_start_epoch": 1, "eval_every": 1})

commands = [
    "set -uo pipefail",
    "cd /mnt/code",
    "export WANDB__SERVICE_WAIT=300",
    "export WANDB_START_METHOD=thread",
    "export WANDB_CONSOLE=off",
    "export RUN_ROOT=/tmp/retinal-beam-run",
    "export OUT_TMP=$RUN_ROOT/outputs",
    "mkdir -p $OUT_TMP /outputs",
    "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader",
    "python3 -m py_compile train.py eval.py",
    "python3 - <<'CHECK'\nfrom pathlib import Path\nfor d in ['data/CHASEDB_1','data/DRIVE','data/STARE_F1','data/STARE_F5']:\n    print(d, Path(d).exists())\nCHECK",
    "echo BEAM_SAFE_TASKS " + shlex.quote(str(len(expanded_tasks))) + " SMOKE=" + shlex.quote(str(SMOKE)),
]

for index, task in enumerate(expanded_tasks, start=1):
    exp_id = str(task.get("experiment_id", config["experiment_id"]))
    cmd = [
        "python", "-u", "train.py",
        "--model", str(task["model"]),
        "--seed", str(task["seed"]),
        "--experiment-id", exp_id,
        "--datasets", str(task["dataset"]),
        "--data-root", "data",
        "--output-root", "$OUT_TMP",
        "--wandb-entity", str(config["wandb_entity"]),
        "--wandb-project", str(config["wandb_project"]),
    ]
    for key, value in common.items():
        cmd.extend(flag(key, value))
    cmd.append("--resume")
    quoted_cmd = " ".join(shlex.quote(part) if part != "$OUT_TMP" else part for part in cmd)
    commands.extend([
        "rm -rf $OUT_TMP && mkdir -p $OUT_TMP",
        "if [ -d " + shlex.quote(f"/outputs/{exp_id}") + " ]; then cp -a " + shlex.quote(f"/outputs/{exp_id}") + " $OUT_TMP/ || true; fi",
        "echo RUN_TASK " + shlex.quote(f"{index}/{len(expanded_tasks)} {task['model']} {task['dataset']} seed={task['seed']}"),
        "status=0; " + quoted_cmd + " || status=$?; cp -a $OUT_TMP/. /outputs/ || true; sync || true; if [ \"$status\" -ne 0 ]; then echo TASK_FAILED_STATUS=$status; exit $status; fi",
    ])
commands.append("echo BEAM_SAFE_DONE")

train_cmd = "\n".join(commands)


@function(
    app="retinal-baseline-pending",
    name="retinal-baseline-beam-safe-a10g",
    gpu="A10G",
    cpu=8,
    memory="48Gi",
    image=image,
    volumes=[outputs],
    secrets=["WANDB_API_KEY"],
    timeout=-1,
    retries=0,
)
def run():
    import subprocess

    completed = subprocess.run(["bash", "-lc", train_cmd], cwd="/mnt/code", check=False)
    if completed.returncode:
        raise SystemExit(completed.returncode)
    return {"status": "completed", "tasks": len(expanded_tasks), "smoke": SMOKE}

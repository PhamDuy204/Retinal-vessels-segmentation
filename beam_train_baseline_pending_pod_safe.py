from __future__ import annotations

import json
import os
import shlex
from pathlib import Path

from beam import Image, Pod, Volume

CONFIG = os.environ.get("BEAM_BASELINE_CONFIG", "configs/baseline_beam1b_pending_eval35_e60.json")
PENDING = os.environ.get("BEAM_BASELINE_PENDING", "configs/baseline_beam1b_pending_eval35_e60.pending_tasks.json")
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

if TASK_LIMIT > 0:
    expanded_tasks = expanded_tasks[TASK_OFFSET : TASK_OFFSET + TASK_LIMIT]
elif TASK_OFFSET:
    expanded_tasks = expanded_tasks[TASK_OFFSET:]
if SMOKE:
    expanded_tasks = expanded_tasks[:1]

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
if os.environ.get("BEAM_EPOCHS_OVERRIDE"):
    common["epochs"] = int(os.environ["BEAM_EPOCHS_OVERRIDE"])
if os.environ.get("BEAM_EVAL_START_OVERRIDE"):
    common["eval_start_epoch"] = int(os.environ["BEAM_EVAL_START_OVERRIDE"])

commands = [
    "set -uo pipefail",
    "cd /mnt/code",
    "export WANDB__SERVICE_WAIT=300",
    "export WANDB_START_METHOD=thread",
    "export WANDB_CONSOLE=off",
    "export RUN_ROOT=/tmp/retinal-beam-run",
    "export OUT_TMP=$RUN_ROOT/outputs",
    "mkdir -p $OUT_TMP /outputs/beam-debug",
    "exec > >(tee -a /outputs/beam-debug/baseline_safe.log) 2>&1",
    "date -u",
    "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader",
    "python3 -m py_compile train.py eval.py",
    "python3 - <<'CHECK'\nfrom pathlib import Path\nfor d in ['data/CHASEDB_1','data/DRIVE','data/STARE_F1','data/STARE_F5']:\n    print(d, Path(d).exists())\nCHECK",
    "echo BEAM_SAFE_TASKS " + shlex.quote(str(len(expanded_tasks))) + " SMOKE=" + shlex.quote(str(SMOKE)) + " HOLD_AFTER=" + shlex.quote(str(HOLD_AFTER)),
]

for index, task in enumerate(expanded_tasks, start=1):
    exp_id = str(task.get("experiment_id", config["experiment_id"]))
    cmd = [
        "python", "-X", "faulthandler", "-u", "train.py",
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
        "echo CMD " + shlex.quote(quoted_cmd),
        "echo BEFORE_TRAIN",
        "status=0; " + quoted_cmd + " || status=$?; echo AFTER_TRAIN_STATUS=$status; cp -a $OUT_TMP/. /outputs/ || true; sync || true; if [ \"$status\" -ne 0 ]; then echo TASK_FAILED_STATUS=$status; " + ("echo HOLD_AFTER_FAILURE; sleep " + shlex.quote(str(HOLD_SECONDS)) + "; " if HOLD_AFTER else "") + "exit $status; fi",
    ])
commands.append("echo BEAM_SAFE_DONE")
if HOLD_AFTER:
    commands.append("echo HOLD_AFTER_DONE; sleep " + shlex.quote(str(HOLD_SECONDS)))

train_cmd = "\n".join(commands)

pod = Pod(
    app="retinal-baseline-pending",
    name="retinal-baseline-beam-safe-a10g",
    gpu="A10G",
    cpu=8,
    memory=49152,
    image=image,
    volumes=[outputs],
    secrets=["WANDB_API_KEY"],
    keep_warm_seconds=0,
    entrypoint=["bash", "-lc", train_cmd],
)

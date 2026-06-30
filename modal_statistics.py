"""Submit one Modal container per model × dataset × seed statistical task."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import modal


APP_NAME = "retinal-vessels-statistics"
VOLUME_NAME = os.environ.get("MODAL_VOLUME_NAME", "retinal-vessels-statistics")
WANDB_SECRET_NAME = os.environ.get("MODAL_WANDB_SECRET", "wandb-secret")
GPU_TYPE = os.environ.get("MODAL_GPU_TYPE", "L4")
MAX_CONTAINERS = int(os.environ.get("MODAL_MAX_CONTAINERS", "4"))
CPU_CORES = float(os.environ.get("MODAL_CPU_CORES", "4"))
MEMORY_MB = int(os.environ.get("MODAL_MEMORY_MB", "16384"))
RETRIES = int(os.environ.get("MODAL_RETRIES", "1"))
TASK_TIMEOUT_SECONDS = int(os.environ.get("MODAL_TASK_TIMEOUT", str(24 * 60 * 60)))
CODE_ROOT = "/root/retinal-vessels"
VOLUME_ROOT = "/retinal-volume"
LOCAL_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = LOCAL_ROOT / "configs" / "statistics.json"
FLAG_NAMES = {
    "batch_size": "--batch_size",
    "epochs": "--epochs",
    "loss": "--loss",
    "learning_rate": "--learning_rate",
    "patches": "--patches",
    "patch_size": "--patch_size",
    "train_type": "--train_type",
    "chunk_size": "--chunk_size",
    "type_split": "--type_split",
    "wandb_mode": "--wandb-mode",
    "log_artifacts": "--log-artifacts",
}


app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
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
    # Mamba's official installation requires CUDA PyTorch to be installed
    # first and build isolation to be disabled, otherwise pip can pull a
    # different temporary Torch/CUDA toolchain. Install the matching release
    # wheels directly because their source metadata assumes nvcc is present.
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
    .add_local_dir(
        str(LOCAL_ROOT),
        remote_path=CODE_ROOT,
        copy=True,
        ignore=ignore,
    )
)


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def common_args_to_cli(common_args: dict[str, Any]) -> list[str]:
    command: list[str] = []
    for key, value in common_args.items():
        if value is None or value is False:
            continue
        command.append(FLAG_NAMES.get(key, f"--{key.replace('_', '-')}"))
        if value is not True:
            command.append(str(value))
    return command


def task_command(task: dict[str, Any], config: dict[str, Any]) -> list[str]:
    common = config.get("common_train_args", {})
    remote_common = {
        key: value
        for key, value in common.items()
        if key not in {"data_root", "output_root"}
    }
    command = [
        "python3",
        f"{CODE_ROOT}/train.py",
        "--model",
        str(task["model"]),
        "--seed",
        str(task["seed"]),
        "--experiment-id",
        str(task.get("experiment_id", config["experiment_id"])),
        "--datasets",
        str(task["dataset"]),
        "--data-root",
        f"{VOLUME_ROOT}/data",
        "--output-root",
        f"{VOLUME_ROOT}/outputs",
        "--wandb-entity",
        str(config["wandb_entity"]),
        "--wandb-project",
        str(config["wandb_project"]),
        *common_args_to_cli(remote_common),
        "--resume",
    ]
    return command


@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes={VOLUME_ROOT: volume},
    secrets=[modal.Secret.from_name(WANDB_SECRET_NAME)],
    cpu=CPU_CORES,
    memory=MEMORY_MB,
    max_containers=MAX_CONTAINERS,
    retries=RETRIES,
    timeout=TASK_TIMEOUT_SECONDS,
)
def train_task(payload: dict[str, Any]) -> dict[str, Any]:
    """Run exactly one model/dataset/seed task and persist it before returning."""
    task = payload["task"]
    config = payload["config"]
    if not (Path(VOLUME_ROOT) / "data").is_dir():
        raise FileNotFoundError(
            f"Dataset directory {VOLUME_ROOT}/data is missing; upload it to "
            f"Modal Volume {VOLUME_NAME!r} before submitting tasks"
        )
    command = task_command(task, config)
    try:
        completed = subprocess.run(command, cwd="/", check=False)
    finally:
        # Every task writes to a unique hierarchy below /outputs.
        volume.commit()
    if completed.returncode:
        raise subprocess.CalledProcessError(completed.returncode, command)
    return {
        "model": task["model"],
        "experiment_id": task["experiment_id"],
        "dataset": task["dataset"],
        "seed": task["seed"],
        "returncode": completed.returncode,
    }


def read_pending_file(path: str) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    tasks = payload.get("tasks", payload) if isinstance(payload, dict) else payload
    if not isinstance(tasks, list):
        raise ValueError("pending_tasks.json must contain a list or a {'tasks': [...]} object")
    return tasks


def configured_tasks(config: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "experiment_id": config["experiment_id"],
            "model": model,
            "dataset": dataset,
            "seed": int(seed),
        }
        for model in config["models"]
        for seed in config["seeds"]
        for dataset in config["datasets"]
    ]


def csv_values(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def filter_tasks(
    tasks: list[dict[str, Any]],
    models: str,
    seeds: str,
    datasets: str,
) -> list[dict[str, Any]]:
    chosen_models = set(csv_values(models))
    chosen_seeds = {int(value) for value in csv_values(seeds)}
    chosen_datasets = set(csv_values(datasets))
    return [
        task
        for task in tasks
        if (not chosen_models or str(task["model"]) in chosen_models)
        and (not chosen_seeds or int(task["seed"]) in chosen_seeds)
        and (not chosen_datasets or str(task["dataset"]) in chosen_datasets)
    ]


@app.local_entrypoint()
def main(
    pending_tasks: str = "",
    config: str = str(DEFAULT_CONFIG),
    experiment_id: str = "",
    models: str = "",
    seeds: str = "",
    datasets: str = "",
    max_tasks: int = 0,
    dry_run: bool = False,
    wait: bool = False,
) -> None:
    experiment_config = load_config(config)
    if experiment_id:
        experiment_config["experiment_id"] = experiment_id
    tasks = read_pending_file(pending_tasks) if pending_tasks else configured_tasks(experiment_config)
    if experiment_id:
        for task in tasks:
            task["experiment_id"] = experiment_id
    tasks = filter_tasks(tasks, models, seeds, datasets)
    if max_tasks > 0:
        tasks = tasks[:max_tasks]
    print(
        f"Modal plan: experiment_id={experiment_config['experiment_id']} "
        f"tasks={len(tasks)} gpu={GPU_TYPE} max_containers={MAX_CONTAINERS} "
        f"cpu={CPU_CORES} memory_mb={MEMORY_MB} timeout={TASK_TIMEOUT_SECONDS}s "
        f"retries={RETRIES} volume={VOLUME_NAME}"
    )
    for task in tasks:
        print(
            f"  {task['model']} / {task['dataset']} / seed={task['seed']}"
        )
    if dry_run:
        return
    payloads = [
        {"task": task, "config": experiment_config}
        for task in tasks
    ]
    if wait:
        failures = 0
        for result in train_task.map(
            payloads,
            order_outputs=False,
            return_exceptions=True,
        ):
            if isinstance(result, BaseException):
                failures += 1
                print(f"FAILED: {result}")
            else:
                print(result)
        print(
            f"Completed {len(tasks) - failures}/{len(tasks)} blocking task calls; "
            f"failures={failures}."
        )
        if failures:
            raise RuntimeError(f"{failures} Modal task(s) failed")
    else:
        train_task.spawn_map(payloads)
        print(f"Submitted {len(tasks)} background task calls.")

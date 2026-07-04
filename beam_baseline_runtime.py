from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

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


def csv_values(value):
    return [part.strip() for part in str(value).split(",") if part.strip()]


def copy_tree_contents(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--pending", required=True)
    parser.add_argument("--task-offset", type=int, default=0)
    parser.add_argument("--task-limit", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--hold-after", action="store_true")
    parser.add_argument("--hold-seconds", type=int, default=3600)
    args = parser.parse_args()

    os.environ.setdefault("WANDB__SERVICE_WAIT", "300")
    os.environ.setdefault("WANDB_START_METHOD", "thread")
    os.environ.setdefault("WANDB_CONSOLE", "off")

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    pending_payload = json.loads(Path(args.pending).read_text(encoding="utf-8"))["tasks"]
    tasks = []
    for task in pending_payload:
        for dataset in csv_values(task["dataset"]):
            model = str(task["model"])
            if model.lower() == "our_net":
                continue
            tasks.append({**task, "dataset": dataset})
    if args.task_limit > 0:
        tasks = tasks[args.task_offset : args.task_offset + args.task_limit]
    elif args.task_offset:
        tasks = tasks[args.task_offset:]
    if args.smoke:
        tasks = tasks[:1]

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
    if args.smoke:
        common.update({"epochs": 1, "eval_start_epoch": 1, "eval_every": 1})
    if os.environ.get("BEAM_EPOCHS_OVERRIDE"):
        common["epochs"] = int(os.environ["BEAM_EPOCHS_OVERRIDE"])
    if os.environ.get("BEAM_EVAL_START_OVERRIDE"):
        common["eval_start_epoch"] = int(os.environ["BEAM_EVAL_START_OVERRIDE"])

    run_root = Path("/tmp/retinal-beam-run")
    out_tmp = run_root / "outputs"
    persistent_root = Path("/outputs")
    print("BEAM_RUNTIME_TASKS", len(tasks), "SMOKE", args.smoke, "HOLD_AFTER", args.hold_after, flush=True)
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], check=False)
    subprocess.run([sys.executable, "-m", "py_compile", "train.py", "eval.py"], check=True)

    for index, task in enumerate(tasks, start=1):
        exp_id = str(task.get("experiment_id", config["experiment_id"]))
        if out_tmp.exists():
            shutil.rmtree(out_tmp)
        out_tmp.mkdir(parents=True, exist_ok=True)
        copy_tree_contents(persistent_root / exp_id, out_tmp / exp_id)
        cmd = [
            sys.executable,
            "-X",
            "faulthandler",
            "-u",
            "train.py",
            "--model",
            str(task["model"]),
            "--seed",
            str(task["seed"]),
            "--experiment-id",
            exp_id,
            "--datasets",
            str(task["dataset"]),
            "--data-root",
            "data",
            "--output-root",
            str(out_tmp),
            "--wandb-entity",
            str(config["wandb_entity"]),
            "--wandb-project",
            str(config["wandb_project"]),
        ]
        for key, value in common.items():
            cmd.extend(flag(key, value))
        cmd.append("--resume")
        print("RUN_TASK", f"{index}/{len(tasks)}", task["model"], task["dataset"], f"seed={task['seed']}", flush=True)
        print("CMD", " ".join(shlex.quote(part) for part in cmd), flush=True)
        status = subprocess.run(cmd, cwd="/mnt/code", check=False).returncode
        print("AFTER_TRAIN_STATUS", status, flush=True)
        copy_tree_contents(out_tmp, persistent_root)
        subprocess.run(["sync"], check=False)
        if status:
            print("TASK_FAILED_STATUS", status, flush=True)
            if args.hold_after:
                print("HOLD_AFTER_FAILURE", flush=True)
                subprocess.run(["sleep", str(args.hold_seconds)], check=False)
            return status
    print("BEAM_RUNTIME_DONE", flush=True)
    if args.hold_after:
        print("HOLD_AFTER_DONE", flush=True)
        subprocess.run(["sleep", str(args.hold_seconds)], check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

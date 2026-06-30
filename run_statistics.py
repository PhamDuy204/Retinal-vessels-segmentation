#!/usr/bin/env python3
"""Run the configured model-by-seed matrix, one train.py subprocess at a time."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from statistics_utils import (
    build_dataset_tasks,
    is_completed,
    load_statistics_config,
    pending_tasks,
    result_directory,
    write_json,
)


DEFAULT_CONFIG = Path(__file__).parent / "configs" / "statistics.json"
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
    "output_root": "--output-root",
    "data_root": "--data-root",
    "wandb_mode": "--wandb-mode",
    "log_artifacts": "--log-artifacts",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--max-runs", type=int)
    return parser.parse_args(argv)


def validate_subset(chosen: Iterable[Any] | None, configured: Iterable[Any], label: str) -> None:
    if chosen is None:
        return
    configured_values = set(configured)
    unknown = [value for value in chosen if value not in configured_values]
    if unknown:
        raise ValueError(f"Unknown {label}: {', '.join(map(str, unknown))}")


def common_args_to_cli(common_args: Mapping[str, Any]) -> list[str]:
    command: list[str] = []
    for key, value in common_args.items():
        if value is None or value is False:
            continue
        flag = FLAG_NAMES.get(key, f"--{key.replace('_', '-')}")
        command.append(flag)
        if value is not True:
            command.append(str(value))
    return command


def train_command(
    config: Mapping[str, Any], model: str, seed: int, resume: bool
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).parent / "train.py"),
        "--model",
        model,
        "--seed",
        str(seed),
        "--experiment-id",
        str(config["experiment_id"]),
        "--datasets",
        *map(str, config["datasets"]),
        "--wandb-entity",
        str(config["wandb_entity"]),
        "--wandb-project",
        str(config["wandb_project"]),
        *common_args_to_cli(config.get("common_train_args", {})),
    ]
    if resume:
        command.append("--resume")
    return command


def invocation_pairs(models: Iterable[str], seeds: Iterable[int]) -> list[tuple[str, int]]:
    return [(model, int(seed)) for model in models for seed in seeds]


def run_and_tee(command: list[str], log_handle: Any) -> int:
    rendered = shlex.join(command)
    print(f"$ {rendered}")
    log_handle.write(f"$ {rendered}\n")
    log_handle.flush()
    process = subprocess.Popen(
        command,
        cwd=Path(__file__).parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        log_handle.write(line)
        log_handle.flush()
    return process.wait()


def pair_has_pending_tasks(
    config: Mapping[str, Any], model: str, seed: int, output_root: str
) -> bool:
    return any(
        not is_completed(
            result_directory(
                output_root,
                str(config["experiment_id"]),
                model,
                dataset,
                seed,
            )
        )
        for dataset in config["datasets"]
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_statistics_config(args.config)
    validate_subset(args.models, config["models"], "models")
    validate_subset(args.seeds, [int(seed) for seed in config["seeds"]], "seeds")
    models = list(args.models) if args.models else list(config["models"])
    seeds = list(args.seeds) if args.seeds else [int(seed) for seed in config["seeds"]]
    output_root = str(config.get("common_train_args", {}).get("output_root", "outputs"))

    full_tasks = build_dataset_tasks(config)
    selected_tasks = build_dataset_tasks(config, models=models, seeds=seeds)
    print(
        f"Expected total dataset tasks: {len(full_tasks)} "
        f"({len(config['models'])}×{len(config['seeds'])}×{len(config['datasets'])})"
    )
    print(f"Selected dataset tasks: {len(selected_tasks)}")

    orchestration_dir = (
        Path(output_root) / str(config["experiment_id"]) / "orchestration"
    )
    orchestration_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = orchestration_dir / f"run_statistics_{timestamp}.log"

    pairs = invocation_pairs(models, seeds)
    if args.resume:
        pairs = [
            pair
            for pair in pairs
            if pair_has_pending_tasks(config, pair[0], pair[1], output_root)
        ]
    if args.max_runs is not None:
        pairs = pairs[: max(args.max_runs, 0)]

    failures = 0
    with log_path.open("w", encoding="utf-8") as log_handle:
        for model, seed in pairs:
            command = train_command(config, model, seed, args.resume)
            if args.dry_run:
                rendered = shlex.join(command)
                print(rendered)
                log_handle.write(f"{rendered}\n")
                continue
            return_code = run_and_tee(command, log_handle)
            if return_code:
                failures += 1
                message = f"train.py failed for model={model}, seed={seed}: exit {return_code}"
                print(message, file=sys.stderr)
                log_handle.write(f"{message}\n")

    remaining = pending_tasks(selected_tasks, output_root)
    pending_path = orchestration_dir / "pending_tasks.json"
    write_json(
        pending_path,
        {
            "experiment_id": config["experiment_id"],
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "output_root": output_root,
            "tasks": remaining,
        },
    )
    print(f"Pending dataset tasks: {len(remaining)}")
    print(f"Orchestration log: {log_path}")
    print(f"Pending task file: {pending_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

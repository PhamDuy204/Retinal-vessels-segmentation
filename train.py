from __future__ import annotations

import argparse
import copy
import csv
import fnmatch
import inspect
import math
import os
import queue
import sys
import traceback
import uuid
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import wandb
from torch.multiprocessing import Process, Queue
from tqdm.auto import tqdm

from data_preparation import get_all_training_set, make_data_loader
from eval import eval_for_seg
from load_model import load_loss_class, load_model_class
from set_up_seed import set_seed
from statistics_utils import (
    candidate_is_better,
    is_completed,
    result_directory,
    selection_key,
    write_json,
)
from training_profiler import TrainingProfiler, profile_evaluation
from utils import check_model_forward_args, count_trainable_params


EVALUATION_THRESHOLD = 0.487
SAMPLED_PATCHES_PER_IMAGE = 750
EPOCH_FIELDS = (
    "epoch",
    "loss",
    "acc",
    "f1",
    "iou",
    "recall",
    "specificity",
    "auc",
    "dice",
    "cdice",
    "threshold",
    "lr",
)


def generate_experiment_id(output_root: str | os.PathLike[str]) -> str:
    """Create and reserve a unique experiment directory for an ad-hoc run."""
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    while True:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        experiment_id = f"run-{timestamp}-{uuid.uuid4().hex}"
        try:
            # Atomic directory creation prevents concurrent runs from claiming
            # the same generated ID on a shared filesystem.
            (root / experiment_id).mkdir(exist_ok=False)
        except FileExistsError:
            continue
        return experiment_id


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train retinal vessel segmentation models")
    parser.add_argument("-b", "--batch_size", type=int, default=4)
    parser.add_argument("-e", "--epochs", type=int, default=100)
    parser.add_argument("-lf", "--loss", type=str, default="abe_dice_loss")
    parser.add_argument("-m", "--model", type=str, default="unet")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001)
    parser.add_argument("-p", "--patches", type=int, default=500)
    parser.add_argument("-ps", "--patch_size", type=int, default=64)
    parser.add_argument("-tt", "--train_type", type=str, default="patch")
    parser.add_argument("-ch", "--chunk_size", type=int, default=None)
    parser.add_argument("-ts", "--type_split", type=str, default="window")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--experiment-id",
        default=None,
        help="Stable experiment ID; omitted values receive a reserved timestamp + UUID4 ID",
    )
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Dataset experiment names or shell-style filters (comma-separated values also work)",
    )
    parser.add_argument(
        "--wandb-entity",
        default="phamdinhanhduy-university-of-information-and-technology",
    )
    parser.add_argument("--wandb-project", default="Retinal-Vessels-Segmentation")
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default=os.environ.get("WANDB_MODE", "online"),
    )
    parser.add_argument("--log-artifacts", action="store_true")
    parser.add_argument("--wandb-watch", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable experimental CUDA FP16 autocast for training",
    )
    parser.add_argument(
        "--eval-amp",
        action="store_true",
        help="Enable CUDA FP16 autocast only during evaluation",
    )
    parser.add_argument("--eval-batch-size", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument(
        "--fast-nondeterministic",
        action="store_true",
        help="Enable cuDNN autotuning and nondeterministic CUDA algorithms",
    )
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-steps", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    if args.profile_steps < 1:
        parser.error("--profile-steps must be at least 1")
    if args.eval_batch_size < 0:
        parser.error("--eval-batch-size cannot be negative")
    if args.eval_every < 1:
        parser.error("--eval-every must be at least 1")
    if not args.experiment_id:
        args.experiment_id = generate_experiment_id(args.output_root)
    return args


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def configure_cuda_runtime(fast_nondeterministic: bool) -> None:
    torch.backends.cudnn.benchmark = fast_nondeterministic
    torch.backends.cudnn.deterministic = not fast_nondeterministic
    torch.use_deterministic_algorithms(
        not fast_nondeterministic,
        warn_only=True,
    )


class Tee:
    def __init__(self, *streams: Any):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return False


def normalize_dataset_filters(values: Iterable[str] | None) -> list[str]:
    if not values:
        return []
    return [part for value in values for part in value.split(",") if part]


def filter_datasets(
    datasets: list[dict[str, Any]], filters: Iterable[str] | None
) -> list[dict[str, Any]]:
    patterns = normalize_dataset_filters(filters)
    if not patterns:
        return datasets
    selected = [
        dataset
        for dataset in datasets
        if any(fnmatch.fnmatchcase(dataset["name"], pattern) for pattern in patterns)
    ]
    unmatched = [
        pattern
        for pattern in patterns
        if not any(fnmatch.fnmatchcase(dataset["name"], pattern) for dataset in datasets)
    ]
    if unmatched:
        available = ", ".join(dataset["name"] for dataset in datasets)
        raise ValueError(
            f"Dataset filters matched nothing: {', '.join(unmatched)}. Available: {available}"
        )
    return selected


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: Any,
        val_loader: Any,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        gpu_id: int,
        dataset_name: str,
        run_dir: Path,
        patch: bool,
        args: argparse.Namespace,
        wandb_run: Any,
        model_forward_args: int,
        model_class_name: str,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.gpu_id = gpu_id
        self.dataset_name = dataset_name
        self.run_dir = run_dir
        self.patch = patch
        self.args = args
        self.wandb_run = wandb_run
        self.model_forward_args = model_forward_args
        self.model_class_name = model_class_name

    def train(self) -> dict[str, Any]:
        torch.cuda.set_device(self.gpu_id)
        self.model.cuda()
        if self.args.wandb_watch:
            wandb.watch(self.model, self.criterion, log="all", log_freq=100)

        scaler = torch.amp.GradScaler("cuda", enabled=self.args.amp)
        profiler = TrainingProfiler(
            enabled=self.args.profile,
            steps=self.args.profile_steps,
            output_dir=self.run_dir,
        )
        profiler.start()

        best_row: dict[str, Any] | None = None
        best_params: dict[str, Any] | None = None
        evaluated_rows: list[dict[str, Any]] = []
        metrics_path = self.run_dir / "epoch_metrics.csv"
        with metrics_path.open("w", newline="", encoding="utf-8") as metrics_file:
            writer = csv.DictWriter(metrics_file, fieldnames=EPOCH_FIELDS)
            writer.writeheader()

            for epoch in range(1, self.args.epochs + 1):
                self.model.train()
                training_loss = 0.0
                train_iterator = iter(self.train_loader)
                for _ in tqdm(range(len(self.train_loader))):
                    with profiler.region("data_loading"):
                        sample = next(train_iterator)
                    image, mask, edge = sample.values()
                    if len(image.shape) > 4:
                        image = image.flatten(0, 1)
                        mask = mask.flatten(0, 1)
                        edge = edge.flatten(0, 1)

                    # Copy each patch group once instead of issuing one host-to-device
                    # transfer per micro-batch. Keep the CPU-generated permutation so
                    # seeded runs retain the same patch order.
                    random_index = torch.randperm(image.size(0))
                    device = torch.device("cuda", self.gpu_id)
                    image = image.to(device, non_blocking=True)
                    mask = mask.to(device, non_blocking=True)
                    random_index = random_index.to(device, non_blocking=True)
                    image = image.index_select(0, random_index)
                    mask = mask.index_select(0, random_index)
                    if self.model_forward_args == 2:
                        edge = edge.to(device, non_blocking=True)
                        edge = edge.index_select(0, random_index)
                    else:
                        edge = None
                    if self.args.chunk_size is None:
                        chunk_size = max(
                            min(
                                math.ceil(image.shape[0] / self.args.batch_size),
                                16 * self.args.batch_size,
                            ),
                            1,
                        )
                    else:
                        chunk_size = self.args.chunk_size

                    image_chunks = torch.chunk(image, chunk_size)
                    mask_chunks = torch.chunk(mask, chunk_size)
                    edge_chunks = (
                        torch.chunk(edge, chunk_size)
                        if edge is not None
                        else (None,) * len(image_chunks)
                    )
                    for next_image, next_mask, next_edge in zip(
                        image_chunks, mask_chunks, edge_chunks
                    ):
                        with profiler.region("model_forward"):
                            with torch.amp.autocast("cuda", enabled=self.args.amp):
                                if self.model_forward_args == 2:
                                    predicted_mask = self.model(next_image, next_edge)
                                else:
                                    predicted_mask = self.model(next_image)
                        with profiler.region("loss_forward"):
                            with torch.amp.autocast("cuda", enabled=self.args.amp):
                                loss = self.criterion(predicted_mask, next_mask)
                        loss_value = float(loss.detach())
                        if not math.isfinite(loss_value):
                            outputs = (
                                predicted_mask
                                if isinstance(predicted_mask, (tuple, list))
                                else (predicted_mask,)
                            )
                            finite_ratios = [
                                float(torch.isfinite(output).float().mean())
                                for output in outputs
                            ]
                            amp_hint = (
                                " Disable --amp and rerun; this model is unstable "
                                "under FP16 autocast."
                                if self.args.amp
                                else ""
                            )
                            raise FloatingPointError(
                                f"Non-finite loss on dataset={self.dataset_name}, "
                                f"epoch={epoch}, output_finite_ratios={finite_ratios}."
                                f"{amp_hint}"
                            )
                        self.optimizer.zero_grad(set_to_none=True)
                        with profiler.region("backward"):
                            scaler.scale(loss).backward()
                        with profiler.region("optimizer_step"):
                            scaler.step(self.optimizer)
                            scaler.update()
                        training_loss += loss_value
                        profiler.step()

                # Preserve the existing scheduler/evaluation order.
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]["lr"]
                should_evaluate = (
                    epoch % self.args.eval_every == 0
                    or epoch == self.args.epochs
                )
                if not should_evaluate:
                    self.wandb_run.log(
                        {
                            "epoch": epoch,
                            "loss": training_loss,
                            "lr": current_lr,
                        }
                    )
                    print(
                        f"[Epoch {epoch}/{self.args.epochs}] "
                        f"Dataset: {self.dataset_name} | "
                        f"Loss: {training_loss:.4f} | evaluation skipped"
                    )
                    continue

                def evaluate():
                    return eval_for_seg(
                        self.model,
                        self.val_loader,
                        self.gpu_id,
                        self.patch,
                        self.args.patch_size,
                        self.args.type_split,
                        threshold=EVALUATION_THRESHOLD,
                        non_blocking=True,
                        amp=self.args.eval_amp,
                        profile=self.args.profile and epoch == 1,
                        batch_size=self.args.eval_batch_size,
                    )

                evaluation_result, _evaluation_profile = profile_evaluation(
                    self.args.profile and epoch == 1,
                    self.run_dir,
                    evaluate,
                )
                (
                    acc,
                    f1,
                    iou,
                    recall,
                    specificity,
                    auc,
                    dice,
                    cdice,
                    _roc_threshold,
                ) = evaluation_result
                row = {
                    "epoch": epoch,
                    "loss": training_loss,
                    "acc": acc,
                    "f1": f1,
                    "iou": iou,
                    "recall": recall,
                    "specificity": specificity,
                    "auc": auc,
                    "dice": dice,
                    "cdice": cdice,
                    "threshold": EVALUATION_THRESHOLD,
                    "lr": current_lr,
                }
                evaluated_rows.append(dict(row))
                writer.writerow(row)
                metrics_file.flush()
                self.wandb_run.log(
                    {
                        "epoch": epoch,
                        "loss": training_loss,
                        "val_acc": acc,
                        "val_f1": f1,
                        "val_iou": iou,
                        "val_recall": recall,
                        "val_specificity": specificity,
                        "val_auc": auc,
                        "val_dice": dice,
                        "val_cdice": cdice,
                        "val_threshold": EVALUATION_THRESHOLD,
                        "lr": current_lr,
                    }
                )
                print(
                    f"[Epoch {epoch}/{self.args.epochs}] Dataset: {self.dataset_name} | "
                    f"Loss: {training_loss:.4f} | Acc: {acc:.4f} | F1: {f1:.4f} | "
                    f"IoU: {iou:.4f} | Recall: {recall:.4f} | "
                    f"Specificity: {specificity:.4f} | DiceScore: {dice:.4f} | "
                    f"cDice: {cdice:.4f} | "
                    f"AUC: {auc:.5f}"
                )

                if candidate_is_better(row, best_row):
                    best_row = dict(row)
                    checkpoint_model = getattr(self.model, "_orig_mod", self.model)
                    best_params = copy.deepcopy(checkpoint_model.state_dict())

        profiler.finish()

        if best_row is None or best_params is None:
            raise RuntimeError("Training completed without an evaluated epoch")

        checkpoint_path = self.run_dir / "best.pt"
        torch.save(
            {
                "model_state_dict": best_params,
                "experiment_id": self.args.experiment_id,
                "model": self.args.model,
                "dataset": self.dataset_name,
                "seed": self.args.seed,
                "selected_epoch": int(best_row["epoch"]),
                "selection_key": selection_key(best_row),
            },
            checkpoint_path,
        )
        selected_result = {
            "experiment_id": self.args.experiment_id,
            "model": self.args.model,
            "model_class": self.model_class_name,
            "dataset": self.dataset_name,
            "seed": self.args.seed,
            "selected_epoch": int(best_row["epoch"]),
            "selection_key": list(selection_key(best_row)),
            "wandb_run_id": getattr(self.wandb_run, "id", None),
            "wandb_run_url": getattr(self.wandb_run, "url", None),
            **{field: best_row[field] for field in EPOCH_FIELDS if field != "epoch"},
        }
        write_json(self.run_dir / "selected_result.json", selected_result)

        summary = {
            "experiment_id": self.args.experiment_id,
            "selected_epoch": selected_result["selected_epoch"],
            "best_epoch": selected_result["selected_epoch"],
            "best_selection_key": selected_result["selection_key"],
            "best_selection_rule": "lexicographic(AUC, Recall, F1, cDice)",
            **{f"selected_{metric}": selected_result[metric] for metric in EPOCH_FIELDS[1:]},
            **{f"best_{metric}": selected_result[metric] for metric in EPOCH_FIELDS[1:]},
        }
        self.wandb_run.summary.update(summary)

        if self.args.wandb_mode != "disabled":
            table_columns = [*EPOCH_FIELDS, "is_best_epoch"]
            epoch_selection_table = wandb.Table(
                columns=table_columns,
                data=[
                    [
                        *(row[field] for field in EPOCH_FIELDS),
                        int(row["epoch"]) == selected_result["selected_epoch"],
                    ]
                    for row in evaluated_rows
                ],
            )
            self.wandb_run.log({"epoch_selection_table": epoch_selection_table})

        if self.args.log_artifacts:
            artifact = wandb.Artifact(
                name=(
                    f"{self.args.model}__{self.dataset_name}__seed{self.args.seed}"
                ),
                type="model",
            )
            artifact.add_file(str(checkpoint_path))
            artifact.add_file(str(self.run_dir / "selected_result.json"))
            artifact.add_file(str(metrics_path))
            artifact.add_file(str(Path(__file__).parent / "loss" / f"{self.args.loss}.py"))
            artifact.add_dir(str(Path(__file__).parent / "models" / self.args.model))
            self.wandb_run.log_artifact(artifact)

        return selected_result


def wandb_config(
    args: argparse.Namespace,
    dataset_name: str,
    patch: bool,
    model: torch.nn.Module,
    gpu_id: int,
    model_module_path: str,
) -> dict[str, Any]:
    return {
        "experiment_id": args.experiment_id,
        "model": args.model,
        "model_folder_name": args.model,
        "model_class": type(model).__name__,
        "model_module_path": model_module_path,
        "dataset": dataset_name,
        "seed": args.seed,
        "threshold": EVALUATION_THRESHOLD,
        "loss": args.loss,
        "patch_size": args.patch_size if patch else None,
        "sampled_patches_per_image": SAMPLED_PATCHES_PER_IMAGE if patch else 0,
        "type_split": args.type_split if patch else None,
        "optimizer": "Adam",
        "lr": args.learning_rate,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gpu": gpu_id,
        "num_parameters": count_trainable_params(model),
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "persistent_workers": args.persistent_workers,
        "amp": args.amp,
        "eval_amp": args.eval_amp,
        "eval_batch_size": args.eval_batch_size,
        "eval_every": args.eval_every,
        "fast_nondeterministic": args.fast_nondeterministic,
        "tf32": args.tf32,
        "compile_model": args.compile_model,
    }


def gpu_worker(
    gpu_id: int,
    task_queue: Queue,
    result_queue: Queue,
    datasets: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    torch.cuda.set_device(gpu_id)
    while True:
        try:
            dataset_id = task_queue.get_nowait()
        except queue.Empty:
            break

        info = datasets[dataset_id]
        dataset_name = info["name"]
        run_dir = result_directory(
            args.output_root,
            args.experiment_id,
            args.model,
            dataset_name,
            args.seed,
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        task_metadata = {
            "experiment_id": args.experiment_id,
            "model": args.model,
            "dataset": dataset_name,
            "seed": args.seed,
            "gpu": gpu_id,
        }
        write_json(
            run_dir / "status.json",
            {"status": "running", "started_at": utc_now(), **task_metadata},
        )

        wandb_run = None
        try:
            with (run_dir / "stdout.log").open("a", encoding="utf-8") as log_file:
                stdout_tee = Tee(sys.__stdout__, log_file)
                stderr_tee = Tee(sys.__stderr__, log_file)
                with redirect_stdout(stdout_tee), redirect_stderr(stderr_tee):
                    # A GPU process can run several datasets, so reset every
                    # RNG at the beginning of every dataset task.
                    set_seed(args.seed)
                    # Construct worker-owning DataLoaders inside the GPU process;
                    # multiprocessing DataLoaders cannot safely be pickled from
                    # the parent orchestration process.
                    train_loader = make_data_loader(
                        info["train_loader"].dataset,
                        args.batch_size,
                        True,
                        args.num_workers,
                        args.pin_memory,
                        args.persistent_workers,
                        args.seed,
                    )
                    val_loader = make_data_loader(
                        info["val_loader"].dataset,
                        1,
                        False,
                        args.num_workers,
                        args.pin_memory,
                        args.persistent_workers,
                        args.seed,
                    )
                    patch = bool(info["patches"])

                    # Reset immediately before initialization as well, making
                    # identical model-seed initializations independent of queue order.
                    model_class = load_model_class(args.model)
                    set_seed(args.seed)
                    configure_cuda_runtime(args.fast_nondeterministic)
                    model = model_class(1, 1).cuda()
                    model_module_path = str(Path(inspect.getfile(model_class)).resolve())
                    expected_module_path = str(
                        (Path(__file__).parent / "models" / args.model / f"{args.model}.py").resolve()
                    )
                    num_parameters = count_trainable_params(model)
                    if model_module_path != expected_module_path:
                        raise RuntimeError(
                            f"--model {args.model!r} resolved to {model_module_path}, "
                            f"expected {expected_module_path}"
                        )
                    if args.model == "our_net" and not 900_000 <= num_parameters <= 1_050_000:
                        raise RuntimeError(
                            f"our_net parameter count {num_parameters:,} is outside the expected ~0.96M range"
                        )
                    config = wandb_config(
                        args,
                        dataset_name,
                        patch,
                        model,
                        gpu_id,
                        model_module_path,
                    )
                    print(
                        f"[GPU {gpu_id}] starting experiment_id={args.experiment_id} "
                        f"model={args.model} dataset={dataset_name} seed={args.seed}"
                    )
                    print(f"Model argument: {args.model}")
                    print(f"Model Python module: {model_module_path}")
                    print(f"Trainable parameters: {num_parameters:,}")
                    wandb_run = wandb.init(
                        entity=args.wandb_entity,
                        project=args.wandb_project,
                        group=f"{args.experiment_id}/{args.model}/{dataset_name}",
                        name=f"{args.model}__{dataset_name}__seed{args.seed}",
                        config=config,
                        mode=args.wandb_mode,
                        reinit=True,
                    )
                    criterion = load_loss_class(args.loss)()
                    model_forward_args = check_model_forward_args(model)
                    model_class_name = type(model).__name__
                    torch.backends.cuda.matmul.allow_tf32 = args.tf32
                    torch.backends.cudnn.allow_tf32 = args.tf32
                    if args.tf32:
                        torch.set_float32_matmul_precision("high")
                    if args.compile_model:
                        model = torch.compile(model)
                    # Keep the existing Adam optimizer without weight decay.
                    optimizer = torch.optim.Adam(
                        model.parameters(), lr=args.learning_rate
                    )
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=args.epochs, eta_min=3e-6
                    )
                    trainer = Trainer(
                        model=model,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        criterion=criterion,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        gpu_id=gpu_id,
                        dataset_name=dataset_name,
                        run_dir=run_dir,
                        patch=patch,
                        args=args,
                        wandb_run=wandb_run,
                        model_forward_args=model_forward_args,
                        model_class_name=model_class_name,
                    )
                    selected_result = trainer.train()
                    write_json(
                        run_dir / "status.json",
                        {
                            "status": "completed",
                            "started_at": read_started_at(run_dir),
                            "completed_at": utc_now(),
                            "selected_epoch": selected_result["selected_epoch"],
                            **task_metadata,
                        },
                    )
                    result_queue.put((dataset_name, True, None))
        except Exception as error:
            details = traceback.format_exc()
            print(f"[GPU {gpu_id}] train on {dataset_name} failed: {error}", file=sys.stderr)
            print(details, file=sys.stderr)
            with (run_dir / "stdout.log").open("a", encoding="utf-8") as log_file:
                log_file.write(f"[GPU {gpu_id}] train on {dataset_name} failed: {error}\n")
                log_file.write(details)
            write_json(
                run_dir / "status.json",
                {
                    "status": "failed",
                    "started_at": read_started_at(run_dir),
                    "failed_at": utc_now(),
                    "exception_type": type(error).__name__,
                    "exception": str(error),
                    "traceback": details,
                    **task_metadata,
                },
            )
            result_queue.put((dataset_name, False, str(error)))
        finally:
            if wandb_run is not None:
                wandb.finish()
            torch.cuda.empty_cache()


def read_started_at(run_dir: Path) -> str | None:
    try:
        import json

        with (run_dir / "status.json").open(encoding="utf-8") as handle:
            return json.load(handle).get("started_at")
    except (OSError, ValueError, AttributeError):
        return None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Seed before dataset construction/execution.
    set_seed(args.seed)
    all_datasets = get_all_training_set(
        args.data_root,
        args.batch_size,
        args.patches,
        args.patch_size,
        args.train_type,
        args.type_split,
        args.model,
        0,
        False,
        False,
        None,
        args.seed,
    )
    datasets = filter_datasets(all_datasets, args.datasets)
    if args.resume:
        datasets = [
            dataset
            for dataset in datasets
            if not is_completed(
                result_directory(
                    args.output_root,
                    args.experiment_id,
                    args.model,
                    dataset["name"],
                    args.seed,
                )
            )
        ]

    print(f"Experiment ID: {args.experiment_id}")
    print(f"Dataset tasks to run: {len(datasets)}")
    if not datasets:
        print("All selected dataset tasks are already completed.")
        return 0

    if args.wandb_mode == "online":
        # Uses WANDB_API_KEY when set, otherwise the existing local login.
        wandb.login()

    number_of_gpus = min(torch.cuda.device_count() if torch.cuda.is_available() else 0, 4)
    if number_of_gpus == 0:
        raise RuntimeError("No CUDA GPU found; training requires at least one GPU")

    torch.multiprocessing.set_start_method("spawn", force=True)
    task_queue: Queue = Queue()
    for dataset_id in range(len(datasets)):
        task_queue.put(dataset_id)
    result_queue: Queue = Queue()

    processes: list[Process] = []
    for gpu_id in range(min(number_of_gpus, len(datasets))):
        process = Process(
            target=gpu_worker,
            args=(gpu_id, task_queue, result_queue, datasets, args),
        )
        process.start()
        processes.append(process)
    for process in processes:
        process.join()

    results: list[tuple[str, bool, str | None]] = []
    while True:
        try:
            results.append(result_queue.get_nowait())
        except queue.Empty:
            break
    failures = [result for result in results if not result[1]]
    crashed_processes = [process.exitcode for process in processes if process.exitcode != 0]
    if len(results) != len(datasets) or failures or crashed_processes:
        print(
            f"Completed {len(results) - len(failures)}/{len(datasets)} dataset tasks; "
            f"failures={len(failures)}, crashed_workers={len(crashed_processes)}",
            file=sys.stderr,
        )
        return 1
    print(f"Completed all {len(datasets)} dataset tasks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

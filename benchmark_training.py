#!/usr/bin/env python3
"""Benchmark our_net baseline, exact-safe pipeline, AMP, and torch.compile."""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path
from typing import Any

import torch
from torchmetrics.classification import (
    AUROC,
    Accuracy,
    BinaryF1Score,
    BinaryROC,
    JaccardIndex,
    Recall,
    Specificity,
)
from torchmetrics.segmentation import DiceScore

from data_preparation import get_all_training_set
from eval import centerline_dice, eval_for_seg
from load_model import load_loss_class, load_model_class
from set_up_seed import set_seed
from utils import (
    check_model_forward_args,
    clear_preprocessing_cache,
    count_trainable_params,
    extract_patches_with_target_count,
    mirror_padding_v2,
    preprocessing_cache_info,
    preprocessing_img,
    reverse_to_original_image,
)


THRESHOLD = 0.487


@dataclass(frozen=True)
class Scenario:
    name: str
    loss: str
    non_blocking: bool = False
    amp: bool = False
    tf32: bool = False
    compile_model: bool = False
    experimental: bool = False


@dataclass
class BenchmarkResult:
    scenario: str
    status: str
    median_step_ms: float | None = None
    measured_steps_time_s: float | None = None
    epoch_time_s: float | None = None
    evaluation_time_s: float | None = None
    peak_vram_mb: float | None = None
    throughput_patches_s: float | None = None
    final_loss: float | None = None
    error: str | None = None


class LimitedLoader:
    def __init__(self, loader: Any, limit: int):
        self.loader = loader
        self.limit = min(limit, len(loader)) if limit > 0 else len(loader)

    def __len__(self) -> int:
        return self.limit

    def __iter__(self):
        return islice(iter(self.loader), self.limit)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="our_net")
    parser.add_argument("--dataset", default="DRIVE_patches")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--micro-batch", type=int, default=16)
    parser.add_argument("--eval-batches", type=int, default=0)
    parser.add_argument(
        "--full-epoch",
        action="store_true",
        help="Measure every optimizer step in an epoch instead of estimating from the median",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--output-dir", default="benchmark_results")
    parser.add_argument("--skip-amp", action="store_true")
    parser.add_argument("--skip-tf32", action="store_true")
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--strict-atol", type=float, default=1e-7)
    return parser.parse_args()


def cuda_elapsed_ms(start: torch.cuda.Event, end: torch.cuda.Event) -> float:
    end.synchronize()
    return start.elapsed_time(end)


def first_fixed_batch(loader: Any, micro_batch: int) -> tuple[torch.Tensor, ...]:
    sample = next(iter(loader))
    image, mask, edge = sample.values()
    if image.ndim > 4:
        image = image.flatten(0, 1)
        mask = mask.flatten(0, 1)
        edge = edge.flatten(0, 1)
    count = min(micro_batch, image.shape[0])
    return image[:count].contiguous(), mask[:count].contiguous(), edge[:count].contiguous()


def make_model(model_name: str, state: dict[str, torch.Tensor], compile_model: bool):
    model_class = load_model_class(model_name)
    model = model_class(1, 1).cuda()
    model.load_state_dict(state)
    model.train()
    if compile_model:
        model = torch.compile(model)
    return model


def benchmark_scenario(
    scenario: Scenario,
    args: argparse.Namespace,
    initial_state: dict[str, torch.Tensor],
    cpu_batch: tuple[torch.Tensor, ...],
    epoch_steps: int,
) -> BenchmarkResult:
    try:
        set_seed(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = scenario.tf32
        torch.backends.cudnn.allow_tf32 = scenario.tf32
        torch.set_float32_matmul_precision("high" if scenario.tf32 else "highest")
        model = make_model(args.model, initial_state, scenario.compile_model)
        criterion = load_loss_class(scenario.loss)().cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scaler = torch.amp.GradScaler("cuda", enabled=scenario.amp)
        forward_args = 1 if args.model == "our_net" else check_model_forward_args(model)

        def step() -> float:
            image, mask, edge = (
                tensor.to("cuda", non_blocking=scenario.non_blocking)
                for tensor in cpu_batch
            )
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=scenario.amp):
                prediction = model(image, edge) if forward_args == 2 else model(image)
                loss = criterion(prediction, mask)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            return float(loss.detach())

        for _ in range(args.warmup):
            step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        timings: list[float] = []
        losses: list[float] = []
        wall_start = time.perf_counter()
        measured_steps = epoch_steps if args.full_epoch else args.steps
        for _ in range(measured_steps):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            losses.append(step())
            end.record()
            timings.append(cuda_elapsed_ms(start, end))
        measured_time = time.perf_counter() - wall_start
        median_step_ms = statistics.median(timings)
        return BenchmarkResult(
            scenario=scenario.name,
            status="completed",
            median_step_ms=median_step_ms,
            measured_steps_time_s=measured_time,
            epoch_time_s=(
                measured_time
                if args.full_epoch
                else (median_step_ms / 1000) * epoch_steps
            ),
            peak_vram_mb=torch.cuda.max_memory_allocated() / (1024**2),
            throughput_patches_s=len(cpu_batch[0]) / (median_step_ms / 1000),
            final_loss=losses[-1],
        )
    except Exception as error:
        return BenchmarkResult(
            scenario=scenario.name,
            status="failed",
            error=f"{type(error).__name__}: {error}",
        )
    finally:
        torch.cuda.empty_cache()


def eval_for_seg_baseline(
    model,
    val_loader,
    gpu_id,
    patch=True,
    patch_size=64,
    type_split="window",
):
    """Original evaluation path retained only as a benchmark reference."""
    torch.cuda.set_device(gpu_id)
    accuracy = Accuracy(task="binary").cuda()
    f1 = BinaryF1Score().cuda()
    jaccard = JaccardIndex(task="binary").cuda()
    recall = Recall(task="binary").cuda()
    specificity = Specificity(task="binary").cuda()
    roc = BinaryROC().cuda()
    auroc = AUROC(task="binary").cuda()
    dice = DiceScore(num_classes=2, average="macro").cuda()
    cdice_scores: list[float] = []
    forward_args = check_model_forward_args(model)
    model.eval()
    with torch.inference_mode():
        for sample in val_loader:
            image, mask, edge = sample.values()
            image = mirror_padding_v2(image)
            edge = mirror_padding_v2(edge)
            batch_size, channels, height, width = image.shape
            image, mask, edge = image.cuda(), mask.cuda(), edge.cuda()
            stride = None
            if patch and type_split != "random":
                patch_grid = (
                    (height - patch_size) // 32 + 1,
                    (width - patch_size) // 8 + 1,
                )
                image, stride = extract_patches_with_target_count(
                    image, patch_size, patch_grid
                )
                edge, _ = extract_patches_with_target_count(edge, patch_size, patch_grid)
                if image.ndim > 4:
                    image, edge = image.flatten(0, 1), edge.flatten(0, 1)
            chunk_count = max(image.shape[0] // 128, 1)
            output = []
            for image_chunk, edge_chunk in zip(
                torch.chunk(image, chunk_count, 0), torch.chunk(edge, chunk_count, 0)
            ):
                output.append(
                    model(image_chunk, edge_chunk)
                    if forward_args == 2
                    else model(image_chunk)
                )
            probability = torch.cat(output, 0)
            if stride is not None:
                probability = probability.view(
                    batch_size, -1, 1, patch_size, patch_size
                )
                threshold_probability = reverse_to_original_image(
                    probability, (height, width), patch_size, stride
                )
                probability = reverse_to_original_image(
                    probability, (height, width), patch_size, stride
                )
            else:
                threshold_probability = probability
            target_height, target_width = mask.shape[-2:]
            probability = probability[:, :, :target_height, :target_width].squeeze().flatten()
            threshold_probability = (
                threshold_probability[:, :, :target_height, :target_width]
                .squeeze()
                .flatten()
            )
            target = mask.squeeze().flatten()
            prediction = torch.where(threshold_probability >= THRESHOLD, 1, 0)
            cdice_scores.append(
                centerline_dice(
                    prediction.view(target_height, target_width),
                    target.view(target_height, target_width),
                )
            )
            accuracy.update(prediction, target)
            f1.update(prediction, target)
            jaccard.update(prediction, target)
            recall.update(prediction, target)
            specificity.update(prediction, target)
            auroc.update(probability, target)
            roc.update(probability, target)
            dice.update(
                prediction.unsqueeze(0).unsqueeze(0).long(),
                target.unsqueeze(0).unsqueeze(0).long(),
            )
            torch.cuda.empty_cache()
    false_positive_rate, true_positive_rate, thresholds = roc.compute()
    best_threshold = thresholds[torch.argmax(true_positive_rate - false_positive_rate)]
    return (
        accuracy.compute().item(),
        f1.compute().item(),
        jaccard.compute().item(),
        recall.compute().item(),
        specificity.compute().item(),
        auroc.compute().item(),
        dice.compute().item(),
        sum(cdice_scores) / len(cdice_scores),
        best_threshold.item(),
    )


def timed_evaluation(callback):
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    result = callback()
    torch.cuda.synchronize()
    return result, time.perf_counter() - start


def verify_loss_equivalence(
    args: argparse.Namespace,
    initial_state: dict[str, torch.Tensor],
    cpu_batch: tuple[torch.Tensor, ...],
) -> dict[str, float]:
    model = make_model(args.model, initial_state, False)
    image, mask, _edge = (tensor.cuda() for tensor in cpu_batch)
    with torch.no_grad():
        prediction = model(image)
        baseline = load_loss_class("abe_dice_loss")().cuda()(prediction, mask)
        optimized = load_loss_class("abe_dice_loss_optimized")().cuda()(prediction, mask)
    torch.testing.assert_close(
        optimized, baseline, rtol=0, atol=args.strict_atol
    )
    return {
        "baseline": float(baseline),
        "optimized": float(optimized),
        "absolute_difference": float(torch.abs(baseline - optimized)),
    }


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("benchmark_training.py requires a CUDA GPU")
    set_seed(args.seed)
    model_class = load_model_class(args.model)
    model_path = str(Path(inspect.getfile(model_class)).resolve())
    reference_model = model_class(1, 1).cuda()
    parameter_count = count_trainable_params(reference_model)
    print(f"Model argument: {args.model}")
    print(f"Model Python module: {model_path}")
    print(f"Trainable parameters: {parameter_count:,}")
    if args.model == "our_net" and not 900_000 <= parameter_count <= 1_050_000:
        raise RuntimeError("our_net does not have the expected ~0.96M parameters")

    datasets = get_all_training_set(
        args.data_root,
        batch_size=1,
        num_patches=500,
        patch_size=64,
        training_type="patch",
        type_split="window",
        model_name=args.model,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        seed=args.seed,
    )
    selected = [dataset for dataset in datasets if dataset["name"] == args.dataset]
    if not selected:
        raise ValueError(f"Unknown dataset experiment {args.dataset!r}")
    train_loader, val_loader = selected[0]["train_loader"], selected[0]["val_loader"]

    clear_preprocessing_cache()
    image_path = train_loader.dataset.image_paths[0]
    cold_start = time.perf_counter()
    preprocessing_img(image_path, model_name=args.model)
    cold_time = time.perf_counter() - cold_start
    warm_start = time.perf_counter()
    preprocessing_img(image_path, model_name=args.model)
    cached_time = time.perf_counter() - warm_start
    cache_benchmark = {
        "cold_preprocessing_ms": cold_time * 1000,
        "cached_preprocessing_ms": cached_time * 1000,
        "cache": str(preprocessing_cache_info()),
    }

    cpu_batch = first_fixed_batch(train_loader, args.micro_batch)
    initial_state = {
        key: value.detach().clone() for key, value in reference_model.state_dict().items()
    }
    loss_equivalence = verify_loss_equivalence(args, initial_state, cpu_batch)
    epoch_steps = math.ceil(len(train_loader.dataset) * 750 / len(cpu_batch[0]))

    scenarios = [
        Scenario("baseline", "abe_dice_loss"),
        Scenario("exact_safe_loss", "abe_dice_loss_optimized"),
        Scenario(
            "exact_safe_non_blocking",
            "abe_dice_loss_optimized",
            non_blocking=True,
        ),
        Scenario(
            "exact_safe_pipeline",
            "abe_dice_loss_optimized",
            non_blocking=True,
        ),
    ]
    if not args.skip_amp:
        scenarios.append(
            Scenario(
                "amp_experimental",
                "abe_dice_loss_optimized",
                non_blocking=True,
                amp=True,
                experimental=True,
            )
        )
    if not args.skip_tf32:
        scenarios.append(
            Scenario(
                "tf32_experimental",
                "abe_dice_loss_optimized",
                non_blocking=True,
                tf32=True,
                experimental=True,
            )
        )
    if not args.skip_compile:
        scenarios.append(
            Scenario(
                "compile_experimental",
                "abe_dice_loss_optimized",
                non_blocking=True,
                compile_model=True,
                experimental=True,
            )
        )

    results = [
        benchmark_scenario(scenario, args, initial_state, cpu_batch, epoch_steps)
        for scenario in scenarios
    ]

    limited_val_loader = LimitedLoader(val_loader, args.eval_batches)
    baseline_eval_model = make_model(args.model, initial_state, False)
    baseline_metrics, baseline_eval_time = timed_evaluation(
        lambda: eval_for_seg_baseline(
            baseline_eval_model, limited_val_loader, 0, True, 64, "window"
        )
    )
    optimized_eval_model = make_model(args.model, initial_state, False)
    optimized_metrics, optimized_eval_time = timed_evaluation(
        lambda: eval_for_seg(
            optimized_eval_model,
            limited_val_loader,
            0,
            True,
            64,
            "window",
            threshold=THRESHOLD,
            non_blocking=True,
        )
    )
    torch.testing.assert_close(
        torch.tensor(optimized_metrics[:8]),
        torch.tensor(baseline_metrics[:8]),
        rtol=0,
        atol=args.strict_atol,
    )
    for result in results:
        result.evaluation_time_s = (
            baseline_eval_time if result.scenario == "baseline" else optimized_eval_time
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": args.model,
        "model_module_path": model_path,
        "parameter_count": parameter_count,
        "dataset": args.dataset,
        "seed": args.seed,
        "warmup_steps": args.warmup,
        "measured_steps": epoch_steps if args.full_epoch else args.steps,
        "epoch_steps": epoch_steps,
        "epoch_time_is_estimated_from_median_step": not args.full_epoch,
        "epoch_time_excludes_data_loading": True,
        "cache_benchmark": cache_benchmark,
        "loss_equivalence": loss_equivalence,
        "baseline_metrics": baseline_metrics,
        "optimized_metrics": optimized_metrics,
        "baseline_evaluation_time_s": baseline_eval_time,
        "optimized_evaluation_time_s": optimized_eval_time,
        "results": [asdict(result) for result in results],
    }
    (output_dir / "benchmark.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    fields = list(asdict(results[0]).keys())
    with (output_dir / "benchmark.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

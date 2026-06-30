"""Pure helpers shared by statistical training and reporting scripts."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable, Mapping, Sequence


METRIC_NAMES = (
    "acc",
    "f1",
    "iou",
    "recall",
    "specificity",
    "auc",
    "dice",
    "cdice",
    "threshold",
)


def result_directory(
    output_root: str | os.PathLike[str],
    experiment_id: str,
    model: str,
    dataset: str,
    seed: int,
) -> Path:
    return (
        Path(output_root)
        / experiment_id
        / f"model={model}"
        / f"dataset={dataset}"
        / f"seed={int(seed)}"
    )


def write_json(path: str | os.PathLike[str], payload: Any) -> None:
    """Atomically write JSON so interrupted runs do not leave partial status."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, destination)


def read_status(run_dir: str | os.PathLike[str]) -> dict[str, Any] | None:
    path = Path(run_dir) / "status.json"
    try:
        with path.open(encoding="utf-8") as handle:
            status = json.load(handle)
    except (OSError, ValueError, TypeError):
        return None
    return status if isinstance(status, dict) else None


def is_completed(run_dir: str | os.PathLike[str]) -> bool:
    status = read_status(run_dir)
    return bool(status and status.get("status") == "completed")


def selection_key(metrics: Mapping[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(metrics["auc"]),
        float(metrics["recall"]),
        float(metrics["f1"]),
        float(metrics["cdice"]),
    )


def candidate_is_better(
    candidate: Mapping[str, Any], best: Mapping[str, Any] | None
) -> bool:
    return best is None or selection_key(candidate) > selection_key(best)


def select_best_epoch(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for row in rows:
        candidate = dict(row)
        if candidate_is_better(candidate, best):
            best = candidate
    if best is None:
        raise ValueError("Cannot select a best epoch from no metric rows")
    return best


def load_statistics_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        config = json.load(handle)
    required = ("experiment_id", "models", "seeds", "datasets")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Statistics config is missing: {', '.join(missing)}")
    return config


def build_dataset_tasks(
    config: Mapping[str, Any],
    models: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    datasets: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    chosen_models = list(models) if models is not None else list(config["models"])
    chosen_seeds = [int(seed) for seed in (seeds if seeds is not None else config["seeds"])]
    chosen_datasets = list(datasets) if datasets is not None else list(config["datasets"])
    return [
        {
            "experiment_id": str(config["experiment_id"]),
            "model": str(model),
            "dataset": str(dataset),
            "seed": seed,
        }
        for model in chosen_models
        for seed in chosen_seeds
        for dataset in chosen_datasets
    ]


def pending_tasks(
    tasks: Iterable[Mapping[str, Any]], output_root: str | os.PathLike[str]
) -> list[dict[str, Any]]:
    pending: list[dict[str, Any]] = []
    for task in tasks:
        run_dir = result_directory(
            output_root,
            str(task["experiment_id"]),
            str(task["model"]),
            str(task["dataset"]),
            int(task["seed"]),
        )
        if not is_completed(run_dir):
            pending.append(dict(task))
    return pending


def sample_mean_sd(values: Sequence[float]) -> tuple[float, float]:
    numeric = [float(value) for value in values]
    if not numeric:
        return math.nan, math.nan
    if len(numeric) == 1:
        return numeric[0], math.nan
    return mean(numeric), stdev(numeric)


def format_mean_sd(values: Sequence[float], digits: int = 4) -> str:
    average, standard_deviation = sample_mean_sd(values)
    if math.isnan(standard_deviation):
        return f"{average:.{digits}f} ± NA"
    return f"{average:.{digits}f} ± {standard_deviation:.{digits}f}"


def seed_issues(
    records: Iterable[Mapping[str, Any]],
    models: Sequence[str],
    datasets: Sequence[str],
    expected_seeds: Sequence[int],
) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str, int], int] = {}
    for record in records:
        key = (str(record["model"]), str(record["dataset"]), int(record["seed"]))
        counts[key] = counts.get(key, 0) + 1

    issues: list[dict[str, Any]] = []
    for model in models:
        for dataset in datasets:
            for seed in expected_seeds:
                count = counts.get((str(model), str(dataset), int(seed)), 0)
                if count == 0:
                    issues.append(
                        {
                            "model": model,
                            "dataset": dataset,
                            "seed": int(seed),
                            "issue": "missing",
                            "count": 0,
                        }
                    )
                elif count > 1:
                    issues.append(
                        {
                            "model": model,
                            "dataset": dataset,
                            "seed": int(seed),
                            "issue": "duplicate",
                            "count": count,
                        }
                    )
    return issues

#!/usr/bin/env python3
"""Merge local/cloud selected epochs and summarize the five-seed experiment."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from statistics_utils import (
    METRIC_NAMES,
    format_mean_sd,
    load_statistics_config,
    sample_mean_sd,
    seed_issues,
)


DEFAULT_CONFIG = Path(__file__).parent / "configs" / "statistics.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-dir",
        action="append",
        nargs="+",
        required=True,
        help="One or more roots containing selected_result.json files",
    )
    parser.add_argument("--output-dir", default="statistics_summary")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--experiment-id",
        help="Override config experiment_id when summarizing a cloud/local variant",
    )
    return parser.parse_args(argv)


def flatten(values: Iterable[Iterable[str]]) -> list[str]:
    return [value for group in values for value in group]


def read_selected_result(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    metrics = payload.get("metrics", {})
    record = {
        "experiment_id": str(payload.get("experiment_id", "")),
        "model": str(payload["model"]),
        "dataset": str(payload["dataset"]),
        "seed": int(payload["seed"]),
        "selected_epoch": int(payload["selected_epoch"]),
        "source": str(path),
    }
    for metric in METRIC_NAMES:
        value = payload.get(metric, metrics.get(metric))
        if value is None:
            raise ValueError(f"{path} is missing selected metric {metric!r}")
        record[metric] = float(value)
    return record


def discover_records(
    run_dirs: Iterable[str], experiment_id: str | None = None
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()
    for run_dir in run_dirs:
        for path in sorted(Path(run_dir).rglob("selected_result.json")):
            resolved = path.resolve()
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            if experiment_id is not None:
                try:
                    with path.open(encoding="utf-8") as handle:
                        payload = json.load(handle)
                except (OSError, ValueError, TypeError):
                    continue
                if str(payload.get("experiment_id", "")) != str(experiment_id):
                    continue
            records.append(read_selected_result(path))
    return records


def discover_failures(run_dirs: Iterable[str]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()
    for run_dir in run_dirs:
        for path in sorted(Path(run_dir).rglob("status.json")):
            resolved = path.resolve()
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            try:
                with path.open(encoding="utf-8") as handle:
                    status = json.load(handle)
            except (OSError, ValueError):
                continue
            if status.get("status") == "failed":
                failures.append(
                    {
                        "experiment_id": status.get("experiment_id", ""),
                        "model": status.get("model", ""),
                        "dataset": status.get("dataset", ""),
                        "seed": status.get("seed", ""),
                        "exception_type": status.get("exception_type", ""),
                        "exception": status.get("exception", ""),
                        "failed_at": status.get("failed_at", ""),
                        "source": str(path),
                    }
                )
    return failures


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def filter_records_by_experiment(
    records: Iterable[dict[str, Any]], experiment_id: str
) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if str(record.get("experiment_id", "")) == str(experiment_id)
    ]


def filter_records_to_config_matrix(
    records: Iterable[dict[str, Any]], config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    models = {str(value) for value in config["models"]}
    datasets = {str(value) for value in config["datasets"]}
    seeds = {int(value) for value in config["seeds"]}
    return [
        record
        for record in records
        if str(record["model"]) in models
        and str(record["dataset"]) in datasets
        and int(record["seed"]) in seeds
    ]


def summarize_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[int, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        grouped[
            (record["experiment_id"], record["model"], record["dataset"])
        ][record["seed"]].append(record)

    summaries: list[dict[str, Any]] = []
    for (experiment_id, model, dataset), by_seed in sorted(grouped.items()):
        # Duplicate seeds are reported separately. Use one deterministic record
        # per seed so duplicates can never inflate n or bias the statistics.
        unique_records = [
            sorted(seed_records, key=lambda record: record["source"])[0]
            for _, seed_records in sorted(by_seed.items())
        ]
        summary: dict[str, Any] = {
            "experiment_id": experiment_id,
            "model": model,
            "dataset": dataset,
            "n_seeds": len(unique_records),
        }
        for metric in METRIC_NAMES:
            values = [record[metric] for record in unique_records]
            average, standard_deviation = sample_mean_sd(values)
            summary[f"{metric}_mean"] = average
            summary[f"{metric}_sd"] = standard_deviation
            summary[metric] = format_mean_sd(values)
        summaries.append(summary)
    return summaries


def markdown_table(summaries: list[dict[str, Any]]) -> str:
    headers = ["Experiment", "Model", "Dataset", "n", *METRIC_NAMES]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| "
        + " | ".join(["---", "---", "---", "---:"] + ["---:"] * len(METRIC_NAMES))
        + " |",
    ]
    for row in summaries:
        values = [
            row["experiment_id"], row["model"], row["dataset"], str(row["n_seeds"])
        ] + [
            row[metric] for metric in METRIC_NAMES
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def latex_escape(value: Any) -> str:
    return str(value).replace("\\", "\\textbackslash{}").replace("_", "\\_")


def latex_table(summaries: list[dict[str, Any]]) -> str:
    columns = "lllr" + "r" * len(METRIC_NAMES)
    lines = [
        f"\\begin{{tabular}}{{{columns}}}",
        "\\toprule",
        "Experiment & Model & Dataset & n & "
        + " & ".join(metric.upper() for metric in METRIC_NAMES)
        + " \\\\",
        "\\midrule",
    ]
    for row in summaries:
        values = [
            row["experiment_id"], row["model"], row["dataset"], row["n_seeds"]
        ] + [
            row[metric].replace(" ± ", " $\\pm$ ") for metric in METRIC_NAMES
        ]
        lines.append(" & ".join(latex_escape(value) for value in values) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dirs = flatten(args.runs_dir)
    config = load_statistics_config(args.config)
    experiment_id = str(args.experiment_id or config["experiment_id"])
    # Filter by experiment before validating metrics. Old smoke runs may not
    # even contain metrics introduced by the selected experiment.
    experiment_records = discover_records(run_dirs, experiment_id=experiment_id)
    records = filter_records_to_config_matrix(experiment_records, config)
    discovered_failures = discover_failures(run_dirs)
    failures = [
        failure
        for failure in discovered_failures
        if str(failure.get("experiment_id", "")) == experiment_id
    ]

    counts: dict[tuple[str, str, str, int], int] = defaultdict(int)
    for record in records:
        counts[
            (
                record["experiment_id"],
                record["model"],
                record["dataset"],
                record["seed"],
            )
        ] += 1
    for record in records:
        record["duplicate"] = counts[
            (
                record["experiment_id"],
                record["model"],
                record["dataset"],
                record["seed"],
            )
        ] > 1

    issues = seed_issues(
        records,
        models=config["models"],
        datasets=config["datasets"],
        expected_seeds=[int(seed) for seed in config["seeds"]],
    )
    summaries = summarize_records(records)
    output_dir = Path(args.output_dir)
    run_fields = [
        "experiment_id",
        "model",
        "dataset",
        "seed",
        "selected_epoch",
        *METRIC_NAMES,
        "duplicate",
        "source",
    ]
    summary_fields = ["experiment_id", "model", "dataset", "n_seeds"] + [
        field
        for metric in METRIC_NAMES
        for field in (metric, f"{metric}_mean", f"{metric}_sd")
    ]
    write_csv(output_dir / "runs.csv", records, run_fields)
    write_csv(output_dir / "summary.csv", summaries, summary_fields)
    write_csv(
        output_dir / "missing_runs.csv",
        issues,
        ["model", "dataset", "seed", "issue", "count"],
    )
    write_csv(
        output_dir / "failed_runs.csv",
        failures,
        [
            "experiment_id",
            "model",
            "dataset",
            "seed",
            "exception_type",
            "exception",
            "failed_at",
            "source",
        ],
    )
    (output_dir / "paper_table.md").write_text(
        markdown_table(summaries), encoding="utf-8"
    )
    (output_dir / "paper_table.tex").write_text(
        latex_table(summaries), encoding="utf-8"
    )

    missing_count = sum(issue["issue"] == "missing" for issue in issues)
    duplicate_count = sum(issue["issue"] == "duplicate" for issue in issues)
    print(
        f"Selected result files: {len(records)} for experiment_id={experiment_id!r}; "
        f"ignored outside configured model/dataset/seed matrix: "
        f"{len(experiment_records) - len(records)}; other experiment IDs were "
        "ignored before metric validation"
    )
    print(f"Missing seeds: {missing_count}; duplicate seeds: {duplicate_count}")
    print(f"Failed runs: {len(failures)}")
    print(f"Wrote summaries to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

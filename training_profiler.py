"""Small torch.profiler controller for named training/evaluation phases."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Callable

import torch
from torch.profiler import ProfilerActivity, record_function


TRAINING_PHASES = (
    "data_loading",
    "model_forward",
    "loss_forward",
    "backward",
    "optimizer_step",
)


def _device_time_us(event: Any) -> float:
    for attribute in (
        "device_time_total",
        "cuda_time_total",
        "self_device_time_total",
        "self_cuda_time_total",
    ):
        value = getattr(event, attribute, None)
        if value is not None:
            return float(value)
    return 0.0


def save_profile(
    profiler: torch.profiler.profile,
    output_dir: Path,
    prefix: str,
    phase_names: tuple[str, ...],
    cuda_event_times_ms: dict[str, float] | None = None,
) -> dict[str, dict[str, float]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    events = {event.key: event for event in profiler.key_averages()}
    summary: dict[str, dict[str, float]] = {}
    for phase in phase_names:
        event = events.get(phase)
        summary[phase] = {
            "calls": float(event.count) if event is not None else 0.0,
            "cpu_time_ms": float(event.cpu_time_total) / 1000 if event is not None else 0.0,
            "cuda_time_ms": _device_time_us(event) / 1000 if event is not None else 0.0,
        }
        if cuda_event_times_ms is not None:
            summary[phase]["cuda_time_ms"] = cuda_event_times_ms.get(phase, 0.0)
    (output_dir / f"{prefix}_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / f"{prefix}_table.txt").write_text(
        profiler.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=100
        ),
        encoding="utf-8",
    )
    return summary


class TrainingProfiler:
    def __init__(self, enabled: bool, steps: int, output_dir: Path):
        self.enabled = bool(enabled)
        self.max_steps = max(int(steps), 1)
        self.output_dir = output_dir
        self.completed_steps = 0
        self.active = False
        self.profiler: torch.profiler.profile | None = None
        self.summary: dict[str, dict[str, float]] = {}
        self.cuda_events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = defaultdict(list)

    def start(self) -> None:
        if not self.enabled or self.active or self.profiler is not None:
            return
        self.profiler = torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        )
        self.profiler.start()
        self.active = True

    def region(self, name: str):
        return self._timed_region(name) if self.active else nullcontext()

    @contextmanager
    def _timed_region(self, name: str):
        if name == "data_loading":
            with record_function(name):
                yield
            return
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        with record_function(name):
            yield
        end_event.record()
        self.cuda_events[name].append((start_event, end_event))

    def step(self) -> None:
        if not self.active or self.profiler is None:
            return
        self.completed_steps += 1
        self.profiler.step()
        if self.completed_steps >= self.max_steps:
            self.stop()

    def stop(self) -> None:
        if not self.active or self.profiler is None:
            return
        self.profiler.stop()
        self.active = False
        torch.cuda.synchronize()
        cuda_event_times_ms = {
            phase: sum(start.elapsed_time(end) for start, end in events)
            for phase, events in self.cuda_events.items()
        }
        self.summary = save_profile(
            self.profiler,
            self.output_dir,
            "training_profile",
            TRAINING_PHASES,
            cuda_event_times_ms,
        )
        print("Profiler training phase summary (CPU ms / CUDA ms):")
        for phase, values in self.summary.items():
            print(
                f"  {phase}: {values['cpu_time_ms']:.3f} / "
                f"{values['cuda_time_ms']:.3f}"
            )

    def finish(self) -> None:
        self.stop()


def profile_evaluation(
    enabled: bool,
    output_dir: Path,
    callback: Callable[[], Any],
) -> tuple[Any, dict[str, dict[str, float]]]:
    if not enabled:
        return callback(), {}
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    cpu_start = time.perf_counter()
    start_event.record()
    with record_function("evaluation"):
        result = callback()
    end_event.record()
    end_event.synchronize()
    summary = {
        "evaluation": {
            "calls": 1.0,
            "cpu_time_ms": (time.perf_counter() - cpu_start) * 1000,
            "cuda_time_ms": start_event.elapsed_time(end_event),
        }
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluation_profile_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("Profiler evaluation summary (CPU ms / CUDA ms):")
    for phase, values in summary.items():
        print(
            f"  {phase}: {values['cpu_time_ms']:.3f} / "
            f"{values['cuda_time_ms']:.3f}"
        )
    return result, summary

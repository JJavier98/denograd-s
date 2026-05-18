"""Low-overhead profiling helpers for timing, CUDA memory and model footprint."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, TypeVar

import torch


T = TypeVar("T")


@dataclass(frozen=True)
class MemorySnapshot:
    """Snapshot of current and peak CUDA allocator statistics in bytes."""

    allocated_bytes: int
    reserved_bytes: int
    max_allocated_bytes: int
    max_reserved_bytes: int


@contextmanager
def timed_block():
    """Measure elapsed wall-clock time for a block."""
    start = time.perf_counter()
    payload = {"elapsed_seconds": None}
    try:
        yield payload
    finally:
        payload["elapsed_seconds"] = time.perf_counter() - start


def time_callable(func: Callable[..., T], *args, **kwargs) -> tuple[T, float]:
    """Measure elapsed wall-clock time for a callable."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed


def synchronize(device=None):
    """Synchronize CUDA if available, otherwise no-op."""
    if device is not None:
        device = torch.device(device)
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)
            return
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def reset_cuda_peak_memory(device=None):
    """Reset CUDA peak memory stats if a CUDA device is active."""
    if torch.cuda.is_available():
        if device is None:
            torch.cuda.reset_peak_memory_stats()
        else:
            torch.cuda.reset_peak_memory_stats(device)


def snapshot_cuda_memory(device=None) -> MemorySnapshot | None:
    """Collect current CUDA memory stats if available."""
    if not torch.cuda.is_available():
        return None

    if device is None:
        device_index = torch.cuda.current_device()
    else:
        device_index = torch.device(device).index
        if device_index is None:
            device_index = torch.cuda.current_device()

    return MemorySnapshot(
        allocated_bytes=torch.cuda.memory_allocated(device_index),
        reserved_bytes=torch.cuda.memory_reserved(device_index),
        max_allocated_bytes=torch.cuda.max_memory_allocated(device_index),
        max_reserved_bytes=torch.cuda.max_memory_reserved(device_index),
    )


def model_size_bytes(model) -> dict[str, int | float]:
    """Return basic parameter-count and memory statistics for a torch model."""
    total_params = 0
    nonzero_params = 0
    param_bytes = 0

    for parameter in model.parameters():
        numel = parameter.numel()
        total_params += numel
        param_bytes += numel * parameter.element_size()
        if parameter.is_floating_point() or parameter.is_complex():
            nonzero_params += int(torch.count_nonzero(parameter).item())
        else:
            nonzero_params += numel

    density = (nonzero_params / total_params) if total_params else 0.0
    sparsity = 1.0 - density if total_params else 0.0

    return {
        "total_params": total_params,
        "nonzero_params": nonzero_params,
        "param_bytes": param_bytes,
        "density": density,
        "sparsity": sparsity,
    }


def benchmark_inference(model, example_input, device=None, warmup_runs: int = 3, timed_runs: int = 10):
    """Benchmark inference time and CUDA memory use for a single model/input pair."""
    model.eval()

    if device is None:
        device = example_input.device
    else:
        device = torch.device(device)

    example_input = example_input.to(device)
    model = model.to(device)

    if device.type == "cuda" and torch.cuda.is_available():
        reset_cuda_peak_memory(device)

    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(example_input)
        synchronize(device)

        timings = []
        for _ in range(timed_runs):
            start = time.perf_counter()
            _ = model(example_input)
            synchronize(device)
            timings.append(time.perf_counter() - start)

    memory = snapshot_cuda_memory(device)
    avg_time = sum(timings) / len(timings) if timings else 0.0

    return {
        "avg_inference_seconds": avg_time,
        "timings_seconds": timings,
        "cuda_memory": None if memory is None else memory.__dict__,
    }
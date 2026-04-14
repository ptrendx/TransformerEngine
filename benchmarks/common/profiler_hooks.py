# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Profiler integration for nsys (CUDA profiler API + NVTX) and perf.

nsys usage:
    nsys profile --capture-range=cudaProfilerApi --trace=cuda,nvtx,cudnn,cublas \
        python -m pytest benchmarks/pytorch/bench_linear.py --profile

perf usage (CPU-bound scenarios):
    perf stat python -m pytest benchmarks/pytorch/bench_linear.py -k cpu_bound
    perf record -g python -m pytest benchmarks/pytorch/bench_linear.py -k cpu_bound
"""

from contextlib import contextmanager
from typing import Optional


@contextmanager
def cuda_profiler_region():
    """Context manager that wraps a region with cudaProfilerStart/Stop.

    When used with `nsys profile --capture-range=cudaProfilerApi`, only the
    code inside this context manager will be captured in the profiler trace.
    """
    try:
        import torch
        torch.cuda.cudart().cudaProfilerStart()
    except (ImportError, AttributeError, RuntimeError):
        pass
    try:
        yield
    finally:
        try:
            import torch
            torch.cuda.cudart().cudaProfilerStop()
        except (ImportError, AttributeError, RuntimeError):
            pass


@contextmanager
def nvtx_range(name: str):
    """Context manager that wraps a region with NVTX range markers.

    Visible in nsys timeline as named ranges.
    """
    try:
        import torch
        torch.cuda.nvtx.range_push(name)
    except (ImportError, AttributeError, RuntimeError):
        pass
    try:
        yield
    finally:
        try:
            import torch
            torch.cuda.nvtx.range_pop()
        except (ImportError, AttributeError, RuntimeError):
            pass


@contextmanager
def nvtx_range_jax(name: str):
    """Context manager for NVTX range markers using nvtx module directly (for JAX)."""
    try:
        import nvtx
        rng = nvtx.start_range(message=name)
    except ImportError:
        rng = None
    try:
        yield
    finally:
        if rng is not None:
            import nvtx
            nvtx.end_range(rng)


@contextmanager
def profiled_benchmark(name: str, profile_mode: bool = False):
    """Combined profiler context: CUDA profiler region + NVTX range.

    Use this to wrap the timed portion of a benchmark when in profile mode.
    In non-profile mode, this is a no-op.
    """
    if not profile_mode:
        yield
        return

    with cuda_profiler_region():
        with nvtx_range(name):
            yield


@contextmanager
def emit_nvtx_context(profile_mode: bool = False):
    """Enable torch.autograd.profiler.emit_nvtx for backward pass visibility.

    Wraps the entire forward+backward with NVTX markers so nsys can see
    individual autograd operations.
    """
    if not profile_mode:
        yield
        return

    try:
        import torch
        ctx = torch.autograd.profiler.emit_nvtx(record_shapes=True)
        ctx.__enter__()
    except (ImportError, AttributeError, RuntimeError):
        ctx = None

    try:
        yield
    finally:
        if ctx is not None:
            ctx.__exit__(None, None, None)

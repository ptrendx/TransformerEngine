# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Multi-GPU benchmark utilities: barriers, warmup, and synchronized timing."""

import statistics
import time
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from benchmarks.common.result_types import BenchmarkResult
from benchmarks.config.benchmark_config import BenchmarkConfig


def init_nccl_warmup(group: Optional[dist.ProcessGroup] = None) -> None:
    """Run a trivial all-reduce to prime NCCL internal state and JIT kernels."""
    dummy = torch.zeros(1, device="cuda")
    for _ in range(3):
        dist.all_reduce(dummy, group=group)
    torch.cuda.synchronize()
    dist.barrier(group=group)


def distributed_warmup(
    fn: Callable,
    warmup_iters: int = 5,
    group: Optional[dist.ProcessGroup] = None,
) -> None:
    """Run warmup iterations with a barrier before and after."""
    dist.barrier(group=group)
    for _ in range(warmup_iters):
        fn()
        torch.cuda.synchronize()
    dist.barrier(group=group)


def distributed_measure(
    fn: Callable,
    *,
    config: BenchmarkConfig,
    name: str,
    category: str,
    shape: tuple,
    dtype: str = "bf16",
    recipe: str = "bf16",
    direction: str = "fwd_bwd",
    regime: str = "gpu_bound",
    total_bytes: Optional[int] = None,
    total_flops: Optional[int] = None,
    group: Optional[dist.ProcessGroup] = None,
    metadata: Optional[Dict[str, str]] = None,
) -> Optional[BenchmarkResult]:
    """Measure a distributed benchmark with proper barrier/warmup protocol.

    Returns BenchmarkResult on rank 0, None on other ranks.
    """
    world_size = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)

    # Warmup
    distributed_warmup(fn, warmup_iters=config.warmup_iterations, group=group)

    # Timed phase
    dist.barrier(group=group)
    torch.cuda.synchronize()

    times_us: List[float] = []
    for _ in range(config.num_iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times_us.append(elapsed * 1e6)

    dist.barrier(group=group)

    if rank != 0:
        return None

    median_us = statistics.median(times_us)
    mean_us = statistics.mean(times_us)

    throughput_gbps = None
    if total_bytes is not None and median_us > 0:
        throughput_gbps = total_bytes / (1024**3) / (median_us / 1e6)

    tflops_val = None
    if total_flops is not None and median_us > 0:
        tflops_val = total_flops / 1e12 / (median_us / 1e6)

    return BenchmarkResult(
        name=name,
        category=category,
        framework="pytorch",
        shape=shape,
        dtype=dtype,
        recipe=recipe,
        direction=direction,
        regime=regime,
        median_time_us=median_us,
        mean_time_us=mean_us,
        min_time_us=min(times_us),
        max_time_us=max(times_us),
        std_time_us=statistics.stdev(times_us) if len(times_us) > 1 else 0.0,
        num_iterations=config.num_iterations,
        throughput_gbps=throughput_gbps,
        tflops=tflops_val,
        num_gpus=world_size,
        metadata=metadata or {},
    )

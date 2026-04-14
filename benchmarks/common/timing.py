# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Benchmark timing utilities for PyTorch and JAX."""

import statistics
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from benchmarks.common.result_types import BenchmarkResult
from benchmarks.config.benchmark_config import BenchmarkConfig


class BenchmarkTimer:
    """Timing wrapper using torch.utils.benchmark.Timer for PyTorch benchmarks.

    Handles CUDA synchronization automatically via the torch benchmark infrastructure.
    """

    def __init__(self, config: BenchmarkConfig):
        self.config = config

    def measure(
        self,
        stmt: str,
        globals_dict: dict,
        *,
        label: str,
        sub_label: str = "",
        name: str = "",
        category: str = "",
        framework: str = "pytorch",
        shape: tuple = (),
        dtype: str = "bf16",
        recipe: str = "bf16",
        direction: str = "fwd_bwd",
        regime: str = "gpu_bound",
        total_bytes: Optional[int] = None,
        total_flops: Optional[int] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> BenchmarkResult:
        """Measure using torch.utils.benchmark.Timer.blocked_autorange."""
        import torch.utils.benchmark as benchmark

        timer = benchmark.Timer(
            stmt=stmt,
            globals=globals_dict,
            label=label,
            sub_label=sub_label,
            num_threads=1,
        )

        measurement = timer.blocked_autorange(min_run_time=self.config.min_run_time)

        median_us = measurement.median * 1e6
        mean_us = measurement.mean * 1e6
        times_us = [t * 1e6 for t in measurement.times]

        throughput_gbps = None
        if total_bytes is not None and median_us > 0:
            throughput_gbps = total_bytes / (1024**3) / (measurement.median)

        tflops_val = None
        if total_flops is not None and median_us > 0:
            tflops_val = total_flops / 1e12 / (measurement.median)

        return BenchmarkResult(
            name=name or label,
            category=category,
            framework=framework,
            shape=shape,
            dtype=dtype,
            recipe=recipe,
            direction=direction,
            regime=regime,
            median_time_us=median_us,
            mean_time_us=mean_us,
            min_time_us=min(times_us) if times_us else median_us,
            max_time_us=max(times_us) if times_us else median_us,
            std_time_us=statistics.stdev(times_us) if len(times_us) > 1 else 0.0,
            num_iterations=measurement.number_per_run * len(measurement.times),
            throughput_gbps=throughput_gbps,
            tflops=tflops_val,
            metadata=metadata or {},
        )


class JAXBenchmarkTimer:
    """Manual timing for JAX benchmarks using jax.block_until_ready."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config

    def measure(
        self,
        fn: Callable,
        args: Tuple = (),
        kwargs: Optional[Dict] = None,
        *,
        label: str,
        sub_label: str = "",
        name: str = "",
        category: str = "",
        shape: tuple = (),
        dtype: str = "bf16",
        recipe: str = "bf16",
        direction: str = "fwd_bwd",
        regime: str = "gpu_bound",
        total_bytes: Optional[int] = None,
        total_flops: Optional[int] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> BenchmarkResult:
        """Measure with manual timing and jax.block_until_ready."""
        import jax

        kwargs = kwargs or {}

        # Warmup (also triggers JIT compilation)
        for _ in range(self.config.warmup_iterations):
            result = fn(*args, **kwargs)
            jax.block_until_ready(result)

        # Timed iterations
        times_ns: List[int] = []
        for _ in range(self.config.num_iterations):
            start = time.perf_counter_ns()
            result = fn(*args, **kwargs)
            jax.block_until_ready(result)
            elapsed = time.perf_counter_ns() - start
            times_ns.append(elapsed)

        times_us = [t / 1000.0 for t in times_ns]
        median_us = statistics.median(times_us)
        mean_us = statistics.mean(times_us)

        throughput_gbps = None
        if total_bytes is not None and median_us > 0:
            throughput_gbps = total_bytes / (1024**3) / (median_us / 1e6)

        tflops_val = None
        if total_flops is not None and median_us > 0:
            tflops_val = total_flops / 1e12 / (median_us / 1e6)

        return BenchmarkResult(
            name=name or label,
            category=category,
            framework="jax",
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
            num_iterations=self.config.num_iterations,
            throughput_gbps=throughput_gbps,
            tflops=tflops_val,
            metadata=metadata or {},
        )

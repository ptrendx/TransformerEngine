# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Benchmark configuration system."""

import argparse
import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution."""

    warmup_iterations: int = 5
    min_run_time: float = 5.0  # seconds, for torch.utils.benchmark.Timer.blocked_autorange
    num_iterations: int = 20  # for manual timing loops (JAX, distributed)
    profile_mode: bool = False  # reduces iterations, enables NVTX/profiler markers
    size_filter: str = "all"  # "cpu_bound", "gpu_bound", "all"
    recipes: List[str] = field(default_factory=lambda: ["bf16"])
    database_url: Optional[str] = None
    output_format: str = "stdout"  # "stdout", "csv", "json"
    output_file: Optional[str] = None

    def __post_init__(self):
        # Allow database URL from environment variable
        if self.database_url is None:
            self.database_url = os.environ.get("TE_BENCH_DB_URL")

        # In profile mode, reduce iterations for faster profiler captures
        if self.profile_mode:
            self.warmup_iterations = min(self.warmup_iterations, 3)
            self.min_run_time = min(self.min_run_time, 1.0)
            self.num_iterations = min(self.num_iterations, 5)

    @classmethod
    def from_pytest(cls, pytestconfig) -> "BenchmarkConfig":
        """Construct from pytest config object."""
        return cls(
            warmup_iterations=pytestconfig.getoption("--warmup-iterations", default=5),
            min_run_time=pytestconfig.getoption("--min-run-time", default=5.0),
            num_iterations=pytestconfig.getoption("--num-iterations", default=20),
            profile_mode=pytestconfig.getoption("--profile", default=False),
            size_filter=pytestconfig.getoption("--size-filter", default="all"),
            recipes=pytestconfig.getoption("--recipe", default=["bf16"]),
            database_url=pytestconfig.getoption("--database-url", default=None),
            output_format=pytestconfig.getoption("--output-format", default="stdout"),
            output_file=pytestconfig.getoption("--output-file", default=None),
        )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "BenchmarkConfig":
        """Construct from argparse namespace."""
        return cls(
            warmup_iterations=getattr(args, "warmup_iterations", 5),
            min_run_time=getattr(args, "min_run_time", 5.0),
            num_iterations=getattr(args, "num_iterations", 20),
            profile_mode=getattr(args, "profile", False),
            size_filter=getattr(args, "size_filter", "all"),
            recipes=getattr(args, "recipe", ["bf16"]),
            database_url=getattr(args, "database_url", None),
            output_format=getattr(args, "output_format", "stdout"),
            output_file=getattr(args, "output_file", None),
        )


def add_benchmark_args(parser: argparse.ArgumentParser) -> None:
    """Add standard benchmark arguments to an argparse parser."""
    parser.add_argument(
        "--warmup-iterations", type=int, default=5,
        help="Number of warmup iterations before timing (default: 5)",
    )
    parser.add_argument(
        "--min-run-time", type=float, default=5.0,
        help="Minimum run time in seconds for blocked_autorange (default: 5.0)",
    )
    parser.add_argument(
        "--num-iterations", type=int, default=20,
        help="Number of timed iterations for manual timing loops (default: 20)",
    )
    parser.add_argument(
        "--profile", action="store_true", default=False,
        help="Enable profiling mode (fewer iterations, NVTX markers)",
    )
    parser.add_argument(
        "--size-filter", type=str, default="all",
        choices=["cpu_bound", "gpu_bound", "all"],
        help="Filter benchmark sizes by regime (default: all)",
    )
    parser.add_argument(
        "--recipe", type=str, nargs="+", default=["bf16"],
        help="Quantization recipes to benchmark (default: bf16)",
    )
    parser.add_argument(
        "--database-url", type=str, default=None,
        help="Database URL for result reporting (or set TE_BENCH_DB_URL env var)",
    )
    parser.add_argument(
        "--output-format", type=str, default="stdout",
        choices=["stdout", "csv", "json"],
        help="Output format (default: stdout)",
    )
    parser.add_argument(
        "--output-file", type=str, default=None,
        help="Output file path for csv/json output",
    )

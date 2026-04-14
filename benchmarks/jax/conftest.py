# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Pytest configuration and fixtures for JAX benchmarks."""

import pytest

from benchmarks.config.benchmark_config import BenchmarkConfig
from benchmarks.common.result_types import collect_metadata_jax
from benchmarks.common.timing import JAXBenchmarkTimer
from benchmarks.common.reporter import BenchmarkReporter


def pytest_addoption(parser):
    """Add benchmark-specific CLI options."""
    parser.addoption("--profile", action="store_true", default=False,
                     help="Enable profiling mode")
    parser.addoption("--size-filter", type=str, default="all",
                     choices=["cpu_bound", "gpu_bound", "all"],
                     help="Filter sizes by regime")
    parser.addoption("--database-url", type=str, default=None,
                     help="Database URL for result reporting")
    parser.addoption("--output-format", type=str, default="stdout",
                     choices=["stdout", "csv", "json"],
                     help="Output format")
    parser.addoption("--output-file", type=str, default=None,
                     help="Output file path")
    parser.addoption("--min-run-time", type=float, default=5.0,
                     help="Minimum run time in seconds")
    parser.addoption("--warmup-iterations", type=int, default=5,
                     help="Warmup iterations")
    parser.addoption("--num-iterations", type=int, default=20,
                     help="Number of timed iterations")


@pytest.fixture(scope="session")
def benchmark_config(pytestconfig):
    """Build BenchmarkConfig from pytest options."""
    return BenchmarkConfig(
        warmup_iterations=pytestconfig.getoption("--warmup-iterations"),
        min_run_time=pytestconfig.getoption("--min-run-time"),
        num_iterations=pytestconfig.getoption("--num-iterations"),
        profile_mode=pytestconfig.getoption("--profile"),
        size_filter=pytestconfig.getoption("--size-filter"),
        database_url=pytestconfig.getoption("--database-url"),
        output_format=pytestconfig.getoption("--output-format"),
        output_file=pytestconfig.getoption("--output-file"),
    )


@pytest.fixture(scope="session")
def benchmark_timer(benchmark_config):
    """Provides a JAXBenchmarkTimer instance."""
    return JAXBenchmarkTimer(benchmark_config)


@pytest.fixture(scope="session")
def benchmark_reporter(benchmark_config):
    """Provides a BenchmarkReporter instance with system metadata."""
    metadata = collect_metadata_jax()
    return BenchmarkReporter(benchmark_config, metadata=metadata)

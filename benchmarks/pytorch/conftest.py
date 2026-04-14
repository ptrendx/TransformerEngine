# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Pytest configuration and fixtures for PyTorch benchmarks."""

import pytest
import torch

from benchmarks.config.benchmark_config import BenchmarkConfig
from benchmarks.common.result_types import collect_metadata
from benchmarks.common.timing import BenchmarkTimer
from benchmarks.common.reporter import BenchmarkReporter


def pytest_addoption(parser):
    """Add benchmark-specific CLI options."""
    parser.addoption("--profile", action="store_true", default=False,
                     help="Enable profiling mode (fewer iterations, NVTX markers)")
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
                     help="Minimum run time in seconds for timing")
    parser.addoption("--warmup-iterations", type=int, default=5,
                     help="Warmup iterations before timing")
    parser.addoption("--num-iterations", type=int, default=20,
                     help="Number of timed iterations")
    parser.addoption("--recipe", type=str, nargs="+", default=["bf16"],
                     help="Quantization recipes to benchmark")


@pytest.fixture(scope="session")
def benchmark_config(pytestconfig):
    """Build BenchmarkConfig from pytest options."""
    return BenchmarkConfig(
        warmup_iterations=pytestconfig.getoption("--warmup-iterations"),
        min_run_time=pytestconfig.getoption("--min-run-time"),
        num_iterations=pytestconfig.getoption("--num-iterations"),
        profile_mode=pytestconfig.getoption("--profile"),
        size_filter=pytestconfig.getoption("--size-filter"),
        recipes=pytestconfig.getoption("--recipe"),
        database_url=pytestconfig.getoption("--database-url"),
        output_format=pytestconfig.getoption("--output-format"),
        output_file=pytestconfig.getoption("--output-file"),
    )


@pytest.fixture(scope="session")
def benchmark_timer(benchmark_config):
    """Provides a BenchmarkTimer instance."""
    return BenchmarkTimer(benchmark_config)


@pytest.fixture(scope="session")
def benchmark_reporter(benchmark_config):
    """Provides a BenchmarkReporter instance with system metadata."""
    metadata = collect_metadata()
    return BenchmarkReporter(benchmark_config, metadata=metadata)


def pytest_sessionfinish(session, exitstatus):
    """Finalize reporting at session end."""
    # The reporter finalizes via the fixture's session scope cleanup
    pass


# --- Hardware availability helpers ---

def _check_fp8_available():
    """Check if FP8 is available on current hardware."""
    try:
        from transformer_engine.pytorch.quantization import FP8GlobalStateManager
        available, reason = FP8GlobalStateManager.is_fp8_block_scaling_available()
        return available, reason
    except (ImportError, AttributeError):
        return False, "FP8GlobalStateManager not available"


def _check_mxfp8_available():
    """Check if MXFP8 is available on current hardware."""
    try:
        from transformer_engine.pytorch.quantization import FP8GlobalStateManager
        available, reason = FP8GlobalStateManager.is_mxfp8_available()
        return available, reason
    except (ImportError, AttributeError):
        return False, "FP8GlobalStateManager not available"


def _check_nvfp4_available():
    """Check if NVFP4 is available on current hardware."""
    try:
        from transformer_engine.pytorch.quantization import FP8GlobalStateManager
        available, reason = FP8GlobalStateManager.is_nvfp4_available()
        return available, reason
    except (ImportError, AttributeError):
        return False, "FP8GlobalStateManager not available"


def skip_if_recipe_unavailable(recipe_name: str):
    """Skip test if the requested recipe is not available on current hardware."""
    if recipe_name in ("bf16",):
        return  # Always available
    if recipe_name == "fp8_block":
        available, reason = _check_fp8_available()
        if not available:
            pytest.skip(f"FP8 block scaling not available: {reason}")
    elif recipe_name == "mxfp8":
        available, reason = _check_mxfp8_available()
        if not available:
            pytest.skip(f"MXFP8 not available: {reason}")
    elif recipe_name == "nvfp4":
        available, reason = _check_nvfp4_available()
        if not available:
            pytest.skip(f"NVFP4 not available: {reason}")


def get_recipe(recipe_name: str):
    """Get a recipe object by name."""
    if recipe_name == "bf16":
        return None
    elif recipe_name == "fp8_block":
        from transformer_engine.common.recipe import Float8BlockScaling
        return Float8BlockScaling()
    elif recipe_name == "mxfp8":
        from transformer_engine.common.recipe import MXFP8BlockScaling
        return MXFP8BlockScaling()
    elif recipe_name == "nvfp4":
        from transformer_engine.common.recipe import NVFP4BlockScaling
        return NVFP4BlockScaling()
    else:
        raise ValueError(f"Unknown recipe: {recipe_name}")

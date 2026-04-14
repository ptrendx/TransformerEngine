# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Benchmark result types and metadata collection."""

import os
import socket
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class BenchmarkResult:
    """A single benchmark measurement result."""

    name: str  # e.g., "nvte_quantize" or "te.Linear"
    category: str  # "activation", "normalization", "cast", "gemm", "attention", etc.
    framework: str  # "cpp", "pytorch", "jax"
    shape: tuple
    dtype: str  # "bf16", "fp16", "fp32"
    recipe: str  # "bf16", "fp8_block", "mxfp8", "nvfp4"
    direction: str  # "fwd", "bwd", "fwd_bwd"
    regime: str  # "cpu_bound", "gpu_bound"
    median_time_us: float
    mean_time_us: float
    min_time_us: float
    max_time_us: float
    std_time_us: float
    num_iterations: int
    throughput_gbps: Optional[float] = None
    tflops: Optional[float] = None
    num_gpus: int = 1
    metadata: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to a flat dictionary for serialization."""
        d = {
            "name": self.name,
            "category": self.category,
            "framework": self.framework,
            "shape": str(self.shape),
            "dtype": self.dtype,
            "recipe": self.recipe,
            "direction": self.direction,
            "regime": self.regime,
            "median_time_us": self.median_time_us,
            "mean_time_us": self.mean_time_us,
            "min_time_us": self.min_time_us,
            "max_time_us": self.max_time_us,
            "std_time_us": self.std_time_us,
            "num_iterations": self.num_iterations,
            "num_gpus": self.num_gpus,
        }
        if self.throughput_gbps is not None:
            d["throughput_gbps"] = self.throughput_gbps
        if self.tflops is not None:
            d["tflops"] = self.tflops
        d.update(self.metadata)
        return d


def collect_metadata() -> Dict[str, str]:
    """Collect system metadata for benchmark results."""
    meta = {
        "hostname": socket.gethostname(),
    }

    # Git commit
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode().strip()
        meta["git_commit"] = commit
    except (subprocess.SubprocessError, FileNotFoundError):
        meta["git_commit"] = "unknown"

    # TE version
    try:
        import transformer_engine
        meta["te_version"] = transformer_engine.__version__
    except (ImportError, AttributeError):
        meta["te_version"] = "unknown"

    # CUDA info via torch (if available)
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            meta["gpu_model"] = props.name
            meta["gpu_memory_gb"] = f"{props.total_mem / 1024**3:.1f}"
            meta["cuda_version"] = torch.version.cuda or "unknown"
            meta["gpu_count"] = str(torch.cuda.device_count())
            meta["compute_capability"] = f"{props.major}.{props.minor}"
    except (ImportError, RuntimeError):
        pass

    # CUDA driver version from nvidia-smi
    try:
        driver = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader", "-i", "0"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode().strip()
        meta["driver_version"] = driver
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    return meta


def collect_metadata_jax() -> Dict[str, str]:
    """Collect system metadata using JAX instead of PyTorch."""
    meta = {
        "hostname": socket.gethostname(),
    }

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode().strip()
        meta["git_commit"] = commit
    except (subprocess.SubprocessError, FileNotFoundError):
        meta["git_commit"] = "unknown"

    try:
        import transformer_engine
        meta["te_version"] = transformer_engine.__version__
    except (ImportError, AttributeError):
        meta["te_version"] = "unknown"

    try:
        import jax
        devices = jax.devices("gpu")
        if devices:
            meta["gpu_count"] = str(len(devices))
            meta["gpu_model"] = str(devices[0].device_kind)
    except (ImportError, RuntimeError):
        pass

    try:
        driver = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader", "-i", "0"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode().strip()
        meta["driver_version"] = driver
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    return meta

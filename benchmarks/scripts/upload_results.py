#!/usr/bin/env python3
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Upload benchmark results from JSON files to a database.

Parses Google Benchmark JSON output (C++ benchmarks) and custom JSON output
(Python benchmarks) and uploads them using the database backend.

Usage:
    python benchmarks/scripts/upload_results.py \
        --cpp results_cpp.json \
        --python results_python.json \
        --database-url "http://db:8080/results"
"""

import argparse
import json
import sys

from benchmarks.common.result_types import BenchmarkResult, collect_metadata
from benchmarks.common.database_backend import create_backend


def parse_google_benchmark_json(filepath: str) -> list:
    """Parse Google Benchmark JSON output into BenchmarkResult objects."""
    with open(filepath) as f:
        data = json.load(f)

    results = []
    context = data.get("context", {})
    metadata = collect_metadata()

    for bench in data.get("benchmarks", []):
        name = bench.get("name", "unknown")
        # Parse shape from name (e.g., "BM_nvte_quantize/8192/4096")
        parts = name.split("/")
        bench_name = parts[0]
        shape = tuple(int(p) for p in parts[1:] if p.isdigit()) if len(parts) > 1 else ()

        # Determine category from name
        category = "unknown"
        for cat in ["cast", "activation", "normalization", "gemm", "transpose",
                     "softmax", "attn", "rope", "hadamard", "swizzle",
                     "router", "multi_tensor"]:
            if cat in bench_name.lower():
                category = cat
                break

        # Determine regime from size
        total_elements = 1
        for s in shape:
            total_elements *= s
        regime = "cpu_bound" if total_elements < 10000 else "gpu_bound"

        time_us = bench.get("real_time", bench.get("cpu_time", 0))
        if bench.get("time_unit") == "ns":
            time_us /= 1000.0

        throughput_gbps = None
        bytes_per_sec = bench.get("bytes_per_second")
        if bytes_per_sec:
            throughput_gbps = bytes_per_sec / (1024**3)

        results.append(BenchmarkResult(
            name=bench_name,
            category=category,
            framework="cpp",
            shape=shape,
            dtype="bf16",
            recipe="native",
            direction="fwd",
            regime=regime,
            median_time_us=time_us,
            mean_time_us=time_us,
            min_time_us=time_us,
            max_time_us=time_us,
            std_time_us=0.0,
            num_iterations=bench.get("iterations", 1),
            throughput_gbps=throughput_gbps,
            metadata=metadata,
        ))

    return results


def parse_python_results_json(filepath: str) -> list:
    """Parse Python benchmark JSON output into BenchmarkResult objects."""
    with open(filepath) as f:
        data = json.load(f)

    results = []
    for entry in data:
        shape = entry.get("shape", "()")
        if isinstance(shape, str):
            # Parse string like "(8192, 4096)"
            shape = tuple(int(x.strip()) for x in shape.strip("()").split(",") if x.strip())

        results.append(BenchmarkResult(
            name=entry.get("name", "unknown"),
            category=entry.get("category", "unknown"),
            framework=entry.get("framework", "unknown"),
            shape=shape,
            dtype=entry.get("dtype", "bf16"),
            recipe=entry.get("recipe", "bf16"),
            direction=entry.get("direction", "fwd"),
            regime=entry.get("regime", "unknown"),
            median_time_us=entry.get("median_time_us", 0),
            mean_time_us=entry.get("mean_time_us", 0),
            min_time_us=entry.get("min_time_us", 0),
            max_time_us=entry.get("max_time_us", 0),
            std_time_us=entry.get("std_time_us", 0),
            num_iterations=entry.get("num_iterations", 0),
            throughput_gbps=entry.get("throughput_gbps"),
            tflops=entry.get("tflops"),
            num_gpus=entry.get("num_gpus", 1),
            metadata=entry.get("metadata", {}),
        ))

    return results


def main():
    parser = argparse.ArgumentParser(description="Upload benchmark results to database")
    parser.add_argument("--cpp", type=str, default=None,
                        help="Path to Google Benchmark JSON output")
    parser.add_argument("--python", type=str, default=None,
                        help="Path to Python benchmark JSON output")
    parser.add_argument("--database-url", type=str, required=True,
                        help="Database URL")
    args = parser.parse_args()

    if not args.cpp and not args.python:
        print("No result files specified. Use --cpp and/or --python.")
        sys.exit(1)

    backend = create_backend(args.database_url)
    all_results = []

    if args.cpp:
        print(f"Parsing C++ results from {args.cpp}")
        all_results.extend(parse_google_benchmark_json(args.cpp))

    if args.python:
        print(f"Parsing Python results from {args.python}")
        all_results.extend(parse_python_results_json(args.python))

    if all_results:
        print(f"Uploading {len(all_results)} results to {args.database_url}")
        backend.report(all_results)
    else:
        print("No results to upload.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Worker script for distributed benchmarks. Launched via torchrun.

Usage:
    # All distributed benchmarks, 4 GPUs
    torchrun --nproc_per_node=4 benchmarks/pytorch/distributed/run_distributed.py --all

    # Specific benchmark
    torchrun --nproc_per_node=8 benchmarks/pytorch/distributed/run_distributed.py \
        --benchmark tensor_parallel

    # With profiling
    nsys profile --capture-range=cudaProfilerApi \
        torchrun --nproc_per_node=4 benchmarks/pytorch/distributed/run_distributed.py \
        --benchmark tensor_parallel --profile

    # Specific recipes
    torchrun --nproc_per_node=4 benchmarks/pytorch/distributed/run_distributed.py \
        --benchmark tensor_parallel --recipe bf16 mxfp8
"""

import argparse
import sys

import torch
import torch.distributed as dist

from benchmarks.config.benchmark_config import BenchmarkConfig, add_benchmark_args
from benchmarks.common.result_types import collect_metadata
from benchmarks.common.reporter import BenchmarkReporter
from benchmarks.common.profiler_hooks import profiled_benchmark


BENCHMARKS = {
    "tensor_parallel": "benchmarks.pytorch.distributed.bench_tensor_parallel",
    "sequence_parallel": "benchmarks.pytorch.distributed.bench_sequence_parallel",
    "context_parallel": "benchmarks.pytorch.distributed.bench_context_parallel",
    "comm_gemm_overlap": "benchmarks.pytorch.distributed.bench_comm_gemm_overlap",
}


def main():
    parser = argparse.ArgumentParser(description="Distributed TE benchmarks")
    add_benchmark_args(parser)
    parser.add_argument(
        "--benchmark", type=str, default=None,
        choices=list(BENCHMARKS.keys()),
        help="Which benchmark to run (default: all)",
    )
    parser.add_argument(
        "--all", action="store_true", default=False,
        help="Run all distributed benchmarks",
    )
    args = parser.parse_args()

    # Initialize process group
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    config = BenchmarkConfig.from_args(args)
    metadata = collect_metadata()
    metadata["world_size"] = str(world_size)
    reporter = BenchmarkReporter(config, metadata=metadata)

    if rank == 0:
        print(f"Running distributed benchmarks with {world_size} GPUs")
        print(f"Config: size_filter={config.size_filter}, "
              f"recipes={config.recipes}, profile={config.profile_mode}")
        print()

    # Determine which benchmarks to run
    if args.all or args.benchmark is None:
        bench_names = list(BENCHMARKS.keys())
    else:
        bench_names = [args.benchmark]

    with profiled_benchmark("distributed_benchmarks", config.profile_mode):
        for bench_name in bench_names:
            if rank == 0:
                print(f"\n{'='*60}")
                print(f"Running: {bench_name}")
                print(f"{'='*60}")

            module_path = BENCHMARKS[bench_name]
            # Import and run the benchmark module
            import importlib
            bench_module = importlib.import_module(module_path)
            bench_module.run(config, reporter)

            dist.barrier()

    if rank == 0:
        reporter.finalize()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

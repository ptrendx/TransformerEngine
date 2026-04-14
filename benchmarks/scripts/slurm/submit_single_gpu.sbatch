#!/bin/bash
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# Slurm batch script for single-GPU benchmarks inside enroot container.
#
# Usage:
#   sbatch benchmarks/scripts/slurm/submit_single_gpu.sbatch
#
# Environment variables:
#   TE_BENCH_CONTAINER_IMAGE  - Container image (default: nvcr.io/nvidia/pytorch:26.03-py3)
#   TE_BENCH_TE_PATH          - TE install path inside container (default: /opt/transformerengine)
#   TE_BENCH_DB_URL            - Database URL for result reporting (optional)
#   TE_BENCH_SIZE_FILTER       - Size filter: cpu_bound, gpu_bound, all (default: all)
#   TE_BENCH_RECIPES           - Space-separated recipes (default: bf16)

#SBATCH --job-name=te-bench-1gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=te_bench_1gpu_%j.out
#SBATCH --error=te_bench_1gpu_%j.err

# Container configuration via pyxis/enroot
#SBATCH --container-image=${TE_BENCH_CONTAINER_IMAGE:-nvcr.io/nvidia/pytorch:26.03-py3}
#SBATCH --container-mounts=${TE_BENCH_TE_PATH:-/opt/transformerengine}:/workspace/te

set -e

: ${TE_BENCH_SIZE_FILTER:=all}
: ${TE_BENCH_RECIPES:=bf16}

cd /workspace/te

echo "=== System Info ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python3 -c "import transformer_engine; print(f'TE version: {transformer_engine.__version__}')"
echo ""

EXTRA_ARGS=()
if [ -n "$TE_BENCH_DB_URL" ]; then
    EXTRA_ARGS+=(--database-url "$TE_BENCH_DB_URL")
fi

# C++ benchmarks
echo "=== C++ Benchmarks ==="
bash benchmarks/scripts/run_cpp.sh "" \
    --benchmark_format=json \
    --benchmark_out=/workspace/te/benchmark_results_cpp.json \
    || echo "Warning: C++ benchmarks failed"

# PyTorch benchmarks
echo "=== PyTorch Benchmarks ==="
python3 -m pytest benchmarks/pytorch/ -v \
    --size-filter "$TE_BENCH_SIZE_FILTER" \
    --recipe $TE_BENCH_RECIPES \
    "${EXTRA_ARGS[@]}" \
    || echo "Warning: PyTorch benchmarks failed"

# JAX benchmarks (if JAX is installed)
if python3 -c "import jax" 2>/dev/null; then
    echo "=== JAX Benchmarks ==="
    python3 -m pytest benchmarks/jax/ -v \
        --size-filter "$TE_BENCH_SIZE_FILTER" \
        "${EXTRA_ARGS[@]}" \
        || echo "Warning: JAX benchmarks failed"
else
    echo "=== JAX not installed, skipping JAX benchmarks ==="
fi

# Upload C++ results if DB URL set
if [ -n "$TE_BENCH_DB_URL" ] && [ -f benchmark_results_cpp.json ]; then
    python3 benchmarks/scripts/upload_results.py \
        --cpp benchmark_results_cpp.json \
        --database-url "$TE_BENCH_DB_URL"
fi

echo ""
echo "=== Benchmark complete ==="

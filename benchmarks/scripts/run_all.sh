#!/bin/bash
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# Run the complete benchmark suite: C++, PyTorch, and JAX.
#
# Usage:
#   bash benchmarks/scripts/run_all.sh [--skip-cpp] [--skip-pytorch] [--skip-jax]
#                                      [--skip-distributed] [--database-url URL]
#                                      [extra_args...]
#
# Examples:
#   bash benchmarks/scripts/run_all.sh
#   bash benchmarks/scripts/run_all.sh --skip-jax --size-filter gpu_bound
#   bash benchmarks/scripts/run_all.sh --database-url "http://db:8080/results"

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

SKIP_CPP=0
SKIP_PYTORCH=0
SKIP_JAX=0
SKIP_DISTRIBUTED=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-cpp) SKIP_CPP=1; shift ;;
        --skip-pytorch) SKIP_PYTORCH=1; shift ;;
        --skip-jax) SKIP_JAX=1; shift ;;
        --skip-distributed) SKIP_DISTRIBUTED=1; shift ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

cd "$REPO_DIR"

echo "================================================"
echo "  Transformer Engine Benchmark Suite"
echo "================================================"
echo ""

# C++ Benchmarks
if [ "$SKIP_CPP" -eq 0 ]; then
    echo ">>> C++ Benchmarks"
    bash benchmarks/scripts/run_cpp.sh "" \
        --benchmark_format=json \
        --benchmark_out=benchmark_results_cpp.json \
        || echo "Warning: C++ benchmarks failed"
    echo ""
fi

# PyTorch Benchmarks
if [ "$SKIP_PYTORCH" -eq 0 ]; then
    echo ">>> PyTorch Benchmarks"
    bash benchmarks/scripts/run_pytorch.sh "${EXTRA_ARGS[@]}" \
        || echo "Warning: PyTorch benchmarks failed"
    echo ""
fi

# JAX Benchmarks
if [ "$SKIP_JAX" -eq 0 ]; then
    echo ">>> JAX Benchmarks"
    bash benchmarks/scripts/run_jax.sh "${EXTRA_ARGS[@]}" \
        || echo "Warning: JAX benchmarks failed"
    echo ""
fi

# Distributed Benchmarks (requires multiple GPUs)
if [ "$SKIP_DISTRIBUTED" -eq 0 ]; then
    NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
    if [ "$NUM_GPUS" -gt 1 ]; then
        echo ">>> Distributed Benchmarks ($NUM_GPUS GPUs)"
        torchrun --nproc_per_node="$NUM_GPUS" \
            benchmarks/pytorch/distributed/run_distributed.py --all \
            "${EXTRA_ARGS[@]}" \
            || echo "Warning: Distributed benchmarks failed"
    else
        echo ">>> Skipping distributed benchmarks (only $NUM_GPUS GPU available)"
    fi
    echo ""
fi

# Upload results to database (if URL provided)
DB_URL=""
for arg in "${EXTRA_ARGS[@]}"; do
    if [[ "$arg" == --database-url=* ]]; then
        DB_URL="${arg#--database-url=}"
    fi
done
if [ -n "$DB_URL" ] && [ -f benchmark_results_cpp.json ]; then
    echo ">>> Uploading C++ results to database"
    python3 benchmarks/scripts/upload_results.py \
        --cpp benchmark_results_cpp.json \
        --database-url "$DB_URL" \
        || echo "Warning: Result upload failed"
fi

echo ""
echo "================================================"
echo "  Benchmark suite complete"
echo "================================================"

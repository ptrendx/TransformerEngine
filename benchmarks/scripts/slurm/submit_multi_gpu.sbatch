#!/bin/bash
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# Slurm batch script for multi-GPU distributed benchmarks inside enroot container.
#
# Usage:
#   sbatch benchmarks/scripts/slurm/submit_multi_gpu.sbatch
#
# Environment variables:
#   TE_BENCH_CONTAINER_IMAGE  - Container image (default: nvcr.io/nvidia/pytorch:26.03-py3)
#   TE_BENCH_TE_PATH          - TE install path inside container (default: /opt/transformerengine)
#   TE_BENCH_DB_URL            - Database URL for result reporting (optional)
#   TE_BENCH_SIZE_FILTER       - Size filter (default: gpu_bound)
#   TE_BENCH_RECIPES           - Space-separated recipes (default: bf16)
#   TE_BENCH_DISTRIBUTED_BENCH - Specific benchmark name or "all" (default: all)

#SBATCH --job-name=te-bench-multi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=04:00:00
#SBATCH --output=te_bench_multi_%j.out
#SBATCH --error=te_bench_multi_%j.err

# Container configuration via pyxis/enroot
#SBATCH --container-image=${TE_BENCH_CONTAINER_IMAGE:-nvcr.io/nvidia/pytorch:26.03-py3}
#SBATCH --container-mounts=${TE_BENCH_TE_PATH:-/opt/transformerengine}:/workspace/te

set -e

: ${TE_BENCH_SIZE_FILTER:=gpu_bound}
: ${TE_BENCH_RECIPES:=bf16}
: ${TE_BENCH_DISTRIBUTED_BENCH:=all}

cd /workspace/te

echo "=== System Info ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python3 -c "import transformer_engine; print(f'TE version: {transformer_engine.__version__}')"
echo "GPUs per node: ${SLURM_GPUS_PER_NODE}"
echo ""

DIST_ARGS=(--size-filter "$TE_BENCH_SIZE_FILTER" --recipe $TE_BENCH_RECIPES)

if [ "$TE_BENCH_DISTRIBUTED_BENCH" = "all" ]; then
    DIST_ARGS+=(--all)
else
    DIST_ARGS+=(--benchmark "$TE_BENCH_DISTRIBUTED_BENCH")
fi

if [ -n "$TE_BENCH_DB_URL" ]; then
    DIST_ARGS+=(--database-url "$TE_BENCH_DB_URL")
fi

echo "=== Distributed Benchmarks ==="
srun --ntasks=1 torchrun --nproc_per_node=${SLURM_GPUS_PER_NODE} \
    benchmarks/pytorch/distributed/run_distributed.py \
    "${DIST_ARGS[@]}"

echo ""
echo "=== Multi-GPU benchmark complete ==="

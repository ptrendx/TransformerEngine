#!/bin/bash
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# Launch an interactive Slurm session inside an enroot container for benchmarking.
#
# Usage:
#   bash benchmarks/scripts/slurm/interactive.sh [num_gpus] [time_limit]
#
# Examples:
#   bash benchmarks/scripts/slurm/interactive.sh           # 1 GPU, 1 hour
#   bash benchmarks/scripts/slurm/interactive.sh 4         # 4 GPUs, 1 hour
#   bash benchmarks/scripts/slurm/interactive.sh 8 02:00:00  # 8 GPUs, 2 hours

NUM_GPUS=${1:-1}
TIME_LIMIT=${2:-01:00:00}

: ${TE_BENCH_CONTAINER_IMAGE:=nvcr.io/nvidia/pytorch:26.03-py3}
TE_MOUNT="${TE_BENCH_TE_PATH:-$(pwd)}:/workspace/te"

echo "Requesting interactive session: ${NUM_GPUS} GPU(s), ${TIME_LIMIT} time limit"
echo "Container: ${TE_BENCH_CONTAINER_IMAGE}"
echo "Mount: ${TE_MOUNT}"

srun --nodes=1 \
     --ntasks=1 \
     --gpus-per-node="$NUM_GPUS" \
     --time="$TIME_LIMIT" \
     --pty \
     --container-image="$TE_BENCH_CONTAINER_IMAGE" \
     --container-mounts="$TE_MOUNT" \
     bash -c "cd /workspace/te && exec bash"

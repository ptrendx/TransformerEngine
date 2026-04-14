#!/bin/bash
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# Run PyTorch benchmarks.
#
# Usage:
#   bash benchmarks/scripts/run_pytorch.sh [extra_pytest_args...]
#
# Examples:
#   bash benchmarks/scripts/run_pytorch.sh                      # Run all
#   bash benchmarks/scripts/run_pytorch.sh -k "linear"          # Filter by name
#   bash benchmarks/scripts/run_pytorch.sh --size-filter gpu_bound
#   bash benchmarks/scripts/run_pytorch.sh --profile -k "fp8_block and gpu_bound"

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO_DIR"

echo "=== Running PyTorch benchmarks ==="
python3 -m pytest benchmarks/pytorch/ -v "$@"

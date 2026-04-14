#!/bin/bash
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# Run JAX benchmarks.
#
# Usage:
#   bash benchmarks/scripts/run_jax.sh [extra_pytest_args...]
#
# Examples:
#   bash benchmarks/scripts/run_jax.sh                          # Run all
#   bash benchmarks/scripts/run_jax.sh -k "dense"               # Filter by name
#   bash benchmarks/scripts/run_jax.sh --size-filter gpu_bound

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO_DIR"

echo "=== Running JAX benchmarks ==="
python3 -m pytest benchmarks/jax/ -v "$@"

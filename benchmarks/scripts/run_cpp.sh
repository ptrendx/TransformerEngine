#!/bin/bash
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# Build and run C++ benchmarks.
#
# Usage:
#   bash benchmarks/scripts/run_cpp.sh [benchmark_filter] [extra_args...]
#
# Examples:
#   bash benchmarks/scripts/run_cpp.sh                          # Run all
#   bash benchmarks/scripts/run_cpp.sh "BM_nvte_quantize.*"     # Filter by name
#   bash benchmarks/scripts/run_cpp.sh "" --benchmark_format=json --benchmark_out=results.json

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="$(cd "$SCRIPT_DIR/../cpp" && pwd)"
BUILD_DIR="${BENCH_DIR}/build"

FILTER="${1:-}"
shift 2>/dev/null || true

echo "=== Building C++ benchmarks ==="
cmake -GNinja -B "$BUILD_DIR" -S "$BENCH_DIR"
cmake --build "$BUILD_DIR"

echo ""
echo "=== Running C++ benchmarks ==="
BENCH_ARGS=()
if [ -n "$FILTER" ]; then
    BENCH_ARGS+=(--benchmark_filter="$FILTER")
fi
BENCH_ARGS+=("$@")

"$BUILD_DIR/operator/bench_operator" "${BENCH_ARGS[@]}"

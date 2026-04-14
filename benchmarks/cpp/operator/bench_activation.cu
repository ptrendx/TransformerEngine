/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <benchmark/benchmark.h>
#include <cuda_runtime.h>

#include <transformer_engine/activation.h>
#include "../bench_common.h"

using namespace transformer_engine;
using namespace test;

namespace {

// ---------------------------------------------------------------------------
// GeLU forward (bf16 -> bf16)
// ---------------------------------------------------------------------------
static void BM_nvte_gelu(benchmark::State &state) {
  const size_t rows = static_cast<size_t>(state.range(0));
  const size_t cols = static_cast<size_t>(state.range(1));

  Tensor input("input", {rows, cols}, DType::kBFloat16);
  Tensor output("output", {rows, cols}, DType::kBFloat16);

  fillUniform(&input);

  cudaStream_t stream = nullptr;

  // Warmup
  nvte_gelu(input.data(), output.data(), stream);
  cudaStreamSynchronize(stream);

  for (auto _ : state) {
    bench::CudaEventTimer timer(state, stream);
    nvte_gelu(input.data(), output.data(), stream);
  }

  const size_t total_bytes = rows * cols * 2 * sizeof(nv_bfloat16);  // read + write
  bench::SetThroughputCounters(state, total_bytes);
}

// ---------------------------------------------------------------------------
// ReLU forward (bf16 -> bf16)
// ---------------------------------------------------------------------------
static void BM_nvte_relu(benchmark::State &state) {
  const size_t rows = static_cast<size_t>(state.range(0));
  const size_t cols = static_cast<size_t>(state.range(1));

  Tensor input("input", {rows, cols}, DType::kBFloat16);
  Tensor output("output", {rows, cols}, DType::kBFloat16);

  fillUniform(&input);

  cudaStream_t stream = nullptr;

  // Warmup
  nvte_relu(input.data(), output.data(), stream);
  cudaStreamSynchronize(stream);

  for (auto _ : state) {
    bench::CudaEventTimer timer(state, stream);
    nvte_relu(input.data(), output.data(), stream);
  }

  const size_t total_bytes = rows * cols * 2 * sizeof(nv_bfloat16);
  bench::SetThroughputCounters(state, total_bytes);
}

// ---------------------------------------------------------------------------
// SiLU forward (bf16 -> bf16)
// ---------------------------------------------------------------------------
static void BM_nvte_silu(benchmark::State &state) {
  const size_t rows = static_cast<size_t>(state.range(0));
  const size_t cols = static_cast<size_t>(state.range(1));

  Tensor input("input", {rows, cols}, DType::kBFloat16);
  Tensor output("output", {rows, cols}, DType::kBFloat16);

  fillUniform(&input);

  cudaStream_t stream = nullptr;

  // Warmup
  nvte_silu(input.data(), output.data(), stream);
  cudaStreamSynchronize(stream);

  for (auto _ : state) {
    bench::CudaEventTimer timer(state, stream);
    nvte_silu(input.data(), output.data(), stream);
  }

  const size_t total_bytes = rows * cols * 2 * sizeof(nv_bfloat16);
  bench::SetThroughputCounters(state, total_bytes);
}

// ---------------------------------------------------------------------------
// SwiGLU forward (bf16 -> bf16)
// Input is [rows, 2*cols], output is [rows, cols] (gated activation).
// ---------------------------------------------------------------------------
static void BM_nvte_swiglu(benchmark::State &state) {
  const size_t rows = static_cast<size_t>(state.range(0));
  const size_t cols = static_cast<size_t>(state.range(1));

  // SwiGLU: input has 2x the output columns
  Tensor input("input", {rows, 2 * cols}, DType::kBFloat16);
  Tensor output("output", {rows, cols}, DType::kBFloat16);

  fillUniform(&input);

  cudaStream_t stream = nullptr;

  // Warmup
  nvte_swiglu(input.data(), output.data(), stream);
  cudaStreamSynchronize(stream);

  for (auto _ : state) {
    bench::CudaEventTimer timer(state, stream);
    nvte_swiglu(input.data(), output.data(), stream);
  }

  // Read 2*cols per row, write cols per row
  const size_t total_bytes = rows * cols * 3 * sizeof(nv_bfloat16);
  bench::SetThroughputCounters(state, total_bytes);
}

// ---------------------------------------------------------------------------
// Benchmark registration
// ---------------------------------------------------------------------------

// Standard (non-gated) activation sizes
#define REGISTER_ACT_BENCHMARK(name)                          \
  BENCHMARK(name)                                             \
      ->Args({16, 64})                                        \
      ->Args({32, 128})                                       \
      ->Args({8192, 4096})                                    \
      ->Args({16384, 12288})                                  \
      ->UseManualTime()                                       \
      ->Unit(benchmark::kMicrosecond);

REGISTER_ACT_BENCHMARK(BM_nvte_gelu)
REGISTER_ACT_BENCHMARK(BM_nvte_relu)
REGISTER_ACT_BENCHMARK(BM_nvte_silu)

// SwiGLU: Args specify output cols; input cols are 2x that.
REGISTER_ACT_BENCHMARK(BM_nvte_swiglu)

}  // namespace

BENCHMARK_MAIN();

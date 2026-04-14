/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <benchmark/benchmark.h>
#include <cuda_runtime.h>

#include <transformer_engine/hadamard_transform.h>
#include "../bench_common.h"

using namespace transformer_engine;
using namespace test;

namespace {

// ---------------------------------------------------------------------------
// Hadamard transform (bf16, cols must be power of 2)
// ---------------------------------------------------------------------------
static void BM_HadamardTransform(benchmark::State &state) {
  const size_t rows = static_cast<size_t>(state.range(0));
  const size_t cols = static_cast<size_t>(state.range(1));

  Tensor input("input", {rows, cols}, DType::kBFloat16);
  Tensor output("output", {rows, cols}, DType::kBFloat16);

  fillUniform(&input);

  // random_sign_mask and random_sign_mask_t are 16-bit sign masks.
  // Use 0 for benchmarking (no randomized signs).
  const int random_sign_mask = 0;
  const int random_sign_mask_t = 0;

  cudaStream_t stream = nullptr;

  // Warmup
  nvte_hadamard_transform(input.data(), output.data(),
                          random_sign_mask, random_sign_mask_t, stream);
  cudaStreamSynchronize(stream);

  for (auto _ : state) {
    bench::CudaEventTimer timer(state, stream);
    nvte_hadamard_transform(input.data(), output.data(),
                            random_sign_mask, random_sign_mask_t, stream);
  }

  // Read input + write output, both bf16.
  const size_t total_bytes = rows * cols * 2 * sizeof(nv_bfloat16);
  bench::SetThroughputCounters(state, total_bytes);
}

// Small sizes (cols must be power of 2)
BENCHMARK(BM_HadamardTransform)
    ->Args({16, 64})
    ->Args({32, 128})
    ->Args({64, 256})
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

// Large sizes
BENCHMARK(BM_HadamardTransform)
    ->Args({2048, 4096})
    ->Args({8192, 4096})
    ->Args({16384, 4096})
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

}  // namespace

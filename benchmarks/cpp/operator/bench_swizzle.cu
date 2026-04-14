/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <benchmark/benchmark.h>
#include <cuda_runtime.h>

#include <transformer_engine/swizzle.h>
#include "../bench_common.h"

using namespace transformer_engine;
using namespace test;

namespace {

// ---------------------------------------------------------------------------
// Swizzle scaling factors (MXFP8 1D scaling, rowwise)
// ---------------------------------------------------------------------------
// This kernel swizzles FP8 scaling factors into the interleaved layout
// required for efficient GEMM access. We create FP8 tensors with MXFP8
// 1D scaling (rowwise) so that they have properly-shaped scale_inv
// buffers, then benchmark the swizzle.
static void BM_SwizzleScalingFactors(benchmark::State &state) {
  const size_t rows = static_cast<size_t>(state.range(0));
  const size_t cols = static_cast<size_t>(state.range(1));

  Tensor input("input", {rows, cols}, DType::kFloat8E4M3,
               /*rowwise=*/true, /*columnwise=*/false,
               NVTE_MXFP8_1D_SCALING);
  Tensor output("output", {rows, cols}, DType::kFloat8E4M3,
                /*rowwise=*/true, /*columnwise=*/false,
                NVTE_MXFP8_1D_SCALING);
  output.set_with_gemm_swizzled_scales(true);

  fillUniform(&input);

  cudaStream_t stream = nullptr;

  // Warmup
  nvte_swizzle_scaling_factors(input.data(), output.data(), stream);
  cudaStreamSynchronize(stream);

  for (auto _ : state) {
    bench::CudaEventTimer timer(state, stream);
    nvte_swizzle_scaling_factors(input.data(), output.data(), stream);
  }

  // The swizzle operates on scale_inv factors. Report the FP8 data tensor
  // bytes as a throughput proxy.
  const size_t total_bytes = rows * cols;  // FP8 = 1 byte per element
  bench::SetThroughputCounters(state, total_bytes);
}

// Small sizes (rows and cols should be multiples of 128 for proper alignment)
BENCHMARK(BM_SwizzleScalingFactors)
    ->Args({128, 128})
    ->Args({128, 256})
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

// Large sizes
BENCHMARK(BM_SwizzleScalingFactors)
    ->Args({2048, 4096})
    ->Args({8192, 4096})
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

}  // namespace

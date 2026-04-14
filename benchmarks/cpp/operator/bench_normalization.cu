/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <benchmark/benchmark.h>
#include <cuda_runtime.h>

#include <transformer_engine/normalization.h>
#include "../bench_common.h"

using namespace transformer_engine;
using namespace test;

namespace {

// Cached SM count so we only query once.
int getMultiProcessorCount() {
  static int mp_count = [] {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    return prop.multiProcessorCount;
  }();
  return mp_count;
}

// ---------------------------------------------------------------------------
// LayerNorm forward (bf16 -> bf16)
// ---------------------------------------------------------------------------
static void BM_nvte_layernorm_fwd(benchmark::State &state) {
  const size_t rows = static_cast<size_t>(state.range(0));
  const size_t hidden = static_cast<size_t>(state.range(1));
  const float epsilon = 1e-5f;
  const bool zero_centered_gamma = false;
  const int sm_count = getMultiProcessorCount();

  Tensor input("input", {rows, hidden}, DType::kBFloat16);
  Tensor gamma("gamma", {hidden}, DType::kBFloat16);
  Tensor beta("beta", {hidden}, DType::kBFloat16);
  Tensor output("output", {rows, hidden}, DType::kBFloat16);
  Tensor mu("mu", {rows}, DType::kFloat32);
  Tensor rsigma("rsigma", {rows}, DType::kFloat32);

  fillUniform(&input);
  fillUniform(&gamma);
  fillUniform(&beta);

  cudaStream_t stream = nullptr;

  // First call with empty workspace to query required size.
  Tensor workspace;
  nvte_layernorm_fwd(input.data(), gamma.data(), beta.data(), epsilon,
                     output.data(), mu.data(), rsigma.data(),
                     workspace.data(), sm_count, zero_centered_gamma, stream);
  // Allocate workspace with the shape/type the kernel reported.
  workspace = Tensor("workspace", workspace.rowwise_shape(), workspace.dtype());

  // Warmup
  nvte_layernorm_fwd(input.data(), gamma.data(), beta.data(), epsilon,
                     output.data(), mu.data(), rsigma.data(),
                     workspace.data(), sm_count, zero_centered_gamma, stream);
  cudaStreamSynchronize(stream);

  for (auto _ : state) {
    bench::CudaEventTimer timer(state, stream);
    nvte_layernorm_fwd(input.data(), gamma.data(), beta.data(), epsilon,
                       output.data(), mu.data(), rsigma.data(),
                       workspace.data(), sm_count, zero_centered_gamma, stream);
  }

  // Bytes: read input + gamma + beta, write output + mu + rsigma
  const size_t total_bytes =
      rows * hidden * sizeof(nv_bfloat16) * 2           // input read + output write
      + hidden * sizeof(nv_bfloat16) * 2                 // gamma + beta read
      + rows * sizeof(float) * 2;                        // mu + rsigma write
  bench::SetThroughputCounters(state, total_bytes);
}

// ---------------------------------------------------------------------------
// RMSNorm forward (bf16 -> bf16)
// ---------------------------------------------------------------------------
static void BM_nvte_rmsnorm_fwd(benchmark::State &state) {
  const size_t rows = static_cast<size_t>(state.range(0));
  const size_t hidden = static_cast<size_t>(state.range(1));
  const float epsilon = 1e-5f;
  const bool zero_centered_gamma = false;
  const int sm_count = getMultiProcessorCount();

  Tensor input("input", {rows, hidden}, DType::kBFloat16);
  Tensor gamma("gamma", {hidden}, DType::kBFloat16);
  Tensor output("output", {rows, hidden}, DType::kBFloat16);
  Tensor rsigma("rsigma", {rows}, DType::kFloat32);

  fillUniform(&input);
  fillUniform(&gamma);

  cudaStream_t stream = nullptr;

  // First call with empty workspace to query required size.
  Tensor workspace;
  nvte_rmsnorm_fwd(input.data(), gamma.data(), epsilon,
                   output.data(), rsigma.data(),
                   workspace.data(), sm_count, zero_centered_gamma, stream);
  // Allocate workspace with the shape/type the kernel reported.
  workspace = Tensor("workspace", workspace.rowwise_shape(), workspace.dtype());

  // Warmup
  nvte_rmsnorm_fwd(input.data(), gamma.data(), epsilon,
                   output.data(), rsigma.data(),
                   workspace.data(), sm_count, zero_centered_gamma, stream);
  cudaStreamSynchronize(stream);

  for (auto _ : state) {
    bench::CudaEventTimer timer(state, stream);
    nvte_rmsnorm_fwd(input.data(), gamma.data(), epsilon,
                     output.data(), rsigma.data(),
                     workspace.data(), sm_count, zero_centered_gamma, stream);
  }

  // Bytes: read input + gamma, write output + rsigma
  const size_t total_bytes =
      rows * hidden * sizeof(nv_bfloat16) * 2           // input read + output write
      + hidden * sizeof(nv_bfloat16)                     // gamma read
      + rows * sizeof(float);                            // rsigma write
  bench::SetThroughputCounters(state, total_bytes);
}

// ---------------------------------------------------------------------------
// Benchmark registration
// ---------------------------------------------------------------------------

// Args: {rows, hidden_size}
// Small sizes (CPU-bound) and large sizes (GPU-bound).
#define REGISTER_NORM_BENCHMARK(name)                         \
  BENCHMARK(name)                                             \
      ->Args({16, 64})                                        \
      ->Args({32, 256})                                       \
      ->Args({2048, 4096})                                    \
      ->Args({8192, 4096})                                    \
      ->Args({2048, 12288})                                   \
      ->Args({16384, 12288})                                  \
      ->UseManualTime()                                       \
      ->Unit(benchmark::kMicrosecond);

REGISTER_NORM_BENCHMARK(BM_nvte_layernorm_fwd)
REGISTER_NORM_BENCHMARK(BM_nvte_rmsnorm_fwd)

}  // namespace

BENCHMARK_MAIN();

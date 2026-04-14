/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <benchmark/benchmark.h>
#include <cuda_runtime.h>

#include <transformer_engine/fused_rope.h>
#include "../bench_common.h"

using namespace transformer_engine;
using namespace test;

namespace {

// ---------------------------------------------------------------------------
// Fused RoPE forward (SBHD format, bf16)
// ---------------------------------------------------------------------------
static void BM_FusedRopeForward(benchmark::State &state) {
  const size_t s = static_cast<size_t>(state.range(0));
  const size_t b = static_cast<size_t>(state.range(1));
  const size_t h = static_cast<size_t>(state.range(2));
  const size_t d = static_cast<size_t>(state.range(3));

  Tensor input("input", {s, b, h, d}, DType::kBFloat16);
  Tensor freqs("freqs", {s, 1, 1, d}, DType::kFloat32);
  Tensor output("output", {s, b, h, d}, DType::kBFloat16);

  // Empty tensors for cu_seqlens and start_positions (not used for SBHD format).
  TensorWrapper cu_seqlens;
  TensorWrapper start_positions;

  fillUniform(&input);
  fillUniform(&freqs);

  // SBHD contiguous strides.
  const int stride_s = static_cast<int>(b * h * d);
  const int stride_b = static_cast<int>(h * d);
  const int stride_h = static_cast<int>(d);
  const int stride_d = 1;
  const int d2 = static_cast<int>(d);

  cudaStream_t stream = nullptr;

  // Warmup
  nvte_fused_rope_forward(input.data(), cu_seqlens.data(), freqs.data(),
                          start_positions.data(), output.data(),
                          NVTE_SBHD, /*interleaved=*/false,
                          /*cp_size=*/1, /*cp_rank=*/0,
                          static_cast<int>(s), static_cast<int>(b),
                          static_cast<int>(h), static_cast<int>(d), d2,
                          stride_s, stride_b, stride_h, stride_d, stream);
  cudaStreamSynchronize(stream);

  for (auto _ : state) {
    bench::CudaEventTimer timer(state, stream);
    nvte_fused_rope_forward(input.data(), cu_seqlens.data(), freqs.data(),
                            start_positions.data(), output.data(),
                            NVTE_SBHD, /*interleaved=*/false,
                            /*cp_size=*/1, /*cp_rank=*/0,
                            static_cast<int>(s), static_cast<int>(b),
                            static_cast<int>(h), static_cast<int>(d), d2,
                            stride_s, stride_b, stride_h, stride_d, stream);
  }

  // Input + freqs read, output written.
  const size_t input_bytes = s * b * h * d * sizeof(nv_bfloat16);
  const size_t freqs_bytes = s * d * sizeof(float);
  const size_t output_bytes = s * b * h * d * sizeof(nv_bfloat16);
  bench::SetThroughputCounters(state, input_bytes + freqs_bytes + output_bytes);
}

// ---------------------------------------------------------------------------
// Fused RoPE backward (SBHD format, bf16)
// ---------------------------------------------------------------------------
static void BM_FusedRopeBackward(benchmark::State &state) {
  const size_t s = static_cast<size_t>(state.range(0));
  const size_t b = static_cast<size_t>(state.range(1));
  const size_t h = static_cast<size_t>(state.range(2));
  const size_t d = static_cast<size_t>(state.range(3));

  Tensor output_grads("output_grads", {s, b, h, d}, DType::kBFloat16);
  Tensor freqs("freqs", {s, 1, 1, d}, DType::kFloat32);
  Tensor input_grads("input_grads", {s, b, h, d}, DType::kBFloat16);

  TensorWrapper cu_seqlens;
  TensorWrapper start_positions;

  fillUniform(&output_grads);
  fillUniform(&freqs);

  const int stride_s = static_cast<int>(b * h * d);
  const int stride_b = static_cast<int>(h * d);
  const int stride_h = static_cast<int>(d);
  const int stride_d = 1;
  const int d2 = static_cast<int>(d);

  cudaStream_t stream = nullptr;

  // Warmup
  nvte_fused_rope_backward(output_grads.data(), cu_seqlens.data(), freqs.data(),
                           start_positions.data(), input_grads.data(),
                           NVTE_SBHD, /*interleaved=*/false,
                           /*cp_size=*/1, /*cp_rank=*/0,
                           static_cast<int>(s), static_cast<int>(b),
                           static_cast<int>(h), static_cast<int>(d), d2,
                           stride_s, stride_b, stride_h, stride_d, stream);
  cudaStreamSynchronize(stream);

  for (auto _ : state) {
    bench::CudaEventTimer timer(state, stream);
    nvte_fused_rope_backward(output_grads.data(), cu_seqlens.data(), freqs.data(),
                             start_positions.data(), input_grads.data(),
                             NVTE_SBHD, /*interleaved=*/false,
                             /*cp_size=*/1, /*cp_rank=*/0,
                             static_cast<int>(s), static_cast<int>(b),
                             static_cast<int>(h), static_cast<int>(d), d2,
                             stride_s, stride_b, stride_h, stride_d, stream);
  }

  const size_t grad_bytes = s * b * h * d * sizeof(nv_bfloat16);
  const size_t freqs_bytes = s * d * sizeof(float);
  bench::SetThroughputCounters(state, grad_bytes + freqs_bytes + grad_bytes);
}

// Small sizes: (s, b, h, d)
BENCHMARK(BM_FusedRopeForward)
    ->Args({32, 1, 4, 64})
    ->Args({64, 2, 8, 64})
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

// Large sizes
BENCHMARK(BM_FusedRopeForward)
    ->Args({2048, 2, 32, 128})
    ->Args({8192, 2, 32, 128})
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

// Small sizes
BENCHMARK(BM_FusedRopeBackward)
    ->Args({32, 1, 4, 64})
    ->Args({64, 2, 8, 64})
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

// Large sizes
BENCHMARK(BM_FusedRopeBackward)
    ->Args({2048, 2, 32, 128})
    ->Args({8192, 2, 32, 128})
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

}  // namespace

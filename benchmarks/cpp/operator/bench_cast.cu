/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <benchmark/benchmark.h>
#include <cuda_runtime.h>

#include <transformer_engine/cast.h>
#include "../bench_common.h"

using namespace transformer_engine;
using namespace test;

namespace {

// ---------------------------------------------------------------------------
// nvte_quantize -- delayed tensor scaling (bf16 -> fp8e4m3)
// ---------------------------------------------------------------------------
static void BM_nvte_quantize_delayed(benchmark::State &state) {
  const size_t rows = static_cast<size_t>(state.range(0));
  const size_t cols = static_cast<size_t>(state.range(1));

  Tensor input("input", {rows, cols}, DType::kBFloat16);
  Tensor output("output", {rows, cols}, DType::kFloat8E4M3,
                /*rowwise=*/true, /*columnwise=*/false,
                NVTE_DELAYED_TENSOR_SCALING);

  fillUniform(&input);
  setRandomScale(&output);

  cudaStream_t stream = nullptr;

  // Warmup
  nvte_quantize(input.data(), output.data(), stream);
  cudaStreamSynchronize(stream);

  for (auto _ : state) {
    bench::CudaEventTimer timer(state, stream);
    nvte_quantize(input.data(), output.data(), stream);
  }

  const size_t total_bytes = rows * cols * (sizeof(nv_bfloat16) + sizeof(__nv_fp8_e4m3));
  bench::SetThroughputCounters(state, total_bytes);
}

// ---------------------------------------------------------------------------
// nvte_quantize -- block 1D scaling (bf16 -> fp8e4m3)
// ---------------------------------------------------------------------------
static void BM_nvte_quantize_block1d(benchmark::State &state) {
  const size_t rows = static_cast<size_t>(state.range(0));
  const size_t cols = static_cast<size_t>(state.range(1));

  Tensor input("input", {rows, cols}, DType::kBFloat16);
  Tensor output("output", {rows, cols}, DType::kFloat8E4M3,
                /*rowwise=*/true, /*columnwise=*/false,
                NVTE_BLOCK_SCALING_1D);

  fillUniform(&input);

  cudaStream_t stream = nullptr;

  // Warmup
  nvte_quantize(input.data(), output.data(), stream);
  cudaStreamSynchronize(stream);

  for (auto _ : state) {
    bench::CudaEventTimer timer(state, stream);
    nvte_quantize(input.data(), output.data(), stream);
  }

  const size_t total_bytes = rows * cols * (sizeof(nv_bfloat16) + sizeof(__nv_fp8_e4m3));
  bench::SetThroughputCounters(state, total_bytes);
}

// ---------------------------------------------------------------------------
// nvte_quantize -- MXFP8 scaling (bf16 -> fp8e4m3)
// ---------------------------------------------------------------------------
static void BM_nvte_quantize_mxfp8(benchmark::State &state) {
  const size_t rows = static_cast<size_t>(state.range(0));
  const size_t cols = static_cast<size_t>(state.range(1));

  Tensor input("input", {rows, cols}, DType::kBFloat16);
  Tensor output("output", {rows, cols}, DType::kFloat8E4M3,
                /*rowwise=*/true, /*columnwise=*/false,
                NVTE_MXFP8_1D_SCALING);

  fillUniform(&input);

  cudaStream_t stream = nullptr;

  // Warmup
  nvte_quantize(input.data(), output.data(), stream);
  cudaStreamSynchronize(stream);

  for (auto _ : state) {
    bench::CudaEventTimer timer(state, stream);
    nvte_quantize(input.data(), output.data(), stream);
  }

  const size_t total_bytes = rows * cols * (sizeof(nv_bfloat16) + sizeof(__nv_fp8_e4m3));
  bench::SetThroughputCounters(state, total_bytes);
}

// ---------------------------------------------------------------------------
// nvte_dequantize (fp8e4m3 -> bf16)
// ---------------------------------------------------------------------------
static void BM_nvte_dequantize(benchmark::State &state) {
  const size_t rows = static_cast<size_t>(state.range(0));
  const size_t cols = static_cast<size_t>(state.range(1));

  // Create an FP8 input by quantizing a BF16 source tensor.
  Tensor source("source", {rows, cols}, DType::kBFloat16);
  Tensor fp8_input("fp8_input", {rows, cols}, DType::kFloat8E4M3,
                   /*rowwise=*/true, /*columnwise=*/false,
                   NVTE_DELAYED_TENSOR_SCALING);
  Tensor output("output", {rows, cols}, DType::kBFloat16);

  fillUniform(&source);
  setRandomScale(&fp8_input);
  nvte_quantize(source.data(), fp8_input.data(), nullptr);
  cudaDeviceSynchronize();

  cudaStream_t stream = nullptr;

  // Warmup
  nvte_dequantize(fp8_input.data(), output.data(), stream);
  cudaStreamSynchronize(stream);

  for (auto _ : state) {
    bench::CudaEventTimer timer(state, stream);
    nvte_dequantize(fp8_input.data(), output.data(), stream);
  }

  const size_t total_bytes = rows * cols * (sizeof(__nv_fp8_e4m3) + sizeof(nv_bfloat16));
  bench::SetThroughputCounters(state, total_bytes);
}

// ---------------------------------------------------------------------------
// Benchmark registration
// ---------------------------------------------------------------------------

// Small sizes (CPU-bound)
// Large sizes (GPU-bound)
#define REGISTER_CAST_BENCHMARK(name)                         \
  BENCHMARK(name)                                             \
      ->Args({16, 128})                                       \
      ->Args({32, 256})                                       \
      ->Args({8192, 4096})                                    \
      ->Args({16384, 12288})                                  \
      ->UseManualTime()                                       \
      ->Unit(benchmark::kMicrosecond);

REGISTER_CAST_BENCHMARK(BM_nvte_quantize_delayed)
REGISTER_CAST_BENCHMARK(BM_nvte_quantize_block1d)
REGISTER_CAST_BENCHMARK(BM_nvte_quantize_mxfp8)
REGISTER_CAST_BENCHMARK(BM_nvte_dequantize)

}  // namespace

BENCHMARK_MAIN();

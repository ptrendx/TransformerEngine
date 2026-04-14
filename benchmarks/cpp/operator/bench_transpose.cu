/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <benchmark/benchmark.h>
#include <cuda_runtime.h>

#include <transformer_engine/transpose.h>
#include "../bench_common.h"

using namespace transformer_engine;

namespace {

// Benchmark for nvte_transpose with BF16 tensors.
// Transposes input of shape (rows, cols) to output of shape (cols, rows).
static void BM_nvte_transpose_bf16(benchmark::State& state) {
    const size_t rows = state.range(0);
    const size_t cols = state.range(1);

    test::Tensor input("input", {rows, cols}, DType::kBFloat16);
    test::Tensor output("output", {cols, rows}, DType::kBFloat16);

    test::fillUniform(&input);

    cudaStream_t stream = 0;

    // Warmup
    nvte_transpose(input.data(), output.data(), stream);
    cudaStreamSynchronize(stream);

    for (auto _ : state) {
        bench::CudaEventTimer timer(state, stream);
        nvte_transpose(input.data(), output.data(), stream);
    }

    // Total bytes: read input + write output, each element is 2 bytes (bf16)
    const size_t total_bytes = rows * cols * 2 * 2;
    bench::SetThroughputCounters(state, total_bytes);
}

// Benchmark for nvte_transpose with FP16 tensors.
static void BM_nvte_transpose_fp16(benchmark::State& state) {
    const size_t rows = state.range(0);
    const size_t cols = state.range(1);

    test::Tensor input("input", {rows, cols}, DType::kFloat16);
    test::Tensor output("output", {cols, rows}, DType::kFloat16);

    test::fillUniform(&input);

    cudaStream_t stream = 0;

    // Warmup
    nvte_transpose(input.data(), output.data(), stream);
    cudaStreamSynchronize(stream);

    for (auto _ : state) {
        bench::CudaEventTimer timer(state, stream);
        nvte_transpose(input.data(), output.data(), stream);
    }

    const size_t total_bytes = rows * cols * 2 * 2;
    bench::SetThroughputCounters(state, total_bytes);
}

// Benchmark for nvte_cast_transpose: casts BF16 input to FP8 and produces
// both rowwise and columnwise (transposed) outputs.
// input: (rows, cols) bf16
// output rowwise: (rows, cols) fp8e4m3
// output columnwise: (cols, rows) fp8e4m3
static void BM_nvte_cast_transpose_bf16_to_fp8(benchmark::State& state) {
    const size_t rows = state.range(0);
    const size_t cols = state.range(1);

    test::Tensor input("input", {rows, cols}, DType::kBFloat16);
    test::Tensor output("output", {rows, cols}, DType::kFloat8E4M3,
                        true /* rowwise */, true /* columnwise */);

    test::fillUniform(&input);
    output.set_scale(1.0f);

    cudaStream_t stream = 0;

    // Warmup
    nvte_cast_transpose(input.data(), output.data(), stream);
    cudaStreamSynchronize(stream);

    for (auto _ : state) {
        bench::CudaEventTimer timer(state, stream);
        nvte_cast_transpose(input.data(), output.data(), stream);
    }

    // Bytes: read bf16 input + write fp8 rowwise + write fp8 columnwise
    const size_t total_bytes = rows * cols * 2 + rows * cols * 1 + rows * cols * 1;
    bench::SetThroughputCounters(state, total_bytes);
}

}  // namespace

// nvte_transpose BF16 - Small sizes (launch overhead)
BENCHMARK(BM_nvte_transpose_bf16)
    ->Args({32, 64})
    ->Args({64, 128})
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

// nvte_transpose BF16 - Large sizes (bandwidth bound)
BENCHMARK(BM_nvte_transpose_bf16)
    ->Args({4096, 4096})
    ->Args({8192, 12288})
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

// nvte_transpose FP16 - Small sizes
BENCHMARK(BM_nvte_transpose_fp16)
    ->Args({32, 64})
    ->Args({64, 128})
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

// nvte_transpose FP16 - Large sizes
BENCHMARK(BM_nvte_transpose_fp16)
    ->Args({4096, 4096})
    ->Args({8192, 12288})
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

// nvte_cast_transpose BF16->FP8 - Small sizes
BENCHMARK(BM_nvte_cast_transpose_bf16_to_fp8)
    ->Args({32, 64})
    ->Args({64, 128})
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

// nvte_cast_transpose BF16->FP8 - Large sizes
BENCHMARK(BM_nvte_cast_transpose_bf16_to_fp8)
    ->Args({4096, 4096})
    ->Args({8192, 12288})
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

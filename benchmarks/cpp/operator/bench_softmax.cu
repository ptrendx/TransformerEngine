/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <benchmark/benchmark.h>
#include <cuda_runtime.h>

#include <transformer_engine/softmax.h>
#include "../bench_common.h"

using namespace transformer_engine;

namespace {

// Benchmark for nvte_scaled_softmax_forward.
// Input shape: (batch, heads, seq_q, seq_kv), dtype: fp16.
static void BM_nvte_scaled_softmax_forward_fp16(benchmark::State& state) {
    const size_t batch  = state.range(0);
    const size_t heads  = state.range(1);
    const size_t seq_q  = state.range(2);
    const size_t seq_kv = state.range(3);

    test::Tensor input("input", {batch, heads, seq_q, seq_kv}, DType::kFloat16);
    test::Tensor output("output", {batch, heads, seq_q, seq_kv}, DType::kFloat16);

    test::fillUniform(&input);

    const float scale_factor = 0.125f;  // 1/sqrt(64)
    cudaStream_t stream = 0;

    // Warmup
    nvte_scaled_softmax_forward(input.data(), output.data(), scale_factor, stream);
    cudaStreamSynchronize(stream);

    for (auto _ : state) {
        bench::CudaEventTimer timer(state, stream);
        nvte_scaled_softmax_forward(input.data(), output.data(), scale_factor, stream);
    }

    // Total bytes: read input + write output, each element is 2 bytes (fp16)
    const size_t num_elements = batch * heads * seq_q * seq_kv;
    const size_t total_bytes = num_elements * 2 * 2;
    bench::SetThroughputCounters(state, total_bytes);
}

// Benchmark for nvte_scaled_softmax_backward.
// Input shapes: (batch, heads, seq_q, seq_kv), dtype: fp16.
static void BM_nvte_scaled_softmax_backward_fp16(benchmark::State& state) {
    const size_t batch  = state.range(0);
    const size_t heads  = state.range(1);
    const size_t seq_q  = state.range(2);
    const size_t seq_kv = state.range(3);

    test::Tensor incoming_grads("incoming_grads", {batch, heads, seq_q, seq_kv}, DType::kFloat16);
    test::Tensor softmax_results("softmax_results", {batch, heads, seq_q, seq_kv}, DType::kFloat16);
    test::Tensor output_grads("output_grads", {batch, heads, seq_q, seq_kv}, DType::kFloat16);

    test::fillUniform(&incoming_grads);
    test::fillUniform(&softmax_results);

    const float scale_factor = 0.125f;
    cudaStream_t stream = 0;

    // Warmup
    nvte_scaled_softmax_backward(incoming_grads.data(), softmax_results.data(),
                                 output_grads.data(), scale_factor, stream);
    cudaStreamSynchronize(stream);

    for (auto _ : state) {
        bench::CudaEventTimer timer(state, stream);
        nvte_scaled_softmax_backward(incoming_grads.data(), softmax_results.data(),
                                     output_grads.data(), scale_factor, stream);
    }

    // Bytes: read incoming_grads + read softmax_results + write output_grads
    const size_t num_elements = batch * heads * seq_q * seq_kv;
    const size_t total_bytes = num_elements * 2 * 3;
    bench::SetThroughputCounters(state, total_bytes);
}

// Benchmark for nvte_scaled_masked_softmax_forward.
// Input shape: (batch, heads, seq_q, seq_kv), mask: (1, 1, seq_q, seq_kv).
static void BM_nvte_scaled_masked_softmax_forward_fp16(benchmark::State& state) {
    const size_t batch  = state.range(0);
    const size_t heads  = state.range(1);
    const size_t seq_q  = state.range(2);
    const size_t seq_kv = state.range(3);

    test::Tensor input("input", {batch, heads, seq_q, seq_kv}, DType::kFloat16);
    test::Tensor mask("mask", {1, 1, seq_q, seq_kv}, DType::kBFloat16);
    test::Tensor output("output", {batch, heads, seq_q, seq_kv}, DType::kFloat16);

    test::fillUniform(&input);
    test::fillUniform(&mask);

    const float scale_factor = 0.125f;
    cudaStream_t stream = 0;

    // Warmup
    nvte_scaled_masked_softmax_forward(input.data(), mask.data(),
                                       output.data(), scale_factor, stream);
    cudaStreamSynchronize(stream);

    for (auto _ : state) {
        bench::CudaEventTimer timer(state, stream);
        nvte_scaled_masked_softmax_forward(input.data(), mask.data(),
                                           output.data(), scale_factor, stream);
    }

    const size_t num_elements = batch * heads * seq_q * seq_kv;
    const size_t mask_elements = seq_q * seq_kv;
    // Bytes: read input + read mask + write output
    const size_t total_bytes = num_elements * 2 + mask_elements * 2 + num_elements * 2;
    bench::SetThroughputCounters(state, total_bytes);
}

// Benchmark for nvte_scaled_aligned_causal_masked_softmax_forward.
// Uses an implicit causal mask aligned to the bottom-right corner.
static void BM_nvte_scaled_causal_softmax_forward_fp16(benchmark::State& state) {
    const size_t batch  = state.range(0);
    const size_t heads  = state.range(1);
    const size_t seq_q  = state.range(2);
    const size_t seq_kv = state.range(3);

    test::Tensor input("input", {batch, heads, seq_q, seq_kv}, DType::kFloat16);
    test::Tensor output("output", {batch, heads, seq_q, seq_kv}, DType::kFloat16);

    test::fillUniform(&input);

    const float scale_factor = 0.125f;
    cudaStream_t stream = 0;

    // Warmup
    nvte_scaled_aligned_causal_masked_softmax_forward(
        input.data(), output.data(), scale_factor, stream);
    cudaStreamSynchronize(stream);

    for (auto _ : state) {
        bench::CudaEventTimer timer(state, stream);
        nvte_scaled_aligned_causal_masked_softmax_forward(
            input.data(), output.data(), scale_factor, stream);
    }

    const size_t num_elements = batch * heads * seq_q * seq_kv;
    const size_t total_bytes = num_elements * 2 * 2;
    bench::SetThroughputCounters(state, total_bytes);
}

// Benchmark for nvte_scaled_aligned_causal_masked_softmax_backward.
static void BM_nvte_scaled_causal_softmax_backward_fp16(benchmark::State& state) {
    const size_t batch  = state.range(0);
    const size_t heads  = state.range(1);
    const size_t seq_q  = state.range(2);
    const size_t seq_kv = state.range(3);

    test::Tensor incoming_grads("incoming_grads", {batch, heads, seq_q, seq_kv}, DType::kFloat16);
    test::Tensor softmax_results("softmax_results", {batch, heads, seq_q, seq_kv}, DType::kFloat16);
    test::Tensor output_grads("output_grads", {batch, heads, seq_q, seq_kv}, DType::kFloat16);

    test::fillUniform(&incoming_grads);
    test::fillUniform(&softmax_results);

    const float scale_factor = 0.125f;
    cudaStream_t stream = 0;

    // Warmup
    nvte_scaled_aligned_causal_masked_softmax_backward(
        incoming_grads.data(), softmax_results.data(),
        output_grads.data(), scale_factor, stream);
    cudaStreamSynchronize(stream);

    for (auto _ : state) {
        bench::CudaEventTimer timer(state, stream);
        nvte_scaled_aligned_causal_masked_softmax_backward(
            incoming_grads.data(), softmax_results.data(),
            output_grads.data(), scale_factor, stream);
    }

    const size_t num_elements = batch * heads * seq_q * seq_kv;
    const size_t total_bytes = num_elements * 2 * 3;
    bench::SetThroughputCounters(state, total_bytes);
}

}  // namespace

// scaled_softmax_forward FP16 - Small sizes (CPU-bound)
BENCHMARK(BM_nvte_scaled_softmax_forward_fp16)
    ->Args({1, 4, 32, 32})
    ->Args({2, 8, 64, 64})
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

// scaled_softmax_forward FP16 - Large sizes (GPU-bound)
BENCHMARK(BM_nvte_scaled_softmax_forward_fp16)
    ->Args({2, 16, 512, 512})
    ->Args({2, 32, 2048, 2048})
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

// scaled_softmax_backward FP16 - Small sizes
BENCHMARK(BM_nvte_scaled_softmax_backward_fp16)
    ->Args({1, 4, 32, 32})
    ->Args({2, 8, 64, 64})
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

// scaled_softmax_backward FP16 - Large sizes
BENCHMARK(BM_nvte_scaled_softmax_backward_fp16)
    ->Args({2, 16, 512, 512})
    ->Args({2, 32, 2048, 2048})
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

// scaled_masked_softmax_forward FP16 - Small sizes
BENCHMARK(BM_nvte_scaled_masked_softmax_forward_fp16)
    ->Args({1, 4, 32, 32})
    ->Args({2, 8, 64, 64})
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

// scaled_masked_softmax_forward FP16 - Large sizes
BENCHMARK(BM_nvte_scaled_masked_softmax_forward_fp16)
    ->Args({2, 16, 512, 512})
    ->Args({2, 32, 2048, 2048})
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

// causal_softmax_forward FP16 - Small sizes
BENCHMARK(BM_nvte_scaled_causal_softmax_forward_fp16)
    ->Args({1, 4, 32, 32})
    ->Args({2, 8, 64, 64})
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

// causal_softmax_forward FP16 - Large sizes
BENCHMARK(BM_nvte_scaled_causal_softmax_forward_fp16)
    ->Args({2, 16, 512, 512})
    ->Args({2, 32, 2048, 2048})
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

// causal_softmax_backward FP16 - Small sizes
BENCHMARK(BM_nvte_scaled_causal_softmax_backward_fp16)
    ->Args({1, 4, 32, 32})
    ->Args({2, 8, 64, 64})
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

// causal_softmax_backward FP16 - Large sizes
BENCHMARK(BM_nvte_scaled_causal_softmax_backward_fp16)
    ->Args({2, 16, 512, 512})
    ->Args({2, 32, 2048, 2048})
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <benchmark/benchmark.h>
#include <cuda_runtime.h>

#include <transformer_engine/gemm.h>
#include "../bench_common.h"

using namespace transformer_engine;

namespace {

// Benchmark for nvte_cublas_gemm with BF16 inputs and output.
// Computes D = A * B where A is (M, K), B is (K, N), D is (M, N).
static void BM_nvte_cublas_gemm_bf16(benchmark::State& state) {
    const size_t M = state.range(0);
    const size_t K = state.range(1);
    const size_t N = state.range(2);

    test::Tensor A("A", {M, K}, DType::kBFloat16);
    test::Tensor B("B", {K, N}, DType::kBFloat16);
    test::Tensor D("D", {M, N}, DType::kBFloat16);
    test::Tensor bias("bias", {0}, DType::kBFloat16);
    test::Tensor pre_gelu_out("pre_gelu_out", {0}, DType::kBFloat16);
    test::Tensor workspace("workspace", {33554432}, DType::kByte);

    test::fillUniform(&A);
    test::fillUniform(&B);

    cudaStream_t stream = 0;

    // Warmup
    nvte_cublas_gemm(A.data(), B.data(), D.data(),
                     bias.data(), pre_gelu_out.data(),
                     false, false, false,
                     workspace.data(),
                     false, false, 0, stream);
    cudaStreamSynchronize(stream);

    for (auto _ : state) {
        bench::CudaEventTimer timer(state, stream);
        nvte_cublas_gemm(A.data(), B.data(), D.data(),
                         bias.data(), pre_gelu_out.data(),
                         false, false, false,
                         workspace.data(),
                         false, false, 0, stream);
    }

    // 2*M*N*K FLOPs for matrix multiplication
    const double flops = 2.0 * static_cast<double>(M) * static_cast<double>(N)
                             * static_cast<double>(K);
    state.counters["TFLOPS"] = benchmark::Counter(
        flops, benchmark::Counter::kIsIterationInvariant,
        benchmark::Counter::OneK::kIs1000);

    // Total bytes: A(M*K) + B(K*N) + D(M*N), each element is 2 bytes (bf16)
    const size_t total_bytes = (M * K + K * N + M * N) * 2;
    bench::SetThroughputCounters(state, total_bytes);
}

// Benchmark for nvte_cublas_gemm_v2 with BF16 inputs and output.
// Computes D = alpha * A * B where A is (M, K), B is (K, N), D is (M, N).
static void BM_nvte_cublas_gemm_v2_bf16(benchmark::State& state) {
    const size_t M = state.range(0);
    const size_t K = state.range(1);
    const size_t N = state.range(2);

    test::Tensor A("A", {M, K}, DType::kBFloat16);
    test::Tensor B("B", {K, N}, DType::kBFloat16);
    test::Tensor C("C", {0}, DType::kBFloat16);
    test::Tensor D("D", {M, N}, DType::kBFloat16);
    test::Tensor workspace("workspace", {33554432}, DType::kByte);

    test::fillUniform(&A);
    test::fillUniform(&B);

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cudaStream_t stream = 0;

    MatmulConfigWrapper config;

    // Warmup
    nvte_cublas_gemm_v2(false, false, &alpha,
                        A.data(), B.data(),
                        &beta, C.data(), D.data(),
                        workspace.data(), config, stream);
    cudaStreamSynchronize(stream);

    for (auto _ : state) {
        bench::CudaEventTimer timer(state, stream);
        nvte_cublas_gemm_v2(false, false, &alpha,
                            A.data(), B.data(),
                            &beta, C.data(), D.data(),
                            workspace.data(), config, stream);
    }

    const double flops = 2.0 * static_cast<double>(M) * static_cast<double>(N)
                             * static_cast<double>(K);
    state.counters["TFLOPS"] = benchmark::Counter(
        flops, benchmark::Counter::kIsIterationInvariant,
        benchmark::Counter::OneK::kIs1000);

    const size_t total_bytes = (M * K + K * N + M * N) * 2;
    bench::SetThroughputCounters(state, total_bytes);
}

}  // namespace

// Small sizes (CPU-bound / launch overhead dominated)
BENCHMARK(BM_nvte_cublas_gemm_bf16)
    ->Args({16, 64, 64})
    ->Args({32, 128, 128})
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

// Large sizes (GPU-bound / compute dominated)
BENCHMARK(BM_nvte_cublas_gemm_bf16)
    ->Args({4096, 4096, 4096})
    ->Args({8192, 4096, 16384})
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

// Small sizes for v2 API
BENCHMARK(BM_nvte_cublas_gemm_v2_bf16)
    ->Args({16, 64, 64})
    ->Args({32, 128, 128})
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

// Large sizes for v2 API
BENCHMARK(BM_nvte_cublas_gemm_v2_bf16)
    ->Args({4096, 4096, 4096})
    ->Args({8192, 4096, 16384})
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

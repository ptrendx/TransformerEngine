/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <benchmark/benchmark.h>

#include <transformer_engine/fused_attn.h>
#include "../bench_common.h"

using namespace transformer_engine;
using namespace test;

namespace {

/**
 * Fused attention benchmarks.
 *
 * Note: The fused attention C API is highly complex, requiring cuDNN
 * workspace allocation, backend selection, and many configuration
 * parameters. For comprehensive attention benchmarking, use the
 * PyTorch-level benchmarks (benchmarks/pytorch/bench_attention.py)
 * which handle all the setup automatically.
 *
 * This file provides a minimal C API benchmark for the most common
 * configuration: self-attention with causal mask.
 */

static int get_sm_count() {
    static int sm_count = 0;
    if (sm_count == 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        sm_count = prop.multiProcessorCount;
    }
    return sm_count;
}

static void BM_nvte_fused_attn_fwd(benchmark::State& state) {
    const size_t batch = state.range(0);
    const size_t num_heads = state.range(1);
    const size_t seq_len = state.range(2);
    const size_t head_dim = state.range(3);

    // Q, K, V as separate tensors: (batch, seq_len, num_heads, head_dim) bf16
    Tensor Q("Q", {batch, seq_len, num_heads, head_dim}, DType::kBFloat16);
    Tensor K("K", {batch, seq_len, num_heads, head_dim}, DType::kBFloat16);
    Tensor V("V", {batch, seq_len, num_heads, head_dim}, DType::kBFloat16);
    Tensor O("O", {batch, seq_len, num_heads, head_dim}, DType::kBFloat16);

    fillUniform(&Q);
    fillUniform(&K);
    fillUniform(&V);

    // Softmax auxiliary output for backward
    Tensor softmax_aux("softmax_aux", {batch, num_heads, seq_len, seq_len},
                       DType::kFloat32);

    // Bias - none
    // cu_seqlens for variable-length - use fixed lengths
    std::vector<int32_t> cu_seqlens_host(batch + 1);
    for (size_t i = 0; i <= batch; ++i) {
        cu_seqlens_host[i] = static_cast<int32_t>(i * seq_len);
    }
    Tensor cu_seqlens_q("cu_seqlens_q", {batch + 1}, DType::kInt32);
    Tensor cu_seqlens_kv("cu_seqlens_kv", {batch + 1}, DType::kInt32);
    cudaMemcpy(cu_seqlens_q.rowwise_dptr(), cu_seqlens_host.data(),
               (batch + 1) * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(cu_seqlens_kv.rowwise_dptr(), cu_seqlens_host.data(),
               (batch + 1) * sizeof(int32_t), cudaMemcpyHostToDevice);

    Tensor rng_state("rng_state", {2}, DType::kInt64);
    Tensor workspace("workspace", {0}, DType::kByte);

    cudaStream_t stream = 0;

    // Query workspace size
    NVTEFusedAttnBackend backend;
    NVTE_Fused_Attn_Backend fused_attn_backend = nvte_get_fused_attn_backend(
        DType::kBFloat16, DType::kBFloat16,
        NVTE_QKV_Layout::NVTE_BSHD_BSHD_BSHD,
        NVTE_Bias_Type::NVTE_NO_BIAS,
        NVTE_Mask_Type::NVTE_CAUSAL_BOTTOM_RIGHT_MASK,
        0.0f,  // dropout
        num_heads, num_heads,  // num_heads_q, num_heads_kv
        seq_len, seq_len,
        head_dim, head_dim,
        -1, -1,  // window_size
        0);  // max_seqlen_kv

    if (fused_attn_backend == NVTE_Fused_Attn_Backend::NVTE_No_Backend) {
        state.SkipWithMessage("No fused attention backend available for this config");
        return;
    }

    // Forward pass benchmark
    float attn_scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    nvte_fused_attn_fwd(
        Q.data(), K.data(), V.data(),
        nullptr,  // bias
        softmax_aux.data(),
        nullptr,  // rng_state
        O.data(),
        nullptr,  // auxiliary output
        cu_seqlens_q.data(), cu_seqlens_kv.data(),
        nullptr,  // cu_seqlens_q_padded
        nullptr,  // cu_seqlens_kv_padded
        nullptr,  // page_table
        nullptr,  // k_table
        nullptr,  // v_table
        NVTE_QKV_Layout::NVTE_BSHD_BSHD_BSHD,
        NVTE_Bias_Type::NVTE_NO_BIAS,
        NVTE_Mask_Type::NVTE_CAUSAL_BOTTOM_RIGHT_MASK,
        NVTE_Softmax_Type::NVTE_VANILLA_SOFTMAX,
        attn_scale,
        0.0f,  // dropout
        0,     // seed
        false, // is_training
        -1, -1,  // window_size
        0,     // max_seqlen_kv
        workspace.data(),
        stream);
    cudaStreamSynchronize(stream);

    for (auto _ : state) {
        bench::CudaEventTimer timer(state, stream);
        nvte_fused_attn_fwd(
            Q.data(), K.data(), V.data(),
            nullptr, softmax_aux.data(),
            nullptr, O.data(), nullptr,
            cu_seqlens_q.data(), cu_seqlens_kv.data(),
            nullptr, nullptr,
            nullptr, nullptr, nullptr,
            NVTE_QKV_Layout::NVTE_BSHD_BSHD_BSHD,
            NVTE_Bias_Type::NVTE_NO_BIAS,
            NVTE_Mask_Type::NVTE_CAUSAL_BOTTOM_RIGHT_MASK,
            NVTE_Softmax_Type::NVTE_VANILLA_SOFTMAX,
            attn_scale, 0.0f, 0, false,
            -1, -1, 0,
            workspace.data(), stream);
    }

    // Throughput: Q+K+V reads + O write, each is batch*seq*heads*hdim*2 bytes (bf16)
    size_t tensor_bytes = batch * seq_len * num_heads * head_dim * 2;
    bench::SetThroughputCounters(state, 4 * tensor_bytes);
    state.counters["batch"] = batch;
    state.counters["heads"] = num_heads;
    state.counters["seq_len"] = seq_len;
    state.counters["head_dim"] = head_dim;
}

// GPU-bound sizes only (attention is always GPU-bound for meaningful sizes)
BENCHMARK(BM_nvte_fused_attn_fwd)
    ->Args({2, 16, 512, 64})
    ->Args({2, 16, 2048, 128})
    ->Args({2, 32, 2048, 128})
    ->Args({2, 32, 8192, 128})
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

}  // namespace

/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <benchmark/benchmark.h>
#include <cuda_runtime.h>

#include <transformer_engine/fused_router.h>
#include "../bench_common.h"

using namespace transformer_engine;
using namespace test;

namespace {

// ---------------------------------------------------------------------------
// Fused topk with score function - forward (softmax, fp32)
// ---------------------------------------------------------------------------
static void BM_FusedTopkForward(benchmark::State &state) {
  const int num_tokens = static_cast<int>(state.range(0));
  const int num_experts = static_cast<int>(state.range(1));
  const int topk = static_cast<int>(state.range(2));

  Tensor logits("logits",
                {static_cast<size_t>(num_tokens), static_cast<size_t>(num_experts)},
                DType::kFloat32);
  Tensor probs("probs",
               {static_cast<size_t>(num_tokens), static_cast<size_t>(num_experts)},
               DType::kFloat32);
  Tensor routing_map("routing_map",
                     {static_cast<size_t>(num_tokens), static_cast<size_t>(num_experts)},
                     DType::kByte);
  Tensor intermediate_output("intermediate_output",
                             {static_cast<size_t>(num_tokens),
                              static_cast<size_t>(num_experts)},
                             DType::kFloat32);

  // Empty expert_bias tensor (not used with softmax score function).
  TensorWrapper expert_bias;

  fillUniform(&logits);

  // score_function: 1 = softmax
  const int score_function = 1;
  const int use_pre_softmax = 1;
  const int num_groups = -1;        // no grouped topk
  const int group_topk = -1;        // no grouped topk
  const float scaling_factor = 1.0f;

  cudaStream_t stream = nullptr;

  // Warmup
  nvte_fused_topk_with_score_function_forward(
      logits.data(), num_tokens, num_experts, topk, use_pre_softmax,
      num_groups, group_topk, scaling_factor, score_function,
      expert_bias.data(), probs.data(), routing_map.data(),
      intermediate_output.data(), stream);
  cudaStreamSynchronize(stream);

  for (auto _ : state) {
    bench::CudaEventTimer timer(state, stream);
    nvte_fused_topk_with_score_function_forward(
        logits.data(), num_tokens, num_experts, topk, use_pre_softmax,
        num_groups, group_topk, scaling_factor, score_function,
        expert_bias.data(), probs.data(), routing_map.data(),
        intermediate_output.data(), stream);
  }

  // Read logits, write probs + routing_map + intermediate_output.
  const size_t logits_bytes = static_cast<size_t>(num_tokens) * num_experts * sizeof(float);
  const size_t probs_bytes = static_cast<size_t>(num_tokens) * num_experts * sizeof(float);
  const size_t routing_bytes = static_cast<size_t>(num_tokens) * num_experts * 1;
  const size_t intermediate_bytes = static_cast<size_t>(num_tokens) * num_experts * sizeof(float);
  bench::SetThroughputCounters(state,
                               logits_bytes + probs_bytes + routing_bytes + intermediate_bytes);
}

// ---------------------------------------------------------------------------
// Fused topk with score function - backward (softmax, fp32)
// ---------------------------------------------------------------------------
static void BM_FusedTopkBackward(benchmark::State &state) {
  const int num_tokens = static_cast<int>(state.range(0));
  const int num_experts = static_cast<int>(state.range(1));
  const int topk = static_cast<int>(state.range(2));

  Tensor routing_map("routing_map",
                     {static_cast<size_t>(num_tokens), static_cast<size_t>(num_experts)},
                     DType::kByte);
  Tensor intermediate_output("intermediate_output",
                             {static_cast<size_t>(num_tokens),
                              static_cast<size_t>(num_experts)},
                             DType::kFloat32);
  Tensor grad_probs("grad_probs",
                    {static_cast<size_t>(num_tokens), static_cast<size_t>(num_experts)},
                    DType::kFloat32);
  Tensor grad_logits("grad_logits",
                     {static_cast<size_t>(num_tokens), static_cast<size_t>(num_experts)},
                     DType::kFloat32);

  fillUniform(&intermediate_output);
  fillUniform(&grad_probs);

  const int score_function = 1;  // softmax
  const int use_pre_softmax = 1;
  const float scaling_factor = 1.0f;

  cudaStream_t stream = nullptr;

  // Warmup
  nvte_fused_topk_with_score_function_backward(
      routing_map.data(), intermediate_output.data(), grad_probs.data(),
      num_tokens, num_experts, topk, use_pre_softmax,
      scaling_factor, score_function, grad_logits.data(), stream);
  cudaStreamSynchronize(stream);

  for (auto _ : state) {
    bench::CudaEventTimer timer(state, stream);
    nvte_fused_topk_with_score_function_backward(
        routing_map.data(), intermediate_output.data(), grad_probs.data(),
        num_tokens, num_experts, topk, use_pre_softmax,
        scaling_factor, score_function, grad_logits.data(), stream);
  }

  const size_t routing_bytes = static_cast<size_t>(num_tokens) * num_experts * 1;
  const size_t intermediate_bytes = static_cast<size_t>(num_tokens) * num_experts * sizeof(float);
  const size_t grad_probs_bytes = static_cast<size_t>(num_tokens) * num_experts * sizeof(float);
  const size_t grad_logits_bytes = static_cast<size_t>(num_tokens) * num_experts * sizeof(float);
  bench::SetThroughputCounters(state,
                               routing_bytes + intermediate_bytes +
                               grad_probs_bytes + grad_logits_bytes);
}

// Small sizes: (num_tokens, num_experts, topk)
BENCHMARK(BM_FusedTopkForward)
    ->Args({32, 8, 2})
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

// Large sizes
BENCHMARK(BM_FusedTopkForward)
    ->Args({4096, 64, 2})
    ->Args({16384, 64, 4})
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

// Small sizes
BENCHMARK(BM_FusedTopkBackward)
    ->Args({32, 8, 2})
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

// Large sizes
BENCHMARK(BM_FusedTopkBackward)
    ->Args({4096, 64, 2})
    ->Args({16384, 64, 4})
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

}  // namespace

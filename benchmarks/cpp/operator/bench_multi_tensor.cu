/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <benchmark/benchmark.h>
#include <cuda_runtime.h>

#include <transformer_engine/multi_tensor.h>
#include "../bench_common.h"

using namespace transformer_engine;
using namespace test;

namespace {

// ---------------------------------------------------------------------------
// Multi-tensor L2 norm (fp32)
// ---------------------------------------------------------------------------
// Create N tensors of a given size, pack them into the tensor_lists format,
// and benchmark the L2 norm computation.
static void BM_MultiTensorL2Norm(benchmark::State &state) {
  const int num_tensors = static_cast<int>(state.range(0));
  const size_t tensor_size = static_cast<size_t>(state.range(1));
  const int chunk_size = 2048;

  // Create the input tensors.
  std::vector<std::unique_ptr<Tensor>> tensors;
  tensors.reserve(num_tensors);
  for (int i = 0; i < num_tensors; ++i) {
    auto t = std::make_unique<Tensor>(
        "tensor_" + std::to_string(i), std::vector<size_t>{tensor_size},
        DType::kFloat32);
    fillUniform(t.get());
    tensors.push_back(std::move(t));
  }

  // Build the tensor_lists structure: NVTETensor** with shape [1][num_tensors].
  // The L2 norm API expects tensor_lists[0][i] for each input tensor.
  std::vector<NVTETensor> tensor_ptrs(num_tensors);
  for (int i = 0; i < num_tensors; ++i) {
    tensor_ptrs[i] = tensors[i]->data();
  }
  NVTETensor *tensor_list_row = tensor_ptrs.data();
  NVTETensor *tensor_lists[1] = {tensor_list_row};

  const size_t num_tensor_lists = 1;
  const size_t num_tensors_per_list = static_cast<size_t>(num_tensors);

  // Compute max_chunks_per_tensor.
  int max_chunks_per_tensor = 0;
  for (int i = 0; i < num_tensors; ++i) {
    int chunks = static_cast<int>((tensor_size + chunk_size - 1) / chunk_size);
    if (chunks > max_chunks_per_tensor) max_chunks_per_tensor = chunks;
  }

  // noop_flag: single-element int32 tensor initialized to 0.
  Tensor noop_flag("noop_flag", {1}, DType::kInt32);

  // Scratch space: output tensor of size 320 floats (same as PyTorch wrapper).
  Tensor output("output", {320}, DType::kFloat32);

  // Per-tensor outputs (not used, per_tensor=false).
  Tensor output_per_tensor("output_per_tensor", {1}, DType::kFloat32);
  Tensor ret_per_tensor("ret_per_tensor", {1}, DType::kFloat32);

  // Return value: single float.
  Tensor ret("ret", {1}, DType::kFloat32);

  const int per_tensor = 0;

  cudaStream_t stream = nullptr;

  // Warmup
  nvte_multi_tensor_l2norm_cuda(chunk_size, noop_flag.data(), tensor_lists,
                                num_tensor_lists, num_tensors_per_list,
                                output.data(), output_per_tensor.data(),
                                ret.data(), ret_per_tensor.data(),
                                per_tensor, max_chunks_per_tensor, stream);
  cudaStreamSynchronize(stream);

  for (auto _ : state) {
    bench::CudaEventTimer timer(state, stream);
    nvte_multi_tensor_l2norm_cuda(chunk_size, noop_flag.data(), tensor_lists,
                                  num_tensor_lists, num_tensors_per_list,
                                  output.data(), output_per_tensor.data(),
                                  ret.data(), ret_per_tensor.data(),
                                  per_tensor, max_chunks_per_tensor, stream);
  }

  // Total bytes read: all input tensor data (fp32 = 4 bytes per element).
  const size_t total_bytes = static_cast<size_t>(num_tensors) * tensor_size * sizeof(float);
  bench::SetThroughputCounters(state, total_bytes);
}

// Small sizes: (num_tensors, elements_per_tensor)
BENCHMARK(BM_MultiTensorL2Norm)
    ->Args({4, 1024})
    ->Args({8, 4096})
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

// Large sizes
BENCHMARK(BM_MultiTensorL2Norm)
    ->Args({32, 1048576})    // 32 tensors of 1M elements
    ->Args({64, 4194304})    // 64 tensors of 4M elements
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

}  // namespace

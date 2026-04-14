/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#pragma once

#include <benchmark/benchmark.h>
#include <cuda_runtime.h>
#include <nvToolsExt.h>

#include "test_common.h"

namespace bench {

/**
 * RAII CUDA event timer for use with Google Benchmark's UseManualTime().
 *
 * Usage inside benchmark loop:
 *   for (auto _ : state) {
 *       CudaEventTimer timer(state, stream);
 *       nvte_some_kernel(..., stream);
 *   }
 */
class CudaEventTimer {
 public:
  CudaEventTimer(benchmark::State& state, cudaStream_t stream = 0)
      : state_(state), stream_(stream) {
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
    cudaEventRecord(start_, stream_);
  }

  ~CudaEventTimer() {
    cudaEventRecord(stop_, stream_);
    cudaEventSynchronize(stop_);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start_, stop_);
    state_.SetIterationTime(ms / 1000.0);
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }

 private:
  benchmark::State& state_;
  cudaStream_t stream_;
  cudaEvent_t start_;
  cudaEvent_t stop_;
};

/**
 * Set throughput counters on a benchmark state.
 */
inline void SetThroughputCounters(benchmark::State& state,
                                  size_t total_bytes) {
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                          static_cast<int64_t>(total_bytes));
}

/**
 * NVTX range RAII guard for profiler visibility.
 */
class NvtxRange {
 public:
  explicit NvtxRange(const char* name) { nvtxRangePushA(name); }
  ~NvtxRange() { nvtxRangePop(); }
};

/**
 * Helper to compute element count from a shape vector.
 */
inline size_t num_elements(const std::vector<size_t>& shape) {
  return test::product(shape);
}

/**
 * Compute bytes for a given shape and type.
 */
inline size_t type_size_bits(transformer_engine::DType dtype) {
  return test::typeToNumBits(dtype);
}

}  // namespace bench

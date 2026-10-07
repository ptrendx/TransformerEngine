/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_GEMM_BATCH_INVARIANT_FFI_H_
#define TRANSFORMER_ENGINE_COMMON_GEMM_BATCH_INVARIANT_FFI_H_

// Native GEMM dispatch remains optional in builds without TVM FFI headers.
#if __has_include(<tvm/ffi/function.h>)
#include <cuda_runtime_api.h>

#include <cstdint>
#include <string>

#include "common/tvm_ffi_bridge.h"

namespace transformer_engine::batch_invariant_ffi {

inline tvm_ffi_bridge::TVMFFIKernel get_batch_invariant_ffi(const std::string& name) {
  static auto& cache = tvm_ffi_bridge::TVMFFIConfigCache::create();
  auto kernel = cache.get_registered(name);
  NVTE_CHECK(kernel.has_value(), "TVM FFI kernel is not registered: ", name);
  return *kernel;
}

inline void launch_batch_invariant_ffi(const tvm_ffi_bridge::TVMFFIKernel& kernel, void* a, void* b,
                                       void* c, int64_t m, int64_t n, int64_t k, int64_t a_stride,
                                       int device, cudaStream_t stream) {
  const NVTEBasicTensor input{
      a, kNVTEBFloat16, {{static_cast<size_t>(m), static_cast<size_t>(k)}, 2}};
  const NVTEBasicTensor weight{
      b, kNVTEBFloat16, {{static_cast<size_t>(n), static_cast<size_t>(k)}, 2}};
  const NVTEBasicTensor output{
      c, kNVTEBFloat16, {{static_cast<size_t>(m), static_cast<size_t>(n)}, 2}};
  const int64_t strides[2] = {a_stride, 1};
  tvm_ffi_bridge::DLTensorWrapper mA(input, false, device, strides);
  tvm_ffi_bridge::DLTensorWrapper mB(weight, false, device);
  tvm_ffi_bridge::DLTensorWrapper mC(output, false, device);
  auto result = kernel(&mA, &mB, &mC, reinterpret_cast<int64_t>(stream));
  NVTE_CHECK(result.type_index() == kTVMFFINone, "Unexpected GEMM return value");
}

}  // namespace transformer_engine::batch_invariant_ffi
#endif
#endif  // TRANSFORMER_ENGINE_COMMON_GEMM_BATCH_INVARIANT_FFI_H_

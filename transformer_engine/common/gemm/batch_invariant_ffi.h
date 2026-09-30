/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_GEMM_BATCH_INVARIANT_FFI_H_
#define TRANSFORMER_ENGINE_COMMON_GEMM_BATCH_INVARIANT_FFI_H_

// Native GEMM dispatch remains optional in builds without TVM FFI headers.
#if __has_include(<tvm/ffi/c_api.h>)
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <tvm/ffi/c_api.h>

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "common/util/cutedsl_launch.h"
#include "common/util/logging.h"

namespace transformer_engine::batch_invariant_ffi {

struct BatchInvariantFFI {
  using GetGlobal = int (*)(const TVMFFIByteArray*, TVMFFIObjectHandle*);
  using Call = int (*)(TVMFFIObjectHandle, TVMFFIAny*, int32_t, TVMFFIAny*);
  using DecRef = int (*)(TVMFFIObjectHandle);
  using MoveError = void (*)(TVMFFIObjectHandle*);
  TVMFFIObjectHandle function = nullptr;
  Call call;
  DecRef dec_ref;
  MoveError move_error;
  std::once_flag first_launch;

  explicit BatchInvariantFFI(const std::string& name) {
    static void* library = dlopen("libtvm_ffi.so", RTLD_NOW | RTLD_GLOBAL);
    NVTE_CHECK(library != nullptr, "Could not load libtvm_ffi.so");
    auto get_global = reinterpret_cast<GetGlobal>(dlsym(library, "TVMFFIFunctionGetGlobal"));
    call = reinterpret_cast<Call>(dlsym(library, "TVMFFIFunctionCall"));
    dec_ref = reinterpret_cast<DecRef>(dlsym(library, "TVMFFIObjectDecRef"));
    move_error = reinterpret_cast<MoveError>(dlsym(library, "TVMFFIErrorMoveFromRaised"));
    NVTE_CHECK(get_global && call && dec_ref && move_error, "Missing TVM FFI symbols");
    TVMFFIByteArray key{name.data(), name.size()};
    check(get_global(&key, &function));
    NVTE_CHECK(function != nullptr, "TVM FFI kernel is not registered: ", name);
  }

  ~BatchInvariantFFI() {
    if (function) dec_ref(function);
  }

  void check(int status) const {
    if (status == 0) return;
    TVMFFIObjectHandle error = nullptr;
    move_error(&error);
    std::string message = "TVM FFI call failed";
    if (error) {
      auto cell = reinterpret_cast<TVMFFIErrorCell*>(static_cast<TVMFFIObject*>(error) + 1);
      message.assign(cell->message.data, cell->message.size);
      dec_ref(error);
    }
    NVTE_ERROR(message);
  }
};

inline std::shared_ptr<BatchInvariantFFI> get_batch_invariant_ffi(const std::string& name) {
  static std::mutex mutex;
  static std::unordered_map<std::string, std::shared_ptr<BatchInvariantFFI>> kernels;
  std::lock_guard<std::mutex> lock(mutex);
  auto& kernel = kernels[name];
  if (!kernel) kernel = std::make_shared<BatchInvariantFFI>(name);
  return kernel;
}

inline void launch_batch_invariant_ffi(BatchInvariantFFI& kernel, void* a, void* b, void* c,
                                       int64_t m, int64_t n, int64_t k, int64_t a_stride,
                                       int device, cudaStream_t stream) {
  int64_t shapes[3][2] = {{m, k}, {n, k}, {m, n}};
  int64_t strides[3][2] = {{a_stride, 1}, {k, 1}, {n, 1}};
  void* pointers[3] = {a, b, c};
  DLTensor tensors[3] = {};
  TVMFFIAny args[4] = {};
  for (int i = 0; i < 3; ++i) {
    tensors[i] = {pointers[i], {kDLCUDA, device}, 2, {kDLBfloat, 16, 1}, shapes[i], strides[i], 0};
    args[i].type_index = kTVMFFIDLTensorPtr;
    args[i].v_ptr = &tensors[i];
  }
  args[3].type_index = kTVMFFIInt;
  args[3].v_int64 = reinterpret_cast<int64_t>(stream);
  auto launch = [&] {
    TVMFFIAny result = {};
    kernel.check(kernel.call(kernel.function, args, 4, &result));
    NVTE_CHECK(result.type_index == kTVMFFINone, "Unexpected GEMM return value");
  };
  // Serial initial entry, then no extra launch locking in steady state.
  bool launched = false;
  std::call_once(kernel.first_launch, [&] {
    std::lock_guard<std::mutex> lock(tvm_ffi_bridge::first_cutedsl_launch_mutex());
    launch();
    launched = true;
  });
  if (!launched) launch();
}

}  // namespace transformer_engine::batch_invariant_ffi
#endif
#endif  // TRANSFORMER_ENGINE_COMMON_GEMM_BATCH_INVARIANT_FFI_H_

/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_UTIL_CUTEDSL_LAUNCH_H_
#define TRANSFORMER_ENGINE_COMMON_UTIL_CUTEDSL_LAUNCH_H_

#include <mutex>

namespace transformer_engine::tvm_ffi_bridge {

// CuTeDSL entrypoints must serialize their initial launch, including across
// different operations and compiled configurations.
// The definition lives in the common library so framework and common dispatch
// share the same mutex even when they are in different shared objects.
std::mutex& first_cutedsl_launch_mutex();

}  // namespace transformer_engine::tvm_ffi_bridge

#endif  // TRANSFORMER_ENGINE_COMMON_UTIL_CUTEDSL_LAUNCH_H_

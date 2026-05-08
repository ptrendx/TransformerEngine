/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file group_quantize_fp8.cuh
 *  \brief CUDA kernels to quantize grouped tensors to FP8 tensor scaling.
 */

#ifndef TRANSFORMER_ENGINE_GROUP_QUANTIZE_FP8_CUH_
#define TRANSFORMER_ENGINE_GROUP_QUANTIZE_FP8_CUH_

#include <cuda_runtime.h>
#include <transformer_engine/transformer_engine.h>

#include <cstddef>
#include <cstdint>
#include <limits>

#include "../../common.h"
#include "../../recipe/recipe_common.cuh"
#include "../../util/math.h"
#include "../../utils.cuh"

namespace transformer_engine {
namespace dispatch {
namespace fp8 {
namespace group_quantize_kernel {

constexpr int THREADS_PER_BLOCK = 256;
constexpr int TARGET_BYTES_PER_THREAD = 16;

template <typename IType>
struct ElemsPerThread {
  static constexpr int value =
      (TARGET_BYTES_PER_THREAD / static_cast<int>(sizeof(IType)) > 0)
          ? TARGET_BYTES_PER_THREAD / static_cast<int>(sizeof(IType))
          : 1;
};

__global__ void initialize_current_scaling_metadata_kernel(float *amax, float *scale,
                                                           float *scale_inv,
                                                           const size_t num_tensors,
                                                           const float *noop) {
  if (noop != nullptr && noop[0] == 1.0f) {
    return;
  }

  const size_t tensor_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (tensor_id >= num_tensors) {
    return;
  }

  amax[tensor_id] = 0.0f;
  scale[tensor_id] = 1.0f;
  scale_inv[tensor_id] = 1.0f;
}

__device__ __forceinline__ size_t tensor_start_offset(
    const size_t tensor_id, const int64_t *const __restrict__ offsets,
    const size_t uniform_tensor_elements) {
  return (offsets != nullptr) ? static_cast<size_t>(offsets[tensor_id])
                              : tensor_id * uniform_tensor_elements;
}

__device__ __forceinline__ size_t tensor_payload_elements(
    const size_t tensor_id, const int64_t *const __restrict__ first_dims, const size_t uniform_rows,
    const size_t last_dim) {
  const size_t rows =
      (first_dims != nullptr) ? static_cast<size_t>(first_dims[tensor_id]) : uniform_rows;
  return rows * last_dim;
}

template <bool IS_ACT, typename ParamOP, float (*OP)(float, const ParamOP &), typename IType,
          int ELEMS_PER_THREAD>
__global__ void grouped_amax_kernel(const IType *const __restrict__ input,
                                    float *const __restrict__ amax,
                                    const size_t uniform_tensor_elements,
                                    const size_t uniform_rows, const size_t last_dim,
                                    const int64_t *const __restrict__ offsets,
                                    const int64_t *const __restrict__ first_dims,
                                    const float *noop) {
  if (noop != nullptr && noop[0] == 1.0f) {
    return;
  }

  const size_t tensor_id = blockIdx.y;
  const size_t local_base =
      (blockIdx.x * blockDim.x + threadIdx.x) * static_cast<size_t>(ELEMS_PER_THREAD);
  const size_t tensor_elements =
      tensor_payload_elements(tensor_id, first_dims, uniform_rows, last_dim);
  if (local_base >= tensor_elements) {
    return;
  }

  const size_t tensor_start = tensor_start_offset(tensor_id, offsets, uniform_tensor_elements);
  float thread_amax = 0.0f;

#pragma unroll
  for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
    const size_t local_idx = local_base + static_cast<size_t>(i);
    if (local_idx < tensor_elements) {
      float elt = static_cast<float>(input[tensor_start + local_idx]);
      if constexpr (IS_ACT) {
        elt = OP(elt, {});
      }
      __builtin_assume(thread_amax >= 0.0f);
      thread_amax = fmaxf(thread_amax, fabsf(elt));
    }
  }

  const int warp_id = threadIdx.x / THREADS_PER_WARP;
  thread_amax = reduce_max<THREADS_PER_BLOCK / THREADS_PER_WARP>(thread_amax, warp_id);
  if (threadIdx.x == 0) {
    atomicMaxFloat(amax + tensor_id, thread_amax);
  }
}

template <typename OType>
__global__ void compute_grouped_scale_kernel(const float *const __restrict__ amax,
                                             float *const __restrict__ scale,
                                             float *const __restrict__ scale_inv,
                                             const size_t num_tensors,
                                             const bool force_pow_2_scales,
                                             const float epsilon, const float *noop) {
  if (noop != nullptr && noop[0] == 1.0f) {
    return;
  }

  const size_t tensor_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (tensor_id >= num_tensors) {
    return;
  }

  const float scale_val = compute_scale_from_amax(
      amax[tensor_id], Quantized_Limits<OType>::max_norm, force_pow_2_scales, epsilon,
      std::numeric_limits<float>::max());
  scale[tensor_id] = scale_val;
  reciprocal<float>(scale_inv + tensor_id, scale_val);
}

template <bool IS_ACT, typename ParamOP, float (*OP)(float, const ParamOP &), typename IType,
          typename OType, int ELEMS_PER_THREAD>
__global__ void grouped_cast_fp8_kernel(const IType *const __restrict__ input,
                                        OType *const __restrict__ output,
                                        const float *const __restrict__ scale,
                                        const size_t uniform_tensor_elements,
                                        const size_t uniform_rows, const size_t last_dim,
                                        const int64_t *const __restrict__ offsets,
                                        const int64_t *const __restrict__ first_dims,
                                        const float *noop) {
  if (noop != nullptr && noop[0] == 1.0f) {
    return;
  }

  const size_t tensor_id = blockIdx.y;
  const size_t local_base =
      (blockIdx.x * blockDim.x + threadIdx.x) * static_cast<size_t>(ELEMS_PER_THREAD);
  const size_t tensor_elements =
      tensor_payload_elements(tensor_id, first_dims, uniform_rows, last_dim);
  if (local_base >= tensor_elements) {
    return;
  }

  const size_t tensor_start = tensor_start_offset(tensor_id, offsets, uniform_tensor_elements);
  const float scale_val = scale[tensor_id];

#pragma unroll
  for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
    const size_t local_idx = local_base + static_cast<size_t>(i);
    if (local_idx < tensor_elements) {
      float elt = static_cast<float>(input[tensor_start + local_idx]);
      if constexpr (IS_ACT) {
        elt = OP(elt, {});
      }
      output[tensor_start + local_idx] = static_cast<OType>(elt * scale_val);
    }
  }
}

}  // namespace group_quantize_kernel

template <bool IS_ACT, typename ParamOP, float (*OP)(float, const ParamOP &)>
void group_quantize(const GroupedTensor *input, const Tensor *noop, GroupedTensor *output,
                    const QuantizationConfig *quant_config, cudaStream_t stream) {
  using namespace group_quantize_kernel;

  checkCuDriverContext(stream);
  CheckNoopTensor(*noop, "cast_noop");
  CheckInputGroupedTensor(*input, "group_fp8_quantize_input");
  CheckOutputGroupedTensor(*output, "group_fp8_quantize_output", false);

  NVTE_CHECK(input->num_tensors == output->num_tensors,
             "Number of input and output tensors must be same.");
  NVTE_CHECK(input->has_data(), "Cannot quantize grouped tensor without rowwise data.");
  NVTE_CHECK(output->has_data(), "Grouped FP8 current scaling requires rowwise output data.");
  NVTE_CHECK(!output->has_columnwise_data(),
             "Grouped FP8 current scaling does not support columnwise output yet.");
  NVTE_CHECK(output->all_same_last_dim(),
             "Grouped FP8 current scaling requires a common last dimension.");
  NVTE_CHECK(input->logical_shape.ndim == 2 && output->logical_shape.ndim == 2,
             "Grouped FP8 current scaling requires 2D logical shapes.");
  NVTE_CHECK(input->logical_shape.data[0] == output->logical_shape.data[0] &&
                 input->logical_shape.data[1] == output->logical_shape.data[1],
             "Input and output grouped logical shapes must match.");
  NVTE_CHECK(is_fp8_dtype(output->dtype()), "Output must have FP8 type.");
  NVTE_CHECK(!is_fp8_dtype(input->dtype()), "Input must be in higher precision.");

  const size_t num_tensors = output->num_tensors;
  const size_t first_logical_dim = output->logical_shape.data[0];
  const size_t last_logical_dim = output->logical_shape.data[1];
  const size_t total_elements = first_logical_dim * last_logical_dim;

  NVTE_CHECK(input->data.numel() == total_elements,
             "Input grouped data size must match the grouped logical shape.");
  NVTE_CHECK(output->data.numel() == total_elements,
             "Output grouped data size must match the grouped logical shape.");
  NVTE_CHECK(output->amax.dptr != nullptr && output->amax.dtype == DType::kFloat32 &&
                 output->amax.numel() == num_tensors,
             "Grouped FP8 current scaling requires one FP32 amax per tensor.");
  NVTE_CHECK(output->scale.dptr != nullptr && output->scale.dtype == DType::kFloat32 &&
                 output->scale.numel() == num_tensors,
             "Grouped FP8 current scaling requires one FP32 scale per tensor.");
  NVTE_CHECK(output->scale_inv.dptr != nullptr && output->scale_inv.dtype == DType::kFloat32 &&
                 output->scale_inv.numel() == num_tensors,
             "Grouped FP8 current scaling requires one FP32 scale_inv per tensor.");

  const bool varying_first_dim = output->first_dims.has_data();
  NVTE_CHECK(!varying_first_dim || output->tensor_offsets.has_data(),
             "Grouped FP8 current scaling requires tensor_offsets when first_dims is set.");
  NVTE_CHECK(varying_first_dim || first_logical_dim % num_tensors == 0,
             "Uniform grouped FP8 current scaling requires first dimension divisible by "
             "num_tensors.");

  const size_t uniform_rows = varying_first_dim ? 0 : first_logical_dim / num_tensors;
  const size_t uniform_tensor_elements = uniform_rows * last_logical_dim;
  const size_t max_tensor_elements =
      varying_first_dim ? total_elements : uniform_tensor_elements;
  const int64_t *const offsets_ptr =
      varying_first_dim ? reinterpret_cast<const int64_t *>(output->tensor_offsets.dptr) : nullptr;
  const int64_t *const first_dims_ptr =
      varying_first_dim ? reinterpret_cast<const int64_t *>(output->first_dims.dptr) : nullptr;
  const float *const noop_ptr = reinterpret_cast<const float *>(noop->data.dptr);
  float *const amax_ptr = reinterpret_cast<float *>(output->amax.dptr);
  float *const scale_ptr = reinterpret_cast<float *>(output->scale.dptr);
  float *const scale_inv_ptr = reinterpret_cast<float *>(output->scale_inv.dptr);

  const bool force_pow_2_scales = quant_config != nullptr && quant_config->force_pow_2_scales;
  const float amax_epsilon = quant_config != nullptr ? quant_config->amax_epsilon : 0.0f;

  const dim3 metadata_grid(DIVUP(num_tensors, static_cast<size_t>(THREADS_PER_BLOCK)));
  initialize_current_scaling_metadata_kernel<<<metadata_grid, THREADS_PER_BLOCK, 0, stream>>>(
      amax_ptr, scale_ptr, scale_inv_ptr, num_tensors, noop_ptr);
  NVTE_CHECK_CUDA(cudaGetLastError());

  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      input->dtype(), IType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
          output->dtype(), OType,
          constexpr int elems_per_thread = ElemsPerThread<IType>::value;
          const dim3 grid(
              DIVUP(max_tensor_elements,
                    static_cast<size_t>(THREADS_PER_BLOCK * elems_per_thread)),
              num_tensors);
          grouped_amax_kernel<IS_ACT, ParamOP, OP, IType, elems_per_thread>
          <<<grid, THREADS_PER_BLOCK, 0, stream>>>(
              reinterpret_cast<const IType *>(input->data.dptr), amax_ptr,
              uniform_tensor_elements, uniform_rows, last_logical_dim, offsets_ptr, first_dims_ptr,
              noop_ptr);
          NVTE_CHECK_CUDA(cudaGetLastError());

          compute_grouped_scale_kernel<OType>
          <<<metadata_grid, THREADS_PER_BLOCK, 0, stream>>>(
              amax_ptr, scale_ptr, scale_inv_ptr, num_tensors, force_pow_2_scales, amax_epsilon,
              noop_ptr);
          NVTE_CHECK_CUDA(cudaGetLastError());

          grouped_cast_fp8_kernel<IS_ACT, ParamOP, OP, IType, OType, elems_per_thread>
          <<<grid, THREADS_PER_BLOCK, 0, stream>>>(
              reinterpret_cast<const IType *>(input->data.dptr),
              reinterpret_cast<OType *>(output->data.dptr), scale_ptr, uniform_tensor_elements,
              uniform_rows, last_logical_dim, offsets_ptr, first_dims_ptr, noop_ptr);
          NVTE_CHECK_CUDA(cudaGetLastError()););  // NOLINT(*)
  );                                             // NOLINT(*)
}

}  // namespace fp8
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_GROUP_QUANTIZE_FP8_CUH_

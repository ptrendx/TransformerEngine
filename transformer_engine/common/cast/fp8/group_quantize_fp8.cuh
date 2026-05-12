/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file group_quantize_fp8.cuh
 *  \brief CUDA kernels to quantize grouped tensors to FP8 with per-tensor current scaling.
 */

#ifndef TRANSFORMER_ENGINE_GROUP_QUANTIZE_FP8_CUH_
#define TRANSFORMER_ENGINE_GROUP_QUANTIZE_FP8_CUH_

#include <cuda.h>
#include <cuda_runtime.h>
#include <transformer_engine/transformer_engine.h>

#include <cfloat>
#include <cstddef>
#include <type_traits>

#include "../../common.h"
#include "../../recipe/recipe_common.cuh"
#include "../../util/math.h"
#include "../../util/ptx.cuh"
#include "../../utils.cuh"
#include "../core/common.cuh"

namespace transformer_engine {
namespace dispatch {
namespace fp8 {
namespace group_quantize_current_scaling {

namespace kernel {

constexpr size_t THREADS_PER_BLOCK = 256;
constexpr size_t ELEMENTS_PER_THREAD = 256;
constexpr size_t ELEMENTS_PER_BLOCK = THREADS_PER_BLOCK * ELEMENTS_PER_THREAD;
constexpr size_t VECTOR_LOAD_BYTES = 32;

template <ShapeRepresentation SHAPE_REP>
__device__ __forceinline__ size_t get_actual_elements(
    const size_t first_logical_dim, const size_t last_logical_dim,
    const int64_t *const __restrict__ offsets_ptr, const size_t num_tensors) {
  if constexpr (SHAPE_REP == ShapeRepresentation::SAME_BOTH_DIMS) {
    return first_logical_dim * last_logical_dim;
  } else {
    return static_cast<size_t>(offsets_ptr[num_tensors]);
  }
}

template <ShapeRepresentation SHAPE_REP>
__device__ __forceinline__ size_t get_tensor_numel(
    const size_t tensor_id, const size_t first_logical_dim, const size_t last_logical_dim,
    const int64_t *const __restrict__ first_dims_ptr, const size_t num_tensors) {
  if constexpr (SHAPE_REP == ShapeRepresentation::SAME_BOTH_DIMS) {
    return (first_logical_dim / num_tensors) * last_logical_dim;
  } else {
    return static_cast<size_t>(first_dims_ptr[tensor_id]) * last_logical_dim;
  }
}

template <ShapeRepresentation SHAPE_REP>
__device__ __forceinline__ size_t get_tensor_start(
    const size_t tensor_id, const size_t first_logical_dim, const size_t last_logical_dim,
    const int64_t *const __restrict__ offsets_ptr, const size_t num_tensors) {
  if constexpr (SHAPE_REP == ShapeRepresentation::SAME_BOTH_DIMS) {
    return tensor_id * (first_logical_dim / num_tensors) * last_logical_dim;
  } else {
    return static_cast<size_t>(offsets_ptr[tensor_id]);
  }
}

template <ShapeRepresentation SHAPE_REP>
__device__ __forceinline__ size_t get_tensor_id(
    const size_t offset, const size_t first_logical_dim, const size_t last_logical_dim,
    const int64_t *const __restrict__ offsets_ptr, const size_t num_tensors) {
  if constexpr (SHAPE_REP == ShapeRepresentation::SAME_BOTH_DIMS) {
    const size_t elems_per_tensor = (first_logical_dim / num_tensors) * last_logical_dim;
    if (elems_per_tensor == 0) {
      return 0;
    }
    const size_t tensor_id = offset / elems_per_tensor;
    return (tensor_id < num_tensors) ? tensor_id : num_tensors - 1;
  } else {
    size_t low = 1;
    size_t high = num_tensors;
    while (low < high) {
      const size_t mid = low + (high - low) / 2;
      const size_t mid_offset = static_cast<size_t>(offsets_ptr[mid]);
      if (mid_offset <= offset) {
        low = mid + 1;
      } else {
        high = mid;
      }
    }
    return low - 1;
  }
}

template <ShapeRepresentation SHAPE_REP>
__device__ __forceinline__ bool block_is_inside_one_tensor(
    const size_t block_start, const size_t block_end, const size_t tensor_id,
    const size_t first_logical_dim, const size_t last_logical_dim,
    const int64_t *const __restrict__ offsets_ptr,
    const int64_t *const __restrict__ first_dims_ptr, const size_t num_tensors) {
  const size_t tensor_start =
      get_tensor_start<SHAPE_REP>(tensor_id, first_logical_dim, last_logical_dim, offsets_ptr,
                                  num_tensors);
  const size_t tensor_end =
      tensor_start + get_tensor_numel<SHAPE_REP>(tensor_id, first_logical_dim, last_logical_dim,
                                                 first_dims_ptr, num_tensors);
  return block_start >= tensor_start && block_end <= tensor_end;
}

__device__ __forceinline__ void update_amax(float *amax_ptr, const size_t tensor_id,
                                            const float value) {
  if (value > 0.0f) {
    atomicMaxFloat(&amax_ptr[tensor_id], value);
  }
}

template <typename IType, typename OType>
__device__ __forceinline__ void cast_4_scaled(const IType *input, OType *output,
                                              const float scale, const size_t count) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  if (count == 4) {
    const ptx::floatx4 scale4 = {scale, scale, scale, scale};
    if constexpr (std::is_same_v<IType, bf16> && std::is_same_v<OType, fp8e4m3>) {
      const ptx::bf16x4 input4 = {input[0], input[1], input[2], input[3]};
      ptx::fp8e4m3x4 output4;
      ptx::mul_cvt_4x(output4, input4, scale4);
      output[0] = output4.x1;
      output[1] = output4.x2;
      output[2] = output4.x3;
      output[3] = output4.x4;
      return;
    } else if constexpr (std::is_same_v<IType, bf16> && std::is_same_v<OType, fp8e5m2>) {
      const ptx::bf16x4 input4 = {input[0], input[1], input[2], input[3]};
      ptx::fp8e5m2x4 output4;
      ptx::mul_cvt_4x(output4, input4, scale4);
      output[0] = output4.x1;
      output[1] = output4.x2;
      output[2] = output4.x3;
      output[3] = output4.x4;
      return;
    } else if constexpr (std::is_same_v<IType, fp16> && std::is_same_v<OType, fp8e4m3>) {
      const ptx::fp16x4 input4 = {input[0], input[1], input[2], input[3]};
      ptx::fp8e4m3x4 output4;
      ptx::mul_cvt_4x(output4, input4, scale4);
      output[0] = output4.x1;
      output[1] = output4.x2;
      output[2] = output4.x3;
      output[3] = output4.x4;
      return;
    } else if constexpr (std::is_same_v<IType, fp16> && std::is_same_v<OType, fp8e5m2>) {
      const ptx::fp16x4 input4 = {input[0], input[1], input[2], input[3]};
      ptx::fp8e5m2x4 output4;
      ptx::mul_cvt_4x(output4, input4, scale4);
      output[0] = output4.x1;
      output[1] = output4.x2;
      output[2] = output4.x3;
      output[3] = output4.x4;
      return;
    } else if constexpr (std::is_same_v<IType, float> && std::is_same_v<OType, fp8e4m3>) {
      const ptx::floatx4 input4 = {input[0], input[1], input[2], input[3]};
      ptx::fp8e4m3x4 output4;
      ptx::mul_cvt_4x(output4, input4, scale4);
      output[0] = output4.x1;
      output[1] = output4.x2;
      output[2] = output4.x3;
      output[3] = output4.x4;
      return;
    } else if constexpr (std::is_same_v<IType, float> && std::is_same_v<OType, fp8e5m2>) {
      const ptx::floatx4 input4 = {input[0], input[1], input[2], input[3]};
      ptx::fp8e5m2x4 output4;
      ptx::mul_cvt_4x(output4, input4, scale4);
      output[0] = output4.x1;
      output[1] = output4.x2;
      output[2] = output4.x3;
      output[3] = output4.x4;
      return;
    }
  }
#endif
#pragma unroll
  for (size_t elem = 0; elem < 4; ++elem) {
    if (elem < count) {
      output[elem] = static_cast<OType>(static_cast<float>(input[elem]) * scale);
    }
  }
}

template <ShapeRepresentation SHAPE_REP>
__global__ void __launch_bounds__(THREADS_PER_BLOCK)
    zero_grouped_amax_kernel(float *const __restrict__ amax_ptr,
                             const float *const __restrict__ noop_ptr,
                             const size_t num_tensors) {
  if (noop_ptr != nullptr && noop_ptr[0] == 1.0f) {
    return;
  }
  const size_t tensor_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (tensor_id < num_tensors) {
    amax_ptr[tensor_id] = 0.0f;
  }
}

template <typename IType, ShapeRepresentation SHAPE_REP>
__global__ void __launch_bounds__(THREADS_PER_BLOCK)
    grouped_amax_kernel(const IType *const __restrict__ input_ptr,
                        float *const __restrict__ amax_ptr,
                        const float *const __restrict__ noop_ptr, const size_t num_tensors,
                        const size_t first_logical_dim, const size_t last_logical_dim,
                        const int64_t *const __restrict__ offsets_ptr,
                        const int64_t *const __restrict__ first_dims_ptr) {
  if (noop_ptr != nullptr && noop_ptr[0] == 1.0f) {
    return;
  }

  const size_t actual_elements =
      get_actual_elements<SHAPE_REP>(first_logical_dim, last_logical_dim, offsets_ptr, num_tensors);
  const size_t block_start = blockIdx.x * ELEMENTS_PER_BLOCK;
  if (block_start >= actual_elements) {
    return;
  }
  const size_t block_limit = block_start + ELEMENTS_PER_BLOCK;
  const size_t block_end = (block_limit < actual_elements) ? block_limit : actual_elements;
  const size_t tensor_id =
      get_tensor_id<SHAPE_REP>(block_start, first_logical_dim, last_logical_dim, offsets_ptr,
                               num_tensors);

  if (block_is_inside_one_tensor<SHAPE_REP>(block_start, block_end, tensor_id, first_logical_dim,
                                            last_logical_dim, offsets_ptr, first_dims_ptr,
                                            num_tensors)) {
    constexpr size_t VEC_SIZE = VECTOR_LOAD_BYTES / sizeof(IType);
    static_assert(VECTOR_LOAD_BYTES % sizeof(IType) == 0);
    IType thread_amax = 0.0f;
    for (size_t vector_offset = threadIdx.x * VEC_SIZE; vector_offset < ELEMENTS_PER_BLOCK;
         vector_offset += blockDim.x * VEC_SIZE) {
      const size_t idx = block_start + vector_offset;
      if (idx >= block_end) {
        break;
      }
      const size_t remaining = block_end - idx;
      const size_t count = (remaining < VEC_SIZE) ? remaining : VEC_SIZE;
      Vec<IType, VEC_SIZE> input_vec;
      input_vec.load_from_elts(input_ptr, idx, count);
#pragma unroll
      for (size_t elem = 0; elem < VEC_SIZE; ++elem) {
        const IType value = input_vec.data.elt[elem];
        __builtin_assume(thread_amax >= IType{0.0f});
        if constexpr (std::is_same_v<IType, __nv_bfloat16>) {
#if __CUDA_ARCH__ >= 800
          thread_amax = __hmax(__habs(value), thread_amax);
#else
          thread_amax = static_cast<__nv_bfloat16>(
              fmaxf(fabsf(static_cast<float>(value)), static_cast<float>(thread_amax)));
#endif
        } else if constexpr (std::is_same_v<IType, __half>) {
          thread_amax = __hmax(__habs(value), thread_amax);
        } else {
          thread_amax = fmaxf(fabsf(value), thread_amax);
        }
      }
    }
    const int warp_id = threadIdx.x / THREADS_PER_WARP;
    const float block_amax =
        reduce_max<THREADS_PER_BLOCK / THREADS_PER_WARP>(thread_amax, warp_id);
    if (threadIdx.x == 0) {
      update_amax(amax_ptr, tensor_id, block_amax);
    }
  } else {
    for (size_t idx = block_start + threadIdx.x; idx < block_end; idx += blockDim.x) {
      const size_t current_tensor_id =
          get_tensor_id<SHAPE_REP>(idx, first_logical_dim, last_logical_dim, offsets_ptr,
                                   num_tensors);
      update_amax(amax_ptr, current_tensor_id, fabsf(static_cast<float>(input_ptr[idx])));
    }
  }
}

template <typename OType, ShapeRepresentation SHAPE_REP>
__global__ void __launch_bounds__(THREADS_PER_BLOCK)
    grouped_scale_kernel(const float *const __restrict__ amax_ptr,
                         float *const __restrict__ scale_ptr,
                         float *const __restrict__ scale_inv_ptr,
                         float *const __restrict__ columnwise_scale_inv_ptr,
                         const float *const __restrict__ noop_ptr, const size_t num_tensors,
                         const bool force_pow_2_scales, const float epsilon) {
  if (noop_ptr != nullptr && noop_ptr[0] == 1.0f) {
    return;
  }
  const size_t tensor_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (tensor_id >= num_tensors) {
    return;
  }

  const float scale =
      compute_scale_from_amax(amax_ptr[tensor_id], Quantized_Limits<OType>::max_norm,
                              force_pow_2_scales, epsilon, FLT_MAX);
  scale_ptr[tensor_id] = scale;
  const float scale_inv = __frcp_rn(scale);
  if (scale_inv_ptr != nullptr) {
    scale_inv_ptr[tensor_id] = scale_inv;
  }
  if (columnwise_scale_inv_ptr != nullptr) {
    columnwise_scale_inv_ptr[tensor_id] = scale_inv;
  }
}

template <typename IType, typename OType, ShapeRepresentation SHAPE_REP>
__device__ __forceinline__ void cast_one_element(
    const IType *const __restrict__ input_ptr, OType *const __restrict__ output_ptr,
    OType *const __restrict__ columnwise_output_ptr, const float *const __restrict__ scale_ptr,
    const size_t idx, const size_t tensor_id, const size_t tensor_start, const size_t tensor_rows,
    const size_t last_logical_dim) {
  const float scaled_value = static_cast<float>(input_ptr[idx]) * scale_ptr[tensor_id];
  const OType output_value = static_cast<OType>(scaled_value);
  if (output_ptr != nullptr) {
    output_ptr[idx] = output_value;
  }
  if (columnwise_output_ptr != nullptr) {
    const size_t local_idx = idx - tensor_start;
    const size_t row = local_idx / last_logical_dim;
    const size_t col = local_idx - row * last_logical_dim;
    columnwise_output_ptr[tensor_start + col * tensor_rows + row] = output_value;
  }
}

template <typename IType, typename OType, ShapeRepresentation SHAPE_REP>
__global__ void __launch_bounds__(THREADS_PER_BLOCK)
    grouped_cast_kernel(const IType *const __restrict__ input_ptr,
                        OType *const __restrict__ output_ptr,
                        OType *const __restrict__ columnwise_output_ptr,
                        const float *const __restrict__ scale_ptr,
                        const float *const __restrict__ noop_ptr, const size_t num_tensors,
                        const size_t first_logical_dim, const size_t last_logical_dim,
                        const int64_t *const __restrict__ offsets_ptr,
                        const int64_t *const __restrict__ first_dims_ptr) {
  if (noop_ptr != nullptr && noop_ptr[0] == 1.0f) {
    return;
  }

  const size_t actual_elements =
      get_actual_elements<SHAPE_REP>(first_logical_dim, last_logical_dim, offsets_ptr, num_tensors);
  const size_t block_start = blockIdx.x * ELEMENTS_PER_BLOCK;
  if (block_start >= actual_elements) {
    return;
  }
  const size_t block_limit = block_start + ELEMENTS_PER_BLOCK;
  const size_t block_end = (block_limit < actual_elements) ? block_limit : actual_elements;
  const size_t tensor_id =
      get_tensor_id<SHAPE_REP>(block_start, first_logical_dim, last_logical_dim, offsets_ptr,
                               num_tensors);
  if (block_is_inside_one_tensor<SHAPE_REP>(block_start, block_end, tensor_id, first_logical_dim,
                                            last_logical_dim, offsets_ptr, first_dims_ptr,
                                            num_tensors)) {
    const size_t tensor_start =
        get_tensor_start<SHAPE_REP>(tensor_id, first_logical_dim, last_logical_dim, offsets_ptr,
                                    num_tensors);
    const size_t tensor_rows =
        get_tensor_numel<SHAPE_REP>(tensor_id, first_logical_dim, last_logical_dim, first_dims_ptr,
                                    num_tensors) /
        last_logical_dim;
    for (size_t idx = block_start + threadIdx.x; idx < block_end; idx += blockDim.x) {
      cast_one_element<IType, OType, SHAPE_REP>(input_ptr, output_ptr, columnwise_output_ptr,
                                                scale_ptr, idx, tensor_id, tensor_start,
                                                tensor_rows, last_logical_dim);
    }
  } else {
    for (size_t idx = block_start + threadIdx.x; idx < block_end; idx += blockDim.x) {
      const size_t current_tensor_id =
          get_tensor_id<SHAPE_REP>(idx, first_logical_dim, last_logical_dim, offsets_ptr,
                                   num_tensors);
      const size_t tensor_start = get_tensor_start<SHAPE_REP>(
          current_tensor_id, first_logical_dim, last_logical_dim, offsets_ptr, num_tensors);
      const size_t tensor_rows = get_tensor_numel<SHAPE_REP>(
                                     current_tensor_id, first_logical_dim, last_logical_dim,
                                     first_dims_ptr, num_tensors) /
                                 last_logical_dim;
      cast_one_element<IType, OType, SHAPE_REP>(input_ptr, output_ptr, columnwise_output_ptr,
                                                scale_ptr, idx, current_tensor_id, tensor_start,
                                                tensor_rows, last_logical_dim);
    }
  }
}

template <typename IType, typename OType, ShapeRepresentation SHAPE_REP>
__global__ void __launch_bounds__(THREADS_PER_BLOCK)
    grouped_rowwise_cast_kernel(const IType *const __restrict__ input_ptr,
                                OType *const __restrict__ output_ptr,
                                const float *const __restrict__ scale_ptr,
                                const float *const __restrict__ noop_ptr,
                                const size_t num_tensors, const size_t first_logical_dim,
                                const size_t last_logical_dim,
                                const int64_t *const __restrict__ offsets_ptr,
                                const int64_t *const __restrict__ first_dims_ptr) {
  if (noop_ptr != nullptr && noop_ptr[0] == 1.0f) {
    return;
  }

  const size_t actual_elements =
      get_actual_elements<SHAPE_REP>(first_logical_dim, last_logical_dim, offsets_ptr, num_tensors);
  const size_t block_start = blockIdx.x * ELEMENTS_PER_BLOCK;
  if (block_start >= actual_elements) {
    return;
  }
  const size_t block_limit = block_start + ELEMENTS_PER_BLOCK;
  const size_t block_end = (block_limit < actual_elements) ? block_limit : actual_elements;
  const size_t tensor_id =
      get_tensor_id<SHAPE_REP>(block_start, first_logical_dim, last_logical_dim, offsets_ptr,
                               num_tensors);

  if (block_is_inside_one_tensor<SHAPE_REP>(block_start, block_end, tensor_id, first_logical_dim,
                                            last_logical_dim, offsets_ptr, first_dims_ptr,
                                            num_tensors)) {
    constexpr size_t VEC_SIZE = VECTOR_LOAD_BYTES / sizeof(IType);
    static_assert(VECTOR_LOAD_BYTES % sizeof(IType) == 0);
    const float scale = scale_ptr[tensor_id];
    for (size_t vector_offset = threadIdx.x * VEC_SIZE; vector_offset < ELEMENTS_PER_BLOCK;
         vector_offset += blockDim.x * VEC_SIZE) {
      const size_t idx = block_start + vector_offset;
      if (idx >= block_end) {
        break;
      }
      const size_t remaining = block_end - idx;
      const size_t count = (remaining < VEC_SIZE) ? remaining : VEC_SIZE;
      Vec<IType, VEC_SIZE> input_vec;
      Vec<OType, VEC_SIZE> output_vec;
      input_vec.load_from_elts(input_ptr, idx, count);
#pragma unroll
      for (size_t elem = 0; elem < VEC_SIZE; elem += 4) {
        const size_t remaining_vec = (count > elem) ? count - elem : 0;
        const size_t cast_count = (remaining_vec < 4) ? remaining_vec : 4;
        if (cast_count > 0) {
          cast_4_scaled<IType, OType>(&input_vec.data.elt[elem], &output_vec.data.elt[elem],
                                      scale, cast_count);
        }
      }
      output_vec.store_to_elts(output_ptr, idx, count);
    }
  } else {
    for (size_t idx = block_start + threadIdx.x; idx < block_end; idx += blockDim.x) {
      const size_t current_tensor_id =
          get_tensor_id<SHAPE_REP>(idx, first_logical_dim, last_logical_dim, offsets_ptr,
                                   num_tensors);
      const float scaled_value = static_cast<float>(input_ptr[idx]) * scale_ptr[current_tensor_id];
      output_ptr[idx] = static_cast<OType>(scaled_value);
    }
  }
}

}  // namespace kernel

template <ShapeRepresentation SHAPE_REP, typename IType, typename OType>
void launch_group_quantize(const GroupedTensor *input, const Tensor *noop, GroupedTensor *output,
                           const QuantizationConfig *quant_config, cudaStream_t stream) {
  using namespace kernel;

  const size_t num_tensors = input->num_tensors;
  const size_t first_logical_dim = input->logical_shape.data[0];
  const size_t last_logical_dim = input->logical_shape.data[1];
  const size_t allocated_elements = first_logical_dim * last_logical_dim;
  const int64_t *const offsets_ptr = reinterpret_cast<const int64_t *>(output->tensor_offsets.dptr);
  const int64_t *const first_dims_ptr = reinterpret_cast<const int64_t *>(output->first_dims.dptr);
  const float *const noop_ptr = reinterpret_cast<const float *>(noop->data.dptr);

  float *const amax_ptr = reinterpret_cast<float *>(output->amax.dptr);
  float *const scale_ptr = reinterpret_cast<float *>(output->scale.dptr);
  float *const scale_inv_ptr = reinterpret_cast<float *>(output->scale_inv.dptr);
  float *const columnwise_scale_inv_ptr =
      reinterpret_cast<float *>(output->columnwise_scale_inv.dptr);

  const IType *const input_ptr = reinterpret_cast<const IType *>(input->data.dptr);
  OType *const output_ptr = reinterpret_cast<OType *>(output->data.dptr);
  OType *const columnwise_output_ptr = reinterpret_cast<OType *>(output->columnwise_data.dptr);

  const dim3 block(THREADS_PER_BLOCK);
  const dim3 tensor_grid(DIVUP(num_tensors, THREADS_PER_BLOCK));
  zero_grouped_amax_kernel<SHAPE_REP><<<tensor_grid, block, 0, stream>>>(amax_ptr, noop_ptr,
                                                                         num_tensors);
  NVTE_CHECK_CUDA(cudaGetLastError());

  if (allocated_elements > 0) {
    const dim3 data_grid(DIVUP(allocated_elements, ELEMENTS_PER_BLOCK));
    grouped_amax_kernel<IType, SHAPE_REP><<<data_grid, block, 0, stream>>>(
        input_ptr, amax_ptr, noop_ptr, num_tensors, first_logical_dim, last_logical_dim,
        offsets_ptr, first_dims_ptr);
    NVTE_CHECK_CUDA(cudaGetLastError());

    const bool force_pow_2_scales =
        (quant_config != nullptr) ? quant_config->force_pow_2_scales : false;
    const float epsilon = (quant_config != nullptr) ? quant_config->amax_epsilon : 0.0f;
    grouped_scale_kernel<OType, SHAPE_REP><<<tensor_grid, block, 0, stream>>>(
        amax_ptr, scale_ptr, scale_inv_ptr, columnwise_scale_inv_ptr, noop_ptr, num_tensors,
        force_pow_2_scales, epsilon);
    NVTE_CHECK_CUDA(cudaGetLastError());

    if (columnwise_output_ptr == nullptr) {
      grouped_rowwise_cast_kernel<IType, OType, SHAPE_REP><<<data_grid, block, 0, stream>>>(
          input_ptr, output_ptr, scale_ptr, noop_ptr, num_tensors, first_logical_dim,
          last_logical_dim, offsets_ptr, first_dims_ptr);
    } else {
      grouped_cast_kernel<IType, OType, SHAPE_REP><<<data_grid, block, 0, stream>>>(
          input_ptr, output_ptr, columnwise_output_ptr, scale_ptr, noop_ptr, num_tensors,
          first_logical_dim, last_logical_dim, offsets_ptr, first_dims_ptr);
    }
    NVTE_CHECK_CUDA(cudaGetLastError());
  }
}

inline ShapeRepresentation get_shape_representation(const GroupedTensor *output) {
  if (output->all_same_shape()) {
    return ShapeRepresentation::SAME_BOTH_DIMS;
  }
  if (output->all_same_last_dim()) {
    return ShapeRepresentation::VARYING_FIRST_DIM;
  }
  if (output->all_same_first_dim()) {
    return ShapeRepresentation::VARYING_LAST_DIM;
  }
  return ShapeRepresentation::VARYING_BOTH_DIMS;
}

inline void check_grouped_fp8_current_scaling_tensors(const GroupedTensor *input,
                                                      const GroupedTensor *output) {
  NVTE_CHECK(input->num_tensors == output->num_tensors,
             "Number of input and output tensors must be same.");
  NVTE_CHECK(input->logical_shape.ndim == 2, "Input grouped tensor logical shape must be 2D.");
  NVTE_CHECK(output->logical_shape.ndim == 2, "Output grouped tensor logical shape must be 2D.");
  NVTE_CHECK(input->logical_shape.data[0] == output->logical_shape.data[0] &&
                 input->logical_shape.data[1] == output->logical_shape.data[1],
             "Input and output grouped tensor logical shapes must match.");
  NVTE_CHECK(input->has_data(), "Cannot quantize grouped tensor without rowwise input data.");
  NVTE_CHECK(!is_fp8_dtype(input->dtype()), "Input must be in higher precision.");
  NVTE_CHECK(output->has_data() || output->has_columnwise_data(),
             "Either rowwise or columnwise output data need to be allocated.");
  NVTE_CHECK(is_fp8_dtype(output->dtype()), "Output must have FP8 type.");
  NVTE_CHECK(output->amax.dptr != nullptr && output->amax.dtype == DType::kFloat32 &&
                 output->amax.numel() == output->num_tensors,
             "Grouped FP8 current scaling requires one FP32 amax per tensor.");
  NVTE_CHECK(output->scale.dptr != nullptr && output->scale.dtype == DType::kFloat32 &&
                 output->scale.numel() == output->num_tensors,
             "Grouped FP8 current scaling requires one FP32 scale per tensor.");
  if (output->has_data()) {
    NVTE_CHECK(output->scale_inv.dptr != nullptr && output->scale_inv.dtype == DType::kFloat32 &&
                   output->scale_inv.numel() == output->num_tensors,
               "Grouped FP8 current scaling requires one rowwise FP32 scale_inv per tensor.");
  }
  if (output->has_columnwise_data()) {
    NVTE_CHECK(output->columnwise_scale_inv.dptr != nullptr &&
                   output->columnwise_scale_inv.dtype == DType::kFloat32 &&
                   output->columnwise_scale_inv.numel() == output->num_tensors,
               "Grouped FP8 current scaling requires one columnwise FP32 scale_inv per tensor.");
  }

  const ShapeRepresentation shape_rep = get_shape_representation(output);
  NVTE_CHECK(shape_rep == ShapeRepresentation::SAME_BOTH_DIMS ||
                 shape_rep == ShapeRepresentation::VARYING_FIRST_DIM,
             "Grouped FP8 current scaling supports same-shape tensors or varying first "
             "dimensions with a common last dimension.");
  if (shape_rep == ShapeRepresentation::VARYING_FIRST_DIM) {
    NVTE_CHECK(output->first_dims.dptr != nullptr && output->first_dims.dtype == DType::kInt64 &&
                   output->first_dims.numel() == output->num_tensors,
               "Grouped FP8 current scaling requires int64 first_dims for varying first dims.");
    NVTE_CHECK(output->tensor_offsets.dptr != nullptr &&
                   output->tensor_offsets.dtype == DType::kInt64 &&
                   output->tensor_offsets.numel() == output->num_tensors + 1,
               "Grouped FP8 current scaling requires int64 tensor_offsets for varying first dims.");
  }
}

inline bool is_current_scaling_grouped_output(const GroupedTensor *output) {
  return output->scaling_mode == NVTE_DELAYED_TENSOR_SCALING && output->scale.dptr != nullptr;
}

inline void group_quantize(const GroupedTensor *input, const Tensor *noop, GroupedTensor *output,
                           const QuantizationConfig *quant_config, cudaStream_t stream) {
  CheckNoopTensor(*noop, "cast_noop");
  check_grouped_fp8_current_scaling_tensors(input, output);

  const ShapeRepresentation shape_rep = get_shape_representation(output);
  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
      input->dtype(), IType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
          output->dtype(), OType,
          TRANSFORMER_ENGINE_GROUP_TENSOR_SHAPE_REPRESENTATION_SWITCH(
              shape_rep, SHAPE_REP,
              {
                if constexpr (SHAPE_REP != ShapeRepresentation::SAME_BOTH_DIMS &&
                              SHAPE_REP != ShapeRepresentation::VARYING_FIRST_DIM) {
                  NVTE_ERROR("Unsupported grouped FP8 current-scaling shape representation.");
                } else {
                  launch_group_quantize<SHAPE_REP, IType, OType>(input, noop, output,
                                                                 quant_config, stream);
                }
              });););  // NOLINT(*)
}

}  // namespace group_quantize_current_scaling
}  // namespace fp8
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_GROUP_QUANTIZE_FP8_CUH_
